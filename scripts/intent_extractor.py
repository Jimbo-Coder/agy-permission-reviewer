#!/usr/bin/env python3
"""Tier 1 Zero-Token Verbal Grant Extractor.

Scans only `USER_INPUT` entries in `transcript.jsonl` (plus `lastUserInput`) to
extract explicit user authorizations for paths or commands without invoking an LLM.
Distinguishes read-only vs read-write grants and honors negative prohibitions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

USER_REQUEST_RE = re.compile(r"<USER_REQUEST>(.*?)</USER_REQUEST>", re.DOTALL)
PATH_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9._\-:/)\]])(?:~|/(?!/)|\.\./)[A-Za-z0-9._/\-]+")

NEGATIVE_CUE_RE = re.compile(
    r"\b(?:do\s+not|don't|never|avoid|forbid|prohibit|except|not\s+touch|hands\s+off)\b",
    re.IGNORECASE,
)
READONLY_CUE_RE = re.compile(
    r"\b(?:read[- ]?only|inspect\s+only|only\s+read|do\s+not\s+(?:edit|write|modify|push))\b",
    re.IGNORECASE,
)
WRITE_GRANT_CUE_RE = re.compile(
    r"\b(?:edit|write|modify|create|make|update|delete|handle|move|copy\s+to|save\s+to|work\s+in|work\s+on|use)\b",
    re.IGNORECASE,
)
READ_GRANT_CUE_RE = re.compile(
    r"\b(?:read|inspect|check|look\s+at|view|compare|diff|clone|fetch|pull|reference)\b",
    re.IGNORECASE,
)
COMMAND_GRANT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mpirun", re.compile(r"\b(?:mpirun|mpiexec)\b", re.IGNORECASE)),
    ("uv", re.compile(r"\buv\s+run\b", re.IGNORECASE)),
    ("cmake", re.compile(r"\b(?:cmake|ctest|ninja)\b", re.IGNORECASE)),
    ("git_push", re.compile(r"\b(?:push\s+to\s+(?:github|origin|remote)|git\s+push)\b", re.IGNORECASE)),
]


def extract_clean_user_text(raw_content: str) -> str:
    """Strip <USER_REQUEST> and <ADDITIONAL_METADATA> wrappers."""
    match = USER_REQUEST_RE.search(raw_content)
    if match:
        return match.group(1).strip()
    return raw_content.strip()


def normalize_path(path_str: str, cwd: str = "") -> str:
    """Expand `~` and resolve relative paths without requiring the target to exist."""
    expanded = os.path.expanduser(path_str.rstrip(".,;:)'\"`"))
    if not os.path.isabs(expanded) and cwd:
        expanded = os.path.join(cwd, expanded)
    return os.path.normpath(expanded)


def extract_grants_from_text(
    text: str,
    step_index: int = 0,
    cwd: str = "",
) -> dict[str, Any]:
    """Parse a single user message into structured path and command grants/prohibitions."""
    path_grants: list[dict[str, Any]] = []
    denied_paths: list[str] = []
    command_grants: list[str] = []

    clauses = [
        c.strip()
        for c in re.split(r"(?:[.!?\n;]+|,\s+(?:and|but|while|whereas|or)\b)", text, flags=re.IGNORECASE)
        if c.strip()
    ]
    for clause in clauses:
        has_neg = bool(NEGATIVE_CUE_RE.search(clause))
        is_ro = bool(READONLY_CUE_RE.search(clause))
        has_write = bool(WRITE_GRANT_CUE_RE.search(clause))
        has_read = bool(READ_GRANT_CUE_RE.search(clause))

        for match in PATH_TOKEN_RE.finditer(clause):
            raw_path = match.group(0)
            if len(raw_path) < 3:
                continue
            # Single-segment `/word` tokens in prose are chat slash commands (`/compact`, `/config`, `/settings`),
            # not root filesystem grants, unless the user includes a second slash or Known Top-Level OS Dir.
            if (
                raw_path.startswith("/")
                and "/" not in raw_path[1:]
                and raw_path not in ("/tmp", "/opt", "/var", "/usr", "/etc", "/Users", "/Volumes", "/Library")
            ):
                continue
            norm = normalize_path(raw_path, cwd=cwd)
            if has_neg and not is_ro:
                denied_paths.append(norm)
                continue
            mode = "read" if (is_ro or (has_read and not has_write)) else "write"
            path_grants.append(
                {
                    "path": norm,
                    "mode": mode,
                    "step_index": step_index,
                    "clause": clause[:160],
                }
            )

        if not has_neg:
            for cmd_key, pat in COMMAND_GRANT_PATTERNS:
                if pat.search(clause) and cmd_key not in command_grants:
                    command_grants.append(cmd_key)

    return {
        "path_grants": path_grants,
        "denied_paths": denied_paths,
        "command_grants": command_grants,
    }


def get_data_dir() -> Path:
    """Resolve the persistent plugin_data directory consistently across hook, CLI, and MCP calls."""
    env_dir = os.environ.get("PLUGIN_DATA")
    if env_dir:
        base = Path(env_dir)
    else:
        default_dir = (
            Path.home() / ".gemini" / "antigravity-cli" / "plugin_data" / "agy-permission-reviewer"
        )
        try:
            default_dir.mkdir(parents=True, exist_ok=True)
            base = default_dir
        except OSError:
            base = Path(tempfile.gettempdir()) / "agy-permission-reviewer"
    base.mkdir(parents=True, exist_ok=True)
    return base


def safe_conv_id(conversation_id: str) -> str:
    return "".join(c for c in str(conversation_id) if c.isalnum() or c in "-_") or "default"


def _ledger_cache_file(transcript_path: Path) -> Path:
    cache_dir = get_data_dir() / "ledger_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(transcript_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


from datetime import datetime

METADATA_TIME_RE = re.compile(r"The current local time is:\s*([0-9T:\-+Z.]+)")


def _parse_iso_seconds(ts_str: str) -> float | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _extract_step_timestamp(obj: dict[str, Any]) -> float | None:
    ts = _parse_iso_seconds(str(obj.get("created_at", "")))
    if ts is not None:
        return ts
    m = METADATA_TIME_RE.search(str(obj.get("content", "")))
    if m:
        return _parse_iso_seconds(m.group(1))
    return None


def select_active_user_messages(
    entries: list[tuple[int, str, float | None]],
    max_step_gap: int = 20,
    max_seconds_gap: float = 900.0,
    max_count: int = 3,
) -> list[str]:
    """Keep the latest user message plus earlier messages only if they belong to the same
    recent interaction (within `max_step_gap` steps and `max_seconds_gap` seconds, or when
    the newer message is a brief continuation reply like 'yes' / 'continue').
    """
    if not entries:
        return []
    selected: list[tuple[int, str, float | None]] = [entries[-1]]
    for prev in reversed(entries[:-1]):
        if len(selected) >= max_count:
            break
        curr = selected[-1]
        curr_text = curr[1].strip()
        is_short_followup = len(curr_text) < 40 and bool(
            re.match(
                r"^(?:yes|yep|yeah|sure|ok|okay|proceed|continue|keep going|go on|go ahead|do it|sounds good|lgtm)\b",
                curr_text,
                re.IGNORECASE,
            )
        )
        step_gap = curr[0] - prev[0]
        time_gap = (
            (curr[2] - prev[2])
            if (curr[2] is not None and prev[2] is not None and curr[2] >= prev[2])
            else None
        )
        if is_short_followup:
            selected.append(prev)
            continue
        if step_gap > max_step_gap:
            break
        if time_gap is not None and time_gap > max_seconds_gap:
            break
        selected.append(prev)
    selected.reverse()
    return [item[1] for item in selected]


def collect_session_ledger(
    transcript_path: Path | None,
    last_user_input: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    """Incrementally scan USER_INPUT and CHECKPOINT steps in transcript.jsonl and build the active verbal grant ledger."""
    user_messages: list[tuple[int, str, float | None]] = []
    checkpoint_summaries: list[str] = []
    if transcript_path and transcript_path.is_file():
        try:
            st = transcript_path.stat()
            cache_file = _ledger_cache_file(transcript_path)
            start_offset = 0
            if cache_file.is_file():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    if (
                        int(cached.get("inode", -1)) == st.st_ino
                        and 0 <= int(cached.get("offset", 0)) <= st.st_size
                    ):
                        start_offset = int(cached["offset"])
                        user_messages = [
                            (
                                int(item[0]),
                                str(item[1]),
                                float(item[2]) if len(item) > 2 and item[2] is not None else None,
                            )
                            for item in cached.get("user_messages", [])
                        ]
                        checkpoint_summaries = [
                            str(s) for s in cached.get("checkpoint_summaries", [])
                        ]
                except Exception:
                    start_offset = 0

            if st.st_size > start_offset:
                with transcript_path.open("rb") as fh:
                    fh.seek(start_offset)
                    chunk = fh.read()
                last_nl = chunk.rfind(b"\n")
                if last_nl != -1:
                    complete_chunk = chunk[: last_nl + 1]
                    new_offset = start_offset + len(complete_chunk)
                else:
                    complete_chunk = chunk
                    new_offset = start_offset + len(chunk)

                for raw_line in complete_chunk.splitlines():
                    if b'"USER_INPUT"' not in raw_line and b'"CHECKPOINT"' not in raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    step_type = obj.get("type", "")
                    step_idx = int(obj.get("step_index", 0))
                    if step_type == "USER_INPUT":
                        clean = extract_clean_user_text(str(obj.get("content", "")))
                        if clean:
                            user_messages.append((step_idx, clean, _extract_step_timestamp(obj)))
                    elif "CHECKPOINT" in step_type:
                        cp_text = str(obj.get("content", "")).strip()
                        if cp_text:
                            checkpoint_summaries.append(cp_text[:2500])

                try:
                    cache_file.write_text(
                        json.dumps(
                            {
                                "inode": st.st_ino,
                                "offset": new_offset,
                                "user_messages": user_messages,
                                "checkpoint_summaries": checkpoint_summaries[-3:],
                            }
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
        except OSError:
            pass

    if last_user_input:
        clean_last = extract_clean_user_text(last_user_input)
        if clean_last and (not user_messages or user_messages[-1][1] != clean_last):
            next_idx = (user_messages[-1][0] + 1) if user_messages else 0
            user_messages.append((next_idx, clean_last, None))

    path_map: dict[str, dict[str, Any]] = {}
    denied_set: set[str] = set()
    commands_set: set[str] = set()

    for step_idx, msg, _ in user_messages:
        extracted = extract_grants_from_text(msg, step_index=step_idx, cwd=cwd)
        for dp in extracted["denied_paths"]:
            denied_set.add(dp)
            path_map.pop(dp, None)
        for pg in extracted["path_grants"]:
            p = pg["path"]
            if p in denied_set:
                continue
            path_map[p] = pg
        for cg in extracted["command_grants"]:
            commands_set.add(cg)

    recent_msgs = select_active_user_messages(user_messages)
    if checkpoint_summaries and not recent_msgs:
        recent_msgs = [f"[Active Session Summary]: {checkpoint_summaries[-1][:1500]}"]

    return {
        "path_grants": list(path_map.values()),
        "denied_paths": sorted(denied_set),
        "command_grants": sorted(commands_set),
        "recent_user_messages": recent_msgs,
        "all_user_entries": [(step_idx, msg) for step_idx, msg, _ in user_messages],
        "last_user_step_index": user_messages[-1][0] if user_messages else 0,
    }


def verify_cited_user_authorization(
    auth_ref: str,
    all_user_entries: list[tuple[int, str]],
) -> dict[str, Any]:
    """Verify an agent's `# auth: <step# or exact user substring>` against real USER_INPUT records in transcript.jsonl."""
    ref = auth_ref.strip().strip("\"'")
    if not ref:
        return {}
    step_m = re.match(r"^(?:step\s*#?\s*)?(\d+)$", ref, re.IGNORECASE)
    if step_m:
        target_step = int(step_m.group(1))
        for step_idx, msg in all_user_entries:
            if step_idx == target_step:
                return {
                    "verifiedFromTranscript": True,
                    "stepIndex": step_idx,
                    "exactUserMessage": msg[:500],
                }
        return {
            "verifiedFromTranscript": False,
            "warning": f"Agent cited step #{target_step}, which does not exist as a USER_INPUT in transcript.jsonl",
        }
    if len(ref) >= 5:
        needle = ref.lower()
        for step_idx, msg in reversed(all_user_entries):
            if needle in msg.lower():
                return {
                    "verifiedFromTranscript": True,
                    "stepIndex": step_idx,
                    "exactUserMessage": msg[:500],
                }
    return {
        "verifiedFromTranscript": False,
        "warning": f"Agent cited phrase {ref!r}, which was NOT found in any USER_INPUT in transcript.jsonl",
    }


def is_path_under(target: str, parent: str) -> bool:
    """Return True if `target` is equal to or inside directory `parent`."""
    t = os.path.normpath(target)
    p = os.path.normpath(parent)
    if t == p:
        return True
    return t.startswith(p.rstrip("/") + "/")
