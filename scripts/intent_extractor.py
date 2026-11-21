#!/usr/bin/env python3
"""Tier 1 Zero-Token Verbal Grant Extractor.

Scans only `USER_INPUT` entries in `transcript.jsonl` (plus `lastUserInput`) to
extract explicit user authorizations for paths or commands without invoking an LLM.
Distinguishes read-only vs read-write grants and honors negative prohibitions.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any

USER_REQUEST_RE = re.compile(r"<USER_REQUEST>(.*?)</USER_REQUEST>", re.DOTALL)
PATH_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9._\-:/)\]])(?:~|/(?!/)|\.\./)[A-Za-z0-9._/\-]+")
PROSE_PATH_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9._\-:/)\]])(?:~/|/(?!/)|\.\./)[A-Za-z0-9._/\-]+")

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


def _is_valid_prose_grant_target(norm_path: str) -> bool:
    """Reject root/home/shallow system dirs so accidental mentions cannot mint broad grants."""
    p = Path(norm_path)
    if not os.path.isabs(norm_path):
        return False
    if is_path_under(norm_path, "/private/tmp"):
        return len(p.parts) >= 4
    if is_path_under(norm_path, "/tmp"):
        return len(p.parts) >= 3
    home = Path.home()
    try:
        if p == home or p.resolve() == home.resolve():
            return False
    except OSError:
        if p == home:
            return False
    return len(p.parts) >= 3


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

        if has_neg or is_ro or has_write or has_read:
            for match in PROSE_PATH_TOKEN_RE.finditer(clause):
                raw_path = match.group(0)
                if len(raw_path) < 3:
                    continue
                if (
                    raw_path.startswith("/")
                    and "/" not in raw_path[1:]
                    and raw_path not in ("/tmp", "/opt", "/var", "/usr", "/etc", "/Users", "/Volumes", "/Library")
                ):
                    continue
                norm = normalize_path(raw_path, cwd=cwd)
                if not _is_valid_prose_grant_target(norm):
                    continue
                if has_neg and not is_ro:
                    denied_paths.append(norm)
                    continue
                mode = "write" if (has_write and not is_ro and not has_neg) else "read"
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
LEDGER_CACHE_VERSION = 3
SHORT_AFFIRMATION_RE = re.compile(
    r"^(?:yes|yep|yeah|yup|sure|ok|okay|proceed|continue|keep going|go on|go ahead|do it|sounds good|lgtm|approved|i'?ve allowed|already allowed)\b",
    re.IGNORECASE,
)
EXPLICIT_PERMISSION_REPLY_RE = re.compile(
    r"(?:^(?:yes|yep|yeah|yup|sure|ok|okay|proceed|continue|keep going|go on|go ahead|do it|sounds good|lgtm|approved|i'?ve allowed|already allowed)\b"
    r"|\b(?:explicit(?:ly)?\s+(?:permission|approval|approv\w+|authoriz\w+)|give\s+you\s+(?:explicit\s+)?(?:permission|approval)|giving\s+you\s+(?:explicit\s+)?(?:permission|approval)|you\s+have\s+(?:my\s+)?(?:explicit\s+)?(?:permission|approval)|i\s+(?:explicitly\s+)?(?:approve|authorize)\b))",
    re.IGNORECASE,
)

_RELEVANT_STOPWORDS = frozenset(
    {
        "tmp", "bin", "usr", "var", "opt", "etc", "home", "users", "user", "private", "local",
        "src", "lib", "git", "python", "python3", "bash", "zsh", "ssh", "scp", "sftp", "rsync",
        "curl", "wget", "cat", "head", "tail", "grep", "find", "sed", "awk", "echo", "true",
        "false", "test", "tests", "log", "logs", "file", "files", "main", "run", "check",
        "status", "diff", "build", "dist", "out", "data", "config", "json", "yaml", "yml",
        "toml", "txt", "md", "py", "sh", "com", "org", "net", "http", "https", "none", "null",
        "purpose", "auth", "appeal", "review", "deep", "step", "latest", "target", "include",
        "command", "batchmode", "yes", "no",
    }
)


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


def _unpack_ledger_entry(
    entry: tuple[Any, ...],
) -> tuple[int, str, float | None, str]:
    return (
        int(entry[0]),
        str(entry[1]),
        entry[2] if len(entry) > 2 else None,
        str(entry[3]) if len(entry) > 3 and entry[3] is not None else "",
    )


def _select_active_user_entries(
    entries: list[tuple[Any, ...]],
    max_step_gap: int = 20,
    max_seconds_gap: float = 900.0,
    max_count: int = 5,
) -> list[tuple[int, str, float | None, str]]:
    if not entries:
        return []
    selected: list[tuple[int, str, float | None, str]] = [_unpack_ledger_entry(entries[-1])]
    for raw_prev in reversed(entries[:-1]):
        if len(selected) >= max_count:
            break
        curr = selected[-1]
        prev = _unpack_ledger_entry(raw_prev)
        curr_text = curr[1].strip()
        is_short_followup = len(curr_text) < 80 and bool(SHORT_AFFIRMATION_RE.match(curr_text))
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
    return selected


def select_active_user_messages(
    entries: list[tuple[Any, ...]],
    max_step_gap: int = 20,
    max_seconds_gap: float = 900.0,
    max_count: int = 5,
) -> list[str]:
    """Keep the latest user message plus earlier messages belonging to the same recent interaction.

    When a brief affirmative reply or explicit-permission reply follows an agent message,
    attaches a clearly delimited snippet of the agent's preceding proposal so Tier 2 sees what was
    confirmed without ever elevating agent text into user grants or clipping the user's words.
    """
    selected = _select_active_user_entries(
        entries,
        max_step_gap=max_step_gap,
        max_seconds_gap=max_seconds_gap,
        max_count=max_count,
    )
    out: list[str] = []
    for _, u_text, _, agent_tail in selected:
        stripped = u_text.strip()
        is_short = len(stripped) < 80 and bool(SHORT_AFFIRMATION_RE.match(stripped))
        is_explicit_perm = len(stripped) < 260 and bool(EXPLICIT_PERMISSION_REPLY_RE.search(stripped))
        if (is_short or is_explicit_perm) and agent_tail:
            prefix = "[Agent context before reply (not user words): ..."
            mid = "] -> [User]: "
            avail = max(120, min(280, 495 - len(prefix) - len(mid) - len(u_text)))
            snippet = agent_tail[-avail:]
            out.append(f"{prefix}{snippet}{mid}{u_text}")
        else:
            out.append(u_text)
    return out


def collect_session_ledger(
    transcript_path: Path | None,
    last_user_input: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    """Incrementally scan USER_INPUT, PLANNER_RESPONSE, and CHECKPOINT steps in transcript.jsonl."""
    user_messages: list[tuple[int, str, float | None, str]] = []
    checkpoint_summaries: list[str] = []
    last_agent_tail = ""
    latest_step_index = 0
    if transcript_path and transcript_path.is_file():
        try:
            st = transcript_path.stat()
            cache_file = _ledger_cache_file(transcript_path)
            start_offset = 0
            if cache_file.is_file():
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                    if (
                        int(cached.get("version", 1)) == LEDGER_CACHE_VERSION
                        and int(cached.get("inode", -1)) == st.st_ino
                        and 0 <= int(cached.get("offset", 0)) <= st.st_size
                    ):
                        start_offset = int(cached["offset"])
                        last_agent_tail = str(cached.get("last_agent_tail", ""))
                        latest_step_index = int(cached.get("latest_step_index", 0))
                        user_messages = [
                            _unpack_ledger_entry(tuple(item))
                            for item in cached.get("user_messages", [])
                        ]
                        checkpoint_summaries = [
                            str(s) for s in cached.get("checkpoint_summaries", [])
                        ]
                except Exception:
                    start_offset = 0
                    last_agent_tail = ""
                    latest_step_index = 0
                    user_messages = []
                    checkpoint_summaries = []

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
                    if (
                        b'"USER_INPUT"' not in raw_line
                        and b'"CHECKPOINT"' not in raw_line
                        and b'"PLANNER_RESPONSE"' not in raw_line
                    ):
                        continue
                    try:
                        obj = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    step_type = obj.get("type", "")
                    step_idx = int(obj.get("step_index", 0))
                    latest_step_index = max(latest_step_index, step_idx)
                    if step_type == "USER_INPUT":
                        clean = extract_clean_user_text(str(obj.get("content", "")))
                        if clean and not clean.startswith("Auto Permissions"):
                            user_messages.append(
                                (step_idx, clean, _extract_step_timestamp(obj), last_agent_tail)
                            )
                    elif step_type == "PLANNER_RESPONSE":
                        agent_text = " ".join(str(obj.get("content", "")).split()).strip()
                        if agent_text:
                            last_agent_tail = agent_text[-800:]
                    elif "CHECKPOINT" in step_type:
                        cp_text = str(obj.get("content", "")).strip()
                        if cp_text:
                            checkpoint_summaries.append(cp_text[:2500])

                try:
                    cache_file.write_text(
                        json.dumps(
                            {
                                "version": LEDGER_CACHE_VERSION,
                                "inode": st.st_ino,
                                "offset": new_offset,
                                "last_agent_tail": last_agent_tail,
                                "latest_step_index": latest_step_index,
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
        if (
            clean_last
            and not clean_last.startswith("Auto Permissions")
            and (not user_messages or user_messages[-1][1] != clean_last)
        ):
            next_idx = max(
                (user_messages[-1][0] + 1) if user_messages else 0,
                (latest_step_index + 1) if latest_step_index > 0 else 0,
            )
            user_messages.append((next_idx, clean_last, None, last_agent_tail))
            latest_step_index = max(latest_step_index, next_idx)

    path_map: dict[str, dict[str, Any]] = {}
    denied_set: set[str] = set()
    commands_set: set[str] = set()

    # Regex prose-grant extraction is disabled by default (Layer 2 LLM + session_dir_cache handles natural language);
    # opt in via AGY_ENABLE_PROSE_LEDGER=1 only if desired. Still scans strictly clean user text, never agent output.
    enable_prose_ledger = os.environ.get("AGY_ENABLE_PROSE_LEDGER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if enable_prose_ledger:
        for step_idx, msg, _, _ in user_messages:
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

    active_entries = _select_active_user_entries(user_messages)
    active_step_indices = [entry[0] for entry in active_entries]
    recent_msgs = select_active_user_messages(user_messages)
    if checkpoint_summaries and not recent_msgs:
        recent_msgs = [f"[Active Session Summary]: {checkpoint_summaries[-1][:1500]}"]

    return {
        "path_grants": list(path_map.values()),
        "denied_paths": sorted(denied_set),
        "command_grants": sorted(commands_set),
        "recent_user_messages": recent_msgs,
        "all_user_entries": [(step_idx, msg) for step_idx, msg, _, _ in user_messages],
        "all_user_entries_with_context": [
            (step_idx, msg, agent_tail) for step_idx, msg, _, agent_tail in user_messages
        ],
        "last_user_step_index": user_messages[-1][0] if user_messages else 0,
        "latest_step_index": latest_step_index,
        "active_user_step_indices": active_step_indices,
    }


def _extract_query_tokens(query_text: str) -> list[tuple[str, bool]]:
    """Return distinctive lowercase `(token, is_distinctive)` pairs from a tool target/command."""
    tokens: dict[str, bool] = {}

    def _add(tok: str, distinctive: bool = False) -> None:
        t = tok.strip("._-").lower() if not distinctive else tok.lower()
        if not t or t in _RELEVANT_STOPWORDS:
            return
        if distinctive:
            if len(t) >= 3:
                tokens[t] = True
            return
        if len(t) >= 4:
            tokens[t] = tokens.get(t, False)

    for hm in re.finditer(r"(?<![A-Za-z0-9._-])([A-Za-z0-9._-]{3,}):(?:/|\\|~)", query_text):
        host = hm.group(1).lower()
        if host not in ("http", "https", "file", "ssh", "scp", "sftp", "rsync"):
            _add(host, distinctive=True)

    for raw in re.split(r"[^A-Za-z0-9._/-]+", query_text):
        if not raw:
            continue
        if "/" in raw:
            base = raw.rsplit("/", 1)[-1]
            if base and base not in _RELEVANT_STOPWORDS.union({"."} ):
                is_dist = ("." in base or "-" in base) and len(base) >= 3
                _add(base, distinctive=is_dist)
                if "." in base:
                    stem = base.split(".", 1)[0]
                    _add(stem, distinctive=False)
        for part in raw.split("/"):
            if not part:
                continue
            is_dist = ("." in part or "-" in part) and len(part) >= 3
            _add(part, distinctive=is_dist)
            if "." in part:
                _add(part.split(".", 1)[0], distinctive=False)

    return list(tokens.items())


def _token_in_text(tok: str, text_low: str) -> bool:
    if len(tok) <= 4:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", text_low))
    return tok in text_low


def extract_relevant_earlier_user_turns(
    query_text: str,
    all_user_entries_with_context: list[tuple[Any, ...]],
    exclude_step_indices: set[int] | list[int] | None = None,
    max_matches: int = 2,
) -> list[str]:
    """Zero-token retrieval of older USER_INPUT turns relevant to the current command/targets."""
    if not query_text or not all_user_entries_with_context:
        return []
    q_tokens = _extract_query_tokens(query_text)
    if not q_tokens:
        return []
    exclude_set = set(exclude_step_indices or ())
    scored: list[tuple[int, int, str]] = []
    for raw_entry in all_user_entries_with_context:
        sidx = int(raw_entry[0])
        if sidx in exclude_set:
            continue
        msg = str(raw_entry[1])
        agent_tail = str(raw_entry[2]) if len(raw_entry) > 2 and raw_entry[2] is not None else ""
        msg_low = msg.lower()
        tail_low = agent_tail.lower() if agent_tail else ""
        score = 0
        matched_in_msg: set[str] = set()
        for tok, _ in q_tokens:
            if _token_in_text(tok, msg_low):
                score += 2
                matched_in_msg.add(tok)
        has_explicit_confirm = bool(EXPLICIT_PERMISSION_REPLY_RE.search(msg))
        if has_explicit_confirm and tail_low:
            for tok, _ in q_tokens:
                if tok not in matched_in_msg and _token_in_text(tok, tail_low):
                    score += 1
        if score < 2:
            continue
        if agent_tail and has_explicit_confirm:
            formatted = (
                f"[Earlier relevant user turn (step #{sidx}) — Agent proposal (untrusted): "
                f"...{agent_tail[-220:]} -> [User]: {msg[:220]}]"
            )
        else:
            formatted = f"[Earlier relevant user turn (step #{sidx})]: {msg[:420]}"
        scored.append((score, sidx, formatted))

    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max_matches]]


def verify_cited_user_authorization(
    auth_ref: str,
    all_user_entries: list[tuple[Any, ...]],
) -> dict[str, Any]:
    """Verify an agent's `# auth: <step# or exact user substring>` against real USER_INPUT records in transcript.jsonl."""
    ref = auth_ref.strip().strip("\"'")
    if not ref:
        return {}
    unpacked: list[tuple[int, str, str]] = [
        (
            int(entry[0]),
            str(entry[1]),
            str(entry[2]) if len(entry) > 2 and entry[2] is not None else "",
        )
        for entry in all_user_entries
    ]
    step_m = re.match(r"^(?:step\s*#?\s*)?(\d+)$", ref, re.IGNORECASE)
    if step_m:
        target_step = int(step_m.group(1))
        for step_idx, msg, agent_tail in unpacked:
            if step_idx == target_step:
                out: dict[str, Any] = {
                    "verifiedFromTranscript": True,
                    "stepIndex": step_idx,
                    "exactUserMessage": msg[:500],
                }
                if agent_tail:
                    out["precedingAgentProposal"] = agent_tail[-500:]
                return out
        return {
            "verifiedFromTranscript": False,
            "warning": f"Agent cited step #{target_step}, which does not exist as a USER_INPUT in transcript.jsonl",
        }
    if len(ref) >= 5:
        needle = ref.lower()
        for step_idx, msg, agent_tail in reversed(unpacked):
            if needle in msg.lower():
                out = {
                    "verifiedFromTranscript": True,
                    "stepIndex": step_idx,
                    "exactUserMessage": msg[:500],
                }
                if agent_tail:
                    out["precedingAgentProposal"] = agent_tail[-500:]
                return out
    return {
        "verifiedFromTranscript": False,
        "warning": f"Agent cited phrase {ref!r}, which was NOT found in any USER_INPUT in transcript.jsonl",
    }


def is_path_under(target: str, parent: str) -> bool:
    """Return True if `target` is equal to or inside directory `parent`."""
    t = os.path.normcase(os.path.normpath(target))
    p = os.path.normcase(os.path.normpath(parent))
    if t == p:
        return True
    sep = os.sep
    p_clean = p.rstrip("/\\")
    if not p_clean:
        return t.startswith("/")
    if len(p_clean) == 2 and p_clean[1] == ":" and sep == "\\":
        return t.startswith(p_clean + "\\")
    return t.startswith(p_clean + sep)


_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WIN_DRIVE_START_RE = re.compile(r"^[A-Za-z]:[/\\]")
_WIN_DRIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_:/])[A-Za-z]:[/\\][^\s\"';&|<>()]+")
_SHELL_META_CHARS = frozenset("\"'\\;&|<>()\n#")
_PUNCT_CHARS = "();<>|&\n"
_SHELL_OP_RE = re.compile(r"&>>|&>|>>|<<<|<<|<&|<>|>&|>\||\|\||&&|\|&|[();<>|&\n]")
_REDIRECT_OPS = frozenset({"<", ">", ">>", "&>", "&>>", "<>", ">|", "<&", ">&", "<<<"})
_OUTPUT_REDIRECT_OPS = frozenset({">", ">>", "&>", "&>>", "<>", ">|", ">&"})
_HEREDOC_TAIL_RE = re.compile(
    r"(-?)[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|((?:\\.|[^\s;&|<>()#\"'])+))"
)


def extract_paths_from_string(
    s: str,
    cwd: str = "",
    allow_bare_tilde_or_root: bool = False,
) -> list[str]:
    """Extract normalized filesystem paths from a shell argument or Python string value."""
    if not s:
        return []
    if s.startswith(("@/", "@~/", "@../")):
        s = s[1:]
    if _URL_SCHEME_RE.match(s):
        low = s.lower()
        if low.startswith("file://localhost/"):
            s = s[16:]
        elif low.startswith("file://"):
            s = s[7:]
        else:
            return []
        if not s:
            return []
    if allow_bare_tilde_or_root and s in ("~", "/", ".."):
        return [normalize_path(s, cwd=cwd)]
    if s in ("~", ".."):
        return [normalize_path(s, cwd=cwd)]
    is_win_abs = bool(_WIN_DRIVE_START_RE.match(s))
    if (
        (s.startswith(("/", "~/", "../")) or is_win_abs)
        and not s.startswith("//")
        and "\n" not in s
        and len(s.strip("/\\")) > 0
    ):
        if " " not in s and "\t" not in s:
            return [normalize_path(s, cwd=cwd)]
        has_shell_op = any(op in s for op in ("&&", "||", ";", "|"))
        has_whitespace_path = any(
            m.start() > 0 and s[m.start() - 1].isspace()
            for m in PATH_TOKEN_RE.finditer(s)
        ) or any(
            m.start() > 0 and s[m.start() - 1].isspace()
            for m in _WIN_DRIVE_PATH_RE.finditer(s)
        )
        if not has_shell_op and not has_whitespace_path:
            return [normalize_path(s, cwd=cwd)]
    found = [
        normalize_path(m.group(0), cwd=cwd)
        for m in PATH_TOKEN_RE.finditer(s)
        if len(m.group(0)) >= 3
    ]
    if ":" in s:
        for wm in _WIN_DRIVE_PATH_RE.finditer(s):
            found.append(normalize_path(wm.group(0), cwd=cwd))
    return found


@functools.lru_cache(maxsize=64)
def _extract_python_code_paths_cached(code_str: str, cwd: str = "") -> tuple[str, ...]:
    """Parse Python inline code with `ast` to distinguish math division from Path/string paths."""
    import ast

    try:
        tree = ast.parse(code_str)
    except (SyntaxError, ValueError, RecursionError):
        return tuple(
            normalize_path(m.group(0), cwd=cwd)
            for m in PATH_TOKEN_RE.finditer(code_str)
            if len(m.group(0)) >= 3
        )

    def _call_name(func_node: ast.AST) -> str:
        parts: list[str] = []
        cur: ast.AST | None = func_node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return ""

    def _join_path_parts(parts_res: list[tuple[str, bool, list[ast.AST]] | None], raw_args: list[ast.AST]) -> tuple[str, list[ast.AST]]:
        cur = ""
        extra: list[ast.AST] = []
        for idx, (res, raw_node) in enumerate(zip(parts_res, raw_args)):
            if res is None:
                seg = "__EXPR__"
                extra.append(raw_node)
            else:
                seg = res[0]
                extra.extend(res[2])
            if not seg:
                continue
            if idx == 0:
                cur = seg
            elif seg.startswith("/"):
                cur = seg
            elif cur == "/":
                cur = f"/{seg}"
            elif not cur or cur == ".":
                cur = seg
            else:
                cur = f"{cur.rstrip('/')}/{seg}"
        return cur, extra

    def _eval_node(node: ast.AST) -> tuple[str, bool, list[ast.AST]] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, False, []

        if isinstance(node, ast.JoinedStr):
            chunks: list[str] = []
            extra: list[ast.AST] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    chunks.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    fv_res = _eval_node(v.value)
                    if fv_res is not None:
                        chunks.append(fv_res[0])
                        extra.extend(fv_res[2])
                    else:
                        chunks.append("__EXPR__")
                        extra.append(v.value)
            return "".join(chunks), False, extra

        if isinstance(node, ast.Call) and not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args):
            cname = _call_name(node.func)
            if cname in ("Path.home", "pathlib.Path.home") and not node.args:
                return "~", True, []
            if cname in ("Path", "PurePath", "PosixPath", "PurePosixPath", "pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"):
                if not node.args:
                    return ".", True, []
                arg_evals = [_eval_node(a) for a in node.args]
                if arg_evals[0] is not None:
                    joined, extra = _join_path_parts(arg_evals, list(node.args))
                    return joined or ".", True, extra
            if cname in ("os.path.join", "posixpath.join") and node.args:
                arg_evals = [_eval_node(a) for a in node.args]
                if arg_evals[0] is not None and not arg_evals[0][1]:
                    joined, extra = _join_path_parts(arg_evals, list(node.args))
                    return joined, False, extra
            if cname in ("os.path.expanduser", "posixpath.expanduser") and len(node.args) == 1:
                arg_res = _eval_node(node.args[0])
                if arg_res is not None:
                    return arg_res

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                l_res = _eval_node(node.left)
                r_res = _eval_node(node.right)
                # Only treat `/` as path joining when at least one side is a Path object.
                if (l_res and l_res[1]) or (r_res and r_res[1]):
                    extra: list[ast.AST] = []
                    if l_res is None:
                        l_str = "__EXPR__"
                        extra.append(node.left)
                    else:
                        l_str = l_res[0]
                        extra.extend(l_res[2])
                    if r_res is None:
                        r_str = "__EXPR__"
                        extra.append(node.right)
                    else:
                        r_str = r_res[0]
                        extra.extend(r_res[2])
                    if r_str.startswith("/"):
                        joined = r_str
                    elif l_str == "/":
                        joined = f"/{r_str}"
                    elif not l_str or l_str == ".":
                        joined = r_str
                    else:
                        joined = f"{l_str.rstrip('/')}/{r_str}"
                    return joined, True, extra

            if isinstance(node.op, ast.Add):
                l_res = _eval_node(node.left)
                r_res = _eval_node(node.right)
                if l_res is not None and r_res is not None and not l_res[1] and not r_res[1]:
                    return l_res[0] + r_res[0], False, l_res[2] + r_res[2]
                if (
                    l_res is not None
                    and r_res is None
                    and not l_res[1]
                    and l_res[0].startswith(("/", "~/", "../"))
                ):
                    return l_res[0] + "__EXPR__", False, l_res[2] + [node.right]

        return None

    extracted: list[str] = []

    def _walk(node: ast.AST) -> None:
        res = _eval_node(node)
        if res is not None:
            s, is_path_obj, extra_nodes = res
            found = extract_paths_from_string(s, cwd=cwd, allow_bare_tilde_or_root=is_path_obj)
            if not found and is_path_obj and s not in ("", ".") and (bool(cwd) or s.startswith(("~", "/", ".."))):
                found = [normalize_path(s, cwd=cwd)]
            extracted.extend(found)
            for ex in extra_nodes:
                _walk(ex)
            return
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    return tuple(extracted)


def extract_python_code_paths(code_str: str, cwd: str = "") -> list[str]:
    """Return paths extracted from inline Python code via cached AST inspection."""
    return list(_extract_python_code_paths_cached(code_str, cwd))


def _scan_line_for_heredocs(
    line: str,
    in_sq: bool,
    in_dq: bool,
) -> tuple[list[tuple[bool, str]], bool, bool]:
    """Scan one shell line for unquoted `<<` / `<<-` heredoc delimiters while tracking quotes."""
    delims: list[tuple[bool, str]] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_sq:
            if ch == "'":
                in_sq = False
            i += 1
            continue
        if in_dq:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_dq = False
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch == "'":
            in_sq = True
            i += 1
            continue
        if ch == '"':
            in_dq = True
            i += 1
            continue
        if ch == "#" and (i == 0 or line[i - 1].isspace() or line[i - 1] in ";|&()<>"):
            break
        if (
            ch == "<"
            and i + 1 < n
            and line[i + 1] == "<"
            and (i + 2 >= n or line[i + 2] != "<")
            and (i == 0 or line[i - 1] != "<")
        ):
            hm = _HEREDOC_TAIL_RE.match(line[i + 2 :])
            if hm:
                strip_tabs = bool(hm.group(1))
                delim = hm.group(2) or hm.group(3) or re.sub(r"\\(.)", r"\1", hm.group(4))
                delims.append((strip_tabs, delim))
                i += 2 + hm.end()
                continue
        i += 1
    return delims, in_sq, in_dq


def _split_heredocs(code_only: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate heredoc body blocks from shell header lines so literal bodies aren't lexed as shell."""
    if "<<" not in code_only or "\n" not in code_only:
        return code_only, []
    lines = code_only.splitlines()
    shell_lines: list[str] = []
    heredocs: list[tuple[str, str]] = []
    pending_delims: list[tuple[str, bool, str]] = []
    cur_body: list[str] = []
    in_sq = False
    in_dq = False

    for line in lines:
        if pending_delims:
            hdr, strip_tabs, delim = pending_delims[0]
            cand = line.lstrip("\t") if strip_tabs else line
            if cand == delim:
                heredocs.append((hdr, "\n".join(cur_body)))
                cur_body = []
                pending_delims.pop(0)
            else:
                cur_body.append(line)
            continue

        shell_lines.append(line)
        found, in_sq, in_dq = _scan_line_for_heredocs(line, in_sq, in_dq)
        for strip_tabs, delim in found:
            pending_delims.append((line, strip_tabs, delim))

    if pending_delims:
        hdr, _, _ = pending_delims[0]
        heredocs.append((hdr, "\n".join(cur_body)))

    return "\n".join(shell_lines), heredocs


def _strip_shell_comments(text: str) -> str:
    """Strip unquoted POSIX `# ...` comments so apostrophes inside comments never trip shlex."""
    if "#" not in text:
        return text
    out: list[str] = []
    in_sq = False
    in_dq = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_sq:
            out.append(ch)
            if ch == "'":
                in_sq = False
            i += 1
        elif in_dq:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_dq = False
            i += 1
        elif ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(text[i + 1])
            i += 2
        elif ch == "'":
            in_sq = True
            out.append(ch)
            i += 1
        elif ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
        elif ch == "#" and (i == 0 or text[i - 1] in " \t\r\n;|&()<>"):
            nl = text.find("\n", i)
            if nl == -1:
                break
            i = nl
        else:
            out.append(ch)
            i += 1
    return "".join(out)


@functools.lru_cache(maxsize=64)
def parse_shell_structure(code_only: str) -> dict[str, Any] | None:
    """Parse a shell command into immutable argv segments, raw segments, redirects, and heredocs.

    Returns `None` on malformed shell syntax (e.g., unclosed quotes) so callers can fail closed.
    """
    if not code_only or not code_only.strip():
        return {
            "argv_segments": (),
            "raw_segments": (),
            "redirects": (),
            "output_redirects": (),
            "heredocs": (),
        }
    if not any(c in code_only for c in _SHELL_META_CHARS):
        s = code_only.strip()
        return {
            "argv_segments": (tuple(s.split()),) if s else (),
            "raw_segments": (s,) if s else (),
            "redirects": (),
            "output_redirects": (),
            "heredocs": (),
        }

    cleaned_text, heredocs = _split_heredocs(code_only)
    cleaned_text = _strip_shell_comments(cleaned_text)
    lex = shlex.shlex(cleaned_text, posix=True, punctuation_chars=_PUNCT_CHARS)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""

    argv_segments: list[tuple[str, ...]] = []
    raw_segments: list[str] = []
    redirects: list[tuple[str, str]] = []
    output_redirects: list[tuple[str, str]] = []

    cur_argv: list[str] = []
    pending_op: str | None = None
    seg_start = 0
    prev_pos = 0

    def _flush_segment(end_pos: int) -> None:
        if cur_argv:
            argv_segments.append(tuple(cur_argv))
            cur_argv.clear()
        raw_seg = cleaned_text[seg_start:end_pos].strip()
        if raw_seg:
            raw_segments.append(raw_seg)

    try:
        while True:
            tok = lex.get_token()
            if tok is lex.eof:
                break
            cur_pos = lex.instream.tell() - len(lex._pushback_chars)
            raw_slice = cleaned_text[prev_pos:cur_pos]
            tok_start = prev_pos
            prev_pos = cur_pos

            if tok == "\n" and raw_slice.strip(" \t\r") in ("\\\n", "\\\r\n"):
                continue

            stripped_raw = raw_slice.strip(" \t\r")
            is_unquoted_op = (
                bool(tok)
                and all(c in _PUNCT_CHARS for c in tok)
                and stripped_raw == tok
            )

            if not is_unquoted_op:
                if pending_op is not None:
                    op = pending_op
                    pending_op = None
                    if op in ("<<", "<<<"):
                        continue
                    if op in ("<&", ">&") and (tok.isdigit() or tok == "-"):
                        continue
                    redirects.append((op, tok))
                    if op in _OUTPUT_REDIRECT_OPS:
                        output_redirects.append((op, tok))
                else:
                    cur_argv.append(tok)
                continue

            ops = _SHELL_OP_RE.findall(tok)
            if not ops:
                continue
            attached_left = bool(raw_slice) and not raw_slice[0].isspace()
            if (
                (ops[0] in _REDIRECT_OPS or ops[0] == "<<")
                and attached_left
                and cur_argv
                and cur_argv[-1].isdigit()
                and pending_op is None
            ):
                cur_argv.pop()

            has_separator = any(op not in _REDIRECT_OPS and op != "<<" for op in ops)
            for op in ops:
                if op in _REDIRECT_OPS or op == "<<":
                    pending_op = op
                else:
                    pending_op = None
            if has_separator:
                lead_ws = len(raw_slice) - len(raw_slice.lstrip(" \t\r"))
                trail_ws = len(raw_slice) - len(raw_slice.rstrip(" \t\r"))
                sep_start = tok_start + lead_ws
                sep_end = cur_pos - trail_ws
                _flush_segment(sep_start)
                seg_start = sep_end

    except ValueError:
        return None

    _flush_segment(len(cleaned_text))
    return {
        "argv_segments": tuple(argv_segments),
        "raw_segments": tuple(raw_segments),
        "redirects": tuple(redirects),
        "output_redirects": tuple(output_redirects),
        "heredocs": tuple(heredocs),
    }
