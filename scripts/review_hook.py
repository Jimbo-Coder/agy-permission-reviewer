#!/usr/bin/env python3
"""Antigravity PreToolUse Permission Reviewer.

Architecture (adapted from `opencode-auto-permissions` and `onkarbadve/agy-auto`):
1. Tier 0 Deterministic Security Gate & Segment Fast-Allow (~1 ms):
   - Hard-deny for catastrophic deletions, raw disk/partition tools, credential/secret
     snooping, shell init modification, package publishing, git force-push/.git deletion,
     pipe-to-shell/sudo/env hijacks, and protected system/config files.
   - Segment-by-segment Fast-Allow for metadata/status commands (`ls`, `git status`, test
     runners) and workspace/artifact/tmp-scoped operations (keeping content-reading tools
     like `cat`/`grep`/`head`/`tail` out of blind fast-allow unless their target paths
     resolve inside workspace/scratch/tmp, per `policy.ts`).
2. Tier 1 Zero-Token Verbal Grant Ledger (explicit user path/command grants in `transcript.jsonl`).
3. Tier 2 OpenCode `$0` Free Reviewer Pipeline (`opencode/jev-1.13-free` via
   `experimental.generate.decisions` -> `opencode-go/beam-free` / `opencode-go/longcat-2.5-free`
   via `experimental.generate.text` -> 3-strike agent send-back on `low_confidence` /
   `escalation_unsure` before `force_ask` user popup).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any

from intent_extractor import (
    PATH_TOKEN_RE,
    collect_session_ledger,
    get_data_dir,
    is_path_under,
    normalize_path,
    safe_conv_id,
)
from llm_classifier import (
    DEFAULT_DECISION_MODELS,
    DEFAULT_ESCALATION_MODELS,
    DEFAULT_MAX_SEND_BACKS,
    DEFAULT_MIN_CONFIDENCE,
    ROTATE_BYTES,
    apply_unresolved_send_back,
    compute_directory_grant_key,
    compute_request_key,
    compute_signature,
    extract_compact_transcript,
    get_cached_verdict,
    prune_directory_files,
    put_cached_verdict,
    run_escalation_tier,
    run_jev_decision_tier,
)

# Subshell / dynamic evaluation / redirection patterns that disqualify fast-allow
UNSAFE_SHELL_EXPANSION_RE = re.compile(r"`|\$\(|<\(|>\(|\beval\s|\bexec\s")
OUTPUT_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d*>>?|&>>?)\s*([^\s;&|]+)")
COMPOUND_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")

# Read-only inspection, local test/build runners, and local git commands (never push/reset/clean).
# Combined with `not outside_paths`, these fast-allow in <1 ms when all referenced paths are inside
# the active workspace, conversation scratch, or /tmp (including BypassSandbox=true for MPI/uv).
ROUTINE_SEGMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^git\s+(?:-C\s+\S+\s+)?(?:status|diff|log|show|rev-parse|remote\s+-v$|branch(?:\s+(?:--show-current|--list|-a|-v|-vv|-r))*$|add\b|commit\b|stash\b|check-ignore\b)(?:\s|$)"),
    re.compile(r"^uv\s+run\s+(?:python3?|pytest|ruff|mypy)\b"),
    re.compile(r"^python3?\s+(?:-m\s+(?:unittest|pytest|py_compile|ruff|mypy)\b|\S+\.py\b)"),
    re.compile(r"^(?:cmake|ctest|ninja|make)\b"),
    re.compile(r"^(?:npm|pnpm|yarn|bun)\s+(?:test|run\s+(?:test|lint|build|typecheck|check))(?:\s|$)"),
    re.compile(r"^cargo\s+(?:test|check|build)(?:\s|$)"),
    re.compile(r"^go\s+test(?:\s|$)"),
    re.compile(r"^(?:pytest|ruff\s+check|tsc)(?:\s|$)"),
    re.compile(r"^(?:ls|tree|wc|du|df|stat|file|which|pwd|whoami|uname|uptime|nproc|sw_vers|true|false)(?:\s|$)"),
    re.compile(r"^(?:hostname|date)(?:\s+(?:-[sf]|\+\S+))?$"),
]

# Content-reading and scoped file cleanup commands fast-allow only when target paths are explicitly present and in-scope
SCOPED_CONTENT_SEGMENT_RE = re.compile(r"^(?:cat|head|tail|sed|awk|grep|rg|find|sort|uniq|cut|tr|rm\s+-f)\b")

# Directory deletion or destructive working-tree commands must NEVER be fast-allowed by Tier 0 or
# Tier 1 path grants alone; they must always go to Tier 2 (LLM review against user messages) so
# they are blocked unless the user explicitly asked to delete/reset them in the conversation.
DESTRUCTIVE_SCOPE_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:(?:\/bin\/)?rm\s+-[^\s]*[rR]|(?:\/bin\/)?rmdir\b|find\b[^;&|]*\s-delete\b|shutil\.rmtree|git\s+(?:-C\s+\S+\s+)?(?:reset\s+--hard|clean\s+-[^\s]*f))"
)

HARD_DENY_COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Root, home, or parent recursive deletion",
        re.compile(
            r"(?:^|\s)rm\s+-[^\s]*(?:r[^\s]*f|f[^\s]*r)[^\s]*\s+(?:--\s+)?(?:[\"']?/(?:[\"']|\s|$|\*)|[\"']?~(?:[\"']|\s|$|/)|[\"']?\$HOME(?:[\"']|\s|$|/)|[\"']?\.\.(?:[\"']|\s|$|/))"
        ),
    ),
    (
        "Raw disk or partition destruction",
        re.compile(r"\b(?:mkfs|dd\s+if=|diskutil\s+(?:eraseDisk|partitionDisk)|fdisk|wipefs)\b"),
    ),
    (
        "Direct private SSH key or cloud secret file read",
        re.compile(r"(?:\.ssh/id_(?:rsa|ed25519|ecdsa|dsa)|\.aws/credentials|\.netrc|(?:^|/|\s)\.env(?:\.(?:local|prod|production|secret))?(?:\s|$|[\"']))"),
    ),
    (
        "Modifying Antigravity settings.json or config.json directly",
        re.compile(r"\.gemini/(?:antigravity-cli/settings\.json|config/(?:config|permission_reviewer\.local)\.json)"),
    ),
    (
        "Pipe-to-shell remote execution or dynamic library injection hijack",
        re.compile(r"(?:(?:curl|wget)\b[^;&|]*\|\s*(?:ba)?sh\b|(?:LD_PRELOAD|DYLD_INSERT_LIBRARIES)=)"),
    ),
    (
        "System account, firewall, or firmware modification",
        re.compile(r"\b(?:dscl|pwpolicy|pfctl|csrutil|nvram)\s+"),
    ),
]


def load_local_config() -> dict[str, Any]:
    """Load optional user-local policy overrides (`read_only_repos`, `known_ssh_hosts`, etc.)."""
    cfg_path = Path(
        os.environ.get("AGY_PERMISSION_REVIEWER_LOCAL_CONFIG")
        or os.path.expanduser("~/.gemini/config/permission_reviewer.local.json")
    )
    if not cfg_path.is_file():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_local_hard_deny_patterns(local_cfg: dict[str, Any] | None = None) -> list[tuple[str, re.Pattern[str]]]:
    cfg = local_cfg if local_cfg is not None else load_local_config()
    ro_repos = [str(r).strip() for r in cfg.get("read_only_repos", []) if str(r).strip()]
    if not ro_repos:
        return []
    alt = "|".join(re.escape(r) for r in ro_repos)
    pat = re.compile(
        rf"(?:(?:git\s+push|gh\s+(?:issue|pr|repo|release|api))\b[^;&|]*(?:{alt})|(?:{alt})[^;&|]*\bgit\s+push\b)",
        re.IGNORECASE,
    )
    return [
        (
            f"Write/push/PR/issue action targeting read-only repository ({', '.join(ro_repos)})",
            pat,
        )
    ]

# Actions on GitHub / public repositories or registries that modify external/public state
# Always require explicit user confirmation (`force_ask`) rather than silent auto-approval.
# Anchored to command segment start `(?:^|[;&|]\s*)` so commit messages (`git commit -m "... git push ..."`) do not false-positive.
FORCE_ASK_COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "GitHub issue/PR/review/comment/release/gist mutation requires explicit user confirmation",
        re.compile(
            r"(?:^|[;&|]\s*)gh\s+(?:(?:issue|pr|release|gist)\s+(?:create|comment|edit|close|reopen|merge|review|delete|lock|unlock)|repo\s+(?:create|delete|edit|fork|rename|archive)|api\b[^;&|]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--method\s*(?:POST|PUT|PATCH|DELETE)|\s-[fF]\s))"
        ),
    ),
    (
        "Pushing to a remote Git repository modifies external state and requires explicit user confirmation",
        re.compile(r"(?:^|[;&|]\s*)git\s+(?:-C\s+\S+\s+)?push\b"),
    ),
    (
        "Publishing to a public package registry requires explicit user confirmation",
        re.compile(r"(?:^|[;&|]\s*)(?:npm\s+publish|pnpm\s+publish|yarn\s+(?:npm\s+)?publish|cargo\s+publish|twine\s+upload|uv\s+publish)\b"),
    ),
]

PROTECTED_WRITE_PATHS = (
    "/etc",
    "/System",
    "/Library",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.zprofile"),
    os.path.expanduser("~/.bash_profile"),
    os.path.expanduser("~/.profile"),
    os.path.expanduser("~/.gemini/config/permission_reviewer.local.json"),
)

SENSITIVE_READ_PATHS = (
    os.path.expanduser("~/.ssh"),
    os.path.expanduser("~/.aws"),
    os.path.expanduser("~/.gnupg"),
    os.path.expanduser("~/.netrc"),
    os.path.expanduser("~/.gemini/antigravity-cli/settings.json"),
    os.path.expanduser("~/.gemini/config/config.json"),
)

TRUSTED_MCP_SERVERS = (
    "agy-context-guard",
    "agy-context-guard_agy-context-guard",
    "agy-permission-reviewer",
    "agy-permission-reviewer_agy-permission-reviewer",
)


def _audit_file(conversation_id: str) -> Path:
    audit_dir = get_data_dir() / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / f"{safe_conv_id(conversation_id)}.jsonl"


def record_audit_entry(conversation_id: str, entry: dict[str, Any]) -> None:
    try:
        af = _audit_file(conversation_id)
        if af.is_file() and af.stat().st_size > ROTATE_BYTES:
            lines = af.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
            af.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with af.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        prune_directory_files(af.parent, "*.jsonl")
        prune_directory_files(af.parent.parent / "ledger_cache", "*.json")
        prune_directory_files(af.parent.parent, "cache_*.json")
        prune_directory_files(af.parent.parent, "sendbacks_*.json")
    except OSError:
        pass


def strip_purpose_comment(command: str) -> str:
    return re.sub(r"\s+#\s*(?:purpose|auth):.*$", "", command.strip(), flags=re.IGNORECASE)


def strip_commit_message_literals(command: str) -> str:
    """Replace `-m "..."` / `-am '...'` commit message literals so prose inside commit messages
    (e.g. `git commit -am "Require force_ask on git push/registry"`) never triggers `git push` or path false positives.
    """
    return re.sub(
        r"""(\bgit\s+(?:-C\s+\S+\s+)?commit\b[^;&|]*?\s-[a-zA-Z]*m\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""",
        r'\1"__COMMIT_MSG__"',
        command,
    )


def is_safe_scoped_path(path_str: str, norm_workspaces: list[str], artifact_dir: str) -> bool:
    if any(is_path_under(path_str, ws) for ws in norm_workspaces):
        return True
    if artifact_dir and is_path_under(path_str, normalize_path(artifact_dir)):
        return True
    if (
        is_path_under(path_str, "/tmp")
        or is_path_under(path_str, "/private/tmp")
        or is_path_under(path_str, normalize_path(tempfile.gettempdir()))
    ):
        return True
    return False


def is_routine_command_segments(command: str) -> bool:
    code_only = strip_commit_message_literals(strip_purpose_comment(command))
    if not code_only or UNSAFE_SHELL_EXPANSION_RE.search(code_only) or OUTPUT_REDIRECT_RE.search(code_only):
        return False
    segments = [seg.strip() for seg in COMPOUND_SPLIT_RE.split(code_only) if seg.strip()]
    if not segments:
        return False
    return all(any(pat.search(seg) for pat in ROUTINE_SEGMENT_PATTERNS) for seg in segments)


def is_fast_allow_command(
    command: str,
    bypass_sandbox: bool,
    norm_workspaces: list[str],
    artifact_dir: str,
    cwd: str,
) -> tuple[bool, str]:
    """Evaluate compound commands segment-by-segment (adapted from agy-auto Fast-Allow AST gate)."""
    code_only = strip_commit_message_literals(strip_purpose_comment(command))
    if (
        not code_only
        or UNSAFE_SHELL_EXPANSION_RE.search(code_only)
        or DESTRUCTIVE_SCOPE_COMMAND_RE.search(code_only)
    ):
        return False, ""

    # Check output redirections: only allow redirects to /dev/null or inside workspace/artifact/tmp
    for redir_match in OUTPUT_REDIRECT_RE.finditer(code_only):
        target_tok = redir_match.group(1).strip("\"'")
        if target_tok in ("/dev/null", "&1", "&2"):
            continue
        resolved = normalize_path(target_tok, cwd=cwd)
        if not is_safe_scoped_path(resolved, norm_workspaces, artifact_dir):
            return False, ""

    # Check all explicit path tokens in the command
    ext_paths = [
        normalize_path(m.group(0), cwd=cwd)
        for m in PATH_TOKEN_RE.finditer(code_only)
        if len(m.group(0)) >= 3
    ]
    for sp in SENSITIVE_READ_PATHS:
        if any(is_path_under(p, sp) for p in ext_paths):
            return False, ""

    outside_paths = [
        p for p in ext_paths if not is_safe_scoped_path(p, norm_workspaces, artifact_dir)
    ]

    segments = [seg.strip() for seg in COMPOUND_SPLIT_RE.split(code_only) if seg.strip()]
    if not segments:
        return False, ""

    all_routine = all(
        any(pat.search(seg) for pat in ROUTINE_SEGMENT_PATTERNS)
        or (bool(ext_paths) and SCOPED_CONTENT_SEGMENT_RE.search(seg))
        for seg in segments
    )
    if all_routine and not outside_paths and not OUTPUT_REDIRECT_RE.search(code_only):
        return True, "[Tier 0 Policy] Routine read-only or local validation command."

    if not bypass_sandbox and not outside_paths:
        return True, "[Tier 0 Policy] Sandboxed command scoped to workspace/scratch/tmp."

    return False, ""


def check_tier0_static(
    tool_name: str,
    tool_args: dict[str, Any],
    workspace_paths: list[str],
    cwd: str,
    artifact_dir: str = "",
) -> dict[str, Any] | None:
    """Tier 0: Instant deterministic allow/deny (~1 ms)."""
    norm_workspaces = [normalize_path(p) for p in workspace_paths if p]
    local_cfg = load_local_config()
    extra_denied = [normalize_path(str(p)) for p in local_cfg.get("extra_denied_paths", []) if str(p).strip()]
    extra_ro = [normalize_path(str(p)) for p in local_cfg.get("extra_read_only_paths", []) if str(p).strip()]

    if tool_name in ("call_mcp_tool", "list_resources", "read_resource"):
        server_name = str(tool_args.get("ServerName", "")).strip()
        if server_name in TRUSTED_MCP_SERVERS:
            return {
                "decision": "allow",
                "reason": f"[Tier 0 Policy] Trusted local plugin MCP server ({server_name}).",
                "tier": "tier0_static",
            }

    if tool_name in ("read_file", "write_file", "edit_file", "view_file", "write_to_file", "replace_file_content"):
        raw_path = str(
            tool_args.get("Path")
            or tool_args.get("TargetFile")
            or tool_args.get("AbsolutePath")
            or ""
        )
        if not raw_path:
            return None
        target = normalize_path(raw_path, cwd=cwd)

        for sp in list(SENSITIVE_READ_PATHS) + extra_denied:
            if is_path_under(target, sp):
                return {
                    "decision": "deny",
                    "reason": f"[Tier 0 Policy] Access to sensitive path '{target}' is blocked.",
                    "tier": "tier0_static",
                }

        for prot in list(PROTECTED_WRITE_PATHS) + extra_ro:
            if is_path_under(target, prot) and tool_name not in ("read_file", "view_file"):
                return {
                    "decision": "deny",
                    "reason": f"[Tier 0 Policy] Write to protected path '{target}' is blocked.",
                    "tier": "tier0_static",
                }

        if tool_name in ("read_file", "view_file") and any(is_path_under(target, ro) for ro in extra_ro):
            return {
                "decision": "allow",
                "reason": f"[Tier 0 Policy] Read-only path '{target}' allowed by local config.",
                "tier": "tier0_static",
            }

        if is_safe_scoped_path(target, norm_workspaces, artifact_dir):
            return {
                "decision": "allow",
                "reason": "[Tier 0 Policy] Path is inside active workspace or conversation scratch/tmp.",
                "tier": "tier0_static",
            }
        return None

    if tool_name == "run_command":
        cmd = str(tool_args.get("CommandLine", ""))
        sanitized_cmd = strip_commit_message_literals(strip_purpose_comment(cmd))
        for label, pat in list(HARD_DENY_COMMAND_PATTERNS) + build_local_hard_deny_patterns(local_cfg):
            if pat.search(sanitized_cmd):
                return {
                    "decision": "deny",
                    "reason": f"[Tier 0 Policy] Blocked catastrophic command pattern: {label}.",
                    "tier": "tier0_static",
                }
        for label, pat in FORCE_ASK_COMMAND_PATTERNS:
            if pat.search(sanitized_cmd):
                return {
                    "decision": "ask",
                    "reason": f"[Tier 0 Policy] {label}.",
                    "tier": "tier0_force_ask",
                }
        ok, reason = is_fast_allow_command(
            command=cmd,
            bypass_sandbox=bool(tool_args.get("BypassSandbox", False)),
            norm_workspaces=norm_workspaces,
            artifact_dir=artifact_dir,
            cwd=cwd,
        )
        if ok:
            return {
                "decision": "allow",
                "reason": reason,
                "tier": "tier0_static",
            }
    return None


def check_tier1_ledger(
    tool_name: str,
    tool_args: dict[str, Any],
    workspace_paths: list[str],
    cwd: str,
    ledger: dict[str, Any],
) -> dict[str, Any] | None:
    """Tier 1: Zero-token check against verbal grants extracted from USER_INPUT steps."""
    norm_workspaces = [normalize_path(p) for p in workspace_paths if p]
    denied_paths: list[str] = ledger.get("denied_paths", [])
    path_grants: list[dict[str, Any]] = ledger.get("path_grants", [])
    command_grants: list[str] = ledger.get("command_grants", [])

    if tool_name in ("read_file", "write_file", "edit_file", "view_file", "write_to_file", "replace_file_content"):
        raw_path = str(
            tool_args.get("Path")
            or tool_args.get("TargetFile")
            or tool_args.get("AbsolutePath")
            or ""
        )
        if not raw_path:
            return None
        target = normalize_path(raw_path, cwd=cwd)
        needs_write = tool_name in ("write_file", "edit_file", "write_to_file", "replace_file_content")

        for dp in denied_paths:
            if is_path_under(target, dp):
                return {
                    "decision": "deny",
                    "reason": f"[Tier 1 Verbal Ledger] User explicitly prohibited access to '{dp}'.",
                    "tier": "tier1_ledger",
                }

        for pg in path_grants:
            if is_path_under(target, pg["path"]):
                if needs_write and pg["mode"] == "read":
                    return {
                        "decision": "deny",
                        "reason": (
                            f"[Tier 1 Verbal Ledger] User granted READ-ONLY access to '{pg['path']}' "
                            f"(step #{pg['step_index']}); write/edit is blocked."
                        ),
                        "tier": "tier1_ledger",
                    }
                return {
                    "decision": "allow",
                    "reason": (
                        f"[Tier 1 Verbal Ledger] Auto-approved ({pg['mode']}) matching user grant "
                        f"for '{pg['path']}' in step #{pg['step_index']}."
                    ),
                    "tier": "tier1_ledger",
                }
        return None

    if tool_name == "run_command":
        cmd = strip_commit_message_literals(strip_purpose_comment(str(tool_args.get("CommandLine", ""))))
        ext_paths = [
            normalize_path(m.group(0), cwd=cwd)
            for m in PATH_TOKEN_RE.finditer(cmd)
            if len(m.group(0)) >= 3
        ]
        for ep in ext_paths:
            for dp in denied_paths:
                if is_path_under(ep, dp):
                    return {
                        "decision": "deny",
                        "reason": f"[Tier 1 Verbal Ledger] Command references prohibited path '{dp}'.",
                        "tier": "tier1_ledger",
                    }

        outside_paths = [
            p
            for p in ext_paths
            if not any(is_path_under(p, ws) for ws in norm_workspaces)
            and not is_path_under(p, "/tmp")
            and not is_path_under(p, "/private/tmp")
        ]
        if (
            outside_paths
            and not DESTRUCTIVE_SCOPE_COMMAND_RE.search(cmd)
            and all(
                any(is_path_under(op, pg["path"]) for pg in path_grants)
                for op in outside_paths
            )
        ):
            return {
                "decision": "allow",
                "reason": "[Tier 1 Verbal Ledger] External paths in command match user verbal grants.",
                "tier": "tier1_ledger",
            }

        if ("mpirun" in cmd or "mpiexec" in cmd) and "mpirun" in command_grants:
            return {
                "decision": "allow",
                "reason": "[Tier 1 Verbal Ledger] User explicitly requested MPI execution.",
                "tier": "tier1_ledger",
            }
    return None


def build_requested_action(
    tool_name: str,
    tool_args: dict[str, Any],
    workspace_paths: list[str] | None = None,
    cwd: str = "",
    artifact_dir: str = "",
    ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    norm_workspaces = [normalize_path(p) for p in (workspace_paths or []) if p]
    user_messages: list[str] = list((ledger or {}).get("recent_user_messages") or [])
    active_grants = [
        f"{pg.get('mode', 'read')}:{pg.get('path', '')}"
        for pg in (ledger or {}).get("path_grants", [])
        if pg.get("path")
    ]

    if tool_name == "run_command":
        from intent_extractor import verify_cited_user_authorization

        cmd = str(tool_args.get("CommandLine", ""))
        bypass = bool(tool_args.get("BypassSandbox", False))
        sandbox_on = 0 if bypass else 1
        auth_match = re.search(
            r"#\s*auth:\s*(.+?)(?=\s+#\s*purpose:|$)", cmd, flags=re.IGNORECASE
        )
        purpose_match = re.search(
            r"#\s*purpose:\s*(.+?)(?=\s+#\s*auth:|$)", cmd, flags=re.IGNORECASE
        )
        code_only = strip_commit_message_literals(strip_purpose_comment(cmd))
        ext_paths = [
            normalize_path(m.group(0), cwd=cwd)
            for m in PATH_TOKEN_RE.finditer(code_only)
            if len(m.group(0)) >= 3
        ]
        outside_paths = [
            p for p in ext_paths if not is_safe_scoped_path(p, norm_workspaces, artifact_dir)
        ]
        is_destructive_or_delete = bool(DESTRUCTIVE_SCOPE_COMMAND_RE.search(code_only))

        details: dict[str, Any] = {
            "toolSummary": tool_args.get("toolSummary", ""),
            "sandboxOn": sandbox_on,
            "externalPaths": outside_paths,
            "isDestructiveOrDelete": is_destructive_or_delete,
        }
        if active_grants:
            details["activeVerbalGrants"] = active_grants[-6:]
        if purpose_match:
            details["purposeComment"] = purpose_match.group(1).strip()
        if auth_match:
            all_entries = list((ledger or {}).get("all_user_entries") or [])
            details["citedUserAuth"] = verify_cited_user_authorization(
                auth_match.group(1).strip(), all_entries
            )
        return {
            "kind": "shell",
            "targets": [cmd],
            "details": details,
            **({"recent_user_messages": user_messages} if user_messages else {}),
        }

    target = str(
        tool_args.get("Path")
        or tool_args.get("TargetFile")
        or tool_args.get("AbsolutePath")
        or ""
    )
    norm_target = normalize_path(target, cwd=cwd) if target else ""
    in_scope = bool(norm_target and is_safe_scoped_path(norm_target, norm_workspaces, artifact_dir))
    is_edit = tool_name in ("write_file", "edit_file", "write_to_file", "replace_file_content")
    details = {
        "tool": tool_name,
        "targetInWorkspace": in_scope,
    }
    if active_grants:
        details["activeVerbalGrants"] = active_grants[-6:]
    return {
        "kind": "edit" if is_edit else "read",
        "targets": [norm_target] if norm_target else [],
        "details": details,
        **({"recent_user_messages": user_messages} if user_messages else {}),
    }


def evaluate_tool_call(
    payload: dict[str, Any],
    api_caller: Any = None,
) -> dict[str, Any]:
    """Run the 3-tier review pipeline (`Policy` -> `Verbal Ledger` -> `jev-1.13-free` + Escalation + Send-Back)."""
    t0 = time.perf_counter()
    tool_call = payload.get("toolCall") or {}
    tool_name = str(tool_call.get("name", ""))
    tool_args = tool_call.get("args") or {}
    workspace_paths = list(payload.get("workspacePaths") or [])
    cwd = str(payload.get("cwd", ""))
    artifact_dir = str(payload.get("artifactDirectoryPath") or payload.get("artifact_directory_path") or "")
    conversation_id = str(payload.get("conversationId") or payload.get("conversation_id") or "default")
    transcript_str = str(payload.get("transcriptPath") or payload.get("transcript_path") or "")
    if not artifact_dir and "/.system_generated/" in transcript_str:
        artifact_dir = transcript_str.split("/.system_generated/", 1)[0]
    transcript_path = Path(transcript_str) if transcript_str else None
    last_user_input = str(payload.get("lastUserInput", ""))

    res = check_tier0_static(tool_name, tool_args, workspace_paths, cwd, artifact_dir=artifact_dir)
    if res is None:
        ledger = collect_session_ledger(
            transcript_path, last_user_input=last_user_input, cwd=cwd
        )
        res = check_tier1_ledger(tool_name, tool_args, workspace_paths, cwd, ledger)
        if res is None:
            mode = os.environ.get("AGY_REVIEWER_MODE", "opencode").lower()
            if mode == "ledger_only":
                res = {
                    "decision": "ask",
                    "reason": "Tier 2 model review disabled (AGY_REVIEWER_MODE=ledger_only); prompting user.",
                    "tier": "tier2_disabled",
                }
            else:
                req_key = compute_request_key(conversation_id, tool_name, tool_args)
                is_file_tool = tool_name in (
                    "read_file",
                    "write_file",
                    "edit_file",
                    "view_file",
                    "write_to_file",
                    "replace_file_content",
                )
                dir_sig = (
                    compute_signature(
                        conversation_id,
                        int(ledger.get("last_user_step_index", 0)),
                        tool_name,
                        tool_args,
                    )
                    if is_file_tool
                    else None
                )
                # Also check if a routine command (`git -C <dir>`, `unittest -s <dir>`) only references
                # external directories that have already been approved in this conversation.
                cmd_outside_dirs: list[str] = []
                if tool_name == "run_command":
                    cmd_str = str(tool_args.get("CommandLine", ""))
                    if is_routine_command_segments(cmd_str):
                        sanitized = strip_commit_message_literals(strip_purpose_comment(cmd_str))
                        norm_ws = [normalize_path(p) for p in workspace_paths if p]
                        ext_p = [
                            normalize_path(m.group(0), cwd=cwd)
                            for m in PATH_TOKEN_RE.finditer(sanitized)
                            if len(m.group(0)) >= 3
                        ]
                        cmd_outside_dirs = [
                            p for p in ext_p if not is_safe_scoped_path(p, norm_ws, artifact_dir)
                        ]

                cached = get_cached_verdict(conversation_id, req_key)
                if cached is None and dir_sig is not None:
                    sig_cached = get_cached_verdict(conversation_id, dir_sig)
                    if sig_cached is not None and sig_cached.get("effect") == "allow":
                        cached = sig_cached
                if cached is None and is_file_tool:
                    raw_p = str(
                        tool_args.get("Path")
                        or tool_args.get("TargetFile")
                        or tool_args.get("AbsolutePath")
                        or ""
                    )
                    if raw_p:
                        norm_p = normalize_path(raw_p, cwd=cwd)
                        for cd in (str(Path(norm_p).parent), str(Path(norm_p).parent.parent)):
                            dir_verdict = get_cached_verdict(conversation_id, compute_directory_grant_key(cd))
                            if dir_verdict is not None and dir_verdict.get("effect") == "allow":
                                cached = {
                                    "effect": "allow",
                                    "source": "cache",
                                    "model": "session_dir_cache",
                                    "reasonCode": "cached_dir_grant",
                                    "reason": "File operation targets a repository/directory already approved in this session.",
                                }
                                break
                if cached is None and cmd_outside_dirs:
                    # Check if every referenced external path (or its parent dir) has an approved dir_grant
                    all_dirs_approved = True
                    for op in cmd_outside_dirs:
                        cand_dirs = [op, str(Path(op).parent), str(Path(op).parent.parent)]
                        if not any(
                            (get_cached_verdict(conversation_id, compute_directory_grant_key(cd)) or {}).get("effect") == "allow"
                            for cd in cand_dirs
                        ):
                            all_dirs_approved = False
                            break
                    if all_dirs_approved:
                        cached = {
                            "effect": "allow",
                            "source": "cache",
                            "model": "session_dir_cache",
                            "reasonCode": "cached_dir_grant",
                            "reason": "Routine command targets an external directory already approved in this session.",
                        }
                if cached is not None:
                    verdict = cached
                else:
                    requested_action = build_requested_action(
                        tool_name,
                        tool_args,
                        workspace_paths=workspace_paths,
                        cwd=cwd,
                        artifact_dir=artifact_dir,
                        ledger=ledger,
                    )
                    compact_tx = extract_compact_transcript(transcript_path)
                    decision_models = [
                        m.strip()
                        for m in os.environ.get(
                            "AGY_DECISION_MODELS", ",".join(DEFAULT_DECISION_MODELS)
                        ).split(",")
                        if m.strip()
                    ]
                    escalation_models = [
                        m.strip()
                        for m in os.environ.get(
                            "AGY_ESCALATION_MODELS", ",".join(DEFAULT_ESCALATION_MODELS)
                        ).split(",")
                        if m.strip()
                    ]
                    min_conf = float(
                        os.environ.get("AGY_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))
                    )
                    kwargs = {"api_caller": api_caller} if api_caller is not None else {}
                    first = run_jev_decision_tier(
                        models=decision_models,
                        min_confidence=min_conf,
                        user_messages=ledger.get("recent_user_messages", []),
                        compact_transcript=compact_tx,
                        requested_action=requested_action,
                        **kwargs,
                    )
                    if first.get("effect") == "ask" and escalation_models:
                        esc_tx = extract_compact_transcript(
                            transcript_path,
                            max_items=24,
                            max_chars=5500,
                            include_thinking=False,
                        )
                        second = run_escalation_tier(
                            escalation_models=escalation_models,
                            requested_action=requested_action,
                            first_verdict=first,
                            compact_transcript=esc_tx or compact_tx,
                            **kwargs,
                        )
                        verdict = second if second is not None else first
                    else:
                        verdict = first
                    put_cached_verdict(conversation_id, req_key, verdict)

                if verdict.get("effect") == "allow":
                    if dir_sig is not None:
                        put_cached_verdict(conversation_id, dir_sig, verdict)
                    if is_file_tool:
                        raw_p = str(
                            tool_args.get("Path")
                            or tool_args.get("TargetFile")
                            or tool_args.get("AbsolutePath")
                            or ""
                        )
                        if raw_p:
                            norm_p = normalize_path(raw_p, cwd=cwd)
                            put_cached_verdict(
                                conversation_id,
                                compute_directory_grant_key(str(Path(norm_p).parent)),
                                verdict,
                            )
                            put_cached_verdict(
                                conversation_id,
                                compute_directory_grant_key(str(Path(norm_p).parent.parent)),
                                verdict,
                            )
                    for op in cmd_outside_dirs:
                        put_cached_verdict(
                            conversation_id,
                            compute_directory_grant_key(op),
                            verdict,
                        )
                        put_cached_verdict(
                            conversation_id,
                            compute_directory_grant_key(str(Path(op).parent)),
                            verdict,
                        )

                unresolved_mode = os.environ.get("AGY_UNRESOLVED_MODE", "agent")
                max_sb = int(os.environ.get("AGY_MAX_SEND_BACKS", str(DEFAULT_MAX_SEND_BACKS)))
                res = apply_unresolved_send_back(
                    conversation_id=conversation_id,
                    verdict=verdict,
                    unresolved_mode=unresolved_mode,
                    max_send_backs=max_sb,
                )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    target_summary = ""
    if tool_name == "run_command":
        target_summary = str(tool_args.get("CommandLine", ""))[:160]
    else:
        target_summary = str(
            tool_args.get("Path")
            or tool_args.get("TargetFile")
            or tool_args.get("AbsolutePath")
            or tool_args.get("ToolName")
            or ""
        )[:160]
    record_audit_entry(
        conversation_id,
        {
            "stepIdx": payload.get("stepIdx", 0),
            "tool": tool_name,
            "target": target_summary,
            "decision": res["decision"],
            "tier": res.get("tier", "unknown"),
            "reason": res.get("reason", ""),
            "latency_ms": elapsed_ms,
        },
    )
    return {
        "decision": res["decision"],
        "reason": res.get("reason", ""),
    }


def to_antigravity_hook_output(res: dict[str, Any]) -> dict[str, Any]:
    """Translate logical decision ('allow', 'deny', 'ask') to Antigravity PreToolHookResult JSON.

    In Antigravity's `EnsurePermissions`:
    - `"auto_approve"` triggers `hook_auto_approve`, overriding default `Ask` prompts
      without requiring the user to set global settings to Always Allow.
    - `"deny"` blocks the tool call immediately and returns `reason` to the model.
    - `"force_ask"` forces a native user confirmation popup even if settings are Always Allow.
    """
    decision = res.get("decision", "ask")
    if decision == "allow":
        wire_decision = os.environ.get("AGY_HOOK_ALLOW_DECISION", "auto_approve")
    elif decision == "ask":
        wire_decision = "force_ask"
    else:
        wire_decision = decision
    out: dict[str, Any] = {
        "decision": wire_decision,
        "reason": res.get("reason", ""),
    }
    if "overwrite" in res and isinstance(res["overwrite"], dict):
        out["overwrite"] = res["overwrite"]
    return out


def main() -> int:
    try:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
        res = evaluate_tool_call(payload)
        out = to_antigravity_hook_output(res)
        if os.environ.get("AGY_REVIEWER_DEBUG", "0") == "1":
            with open("/tmp/agy_reviewer_debug.log", "a", encoding="utf-8") as dbg:
                dbg.write(
                    json.dumps(
                        {
                            "hookEventName": payload.get("hookEventName"),
                            "toolName": (payload.get("toolCall") or {}).get("name"),
                            "out": out,
                        }
                    )
                    + "\n"
                )
        sys.stdout.write(json.dumps(out) + "\n")
    except Exception as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "decision": "force_ask",
                    "reason": f"[Permission Reviewer Internal Error: {exc}]",
                }
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
