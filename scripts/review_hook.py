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

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time
from typing import Any

from intent_extractor import (
    PATH_TOKEN_RE,
    collect_session_ledger,
    extract_paths_from_string,
    extract_python_code_paths,
    extract_relevant_earlier_user_turns,
    get_data_dir,
    is_path_under,
    normalize_path,
    parse_shell_structure,
    safe_conv_id,
)
from llm_classifier import (
    DEFAULT_DECISION_MODELS,
    DEFAULT_DECISION_TIMEOUT_SEC,
    DEFAULT_DEEP_REVIEW_TIMEOUT_SEC,
    DEFAULT_ESCALATION_MODELS,
    DEFAULT_ESCALATION_TIMEOUT_SEC,
    DEFAULT_MAX_SEND_BACKS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_PAID_FALLBACK_MODELS,
    ROTATE_BYTES,
    apply_unresolved_send_back,
    check_and_record_appeal_quota,
    compute_directory_grant_key,
    compute_request_key,
    compute_signature,
    compute_ssh_host_grant_key,
    extract_compact_transcript,
    find_opencode_binary,
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
DISCARD_REDIRECT_TARGETS = frozenset({"/dev/null", "&1", "&2"})

REVIEWED_TOOLS = frozenset(
    {
        "run_command",
        "read_file",
        "write_file",
        "edit_file",
        "view_file",
        "write_to_file",
        "replace_file_content",
        "delete_file",
        "call_mcp_tool",
        "list_resources",
        "read_resource",
    }
)
REVIEWED_TOOL_PREFIXES = ("mcp_", "browser_")

STANDARD_BIN_PATHS = (
    "/bin",
    "/usr/bin",
    "/sbin",
    "/usr/sbin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/opt/homebrew/Cellar",
    "/opt/homebrew/opt",
    "/opt/homebrew/Caskroom",
    "/Library/Developer/CommandLineTools/usr/bin",
)

_SEGMENT_CMD_PREFIX = (
    r"(?:^|[;&|]\s*)"
    r"(?:export\s+[^;&|]+;\s*|(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s+))*"
    r"(?:/[\w./-]+/)?"
)

_SANDBOX_CMD_PREFIX = (
    r"(?:^|(?<!\\)(?:&&|\|\||;)\s*|(?<!\\)\|\s+|(?:(?:/[\w./-]+/)?(?:ba|z)?sh\s+-[a-zA-Z]*c\s+[\"']\s*))"
    r"(?:export\s+[^;&|]+;\s*|(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s+))*"
    r"(?:/[\w./-]+/)?"
)

# Commands that require outbound network, MPI shared-memory IPC, tmux sockets, or host process
# inspection and deterministically fail inside the macOS seatbelt sandbox (BypassSandbox=false).
SANDBOX_INCOMPATIBLE_COMMAND_RE = re.compile(
    _SANDBOX_CMD_PREFIX
    + r"(?:git\s+(?:-C\s+\S+\s+)?(?:push|pull|fetch|ls-remote|clone\s+(?:https?://|ssh://|git@))\b"
    + r"|gh\s+(?:pr|issue|release|gist|repo|run|api|workflow|search)\b"
    + r"|ssh\s+(?!-[vV](?:\s|$))|scp\s+|sftp\s+|rsync\b[^;&|]*\s+\S+@?\S+:\S+"
    + r"|(?:curl|wget)\b[^;&|]*https?://(?!(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::|/|\s|[\"']|$))"
    + r"|(?:ping|dig|nslookup|traceroute)\s+"
    + r"|(?:npm|pnpm|yarn|cargo|uv)\s+(?:npm\s+)?publish\b|twine\s+upload\b"
    + r"|brew\s+(?:update|upgrade|install|reinstall|fetch|tap)\b"
    + r"|(?:pip3?|uv\s+pip)\s+install\b(?![^;&|]*--(?:offline|no-index))"
    + r"|opencode\s+(?:api|models|service)\b"
    + r"|(?:mpirun|mpiexec|orterun|prterun|srun)\b(?![^;&|]*(?:--version|-V\b|--help|-h\b))"
    + r"|tmux\b(?![^;&|]*-V\b)"
    + r"|(?:ps|pgrep|pkill|killall|top|lsof|launchctl|osascript)\b)"
)
NETWORK_REQUIRED_COMMAND_RE = SANDBOX_INCOMPATIBLE_COMMAND_RE

# Read-only inspection, local test/build runners, and local git commands (never push/reset/clean).
# Combined with `not outside_paths`, these fast-allow in <1 ms when all referenced paths are inside
# the active workspace, conversation scratch, or /tmp (including BypassSandbox=true for MPI/uv).
ROUTINE_SEGMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^git\s+(?:-C\s+\S+\s+)?(?:status|diff|log|show|rev-parse|remote\s+-v$|branch(?:\s+(?:--show-current|--list|-a|-v|-vv|-r))*$|add\b|commit\b|stash\b|check-ignore\b)(?:\s|$)"),
    re.compile(r"^uv\s+run\s+(?:python3?\s+-m\s+(?:unittest|pytest|py_compile|ruff|mypy)|pytest|ruff|mypy)\b"),
    re.compile(r"^python3?\s+-m\s+(?:unittest|pytest|py_compile|ruff|mypy)\b"),
    re.compile(r"^(?:mpirun|mpiexec|orterun|prterun)\b(?![^;&|]*(?:\b(?:ba|z|da|k)?sh\b|\bpython3?\s+-c\b|\bperl\s+-e\b|\bruby\s+-e\b|\bnode\s+-e\b))"),
    re.compile(r"^(?:cmake|ctest|ninja|make)\b"),
    re.compile(r"^(?:clang\+\+|clang|g\+\+|gcc|nvcc|rustc|javac|javap)\b"),
    re.compile(r"^(?:npm|pnpm|yarn|bun)\s+(?:test|run\s+(?:test|lint|build|typecheck|check))(?:\s|$)"),
    re.compile(r"^cargo\s+(?:test|check|build)(?:\s|$)"),
    re.compile(r"^go\s+test(?:\s|$)"),
    re.compile(r"^(?:pytest|ruff\s+check|tsc)(?:\s|$)"),
    re.compile(r"^gh\s+(?:pr\s+(?:list|view|checks|diff|status)|issue\s+(?:list|view|status)|run\s+(?:list|view)|repo\s+view)(?:\s|$)"),
    re.compile(r"^(?:ls|tree|wc|du|df|stat|file|which|pwd|whoami|uname|uptime|nproc|sw_vers|true|false|echo|printf|ps|pgrep)(?:\s|$)"),
    re.compile(r"^(?:hostname|date)(?:\s+(?:-[sf]|\+\S+))?$"),
]

_GH_API_MUTATION_FLAG_RE = re.compile(r"(?:\s|^)(?:-X|--method|-f|-F|--raw-field|--field|--input)\b")
_GIT_COMMIT_SPAN_RE = re.compile(
    r"""(?<![A-Za-z0-9_])(?:/[\w./-]+/)?git\s+(?:-[Cc]\s+\S+\s+)?commit\b(?:[^"'`;&|]+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')*""",
    re.DOTALL,
)
_GIT_COMMIT_MSG_FLAG_RE = re.compile(
    r"""(\s(?:-[a-zA-Z]*m\s*|--message(?:=|\s+)))(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""",
    re.DOTALL,
)

# Content-reading and scoped file cleanup commands fast-allow only when target paths are explicitly present and in-scope
SCOPED_CONTENT_SEGMENT_RE = re.compile(r"^(?:cat|head|tail|sed|awk|grep|rg|find|sort|uniq|cut|tr|rm\s+-f)\b")

# Directory deletion or destructive working-tree commands must NEVER be fast-allowed by Tier 0 or
# Tier 1 path grants alone; they must always go to Tier 2 (LLM review against user messages) so
# they are blocked unless the user explicitly asked to delete/reset them in the conversation.
DESTRUCTIVE_SCOPE_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:(?:/bin/)?rm\s+(?:-[^\s]+\s+)*(?:-[^\s]*[rR]|--recursive\b)|(?:/bin/)?rmdir\b|find\b[^;&|]*\s-delete\b|shutil\.rmtree|git\s+(?:-[Cc]\s+\S+\s+)?(?:reset\s+--hard|clean\s+(?:-[^\s]+\s+)*(?:-[^\s]*f|--force\b)|rm\s+(?:-[^\s]+\s+)*(?:-[^\s]*r\b|--recursive\b)))"
)

_GIT_WRITE_SUBCOMMANDS = frozenset(
    {
        "add",
        "commit",
        "rm",
        "mv",
        "checkout",
        "switch",
        "restore",
        "merge",
        "rebase",
        "cherry-pick",
        "revert",
        "reset",
        "clean",
        "am",
    }
)
_GIT_GLOBAL_ARG_FLAGS = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"}
)


def _is_git_segment_mutating(seg: str) -> tuple[bool, str | None]:
    """Return (is_mutating_git_cmd, explicit_minus_C_repo_arg)."""
    try:
        tokens = shlex.split(seg)
    except ValueError:
        tokens = seg.split()
    if not tokens:
        return False, None
    idx = 0
    while idx < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[idx]):
        idx += 1
    if idx >= len(tokens) or Path(tokens[idx]).name != "git":
        return False, None
    idx += 1
    repo_arg: str | None = None
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in _GIT_GLOBAL_ARG_FLAGS:
            if tok == "-C" and idx + 1 < len(tokens):
                repo_arg = tokens[idx + 1]
            idx += 2
            continue
        if tok.startswith("-C") and len(tok) > 2:
            repo_arg = tok[2:]
            idx += 1
            continue
        if tok.startswith("-"):
            idx += 1
            continue
        break
    if idx >= len(tokens):
        return False, repo_arg
    subcmd = tokens[idx]
    rest = tokens[idx + 1 :]
    if subcmd in _GIT_WRITE_SUBCOMMANDS:
        return True, repo_arg
    if subcmd == "stash":
        return (not rest or rest[0] not in ("list", "show")), repo_arg
    if subcmd == "branch":
        has_mut_flag = any(
            t in ("-d", "-D", "-m", "-M", "--delete", "--move")
            or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "dDmM"))
            for t in rest
        )
        return has_mut_flag, repo_arg
    if subcmd == "tag":
        is_readonly = any(t in ("-l", "--list", "-v", "--verify", "--contains", "--merged", "--no-merged") for t in rest)
        if is_readonly:
            return False, repo_arg
        has_mut_flag = any(
            t in ("-d", "-a", "-f", "-s", "--delete", "--annotate", "--force", "--sign")
            or (t.startswith("-") and not t.startswith("--") and any(c in t[1:] for c in "dafs"))
            for t in rest
        )
        has_positional = any(not t.startswith("-") for t in rest)
        return (has_mut_flag or has_positional), repo_arg
    return False, repo_arg


def _is_workspace_git_write(sanitized_cmd: str, norm_workspaces: list[str], cwd: str) -> bool:
    """Return True if a `.git`-mutating command targets an active workspace repository (blocked in sandbox)."""
    if not norm_workspaces:
        return False
    segments = split_shell_segments(sanitized_cmd)
    expanded: list[str] = []
    for raw_seg in segments:
        s = _strip_path_env_in_segment(raw_seg)
        if not s:
            continue
        shell_m = re.match(r"^(?:/[\w./-]+/)?(?:ba|z)?sh\s+-[a-zA-Z]*c\s+([\"'])(.*)\1$", s, re.DOTALL)
        if shell_m:
            expanded.extend(
                _strip_path_env_in_segment(sub)
                for sub in split_shell_segments(shell_m.group(2))
                if sub.strip()
            )
        else:
            expanded.append(s)
    for seg in expanded:
        is_mut, repo_arg = _is_git_segment_mutating(seg)
        if not is_mut:
            continue
        if repo_arg:
            repo_dir = normalize_path(repo_arg.strip("\"'"), cwd=cwd)
        elif cwd:
            repo_dir = normalize_path(cwd)
        else:
            repo_dir = norm_workspaces[0]
        if any(is_path_under(repo_dir, ws) for ws in norm_workspaces):
            return True
    return False

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
        re.compile(r"(?:\.ssh[/\\]id_(?:rsa|ed25519|ecdsa|dsa)|\.aws[/\\]credentials|\.netrc|(?:^|[/\\]|\s)\.env(?:\.(?:local|prod|production|secret))?(?:\s|$|[\"']))"),
    ),
    (
        "Modifying Antigravity settings.json or config.json directly",
        re.compile(r"\.gemini[/\\](?:antigravity-cli[/\\]settings\.json|config[/\\](?:config|permission_reviewer\.local)\.json)"),
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
    try:
        if not cfg_path.is_file():
            return {}
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
            _SEGMENT_CMD_PREFIX
            + r"gh\s+(?:(?:issue|pr|release|gist)\s+(?:create|comment|edit|close|reopen|merge|review|delete|lock|unlock)|repo\s+(?:create|delete|edit|fork|rename|archive)|api\b[^;&|]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--method\s*(?:POST|PUT|PATCH|DELETE)|\s-[fF]\s|--(?:raw-field|field|input)(?:=|\s)))"
        ),
    ),
    (
        "Pushing to a remote Git repository modifies external state and requires explicit user confirmation",
        re.compile(_SEGMENT_CMD_PREFIX + r"git\s+(?:-C\s+\S+\s+)?push\b"),
    ),
    (
        "Publishing to a public package registry requires explicit user confirmation",
        re.compile(
            _SEGMENT_CMD_PREFIX
            + r"(?:npm\s+publish|pnpm\s+publish|yarn\s+(?:npm\s+)?publish|cargo\s+publish|twine\s+upload|uv\s+publish)\b"
        ),
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
    *((os.environ.get("WINDIR") or r"C:\Windows",) if os.name == "nt" else ()),
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
        prune_directory_files(af.parent.parent, "appeals_*.json")
    except OSError:
        pass


def strip_purpose_comment(command: str) -> str:
    return re.sub(
        r"(?:\n|[ \t]+)#\s*(?:(?:purpose|auth|review)\s*:|appeal\b)[^\n]*$",
        "",
        command.strip(),
        flags=re.IGNORECASE,
    )


def _parse_review_annotations(command: str) -> dict[str, Any]:
    stripped = command.strip()
    tail = stripped.rsplit("\n", 1)[-1] if stripped else ""
    auth_m = re.search(
        r"#\s*auth:\s*(.+?)(?=\s+#\s*(?:purpose|review)\s*:|\s+#\s*appeal\b|$)",
        tail,
        flags=re.IGNORECASE,
    )
    purpose_m = re.search(
        r"#\s*purpose:\s*(.+?)(?=\s+#\s*(?:auth|review)\s*:|\s+#\s*appeal\b|$)",
        tail,
        flags=re.IGNORECASE,
    )
    appeal_requested = bool(
        re.search(
            r"(?:^|\s)#\s*(?:appeal\b|review\s*:\s*(?:deep|escalate)\b)",
            tail,
            flags=re.IGNORECASE,
        )
    )
    return {
        "auth": auth_m.group(1).strip() if auth_m else "",
        "purpose": purpose_m.group(1).strip() if purpose_m else "",
        "appeal_requested": appeal_requested,
    }


def _resolve_model_list(
    env_var: str,
    local_cfg: dict[str, Any],
    cfg_key: str,
    default_models: list[str],
    append_fallbacks: list[str] | None = None,
) -> list[str]:
    """Resolve model list from env -> local config -> defaults, optionally appending free fallbacks."""
    if env_var in os.environ:
        env_val = os.environ[env_var].strip()
        if not env_val:
            return []
        models = [m.strip() for m in env_val.split(",") if m.strip()]
    else:
        raw_cfg = local_cfg.get(cfg_key)
        if isinstance(raw_cfg, list):
            models = [str(m).strip() for m in raw_cfg if str(m).strip()]
        elif isinstance(raw_cfg, str) and raw_cfg.strip():
            models = [m.strip() for m in raw_cfg.split(",") if m.strip()]
        else:
            models = list(default_models)
    if append_fallbacks:
        seen = set(models)
        for fb in append_fallbacks:
            if fb not in seen:
                models.append(fb)
                seen.add(fb)
    return models


def strip_commit_message_literals(command: str) -> str:
    """Replace all `-m "..."` / `--message=...` commit message literals in `git commit` segments
    so prose inside commit messages never triggers `git push`, `gh`, or path false positives.
    """
    return _GIT_COMMIT_SPAN_RE.sub(
        lambda m: _GIT_COMMIT_MSG_FLAG_RE.sub(r'\1"__COMMIT_MSG__"', m.group(0)),
        command,
    )


def is_standard_bin_path(path_str: str) -> bool:
    return any(is_path_under(path_str, bp) for bp in STANDARD_BIN_PATHS)


def _strip_path_env_in_segment(seg: str) -> str:
    s = seg.strip()
    if re.fullmatch(r"export\s+PATH=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)", s):
        return ""
    s = re.sub(r"^(?:export\s+PATH=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s*;\s*)+", "", s)
    s = re.sub(r"^(?:PATH=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s+)+", "", s)
    return s.strip()


_TRANSPARENT_WRAPPERS = frozenset({"env", "command", "nohup", "time", "nice"})
_NESTED_SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_PYTHON_BIN_RE = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
_PYTHON_LAUNCHERS = frozenset({"uv", "mpirun", "mpiexec", "orterun", "prterun", "srun"})
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SED_FILE_CMD_RE = re.compile(r"(?:^|[;\s{\d$/])[wr]\s+([^\s;]+)")
_AWK_DQ_STR_RE = re.compile(r'"((?:\\.|[^"\\])*)"')


def split_shell_segments(code_only: str) -> list[str]:
    parsed = parse_shell_structure(code_only)
    if parsed is not None:
        return list(parsed["raw_segments"])
    return [seg.strip() for seg in COMPOUND_SPLIT_RE.split(code_only) if seg.strip()]


def extract_shell_output_redirects(code_only: str) -> list[str]:
    parsed = parse_shell_structure(code_only)
    if parsed is not None:
        return [
            target
            for _, target in parsed["output_redirects"]
            if target not in DISCARD_REDIRECT_TARGETS
        ]
    return [
        m.group(1).strip("\"'")
        for m in OUTPUT_REDIRECT_RE.finditer(code_only)
        if m.group(1).strip("\"'") not in DISCARD_REDIRECT_TARGETS
    ]


def _is_sensitive_path(norm_path: str) -> bool:
    return any(is_path_under(norm_path, sp) for sp in SENSITIVE_READ_PATHS)


def _append_exec_tok(tok: str, cwd: str, strip_std_bin: bool, out: list[str]) -> None:
    if tok in ("~", "..") or tok.startswith(("/", "~/", "../")):
        norm = normalize_path(tok, cwd=cwd)
        if strip_std_bin and is_standard_bin_path(norm) and not _is_sensitive_path(norm):
            return
        out.append(norm)


def _append_kv_or_arg(tok: str, cwd: str, out: list[str]) -> None:
    if tok.startswith("--jq="):
        return
    if "=" in tok and not tok.startswith(("/", "~/", "../")):
        _, rhs = tok.split("=", 1)
        parts = rhs.split(":") if "://" not in rhs else [rhs]
        for part in parts:
            out.extend(extract_paths_from_string(part, cwd=cwd))
        return
    if re.match(r"^-[A-Za-z](?:/|~/|\.\./).+", tok):
        out.append(normalize_path(tok[2:], cwd=cwd))
        return
    out.extend(extract_paths_from_string(tok, cwd=cwd))


def _extract_default_args(args: tuple[str, ...], cwd: str, out: list[str]) -> None:
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if tok == "--jq":
            i += 2
            continue
        _append_kv_or_arg(tok, cwd, out)
        i += 1


def _consume_leading_wrappers(
    argv: tuple[str, ...],
    cwd: str,
    strip_std_bin: bool,
    out: list[str],
) -> int | None:
    if not argv:
        return None
    idx = 0
    n = len(argv)
    if argv[0] == "export":
        for tok in argv[1:]:
            if _ENV_ASSIGN_RE.match(tok):
                var, val = tok.split("=", 1)
                if not (var == "PATH" and strip_std_bin):
                    for part in (val.split(":") if "://" not in val else [val]):
                        out.extend(extract_paths_from_string(part, cwd=cwd))
            else:
                _append_kv_or_arg(tok, cwd, out)
        return None

    while idx < n and _ENV_ASSIGN_RE.match(argv[idx]):
        var, val = argv[idx].split("=", 1)
        if not (var == "PATH" and strip_std_bin):
            for part in (val.split(":") if "://" not in val else [val]):
                out.extend(extract_paths_from_string(part, cwd=cwd))
        idx += 1

    while idx < n:
        wname = Path(argv[idx]).name
        if wname not in _TRANSPARENT_WRAPPERS:
            break
        _append_exec_tok(argv[idx], cwd, strip_std_bin, out)
        idx += 1
        if wname == "env":
            while idx < n:
                tok = argv[idx]
                if tok in ("-i", "-0", "--ignore-environment", "--null", "--"):
                    idx += 1
                    if tok == "--":
                        break
                elif tok in ("-u", "--unset", "-C", "--chdir"):
                    if tok in ("-C", "--chdir") and idx + 1 < n:
                        _append_kv_or_arg(argv[idx + 1], cwd, out)
                    idx += 2
                elif tok.startswith("-") and tok != "-":
                    idx += 1
                elif _ENV_ASSIGN_RE.match(tok):
                    var, val = tok.split("=", 1)
                    if not (var == "PATH" and strip_std_bin):
                        for part in (val.split(":") if "://" not in val else [val]):
                            out.extend(extract_paths_from_string(part, cwd=cwd))
                    idx += 1
                else:
                    break
        elif wname == "nice":
            if idx < n and argv[idx] == "-n" and idx + 1 < n:
                idx += 2
            elif idx < n and re.fullmatch(r"-\d+", argv[idx]):
                idx += 1
        elif wname == "command":
            if idx < n and argv[idx] in ("-p", "--"):
                idx += 1

    return idx if idx < n else None


def _extract_sed_args(args: tuple[str, ...], cwd: str, out: list[str]) -> None:
    has_explicit_script = False
    saw_positional_script = False
    end_opts = False
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if end_opts or not tok.startswith("-") or tok == "-":
            if not has_explicit_script and not saw_positional_script:
                saw_positional_script = True
                for wm in _SED_FILE_CMD_RE.finditer(tok):
                    out.extend(extract_paths_from_string(wm.group(1), cwd=cwd))
            else:
                _append_kv_or_arg(tok, cwd, out)
            i += 1
            continue
        if tok == "--":
            end_opts = True
            i += 1
            continue
        if tok in ("-e", "--expression"):
            has_explicit_script = True
            if i + 1 < n:
                for wm in _SED_FILE_CMD_RE.finditer(args[i + 1]):
                    out.extend(extract_paths_from_string(wm.group(1), cwd=cwd))
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--expression=") or (tok.startswith("-e") and len(tok) > 2):
            has_explicit_script = True
            expr = tok.split("=", 1)[1] if tok.startswith("--") else tok[2:]
            for wm in _SED_FILE_CMD_RE.finditer(expr):
                out.extend(extract_paths_from_string(wm.group(1), cwd=cwd))
            i += 1
            continue
        if tok in ("-f", "--file"):
            has_explicit_script = True
            if i + 1 < n:
                _append_kv_or_arg(args[i + 1], cwd, out)
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--file=") or (tok.startswith("-f") and len(tok) > 2):
            has_explicit_script = True
            farg = tok.split("=", 1)[1] if tok.startswith("--") else tok[2:]
            out.extend(extract_paths_from_string(farg, cwd=cwd))
            i += 1
            continue
        if tok == "-i":
            next_tok = args[i + 1] if i + 1 < n else None
            rem_pos = 0
            k = i + 1
            while k < n:
                if args[k] in ("-e", "--expression", "-f", "--file"):
                    k += 2
                    continue
                if args[k] != "--" and not args[k].startswith("-"):
                    rem_pos += 1
                k += 1
            if (
                next_tok is not None
                and (next_tok == "" or (next_tok.startswith(".") and "/" not in next_tok))
                and rem_pos >= 2
            ):
                i += 2
                continue
        i += 1


def _extract_awk_args(args: tuple[str, ...], cwd: str, out: list[str]) -> None:
    has_prog_file = False
    saw_inline_prog = False
    end_opts = False
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if end_opts or not tok.startswith("-") or tok == "-":
            if not has_prog_file and not saw_inline_prog:
                saw_inline_prog = True
                for m in _AWK_DQ_STR_RE.finditer(tok):
                    out.extend(extract_paths_from_string(m.group(1), cwd=cwd))
            else:
                _append_kv_or_arg(tok, cwd, out)
            i += 1
            continue
        if tok == "--":
            end_opts = True
            i += 1
            continue
        if tok in ("-f", "--file"):
            has_prog_file = True
            if i + 1 < n:
                _append_kv_or_arg(args[i + 1], cwd, out)
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--file=") or (tok.startswith("-f") and len(tok) > 2):
            has_prog_file = True
            farg = tok.split("=", 1)[1] if tok.startswith("--") else tok[2:]
            out.extend(extract_paths_from_string(farg, cwd=cwd))
            i += 1
            continue
        if tok in ("-F", "--field-separator"):
            i += 2
            continue
        if tok in ("-v", "--assign"):
            if i + 1 < n:
                _append_kv_or_arg(args[i + 1], cwd, out)
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--assign=") or (tok.startswith("-v") and len(tok) > 2):
            val = tok.split("=", 1)[1] if tok.startswith("--assign=") else tok[2:]
            _append_kv_or_arg(val, cwd, out)
            i += 1
            continue
        i += 1


def _extract_jq_args(args: tuple[str, ...], cwd: str, out: list[str]) -> None:
    has_filter_file = False
    saw_inline_filter = False
    end_opts = False
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if end_opts or not tok.startswith("-") or tok == "-":
            if not has_filter_file and not saw_inline_filter:
                saw_inline_filter = True
            else:
                _append_kv_or_arg(tok, cwd, out)
            i += 1
            continue
        if tok == "--":
            end_opts = True
            i += 1
            continue
        if tok in ("-f", "--from-file"):
            has_filter_file = True
            if i + 1 < n:
                _append_kv_or_arg(args[i + 1], cwd, out)
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("--from-file=") or (tok.startswith("-f") and len(tok) > 2):
            has_filter_file = True
            farg = tok.split("=", 1)[1] if tok.startswith("--from-file=") else tok[2:]
            out.extend(extract_paths_from_string(farg, cwd=cwd))
            i += 1
            continue
        if tok == "-L":
            if i + 1 < n:
                _append_kv_or_arg(args[i + 1], cwd, out)
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("-L") and len(tok) > 2:
            out.extend(extract_paths_from_string(tok[2:], cwd=cwd))
            i += 1
            continue
        if tok in ("--arg", "--argjson", "--slurpfile", "--rawfile"):
            if i + 2 < n:
                _append_kv_or_arg(args[i + 2], cwd, out)
                i += 3
            else:
                i = n
            continue
        i += 1


def _extract_python_cmd_args(
    argv: tuple[str, ...],
    cmd_idx: int,
    cmd_name: str,
    cwd: str,
    strip_std_bin: bool,
    out: list[str],
) -> bool:
    if _PYTHON_BIN_RE.fullmatch(cmd_name):
        py_idx: int | None = cmd_idx
    elif cmd_name in _PYTHON_LAUNCHERS:
        py_idx = None
        for k in range(cmd_idx + 1, len(argv)):
            if _PYTHON_BIN_RE.fullmatch(Path(argv[k]).name):
                py_idx = k
                break
        if py_idx is None:
            return False
        for mid in argv[cmd_idx + 1 : py_idx]:
            _append_kv_or_arg(mid, cwd, out)
        _append_exec_tok(argv[py_idx], cwd, strip_std_bin, out)
    else:
        return False

    k = py_idx + 1
    n = len(argv)
    while k < n:
        tok = argv[k]
        if tok in ("-W", "-X", "--check-hash-based-pycs"):
            if k + 1 < n:
                _append_kv_or_arg(argv[k + 1], cwd, out)
                k += 2
            else:
                k += 1
            continue
        if tok == "-m" or not tok.startswith("-") or tok in ("-", "--"):
            break
        m_c = re.fullmatch(r"-[a-zA-Z]*c(.*)", tok)
        if m_c and not tok.startswith("--"):
            if m_c.group(1):
                out.extend(extract_python_code_paths(m_c.group(1), cwd=cwd))
                k += 1
            elif k + 1 < n:
                out.extend(extract_python_code_paths(argv[k + 1], cwd=cwd))
                k += 2
            else:
                k += 1
            break
        k += 1

    _extract_default_args(argv[k:], cwd, out)
    return True


def _extract_argv_paths(
    argv: tuple[str, ...],
    cwd: str,
    strip_std_bin: bool,
    out: list[str],
    depth: int = 0,
) -> None:
    cmd_idx = _consume_leading_wrappers(argv, cwd, strip_std_bin, out)
    if cmd_idx is None:
        return
    cmd_tok = argv[cmd_idx]
    _append_exec_tok(cmd_tok, cwd, strip_std_bin, out)
    cmd_name = Path(cmd_tok).name
    rest = argv[cmd_idx + 1 :]

    if cmd_name in _NESTED_SHELL_NAMES:
        for j, tok in enumerate(rest):
            if re.fullmatch(r"-[a-zA-Z]*c", tok) and j + 1 < len(rest):
                _extract_default_args(rest[:j], cwd, out)
                out.extend(
                    _extract_command_ext_paths(
                        rest[j + 1],
                        cwd=cwd,
                        strip_std_bin=strip_std_bin,
                        _depth=depth + 1,
                    )
                )
                _extract_default_args(rest[j + 2 :], cwd, out)
                return

    if _extract_python_cmd_args(argv, cmd_idx, cmd_name, cwd, strip_std_bin, out):
        return

    if cmd_name in ("sed", "gsed"):
        _extract_sed_args(rest, cwd, out)
    elif cmd_name in ("awk", "gawk", "nawk"):
        _extract_awk_args(rest, cwd, out)
    elif cmd_name == "jq":
        _extract_jq_args(rest, cwd, out)
    else:
        _extract_default_args(rest, cwd, out)


def _extract_command_ext_paths(
    code_only: str,
    cwd: str = "",
    strip_std_bin: bool = True,
    _depth: int = 0,
) -> list[str]:
    """Extract path tokens using stdlib `shlex` + Python `ast.parse`, failing closed on parse errors."""
    if (
        "/" not in code_only
        and "~" not in code_only
        and ".." not in code_only
        and "python" not in code_only.lower()
    ):
        return []

    parsed = parse_shell_structure(code_only)
    if parsed is None or _depth > 4:
        paths: list[str] = []
        for raw_seg in COMPOUND_SPLIT_RE.split(code_only):
            seg = _strip_path_env_in_segment(raw_seg)
            if not seg:
                continue
            m_cmd = re.match(r"^(\S+)(.*)$", seg, re.DOTALL)
            if m_cmd:
                cmd_tok, rest = m_cmd.group(1), m_cmd.group(2)
                if cmd_tok.startswith(("/", "~", "../")):
                    norm_cmd = normalize_path(cmd_tok, cwd=cwd)
                    if strip_std_bin and is_standard_bin_path(norm_cmd) and not _is_sensitive_path(norm_cmd):
                        seg = rest.lstrip()
            for pm in PATH_TOKEN_RE.finditer(seg):
                if len(pm.group(0)) >= 3:
                    paths.append(normalize_path(pm.group(0), cwd=cwd))
        return paths

    paths = []
    for _, target in parsed["redirects"]:
        if target in DISCARD_REDIRECT_TARGETS:
            continue
        paths.append(normalize_path(target, cwd=cwd))

    for argv in parsed["argv_segments"]:
        _extract_argv_paths(argv, cwd=cwd, strip_std_bin=strip_std_bin, out=paths, depth=_depth)

    for header_line, body_text in parsed["heredocs"]:
        if not body_text:
            continue
        head_part = header_line.split("<<", 1)[0].strip()
        head_parsed = parse_shell_structure(head_part) if head_part else None
        head_argvs = head_parsed["argv_segments"] if head_parsed is not None else ()
        head_cmd = ""
        if head_argvs:
            tmp_out: list[str] = []
            c_idx = _consume_leading_wrappers(head_argvs[-1], cwd, strip_std_bin, tmp_out)
            if c_idx is not None:
                head_cmd = Path(head_argvs[-1][c_idx]).name
        if _PYTHON_BIN_RE.fullmatch(head_cmd):
            paths.extend(extract_python_code_paths(body_text, cwd=cwd))
        elif head_cmd in _NESTED_SHELL_NAMES:
            paths.extend(
                _extract_command_ext_paths(
                    body_text,
                    cwd=cwd,
                    strip_std_bin=strip_std_bin,
                    _depth=_depth + 1,
                )
            )
        elif head_cmd in ("cat", "tee") and len(head_argvs) == 1 and "|" not in header_line:
            continue
        else:
            for pm in PATH_TOKEN_RE.finditer(body_text):
                if len(pm.group(0)) >= 3:
                    paths.append(normalize_path(pm.group(0), cwd=cwd))

    return paths


def _normalize_segment_for_routine(seg: str, norm_workspaces: list[str] | None = None) -> str:
    s = _strip_path_env_in_segment(seg)
    if not s or s.startswith("#"):
        return "true"
    s = re.sub(r"(?:(?:\s+\d*|\b)>>?\s*/dev/null|\s+\d*>&\d+)+\s*$", "", s).strip()
    m_cmd = re.match(r"^(/\S+)(\s.*|$)", s, re.DOTALL)
    if m_cmd:
        cmd_tok, rest = m_cmd.group(1), m_cmd.group(2)
        norm_cmd = normalize_path(cmd_tok)
        in_ws = bool(norm_workspaces and any(is_path_under(norm_cmd, ws) for ws in norm_workspaces))
        if is_standard_bin_path(norm_cmd) or in_ws:
            s = Path(norm_cmd).name + rest
    return s


def _matches_routine_segment(norm_seg: str) -> bool:
    if any(pat.search(norm_seg) for pat in ROUTINE_SEGMENT_PATTERNS):
        return True
    if re.match(r"^gh\s+api\b", norm_seg) and not _GH_API_MUTATION_FLAG_RE.search(norm_seg):
        return True
    return False


def _has_non_discard_file_redirect(code_only: str) -> bool:
    return bool(extract_shell_output_redirects(code_only))


_SSH_ARG_FLAGS = frozenset("BbcDEeFIiJLlmOopQRSWw")
_SSH_NOARG_FLAGS = frozenset("1246AaCfGgKkMNnqsTtVvXxYy")
_SSH_SENSITIVE_TOKEN_RE = re.compile(
    r"(?:\.ssh/|\.aws/|\.gnupg/|\.netrc|(?:^|/|\s)\.env(?:\.(?:local|prod|production|secret))?\b|/etc/(?:shadow|master\.passwd))"
)
_SSH_READONLY_SEGMENT_RE = re.compile(
    r"^(?:"
    r"(?:ls|tree|wc|du|df|stat|file|which|pwd|whoami|uname|uptime|nproc|ps|pgrep|nvidia-smi|cat|head|tail|grep|rg|dirname|basename|realpath|readlink)(?:\s|$)"
    r"|git\s+(?:-C\s+\S+\s+)?(?:status|diff|log|show|rev-parse|branch(?:\s+(?:--show-current|-a|-v|-vv|--list))?)(?:\s|$)"
    r"|echo(?:\s|$)"
    r"|true$|false$"
    r")"
)


def _quote_join(tokens: tuple[str, ...] | list[str]) -> str:
    return " ".join(
        shlex.quote(tok) if (not tok or " " in tok or any(c in tok for c in "\"'\\;&|<>()\n#")) else tok
        for tok in tokens
    )


_SCP_SFTP_ARG_OPTS = frozenset(
    {"-c", "-F", "-i", "-J", "-l", "-o", "-P", "-S", "-D", "-b", "-B", "-R", "-s"}
)
_RSYNC_ARG_OPTS = frozenset(
    {
        "-e", "-f", "--rsh", "--rsync-path", "--exclude", "--include", "--exclude-from",
        "--include-from", "--files-from", "--log-file", "--password-file", "--port",
        "--address", "--bwlimit", "--timeout", "--contimeout", "--temp-dir",
        "--compare-dest", "--copy-dest", "--link-dest", "--backup-dir", "--suffix",
        "--chmod", "--chown", "--usermap", "--groupmap", "--iconv", "--max-size",
        "--min-size", "--modify-window", "--block-size", "--out-format", "--info",
        "--debug", "--filter",
    }
)
_REMOTE_URL_HOST_RE = re.compile(r"^(?:scp|sftp|rsync)://([^/]+)(?:/|$)", re.IGNORECASE)


def _normalize_remote_host(raw_host: str, from_url: bool = False) -> str:
    h = raw_host.strip()
    if "@" in h:
        h = h.split("@", 1)[1]
    if h.startswith("[") and "]" in h:
        h = h[1 : h.index("]")]
    elif from_url and ":" in h:
        h = h.split(":", 1)[0]
    h = h.strip().lower()
    if not h or h.startswith("-") or "/" in h or "\\" in h:
        return ""
    return h


def _extract_remote_transfer_hosts(sanitized_cmd: str) -> list[str]:
    """Extract normalized remote hosts from `scp`, `sftp`, or `rsync` command segments."""
    parsed = parse_shell_structure(sanitized_cmd)
    if parsed is None:
        return []
    hosts: list[str] = []
    seen: set[str] = set()
    for argv in parsed["argv_segments"]:
        tmp_out: list[str] = []
        cmd_idx = _consume_leading_wrappers(argv, "", True, tmp_out)
        if cmd_idx is None:
            continue
        cmd_name = Path(argv[cmd_idx]).name
        if cmd_name not in ("scp", "sftp", "rsync"):
            continue
        arg_opts = _RSYNC_ARG_OPTS if cmd_name == "rsync" else _SCP_SFTP_ARG_OPTS
        args = argv[cmd_idx + 1 :]
        i = 0
        n = len(args)
        end_opts = False
        while i < n:
            tok = args[i]
            if not end_opts:
                if tok == "--":
                    end_opts = True
                    i += 1
                    continue
                if tok in arg_opts:
                    i += 2
                    continue
                if tok.startswith("--") and "=" in tok:
                    i += 1
                    continue
                if cmd_name in ("scp", "sftp") and tok.startswith("-") and len(tok) > 2 and tok[:2] in _SCP_SFTP_ARG_OPTS:
                    i += 1
                    continue
                if cmd_name == "rsync" and tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and tok[:2] in ("-e", "-f"):
                    i += 1
                    continue
                if tok.startswith("-") and tok != "-":
                    i += 1
                    continue
            if tok in ("~", ".", "..", "-") or tok.startswith(("/", "~/", "./", "../")):
                i += 1
                continue
            low_tok = tok.lower()
            if low_tok.startswith(("http://", "https://", "file://")):
                i += 1
                continue
            url_m = _REMOTE_URL_HOST_RE.match(tok)
            cand = ""
            if url_m:
                cand = _normalize_remote_host(url_m.group(1), from_url=True)
            elif ":" in tok:
                host_part, _ = tok.split(":", 1)
                if "@" in host_part or len(host_part) >= 2:
                    cand = _normalize_remote_host(host_part)
            elif cmd_name == "sftp" and "@" in tok and "/" not in tok:
                cand = _normalize_remote_host(tok)
            if cand and cand not in seen:
                seen.add(cand)
                hosts.append(cand)
            i += 1
    return hosts


def _parse_ssh_invocation(sanitized_cmd: str) -> tuple[str, str] | None:
    """Parse a single `ssh` invocation into `(normalized_host, remote_command)`."""
    parsed = parse_shell_structure(sanitized_cmd)
    if parsed is None or len(parsed["argv_segments"]) != 1:
        return None
    if any(t not in DISCARD_REDIRECT_TARGETS for _, t in parsed["redirects"]):
        return None
    argv = parsed["argv_segments"][0]
    tmp_out: list[str] = []
    cmd_idx = _consume_leading_wrappers(argv, "", True, tmp_out)
    if cmd_idx is None or Path(argv[cmd_idx]).name != "ssh":
        return None
    args = argv[cmd_idx + 1 :]
    i = 0
    n = len(args)
    dest = ""
    while i < n:
        tok = args[i]
        if tok == "--":
            if i + 1 < n:
                dest = args[i + 1]
                i += 2
            else:
                i = n
            break
        if tok.startswith("-") and len(tok) > 1:
            if tok[1] in _SSH_ARG_FLAGS:
                i += 1 if len(tok) > 2 else 2
                continue
            if all(ch in _SSH_NOARG_FLAGS for ch in tok[1:]):
                i += 1
                continue
            return None
        dest = tok
        i += 1
        break
    if not dest:
        return None
    host = dest.split("@", 1)[1] if "@" in dest else dest
    if host.startswith("[") and "]" in host:
        host = host[1 : host.index("]")]
    host = host.strip().lower()
    if not host or host.startswith("-"):
        return None
    # OpenSSH concatenates trailing remote command arguments with spaces before sending to the remote shell.
    remote_cmd = " ".join(args[i:]).strip() if i < n else ""
    return host, remote_cmd


def _unwrap_wsl_exec(remote_cmd: str) -> str:
    """Unwrap `wsl.exe [--distribution ...] --exec <cmd...>` into its inner command string."""
    parsed = parse_shell_structure(remote_cmd)
    if parsed is None or len(parsed["argv_segments"]) != 1:
        return remote_cmd
    argv = parsed["argv_segments"][0]
    if not argv or Path(argv[0]).name not in ("wsl", "wsl.exe"):
        return remote_cmd
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok in ("-e", "--exec"):
            if i + 1 >= n:
                return ""
            return _quote_join(argv[i + 1 :])
        if tok in ("-d", "--distribution", "-u", "--user", "--cd", "--shell-type"):
            i += 2
            continue
        if tok.startswith(("--distribution=", "--user=", "--cd=")):
            i += 1
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    return remote_cmd


def _matches_force_ask(sanitized_cmd: str, _depth: int = 0) -> str | None:
    """Check `FORCE_ASK_COMMAND_PATTERNS` directly and inside nested `sh -c` / `ssh` / `wsl --exec` wrappers."""
    cleaned = strip_commit_message_literals(strip_purpose_comment(sanitized_cmd))
    for label, pat in FORCE_ASK_COMMAND_PATTERNS:
        if pat.search(cleaned):
            return label
    if _depth >= 3:
        return None
    parsed = parse_shell_structure(cleaned)
    if parsed is not None:
        for argv in parsed["argv_segments"]:
            tmp_out: list[str] = []
            c_idx = _consume_leading_wrappers(argv, "", True, tmp_out)
            if c_idx is None:
                continue
            if Path(argv[c_idx]).name in _NESTED_SHELL_NAMES:
                rest = argv[c_idx + 1 :]
                for j, tok in enumerate(rest):
                    if re.fullmatch(r"-[a-zA-Z]*c", tok) and j + 1 < len(rest):
                        found = _matches_force_ask(rest[j + 1], _depth=_depth + 1)
                        if found:
                            return found
                        break
    ssh_info = _parse_ssh_invocation(cleaned)
    if ssh_info is not None and ssh_info[1]:
        remote_cmd = ssh_info[1]
        found = _matches_force_ask(remote_cmd, _depth=_depth + 1)
        if found:
            return found
        unwrapped = _unwrap_wsl_exec(remote_cmd)
        if unwrapped != remote_cmd:
            found = _matches_force_ask(unwrapped, _depth=_depth + 1)
            if found:
                return found
    return None


def _is_strictly_readonly_remote_cmd(remote_cmd: str) -> bool:
    """Return True only when every segment of `remote_cmd` is strictly read-only inspection."""
    inner = _unwrap_wsl_exec(remote_cmd).strip()
    if (
        not inner
        or UNSAFE_SHELL_EXPANSION_RE.search(inner)
        or DESTRUCTIVE_SCOPE_COMMAND_RE.search(inner)
        or _SSH_SENSITIVE_TOKEN_RE.search(inner)
        or _matches_force_ask(inner)
        or _has_non_discard_file_redirect(inner)
    ):
        return False
    segments = split_shell_segments(inner)
    if not segments:
        return False
    return all(_SSH_READONLY_SEGMENT_RE.search(_normalize_segment_for_routine(seg)) for seg in segments)


def _dir_grant_candidates(path_str: str, include_self: bool = False, max_levels: int = 4) -> list[str]:
    """Return ancestor directories (up to `max_levels`) eligible for `session_dir_cache` grants,
    stopping before root, top-level OS dirs (`/opt`, `/Users`, ...), and `Path.home()`.
    """
    home_path = Path.home().resolve()
    p = Path(path_str)
    raw_cands = ([p] if include_self else []) + list(p.parents)[:max_levels]
    out: list[str] = []
    for cand in raw_cands:
        if len(cand.parts) <= 2 or cand == Path.home() or cand == home_path:
            break
        out.append(str(cand))
    return out


def _has_cached_dir_grant(conversation_id: str, dir_path: str, needed_mode: str) -> bool:
    write_verdict = get_cached_verdict(
        conversation_id, compute_directory_grant_key(dir_path, "write")
    )
    if write_verdict is not None and write_verdict.get("effect") == "allow":
        return True
    if needed_mode == "read":
        read_verdict = get_cached_verdict(
            conversation_id, compute_directory_grant_key(dir_path, "read")
        )
        if read_verdict is not None and read_verdict.get("effect") == "allow":
            return True
    return False


def _command_dir_cache_mode(sanitized_cmd: str, bypass_sandbox: bool) -> str:
    if not bypass_sandbox:
        return "read"
    if _has_non_discard_file_redirect(sanitized_cmd):
        return "write"
    for seg in split_shell_segments(sanitized_cmd):
        cleaned_seg = _strip_path_env_in_segment(seg)
        if not cleaned_seg:
            continue
        is_mut, _ = _is_git_segment_mutating(cleaned_seg)
        if is_mut:
            return "write"
    return "read"


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


def is_routine_command_segments(command: str, norm_workspaces: list[str] | None = None) -> bool:
    code_only = strip_commit_message_literals(strip_purpose_comment(command))
    if not code_only or UNSAFE_SHELL_EXPANSION_RE.search(code_only) or _has_non_discard_file_redirect(code_only):
        return False
    segments = split_shell_segments(code_only)
    if not segments:
        return False
    return all(_matches_routine_segment(_normalize_segment_for_routine(seg, norm_workspaces)) for seg in segments)


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

    # Check output redirections: only allow redirects to /dev/null, &1/&2, or inside workspace/artifact/tmp
    output_redirects = extract_shell_output_redirects(code_only)
    has_file_redirect = bool(output_redirects)
    for target_tok in output_redirects:
        resolved = normalize_path(target_tok, cwd=cwd)
        if not is_safe_scoped_path(resolved, norm_workspaces, artifact_dir):
            return False, ""

    # Check sensitive read paths across all path tokens, and external paths excluding harmless std bin prefixes
    all_raw_paths = _extract_command_ext_paths(code_only, cwd=cwd, strip_std_bin=False)
    for sp in SENSITIVE_READ_PATHS:
        if any(is_path_under(p, sp) for p in all_raw_paths):
            return False, ""

    ext_paths = _extract_command_ext_paths(code_only, cwd=cwd, strip_std_bin=True)
    outside_paths = [
        p for p in ext_paths if not is_safe_scoped_path(p, norm_workspaces, artifact_dir)
    ]

    segments = split_shell_segments(code_only)
    if not segments:
        return False, ""

    all_routine = all(
        _matches_routine_segment(_normalize_segment_for_routine(seg, norm_workspaces))
        or (bool(ext_paths) and bool(SCOPED_CONTENT_SEGMENT_RE.search(_normalize_segment_for_routine(seg, norm_workspaces))))
        for seg in segments
    )
    if all_routine and not outside_paths and not has_file_redirect:
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
    if tool_name and tool_name not in REVIEWED_TOOLS and not tool_name.startswith(REVIEWED_TOOL_PREFIXES):
        return {
            "decision": "allow",
            "reason": f"[Tier 0 Policy] Internal non-filesystem/non-shell tool ({tool_name}).",
            "tier": "tier0_static",
        }
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
        bypass_sandbox = bool(tool_args.get("BypassSandbox", False))
        sanitized_cmd = strip_commit_message_literals(strip_purpose_comment(cmd))
        for label, pat in list(HARD_DENY_COMMAND_PATTERNS) + build_local_hard_deny_patterns(local_cfg):
            if pat.search(sanitized_cmd):
                return {
                    "decision": "deny",
                    "reason": f"[Tier 0 Policy] Blocked catastrophic command pattern: {label}.",
                    "tier": "tier0_static",
                }
        if sys.platform == "darwin" and not bypass_sandbox and (
            SANDBOX_INCOMPATIBLE_COMMAND_RE.search(sanitized_cmd)
            or (
                not DESTRUCTIVE_SCOPE_COMMAND_RE.search(sanitized_cmd)
                and _is_workspace_git_write(sanitized_cmd, norm_workspaces, cwd)
            )
        ):
            return {
                "decision": "deny",
                "reason": (
                    "[Tier 0 Policy] Command requires network, MPI shared-memory IPC, tmux socket, host process, "
                    "or workspace `.git` write permissions that deterministically fail inside the macOS sandbox "
                    "(BypassSandbox=false). Re-issue once with BypassSandbox=true (only after explicit user confirmation "
                    "for public/remote mutations like git push or gh writes)."
                ),
                "tier": "tier0_sandbox_network_mismatch",
            }
        force_ask_label = _matches_force_ask(sanitized_cmd)
        if force_ask_label:
            return {
                "decision": "ask",
                "reason": f"[Tier 0 Policy] {force_ask_label}.",
                "tier": "tier0_force_ask",
            }
        ok, reason = is_fast_allow_command(
            command=cmd,
            bypass_sandbox=bypass_sandbox,
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
    """Tier 1: Zero-token check against opt-in verbal grants extracted from USER_INPUT steps."""
    denied_paths: list[str] = ledger.get("denied_paths", [])
    path_grants: list[dict[str, Any]] = ledger.get("path_grants", [])
    command_grants: list[str] = ledger.get("command_grants", [])
    if not denied_paths and not path_grants and not command_grants:
        return None
    norm_workspaces = [normalize_path(p) for p in workspace_paths if p]

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
        raw_paths = _extract_command_ext_paths(cmd, cwd=cwd, strip_std_bin=False)
        for ep in raw_paths:
            for dp in denied_paths:
                if is_path_under(ep, dp):
                    return {
                        "decision": "deny",
                        "reason": f"[Tier 1 Verbal Ledger] Command references prohibited path '{dp}'.",
                        "tier": "tier1_ledger",
                    }

        ext_paths = _extract_command_ext_paths(cmd, cwd=cwd)
        outside_paths = [
            p
            for p in ext_paths
            if not any(is_path_under(p, ws) for ws in norm_workspaces)
            and not is_path_under(p, "/tmp")
            and not is_path_under(p, "/private/tmp")
        ]
        needs_write = _command_dir_cache_mode(cmd, bool(tool_args.get("BypassSandbox", False))) == "write"
        if (
            outside_paths
            and not DESTRUCTIVE_SCOPE_COMMAND_RE.search(cmd)
            and all(
                any(
                    is_path_under(op, pg["path"]) and (not needs_write or pg["mode"] == "write")
                    for pg in path_grants
                )
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
        annotations = _parse_review_annotations(cmd)
        code_only = strip_commit_message_literals(strip_purpose_comment(cmd))
        ext_paths = _extract_command_ext_paths(code_only, cwd=cwd)
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
        ssh_info = _parse_ssh_invocation(code_only)
        local_cfg = load_local_config()
        known_hosts = {
            str(h).strip().lower()
            for h in local_cfg.get("known_ssh_hosts", [])
            if str(h).strip()
        }
        hpc_hosts = {
            str(h).strip().lower()
            for h in local_cfg.get("shared_hpc_hosts", [])
            if str(h).strip()
        }
        if ssh_info is not None:
            ssh_host = ssh_info[0]
            details["sshHost"] = ssh_host
            if ssh_host in hpc_hosts:
                details["sharedHpcHost"] = True
                details["knownUserSshHost"] = True
            elif ssh_host in known_hosts:
                details["knownUserSshHost"] = True
        else:
            transfer_hosts = _extract_remote_transfer_hosts(code_only)
            if transfer_hosts:
                details["remoteHosts"] = transfer_hosts
                if len(transfer_hosts) == 1:
                    details["sshHost"] = transfer_hosts[0]
                if any(h in hpc_hosts for h in transfer_hosts):
                    details["sharedHpcHost"] = True
                if all(h in (known_hosts | hpc_hosts) for h in transfer_hosts):
                    details["knownUserSshHost"] = True
        if active_grants:
            details["activeVerbalGrants"] = active_grants[-6:]
        if annotations["purpose"]:
            details["purposeComment"] = annotations["purpose"]
        if annotations["auth"]:
            all_entries = list(
                (ledger or {}).get("all_user_entries_with_context")
                or (ledger or {}).get("all_user_entries")
                or []
            )
            details["citedUserAuth"] = verify_cited_user_authorization(
                annotations["auth"], all_entries
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
    if not artifact_dir:
        for sep_marker in ("/.system_generated/", "\\.system_generated\\"):
            if sep_marker in transcript_str:
                artifact_dir = transcript_str.split(sep_marker, 1)[0]
                break
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
                annotations = (
                    _parse_review_annotations(str(tool_args.get("CommandLine", "")))
                    if tool_name == "run_command"
                    else {"auth": "", "purpose": "", "appeal_requested": False}
                )
                appeal_requested = bool(annotations["appeal_requested"])
                is_destructive_cmd = False
                if tool_name == "run_command":
                    sanitized_for_dest = strip_commit_message_literals(
                        strip_purpose_comment(str(tool_args.get("CommandLine", "")))
                    )
                    is_destructive_cmd = bool(DESTRUCTIVE_SCOPE_COMMAND_RE.search(sanitized_for_dest))
                is_file_tool = tool_name in (
                    "read_file",
                    "write_file",
                    "edit_file",
                    "view_file",
                    "write_to_file",
                    "replace_file_content",
                    "delete_file",
                )
                file_mode = (
                    "write"
                    if tool_name in ("write_file", "edit_file", "write_to_file", "replace_file_content", "delete_file")
                    else "read"
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
                # Also check if a routine command (`git -C <dir>`, `unittest -s <dir>`) or a sandboxed
                # non-destructive command only references external directories already approved in this session,
                # or if a strictly read-only SSH command targets a host already approved in this session.
                cmd_outside_dirs: list[str] = []
                cmd_dir_mode = "read"
                ssh_info: tuple[str, str] | None = None
                if tool_name == "run_command":
                    cmd_str = str(tool_args.get("CommandLine", ""))
                    bypass_sb = bool(tool_args.get("BypassSandbox", False))
                    norm_ws = [normalize_path(p) for p in workspace_paths if p]
                    sanitized = strip_commit_message_literals(strip_purpose_comment(cmd_str))
                    ssh_info = _parse_ssh_invocation(sanitized)
                    cmd_dir_mode = _command_dir_cache_mode(sanitized, bypass_sb)
                    if (
                        (is_routine_command_segments(cmd_str, norm_workspaces=norm_ws) or not bypass_sb)
                        and not DESTRUCTIVE_SCOPE_COMMAND_RE.search(sanitized)
                    ):
                        ext_p = _extract_command_ext_paths(sanitized, cwd=cwd)
                        if not any(
                            any(is_path_under(p, sp) for sp in SENSITIVE_READ_PATHS)
                            for p in ext_p
                        ):
                            cmd_outside_dirs = [
                                p for p in ext_p if not is_safe_scoped_path(p, norm_ws, artifact_dir)
                            ]

                cached = None if appeal_requested else get_cached_verdict(conversation_id, req_key)
                if not appeal_requested and cached is None and dir_sig is not None:
                    sig_cached = get_cached_verdict(conversation_id, dir_sig)
                    if sig_cached is not None and sig_cached.get("effect") == "allow":
                        cached = sig_cached
                if cached is not None:
                    cached = {**cached, "source": "cache", "model": "session_cache"}
                if not appeal_requested and cached is None and is_file_tool:
                    raw_p = str(
                        tool_args.get("Path")
                        or tool_args.get("TargetFile")
                        or tool_args.get("AbsolutePath")
                        or ""
                    )
                    if raw_p:
                        norm_p = normalize_path(raw_p, cwd=cwd)
                        for cd in _dir_grant_candidates(norm_p, include_self=False, max_levels=4):
                            if _has_cached_dir_grant(conversation_id, cd, file_mode):
                                cached = {
                                    "effect": "allow",
                                    "source": "cache",
                                    "model": "session_dir_cache",
                                    "reasonCode": "cached_dir_grant",
                                    "reason": f"File operation ({file_mode}) targets a repository/directory already approved in this session.",
                                }
                                break
                if not appeal_requested and cached is None and cmd_outside_dirs:
                    # Check if every referenced external path (or an ancestor up to 4 levels) has an approved dir_grant
                    all_dirs_approved = True
                    for op in cmd_outside_dirs:
                        cand_dirs = _dir_grant_candidates(op, include_self=True, max_levels=4)
                        if not cand_dirs or not any(
                            _has_cached_dir_grant(conversation_id, cd, cmd_dir_mode)
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
                            "reason": f"Routine command ({cmd_dir_mode}) targets an external directory already approved in this session.",
                        }
                if (
                    not appeal_requested
                    and cached is None
                    and ssh_info is not None
                    and ssh_info[1]
                    and _is_strictly_readonly_remote_cmd(ssh_info[1])
                ):
                    ssh_verdict = get_cached_verdict(
                        conversation_id, compute_ssh_host_grant_key(ssh_info[0])
                    )
                    if ssh_verdict is not None and ssh_verdict.get("effect") == "allow":
                        cached = {
                            "effect": "allow",
                            "source": "cache",
                            "model": "session_ssh_readonly_cache",
                            "reasonCode": "cached_ssh_readonly_grant",
                            "reason": "Read-only SSH inspection on a host already approved in this session.",
                        }
                if cached is not None:
                    verdict = cached
                    res = apply_unresolved_send_back(
                        conversation_id=conversation_id,
                        verdict=verdict,
                        unresolved_mode=os.environ.get("AGY_UNRESOLVED_MODE", "agent"),
                        max_send_backs=int(os.environ.get("AGY_MAX_SEND_BACKS", str(DEFAULT_MAX_SEND_BACKS))),
                    )
                else:
                    requested_action = build_requested_action(
                        tool_name,
                        tool_args,
                        workspace_paths=workspace_paths,
                        cwd=cwd,
                        artifact_dir=artifact_dir,
                        ledger=ledger,
                    )
                    local_cfg = load_local_config()
                    decision_models = _resolve_model_list(
                        "AGY_DECISION_MODELS",
                        local_cfg,
                        "decision_models",
                        DEFAULT_DECISION_MODELS,
                    )
                    escalation_models = _resolve_model_list(
                        "AGY_ESCALATION_MODELS",
                        local_cfg,
                        "escalation_models",
                        DEFAULT_ESCALATION_MODELS,
                        append_fallbacks=DEFAULT_ESCALATION_MODELS,
                    )
                    deep_models = _resolve_model_list(
                        "AGY_DEEP_REVIEW_MODELS",
                        local_cfg,
                        "deep_review_models",
                        escalation_models,
                        append_fallbacks=escalation_models,
                    )
                    paid_models = [
                        m.strip()
                        for m in os.environ.get(
                            "AGY_PAID_DECISION_MODELS", ",".join(DEFAULT_PAID_FALLBACK_MODELS)
                        ).split(",")
                        if m.strip()
                    ]
                    min_conf = float(
                        os.environ.get("AGY_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))
                    )
                    decision_timeout = float(
                        os.environ.get("AGY_DECISION_TIMEOUT_SEC", str(DEFAULT_DECISION_TIMEOUT_SEC))
                    )
                    escalation_timeout = float(
                        os.environ.get("AGY_ESCALATION_TIMEOUT_SEC", str(DEFAULT_ESCALATION_TIMEOUT_SEC))
                    )
                    deep_timeout = float(
                        os.environ.get("AGY_DEEP_REVIEW_TIMEOUT_SEC", str(DEFAULT_DEEP_REVIEW_TIMEOUT_SEC))
                    )
                    kwargs = {"api_caller": api_caller} if api_caller is not None else {}

                    def _find_relevant_earlier() -> list[str]:
                        query_text = " ".join(
                            [
                                str(tool_args.get("CommandLine", "")),
                                *(str(t) for t in (requested_action.get("targets") or [])),
                                *(str(p) for p in ((requested_action.get("details") or {}).get("externalPaths") or [])),
                            ]
                        )
                        return extract_relevant_earlier_user_turns(
                            query_text,
                            ledger.get("all_user_entries_with_context") or [],
                            exclude_step_indices=ledger.get("active_user_step_indices") or [],
                        )

                    if appeal_requested:
                        last_user_step = int(ledger.get("last_user_step_index", 0))
                        latest_step = max(
                            int(payload.get("stepIdx") or 0),
                            int(ledger.get("latest_step_index", 0)),
                        )
                        recent_steps_limit = int(os.environ.get("AGY_APPEAL_RECENT_STEPS", "12"))
                        has_user_entries = bool(ledger.get("all_user_entries"))
                        is_recent_user = has_user_entries and (latest_step - last_user_step) <= recent_steps_limit
                        cited_auth = (requested_action.get("details") or {}).get("citedUserAuth") or {}
                        cited_verified = cited_auth.get("verifiedFromTranscript") is True
                        if cited_auth.get("verifiedFromTranscript") is False or not (is_recent_user or cited_verified):
                            res = {
                                "decision": "deny",
                                "reason": (
                                    "Auto Permissions rejected # appeal: deep review requires either a recent user "
                                    "authorization turn or a transcript-verified `# auth: \"<exact phrase>\"` / `# auth: step <N>` citation."
                                ),
                                "tier": "appeal_gate_rejected",
                            }
                        else:
                            code_only_fp = strip_commit_message_literals(
                                strip_purpose_comment(str(tool_args.get("CommandLine", "")))
                            )
                            cmd_fp = hashlib.sha256(code_only_fp.encode("utf-8")).hexdigest()[:24]
                            max_appeals = int(os.environ.get("AGY_MAX_APPEALS_PER_TURN", "4"))
                            allowed_quota, quota_reason = check_and_record_appeal_quota(
                                conversation_id,
                                last_user_step,
                                cmd_fp,
                                max_per_turn=max_appeals,
                            )
                            if not allowed_quota:
                                res = {
                                    "decision": "deny",
                                    "reason": f"Auto Permissions rejected # appeal: {quota_reason}",
                                    "tier": "appeal_limit_exceeded",
                                }
                            else:
                                deep_tx = extract_compact_transcript(
                                    transcript_path,
                                    max_items=22,
                                    max_chars=5500,
                                    include_thinking=False,
                                    user_limit=1400,
                                    agent_limit=1200,
                                    tool_limit=360,
                                )
                                second = run_escalation_tier(
                                    escalation_models=deep_models,
                                    requested_action=requested_action,
                                    first_verdict={"reason": "agent requested deep/appeal review"},
                                    compact_transcript=deep_tx,
                                    timeout_sec=deep_timeout,
                                    appeal_mode=True,
                                    relevant_earlier_turns=_find_relevant_earlier(),
                                    **kwargs,
                                )
                                verdict = (
                                    second
                                    if second is not None
                                    else {
                                        "effect": "ask",
                                        "source": "model",
                                        "reasonCode": "appeal_failed",
                                        "reason": "deep review models unavailable or timed out",
                                    }
                                )
                                if verdict.get("effect") == "ask":
                                    res = {
                                        "decision": "deny",
                                        "reason": (
                                            f"Auto Permissions deep review could not approve this action ({verdict.get('reason', '')}). "
                                            "Stop and confirm with the user in plain chat text rather than retrying."
                                        ),
                                        "tier": verdict.get("model") or verdict.get("reasonCode", "appeal_unsure"),
                                    }
                                else:
                                    res = apply_unresolved_send_back(
                                        conversation_id=conversation_id,
                                        verdict=verdict,
                                        unresolved_mode=os.environ.get("AGY_UNRESOLVED_MODE", "agent"),
                                        max_send_backs=int(os.environ.get("AGY_MAX_SEND_BACKS", str(DEFAULT_MAX_SEND_BACKS))),
                                    )
                    else:
                        compact_tx = extract_compact_transcript(
                            transcript_path,
                            max_items=10,
                            max_chars=2000,
                            include_thinking=False,
                        )
                        first = run_jev_decision_tier(
                            models=decision_models,
                            min_confidence=min_conf,
                            user_messages=ledger.get("recent_user_messages", []),
                            compact_transcript=compact_tx,
                            requested_action=requested_action,
                            timeout_sec=decision_timeout,
                            **kwargs,
                        )
                        if first.get("effect") == "ask" and escalation_models:
                            esc_tx = extract_compact_transcript(
                                transcript_path,
                                max_items=20,
                                max_chars=5000,
                                include_thinking=False,
                                user_limit=1200,
                                agent_limit=900,
                                tool_limit=360,
                            )
                            second = run_escalation_tier(
                                escalation_models=escalation_models,
                                requested_action=requested_action,
                                first_verdict=first,
                                compact_transcript=esc_tx or compact_tx,
                                timeout_sec=escalation_timeout,
                                relevant_earlier_turns=_find_relevant_earlier(),
                                **kwargs,
                            )
                            if second is not None:
                                verdict = second
                            elif (
                                first.get("reasonCode") == "review_failed"
                                and "opencode binary not found" not in str(first.get("reason", ""))
                                and paid_models
                            ):
                                paid_tx = extract_compact_transcript(
                                    transcript_path,
                                    max_items=6,
                                    max_chars=1000,
                                    include_thinking=False,
                                )
                                verdict = run_jev_decision_tier(
                                    models=paid_models,
                                    min_confidence=min_conf,
                                    user_messages=ledger.get("recent_user_messages", [])[-2:],
                                    compact_transcript=paid_tx,
                                    requested_action=requested_action,
                                    timeout_sec=decision_timeout,
                                    **kwargs,
                                )
                            else:
                                verdict = first
                        else:
                            verdict = first

                        if not is_destructive_cmd:
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
                                    for cd in _dir_grant_candidates(norm_p, include_self=False, max_levels=2):
                                        put_cached_verdict(
                                            conversation_id,
                                            compute_directory_grant_key(cd, file_mode),
                                            verdict,
                                        )
                            for op in cmd_outside_dirs:
                                for cd in _dir_grant_candidates(op, include_self=True, max_levels=1):
                                    put_cached_verdict(
                                        conversation_id,
                                        compute_directory_grant_key(cd, cmd_dir_mode),
                                        verdict,
                                    )
                            if ssh_info is not None:
                                put_cached_verdict(
                                    conversation_id,
                                    compute_ssh_host_grant_key(ssh_info[0]),
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


def _find_latest_transcript() -> Path | None:
    home = Path.home() / ".gemini"
    if not home.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for app_dir in ("antigravity-cli", "antigravity", "antigravity-ide"):
        brain_dir = home / app_dir / "brain"
        if not brain_dir.is_dir():
            continue
        for tp in brain_dir.glob("*/.system_generated/logs/transcript.jsonl"):
            try:
                candidates.append((tp.stat().st_mtime, tp))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_reviewer_status(args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble live status, verbal grants, rotated usage/cost summary, and recent audit entries."""
    params = args or {}
    tp_str = params.get("transcript_path")
    tp = Path(tp_str) if tp_str else _find_latest_transcript()
    ledger = collect_session_ledger(tp)
    base = get_data_dir()
    req_cid = str(params.get("conversation_id") or "").strip()
    audit_entries: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    audit_dir = base / "audit"
    if audit_dir.is_dir():
        target_file: Path | None = None
        if req_cid:
            cand = audit_dir / f"{safe_conv_id(req_cid)}.jsonl"
            if cand.is_file():
                target_file = cand
        if target_file is None:
            files = sorted(audit_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                target_file = files[0]
        if target_file is not None:
            all_lines = target_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in all_lines:
                try:
                    obj = json.loads(ln)
                    t_key = str(obj.get("tier", "unknown"))
                    d_key = str(obj.get("decision", "unknown"))
                    tier_counts[t_key] = tier_counts.get(t_key, 0) + 1
                    decision_counts[d_key] = decision_counts.get(d_key, 0) + 1
                except Exception:
                    pass
            for ln in all_lines[-20:]:
                try:
                    audit_entries.append(json.loads(ln))
                except Exception:
                    pass
    usage_summary: dict[str, Any] = {"total_calls": 0, "total_cost_usd": 0.0, "by_model": {}, "by_outcome": {}}
    for usage_file in (base / "usage.jsonl.1", base / "usage.jsonl"):
        if not usage_file.is_file():
            continue
        for ln in usage_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                u = json.loads(ln)
                usage_summary["total_calls"] += 1
                usage_summary["total_cost_usd"] = round(
                    usage_summary["total_cost_usd"] + float(u.get("costUsd", 0.0)), 6
                )
                m_key = str(u.get("model", "unknown"))
                o_key = str(u.get("outcome", "unknown"))
                usage_summary["by_model"][m_key] = usage_summary["by_model"].get(m_key, 0) + 1
                usage_summary["by_outcome"][o_key] = usage_summary["by_outcome"].get(o_key, 0) + 1
            except Exception:
                pass
    compact_ledger = {
        "path_grants": ledger.get("path_grants", []),
        "denied_paths": ledger.get("denied_paths", []),
        "command_grants": ledger.get("command_grants", []),
        "recent_user_messages": ledger.get("recent_user_messages", []),
        "total_user_messages_in_transcript": len(ledger.get("all_user_entries", [])),
    }
    local_cfg = load_local_config()
    decision_models = _resolve_model_list(
        "AGY_DECISION_MODELS", local_cfg, "decision_models", DEFAULT_DECISION_MODELS
    )
    escalation_models = _resolve_model_list(
        "AGY_ESCALATION_MODELS",
        local_cfg,
        "escalation_models",
        DEFAULT_ESCALATION_MODELS,
        append_fallbacks=DEFAULT_ESCALATION_MODELS,
    )
    deep_models = _resolve_model_list(
        "AGY_DEEP_REVIEW_MODELS",
        local_cfg,
        "deep_review_models",
        escalation_models,
        append_fallbacks=escalation_models,
    )
    return {
        "mode": os.environ.get("AGY_REVIEWER_MODE", "opencode"),
        "model": ",".join(decision_models) if decision_models else "disabled",
        "escalation_models": escalation_models,
        "deep_review_models": deep_models,
        "opencode_available": find_opencode_binary() is not None,
        "session_tier_breakdown": tier_counts,
        "session_decision_breakdown": decision_counts,
        "llm_usage_summary": usage_summary,
        "verbal_grant_ledger": compact_ledger,
        "recent_audit_decisions": audit_entries,
    }


def main() -> int:
    try:
        for stream in (sys.stdin, sys.stdout):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
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
