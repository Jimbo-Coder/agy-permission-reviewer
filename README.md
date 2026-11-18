# agy-permission-reviewer

Context-aware `PreToolUse` permission reviewer plugin for Google Antigravity, designed to eliminate repetitive approval popups while enforcing strict security boundaries and zero-token/free-model review cascades.

## Layered Architecture

Naive auto-reviewers feed the entire 50k–200k token conversation transcript to a secondary LLM on every tool call. `agy-permission-reviewer` solves the cost, latency, and safety problem with a fast deterministic front gate paired with an anchored `$0` model cascade:

1. **Layer 1 — Deterministic Static Policy, Mode-Separated Session Caches & Fast-Allow (`0.1–4 ms`, 0 tokens)**:
   - Scoped `PreToolUse` matcher intercepts only permission-bearing tools (`run_command`, file tools, MCP, `browser_*`) so internal planning/task tools add zero overhead.
   - Uses Python stdlib `shlex` tokenization + lazy `ast.parse` inline-Python inspection (with a zero-metacharacter C-speed fast-path and `lru_cache` memoization) so arithmetic division, `sed`/`awk`/`jq` expressions, `cat` heredocs, `2>/dev/null`, and quoted redirects do not trigger false external-path matches, while `Path.home() / ...`, `os.path.join(...)`, `"..." + "..."`, and quoted paths with spaces are accurately resolved.
   - Immediately allows normal in-workspace file operations, routine read-only/module-test/build/local-git commands (`pytest`, `python3 -m unittest`, `make`), and sandboxed commands scoped to the workspace/scratch/tmp (arbitrary unsandboxed `python3 script.py` runs require Layer 2 review).
   - Mode-separated `session_dir_cache` (`read` vs `write`) and strictly read-only `session_ssh_readonly_cache` reuse prior Layer 2 approvals within the session without allowing a `read` grant to escalate to writes or mutating commands.
   - Immediately rejects deterministically sandbox-incompatible commands (outbound network `git push/pull/fetch`, `gh`, `ssh`, `scp`, remote `curl`/`wget`, `brew`/`pip install`, `opencode api`, `publish`; MPI launchers `mpirun`/`mpiexec`; host IPC/process tools `tmux`, `ps`, `pgrep`, `top`, `lsof`, `launchctl`, `osascript`; and workspace `.git`-mutating commands like `git add`/`commit`/`stash`) sent with `BypassSandbox=false` so agents re-issue once with `BypassSandbox=true` instead of wasting a doomed sandbox attempt.
   - Immediately blocks (`"decision": "deny"`) destructive system commands (`rm -rf /`, `mkfs`), credential exfiltration (`.ssh/id_*`, `.aws/credentials`, `.env`), and writes to protected system directories (`/etc`, `/System`, `/usr`, shell rc files).
   - Forces explicit user confirmation (`"decision": "force_ask"`) for public/external state mutations (`git push`, `gh issue/pr/release/repo`, `gh api -X POST/PATCH/...`, `npm/cargo/twine/uv publish`), including nested `sh -c` / `ssh` / `wsl --exec` wrappers.
   - Escalates destructive working-tree actions (`rm -rf`/`--recursive`, `rmdir`, `git reset --hard`, `git clean -f`/`--force`) directly to Layer 2 so they are never silently allowed by path scope alone.

2. **Layer 2 — Context-Anchored OpenCode Reviewer Cascade + 3-Strike Agent Send-Back**:
   - First pass: `opencode/jev-1.13-free` via `opencode api experimental.generate.decisions` (`minConfidence = 0.80`, `3.5s` timeout, \$0, ~220 ms).
   - Incrementally indexes `USER_INPUT` turns in `transcript.jsonl` to pair brief affirmative replies (`"yes"`, `"go ahead"`) with the preceding agent question tail cleanly without letting agent output self-grant permissions, and keeps the brittle English-prose regex grant ledger (`AGY_ENABLE_PROSE_LEDGER=0`) disabled by default.
   - Second-opinion escalation: `opencode-go/beam-free` → `opencode-go/longcat-2.5-free` via `opencode api experimental.generate.text` (`5.0s` cap with `60s` backoff on timeout/errors, \$0).
   - Paid failover: `opencode/jev-1.13` (`3.5s` timeout, ultra-compact transcript) runs only when `jev-1.13-free` errored/timed out/rate-limited (`120s` backoff) **and** free escalation also failed/timed out.
   - Deterministic `# auth: "<exact user phrase>"` (or `# auth: step <N>`) verification against `transcript.jsonl` so earlier user authorizations can be cited without widening the prompt window (unverified quotes are flagged as fake).
   - 3-strike agent send-back: ambiguous or broad commands are returned to the agent with actionable instructions to narrow scope, split chained commands, or add a `# purpose:` / `# auth:` comment before troubling the user.

## Optional Local Policy Configuration

User-specific repository or path rules can be configured in `~/.gemini/config/permission_reviewer.local.json` (or via `AGY_PERMISSION_REVIEWER_LOCAL_CONFIG`) without modifying tracked plugin files. See [permission_reviewer.local.example.json](permission_reviewer.local.example.json):

```json
{
  "read_only_repos": [
    "collaborator-org/reference-repo"
  ],
  "known_ssh_hosts": [
    "cluster-login",
    "dev-workstation"
  ],
  "extra_read_only_paths": [],
  "extra_denied_paths": []
}
```

## Installation

```bash
git clone https://github.com/Jimbo-Coder/agy-permission-reviewer ~/.gemini/config/plugins/agy-permission-reviewer
```

Companion plugin: [agy-context-guard](https://github.com/Jimbo-Coder/agy-context-guard) (context window monitor and self-triggered compaction).
