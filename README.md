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
   - Incrementally indexes `USER_INPUT` and `PLANNER_RESPONSE` turns in `transcript.jsonl` to pair affirmative or explicit-permission replies with the preceding agent proposal tail without letting agent output self-grant permissions, preserves the preceding proposal across turn boundaries, extracts `knownUserSshHost` / `sharedHpcHost` metadata for `ssh`/`scp`/`sftp`/`rsync`, never caches `deny` verdicts across user turns, and keeps the brittle English-prose regex grant ledger (`AGY_ENABLE_PROSE_LEDGER=0`) disabled by default.
   - Second-opinion escalation: `opencode-go/beam-free` → `opencode-go/longcat-2.5-free` via `opencode api experimental.generate.text` (`9.0s` cap with `20s` timeout cooldown / `60s` HTTP error backoff, \$0), using expanded proposal context, zero-token auto-retrieval of relevant earlier user turns, and reasoning-first JSON verdicts.
   - Opt-in `# appeal` / `# review: deep` (`AGY_DEEP_REVIEW_MODELS` or local `"deep_review_models"`, `AGY_DEEP_REVIEW_TIMEOUT_SEC`, default `11.0s`): lets an agent route a sensitive one-time user-authorized command directly to the deeper reviewer under deterministic guardrails (Tier 0 hard-denies remain non-appealable, requires a recent `USER_INPUT` or transcript-verified `# auth:`, capped to 1 appeal per command and 4 per turn, and never cached or promoted into session directory/SSH caches).
   - Paid failover: `opencode/jev-1.13` (`3.5s` timeout, ultra-compact transcript) runs only when `jev-1.13-free` errored/timed out/rate-limited (`120s` backoff) **and** free escalation also failed/timed out.
   - Deterministic `# auth: "<exact user phrase>"` (or `# auth: step <N>`) verification against `transcript.jsonl` (enriched with the preceding agent proposal when present) so earlier user authorizations can be cited without widening the prompt window (unverified quotes are flagged as fake).
   - 3-strike agent send-back: ambiguous or broad commands are returned to the agent with actionable instructions to narrow scope, split chained commands, add a `# purpose:` / `# auth:` comment, or use `# appeal` once if explicitly authorized before troubling the user.

## Optional Local Policy Configuration

User-specific repository, path, or bring-your-own-model rules can be configured in `~/.gemini/config/permission_reviewer.local.json` (or via `AGY_PERMISSION_REVIEWER_LOCAL_CONFIG`) without modifying tracked plugin files. See [permission_reviewer.local.example.json](permission_reviewer.local.example.json):

```json
{
  "read_only_repos": [
    "collaborator-org/reference-repo"
  ],
  "known_ssh_hosts": [
    "dev-workstation"
  ],
  "shared_hpc_hosts": [
    "cluster-login"
  ],
  "extra_read_only_paths": [],
  "extra_denied_paths": [],
  "deep_review_models": []
}
```

**Bring Your Own Models:** Everything works `$0` out of the box using free models. If you have an OpenCode Go subscription or other configured OpenCode providers, you can set `"deep_review_models"` (for example `["opencode-go/claude-haiku-5-5", "opencode-go/gpt-5.6-luna"]`, plus optional `"escalation_models"` / `"decision_models"`) in your local config. Free fallbacks (`beam-free`, `longcat-2.5-free`) are automatically appended, and expired/unauthorized/missing models (`401`/`402`/`403`/`404`/subscription/quota errors) enter a `1h` backoff so the pipeline degrades gracefully back to free models without breaking.

## Installation

Supports macOS, Linux, and Windows (via the polyglot `scripts/run_hook.cmd` hook launcher and automatic native `opencode.exe` resolution):

```bash
# macOS / Linux
git clone https://github.com/Jimbo-Coder/agy-permission-reviewer ~/.gemini/config/plugins/agy-permission-reviewer
```

```powershell
# Windows (PowerShell)
git clone https://github.com/Jimbo-Coder/agy-permission-reviewer "$env:USERPROFILE\.gemini\config\plugins\agy-permission-reviewer"
```

### T3 Code (Antigravity ACP)

T3 Code runs `agy_acp_server` with an isolated `GEMINI_HOME` per Antigravity profile and links global skills, not plugins. After cloning this repo and enabling Antigravity in T3 Code, wire the `PreToolUse` hook into your T3 Code profile(s):

```bash
python3 ~/.gemini/config/plugins/agy-permission-reviewer/scripts/install_t3code.py
```

When running inside Antigravity ACP (SQLite `.db` session trajectories under `$GEMINI_HOME/antigravity-acp`), unresolved `ask` verdicts fail closed to `deny` (since ACP hooks cannot pop up an interactive CLI dialog) and global `~/.gemini/config/permission_reviewer.local.json` is automatically shared.

Companion plugin: [agy-context-guard](https://github.com/Jimbo-Coder/agy-context-guard) (context window monitor and self-triggered compaction).
