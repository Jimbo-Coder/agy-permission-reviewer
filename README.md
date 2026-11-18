# agy-permission-reviewer

Context-aware `PreToolUse` permission reviewer plugin for Google Antigravity, designed to eliminate repetitive approval popups while enforcing strict security boundaries and zero-token/free-model review cascades.

## 3-Tier Cascading Architecture

Naive auto-reviewers feed the entire 50k–200k token conversation transcript to a secondary LLM on every tool call. `agy-permission-reviewer` solves the cost, latency, and safety problem with a 3-tier cascade:

1. **Tier 0 — Deterministic Static Policy & Fast-Allow (~1 ms, 0 tokens)**:
   - Scoped `PreToolUse` matcher intercepts only permission-bearing tools (`run_command`, file tools, MCP, `browser_*`) so internal planning/task tools add zero overhead.
   - Immediately allows normal in-workspace file operations, routine read-only/build/test commands (including prefixed standard binaries and `2>&1`/`2>/dev/null`), and sandboxed commands scoped to the workspace/scratch/tmp.
   - Immediately rejects network-only commands (`git push/pull/fetch`, `gh`, `ssh`, `scp`, `publish`) sent with `BypassSandbox=false` so agents re-issue once with `BypassSandbox=true` instead of triggering double approvals.
   - Immediately blocks (`"decision": "deny"`) destructive system commands (`rm -rf /`, `mkfs`), credential exfiltration (`.ssh/id_*`, `.aws/credentials`, `.env`), and writes to protected system directories (`/etc`, `/System`, `/usr`, shell rc files).
   - Forces explicit user confirmation (`"decision": "force_ask"`) for public/external state mutations (`git push`, `gh issue/pr/release/repo`, `gh api -X POST/PATCH/...`, `npm/cargo/twine/uv publish`).
   - Escalates destructive working-tree actions (`rm -rf`, `rmdir`, `git reset --hard`, `git clean -f`) directly to Tier 2 so they are never silently allowed by path scope alone.

2. **Tier 1 — Zero-Token Verbal Grant Ledger (~2 ms, 0 tokens)**:
   - Incrementally scans `USER_INPUT` lines in `transcript.jsonl` (with `mtime_ns + size` caching).
   - Extracts explicit path and command authorizations stated by the user, distinguishing **read-only** vs **read-write** grants and honoring negative prohibitions (*"do not edit..."*, *"read-only"*).

3. **Tier 2 — OpenCode Reviewer Cascade + Session Cache + 3-Strike Agent Send-Back**:
   - First pass: `opencode/jev-1.13-free` via `opencode api experimental.generate.decisions` (`minConfidence = 0.80`, `3.5s` timeout, \$0, ~220 ms).
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
