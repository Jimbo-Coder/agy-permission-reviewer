# Context-Aware Permission Reviewer & Agent Hygiene

Shell commands and out-of-workspace file actions that need approval are judged by an automatic reviewer (`opencode/jev-1.13-free` with escalation to free chat models) that sees the command text, `toolSummary`, and compact recent conversation, but not your internal reasoning.

1. **No Tautological Tests, Comments, or UI**:
   - **Comments & `# purpose:` hints**: Never write comments that merely restate what the code or command already says (`i += 1  # increment i` or `pytest tests/  # purpose: run pytest on tests`). Only add a comment or `# purpose:` hint when it conveys *why* (intent, invariant, or non-obvious context) that is not already visible from the command and `toolSummary`.
   - **Tests**: Never write tautological tests that only assert mock return values, mirror implementation constants line-for-line, or cannot fail when real behavior breaks. Every test must exercise a real boundary, invariant, or regression mode.
   - **UI / Output**: Avoid noisy status banners, redundant wrapper text, or UI chrome that repeats what the underlying data or tool output already shows.

2. **When Sent Back by Auto Permissions**:
   If a command is refused with an `Auto Permissions did not approve this command` note, do **not** blindly retry the identical command:
   - Narrow the target scope (specific paths instead of broad globs or parent directories)
   - Split chained (`&&`, `;`, `|`) commands into single-purpose steps
   - Run a read-only check first if appropriate
   - Add a non-tautological `# purpose: <why this is needed for the user's goal>` comment if the intent is not obvious from the command itself
   - If the command is authorized by an earlier user message from many turns ago that was filtered out of the active turn, cite it with `# auth: "<exact user phrase>"` (or `# auth: step <N>`); the hook deterministically verifies the quote against `transcript.jsonl` (unverified quotes are flagged as fake)

3. **Persistent SSH & `tmux` Sessions (No SSH Spam)**:
   When working on remote hosts (e.g., cluster login nodes or remote workstations), establish a single persistent session (via `tmux` or a backgrounded `ssh` task driven with `manage_task(Action="send_input")` / SSH ControlMaster) rather than spawning repeated one-off `ssh` commands on every step.

4. **Public/External Mutations vs. Local Safe Workflows**:
   - **Always Confirm Public/External Changes**: Never modify public or external state (`git push`, GitHub issues/PRs/comments/reviews/releases/gists, package publishing, or any external repository) without explicit user confirmation first.
   - **Local Autonomy & Cleanup**: Local code edits, documentation updates, local `git add` / `git commit` / `git branch` / `git stash`, and temporary local backups or scratch files are encouraged to keep development moving smoothly. Always clean up temporary backups, probe files, and scratch artifacts once the work is verified and committed.
   - **Harmless-Only Live Tests**: When testing permission or hook behavior live, only use harmless no-ops (such as `true`, `echo`, or reading a dummy test file) so that even if a command is unexpectedly allowed through, it has zero side effects.

5. **Proactive Workflow & Performance Self-Correction**:
   - Do not wait for the user to point out slow, repetitive, or fragile workflows. If a test, build, generator, or validation script takes minutes when it should take seconds, or if you find yourself repeating a multi-step shell recipe or polling in a loop, stop and consolidate/cache/optimize the tool or harness immediately.
   - Never poll background tasks or subagents in a loop; rely on reactive notifications or single bounded runs.
