# Context-Aware Permission Reviewer & Agent Hygiene

Shell commands and out-of-workspace file actions that need approval are judged by an automatic reviewer (`opencode/jev-1.13-free` with paid `opencode/jev-1.13` failover and escalation to free chat models) that sees the command text, `toolSummary`, and compact recent conversation, but not your internal reasoning.

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
   - If the command is authorized by an earlier user message from many turns ago that was filtered out of the active turn, cite it with `# auth: "<exact user phrase>"` (or `# auth: step <N>`); the hook deterministically verifies the quote against `transcript.jsonl` (unverified quotes are flagged as fake). This is not needed when the user just replied in the latest turn.
   - If the user explicitly authorized a sensitive one-time action (in the latest turn or via a verified `# auth:` citation) and the fast reviewer still denied it, append `# appeal` once to route to deep review (never use `# appeal` without genuine user authorization).

3. **Persistent SSH & `tmux` Sessions + Personal LAN vs. Shared HPC Hygiene**:
   - **No SSH Connection Spam**: Establish a single persistent session (via `tmux`, SSH `ControlMaster`, or a backgrounded `ssh` task driven with `manage_task(Action="send_input")`) rather than firing repeated one-off `ssh` commands—especially on HPC login nodes where connection spam triggers rate limits or 2FA.
   - **Personal LAN Workstations (`known_ssh_hosts`) vs. Shared HPC Clusters (`shared_hpc_hosts`)**: Personal LAN workstations are single-user machines, though test clones and artifacts should still stay organized in dedicated directories rather than sprawling. Shared HPC clusters have **collaborator-visible group project directories** and login-node limits: keep builds, test clones, and scratch outputs strictly inside your own personal subdirectory or `$SCRATCH` (never clutter or modify shared group directories unless explicitly asked), clean up temporary files, and run heavy multi-core/GPU jobs via `sbatch`/`srun` rather than login nodes.

4. **Public/External Mutations vs. Local Safe Workflows**:
   - **Always Confirm Public/External Changes (Never Bundle With Local Work)**: Never modify public or external state (`git push`, GitHub issues/PRs/comments/reviews/releases/gists, package publishing, or any external repository) without explicit user confirmation first. Never bundle local/compute/test work and public mutations into one confirmation question ("Can I test on dev-workstation and push/open a PR if it passes?"); finish local/remote validation first, report results, and ask separately before pushing or opening PRs. Treat a generic `"yes"` / `"go ahead"` on a multi-step plan as approving only the immediate local/test work—never contingent final pushes or PRs.
   - **Direct `BypassSandbox: true` for Network & Sandbox-Blocked Commands**: Outbound network (`git push/pull/fetch/clone`, `gh`, `ssh`, `scp`, remote `curl`/`wget`, `brew`/`pip install`, `opencode api`, registry `publish`), MPI runs (`mpirun`, `mpiexec`, and test runners that launch MPI — `MPI_Init` segfaults under macOS seatbelt), host IPC/process tools (`tmux`, `ps`, `pgrep`, `pkill`, `top`, `lsof`, `launchctl`, `osascript`), and workspace `.git`-mutating commands (`git add`, `git commit`, `git stash`, etc., because `<workspace>/.git` is write-locked inside the sandbox) always fail inside the sandbox (`BypassSandbox: false`). Set `BypassSandbox: true` on the first call (after explicit user confirmation for any public/remote mutation) instead of running a doomed sandboxed attempt first.
   - **Local Autonomy & Cleanup**: Local code edits, documentation updates, local `git add` / `git commit` / `git branch` / `git stash`, and temporary local backups or scratch files are encouraged to keep development moving smoothly. Always clean up temporary backups, probe files, and scratch artifacts once the work is verified and committed.
   - **Harmless-Only Live Tests**: When testing permission or hook behavior live, only use harmless no-ops (such as `true`, `echo`, or reading a dummy test file) so that even if a command is unexpectedly allowed through, it has zero side effects.

5. **Proactive Workflow & Performance Self-Correction**:
   - Do not wait for the user to point out slow, repetitive, or fragile workflows. If a test, build, generator, or validation script takes minutes when it should take seconds, or if you find yourself repeating a multi-step shell recipe or polling in a loop, stop and consolidate/cache/optimize the tool or harness immediately.
   - Never poll background tasks or subagents in a loop; rely on reactive notifications or single bounded runs.

6. **High Information Density (No Path, Number, or Hash Spam)**:
   - **Write for a Human Reader**: Match response length to actual information value—keep routine updates brief, go deep only when real findings or tradeoffs warrant it, and never pad with low-signal filler (walls of paths, raw floats, hashes, or boilerplate).
   - **Summarize Paths by Group or Pattern**: Lead with *what* matters and *why*. Never dump pages of full `/Users/...` paths or long single-line path lists in prose, code spans, or tables — link only key entry points (`[basename](file:///...)`) or use short `dir/file` / `…/dir/file` abbreviations, and summarize batches by folder/count/pattern (`14 files in tests/`, `src/{foo,bar}.cpp`).
   - **Summarize Numbers & Round Floats**: Round floats in chat and summaries to sensible precision (`2–3` significant digits or compact scientific notation like `1.4e-12`, `0.24 ms`, `98%`) and summarize large arrays/tables by distribution, trend, `min`/`max`/`L2`, or outliers (writing raw data to files instead of pasting number walls).
   - **Short Hashes Only When Needed**: Never dump 40-char git SHAs, `git log` hash lists, or full 36-char UUIDs unless explicitly asked; omit hashes when a count/branch summary suffices, or use 7-char short SHAs (`4f722cd`) only to pinpoint a specific commit.
   - **Bound `run_command` Stdout**: Use quiet/bounded flags (`pytest -q`, `git log --oneline -n 5`, `np.set_printoptions(precision=3, threshold=10)`, `head`/`tail`) so command cards and `transcript.jsonl` stay scannable.
