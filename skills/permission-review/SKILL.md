---
name: permission-review
description: >-
  Inspect active verbal permission grants, review audit logs, and configure the
  3-tier PreToolUse permission reviewer.
---

# Context-Aware Permission Review

Use this skill to inspect which out-of-workspace paths or commands have been verbally authorized in the current session, or to audit recent `PreToolUse` decisions.

## Usage

1. Call the `permission_reviewer_status` MCP tool to view:
   - `verbal_grant_ledger.path_grants` (with `read` vs `write` mode and originating `step_index`)
   - `verbal_grant_ledger.denied_paths`
   - `recent_audit_decisions` (showing tier used and latency in ms)
