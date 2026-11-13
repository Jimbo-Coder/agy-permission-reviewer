#!/usr/bin/env python3
"""Zero-dependency Stdio MCP server exposing the permission reviewer's verbal ledger and audit log."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import intent_extractor
import llm_classifier
import review_hook

TOOLS = [
    {
        "name": "permission_reviewer_status",
        "description": (
            "Inspect active verbal permission grants (paths/commands extracted from user "
            "messages) and the recent PreToolUse review audit log."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "transcript_path": {
                    "type": "string",
                    "description": "Optional path to transcript.jsonl.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID for audit log lookup.",
                },
            },
        },
    }
]


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


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agy-permission-reviewer", "version": "0.1.0"},
            },
        }

    if method.startswith("notifications/"):
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "permission_reviewer_status":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        importlib.reload(intent_extractor)
        importlib.reload(llm_classifier)
        importlib.reload(review_hook)
        tp_str = args.get("transcript_path")
        tp = Path(tp_str) if tp_str else _find_latest_transcript()
        ledger = intent_extractor.collect_session_ledger(tp)
        base = intent_extractor.get_data_dir()
        req_cid = str(args.get("conversation_id") or "").strip()
        audit_entries: list[dict[str, Any]] = []
        tier_counts: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        audit_dir = base / "audit"
        if audit_dir.is_dir():
            target_file: Path | None = None
            if req_cid:
                cand = audit_dir / f"{intent_extractor.safe_conv_id(req_cid)}.jsonl"
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
        usage_summary: dict[str, Any] = {"total_calls": 0, "by_model": {}, "by_outcome": {}}
        usage_file = base / "usage.jsonl"
        if usage_file.is_file():
            for ln in usage_file.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]:
                try:
                    u = json.loads(ln)
                    usage_summary["total_calls"] += 1
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
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "mode": os.environ.get("AGY_REVIEWER_MODE", "opencode"),
                                "model": os.environ.get("AGY_DECISION_MODELS", "opencode/jev-1.13-free"),
                                "opencode_available": llm_classifier.find_opencode_binary() is not None,
                                "session_tier_breakdown": tier_counts,
                                "session_decision_breakdown": decision_counts,
                                "llm_usage_summary": usage_summary,
                                "verbal_grant_ledger": compact_ledger,
                                "recent_audit_decisions": audit_entries,
                            },
                            indent=2,
                        ),
                    }
                ],
                "isError": False,
            },
        }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
