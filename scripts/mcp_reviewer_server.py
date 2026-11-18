#!/usr/bin/env python3
"""Zero-dependency Stdio MCP server exposing the permission reviewer's verbal ledger and audit log."""

from __future__ import annotations

import importlib
import json
import sys
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
        status_payload = review_hook.build_reviewer_status(args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(status_payload, indent=2),
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
