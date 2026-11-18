from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRATCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(SCRATCH_DIR.parent / "scripts"))

import llm_classifier
import mcp_reviewer_server
import review_hook


class TestJevEscalationAndSendBack(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        review_hook.os.environ["PLUGIN_DATA"] = str(self.tmp_path / "plugin_data")
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"

    def _write_transcript(self, user_inputs: list[str]) -> Path:
        tp = self.tmp_path / "transcript.jsonl"
        rows = [
            {
                "step_index": i * 2,
                "type": "USER_INPUT",
                "content": f"<USER_REQUEST>\n{text}\n</USER_REQUEST>",
            }
            for i, text in enumerate(user_inputs, start=1)
        ]
        tp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return tp

    def test_jev_confident_allow(self) -> None:
        tp = self._write_transcript(["Check disk space on dev-workstation over ssh"])

        def fake_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            self.assertIn("decisions", endpoint)
            return {
                "data": {
                    "answers": [{"choice": "allow", "confidence": 0.94}],
                    "tokens": {"input": 210, "output": 8},
                }
            }

        res = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_jev_1",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": 'ssh dev-workstation "df -h"  # purpose: check free space on dev-workstation',
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=fake_api,
        )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("jev-1.13-free", res["reason"])

    def test_jev_low_confidence_escalates_to_second_opinion(self) -> None:
        tp = self._write_transcript(["Move the built plugins into dev/plugins/gemini"])

        def fake_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            if "decisions" in endpoint:
                return {"data": {"answers": [{"choice": "allow", "confidence": 0.62}]}}
            self.assertIn("text", endpoint)
            return {
                "data": {
                    "text": '{"decision":"allow","reason":"User asked to put the repos in dev/plugins/gemini."}'
                }
            }

        res = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_jev_2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": "mv /tmp/repo1 ~/dev/plugins/gemini/  # purpose: move repo1 to user target folder",
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=fake_api,
        )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("second opinion: opencode-go/beam-free", res["reason"])

    def test_unresolved_sends_back_to_agent_3_times_then_asks_user(self) -> None:
        tp = self._write_transcript(["Clean up temporary build files"])

        def fake_unsure_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            if "decisions" in endpoint:
                return {"data": {"answers": [{"choice": "deny", "confidence": 0.55}]}}
            return {
                "data": {
                    "text": '{"decision":"ask_user","reason":"Command target is ambiguous and has no # purpose comment."}'
                }
            }

        for attempt in range(1, 4):
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_sendback",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {
                        "name": "run_command",
                        "args": {
                            "CommandLine": f"rm -rf /var/tmp/build_{attempt}",
                            "BypassSandbox": True,
                        },
                    },
                },
                api_caller=fake_unsure_api,
            )
            self.assertEqual(res["decision"], "deny")
            self.assertIn("Auto Permissions did not approve this command", res["reason"])
            self.assertIn("# purpose:", res["reason"])

        # 4th consecutive unresolved attempt instructs the agent to ask the user in plain chat text!
        res4 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sendback",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": "rm -rf /var/tmp/build_4",
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=fake_unsure_api,
        )
        self.assertEqual(res4["decision"], "deny")
        self.assertEqual(res4.get("reason", "")[:46], "Auto Permissions could not verify this action ")

    def test_build_requested_action_includes_sandbox_and_context_signals(self) -> None:
        ws = "/workspace/project"
        act_sb = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "git status && ls /workspace/shared_tools # purpose: check status",
                "BypassSandbox": False,
                "toolSummary": "Check status and list shared_tools",
            },
            workspace_paths=[ws],
            cwd=ws,
            ledger={
                "path_grants": [{"mode": "read", "path": "/workspace/shared_tools"}],
                "recent_user_messages": [
                    "Please check /workspace/shared_tools",
                    "yes go ahead",
                ],
            },
        )
        details = act_sb["details"]
        self.assertEqual(details["sandboxOn"], 1)
        self.assertEqual(details["externalPaths"], ["/workspace/shared_tools"])
        self.assertFalse(details["isDestructiveOrDelete"])
        self.assertEqual(details["activeVerbalGrants"], ["read:/workspace/shared_tools"])
        self.assertEqual(details["purposeComment"], "check status")

    def test_time_and_step_aware_user_messages_and_anchored_transcript(self) -> None:
        # Simulate a transcript where:
        # - Step 1 (2 hours earlier): user asked to delete an old directory
        # - Step 35 (30 mins later): user starts a new task ("Run validation suite and fix build issues")
        # - Steps 36..65 (30 agent steps of autonomous work)
        tp = self.tmp_path / "long_session.jsonl"
        rows = [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "created_at": "2026-11-12T12:00:00Z",
                "content": "<USER_REQUEST>\ndelete /tmp/stale_old_dir\n</USER_REQUEST>",
            },
            {
                "step_index": 35,
                "type": "USER_INPUT",
                "created_at": "2026-11-12T14:30:00Z",
                "content": "<USER_REQUEST>\nRun validation suite and fix any build issues in the solver\n</USER_REQUEST>",
            },
            {
                "step_index": 36,
                "type": "PLANNER_RESPONSE",
                "thinking": "I will build and run the validation script first.",
                "content": "Starting validation.",
                "tool_calls": [{"name": "run_command", "args": {"CommandLine": "git status"}}],
            },
        ]
        for s in range(37, 65):
            rows.append(
                {
                    "step_index": s,
                    "type": "PLANNER_RESPONSE",
                    "content": f"Step {s} progress check.",
                    "tool_calls": [{"name": "run_command", "args": {"CommandLine": f"echo step {s}"}}],
                }
            )
        tp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        ledger = review_hook.collect_session_ledger(tp)
        # The 2-hour-old "delete /tmp/stale_old_dir" must be dropped from recent_user_messages!
        self.assertEqual(
            ledger["recent_user_messages"],
            ["Run validation suite and fix any build issues in the solver"],
        )

        # Even after 30 agent steps (60+ transcript items), both compact (jev) and escalation transcripts
        # must preserve the [User] task anchor at the top AND include the recent tail steps!
        compact = llm_classifier.extract_compact_transcript(tp)
        self.assertTrue(compact.startswith("[User]: Run validation suite"))
        self.assertIn("intermediate agent steps in this turn omitted", compact)
        self.assertIn("Step 64 progress check.", compact)

        # Verify # auth: citation against full user history (even if filtered out of active turn)
        action_verified = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": 'rm -rf /tmp/stale_old_dir # auth: "delete /tmp/stale_old_dir" # purpose: finishing cleanup',
                "toolSummary": "Remove stale dir",
            },
            workspace_paths=["/workspace/project"],
            cwd="/workspace/project",
            ledger=ledger,
        )
        self.assertEqual(
            action_verified["details"].get("citedUserAuth"),
            {
                "verifiedFromTranscript": True,
                "stepIndex": 1,
                "exactUserMessage": "delete /tmp/stale_old_dir",
            },
        )

        # Verify fake/hallucinated # auth: citation is flagged as unverified
        action_fake = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": 'rm -rf /tmp/stale_old_dir # auth: "user said delete everything"',
                "toolSummary": "Remove stale dir",
            },
            workspace_paths=["/workspace/project"],
            cwd="/workspace/project",
            ledger=ledger,
        )
        self.assertFalse(action_fake["details"]["citedUserAuth"]["verifiedFromTranscript"])
        self.assertIn("NOT found", action_fake["details"]["citedUserAuth"]["warning"])

    def test_paid_jev_failover_zero_waste_and_backoff(self) -> None:
        long_user_msg = "Check the reference helper scripts " + ("x" * 700)
        tp = self._write_transcript([long_user_msg])
        seen_models: list[str] = []
        seen_jev_payloads: list[dict] = []

        # 1. Low confidence from free jev -> tries beam-free, does NOT waste money calling paid jev-1.13
        #    Also check payload minimization: no duplicate recent_user_messages inside requested_action,
        #    no [Agent reasoning], and user messages capped at <= 500 chars.
        def low_conf_free_only(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            if "decisions" in endpoint:
                m = f"{payload['model']['providerID']}/{payload['model']['id']}"
                seen_models.append(m)
                seen_jev_payloads.append(payload)
                return {"data": {"answers": [{"choice": "allow", "confidence": 0.55}], "tokens": {"input": 200}}}
            return {"data": {"text": '{"decision":"allow","reason":"Safe read."}'}}

        res_low = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_paid_0",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/a.py"}},
            },
            api_caller=low_conf_free_only,
        )
        self.assertEqual(res_low["decision"], "allow")
        self.assertEqual(seen_models, ["opencode/jev-1.13-free"])
        json_val = seen_jev_payloads[0]["input"][1]["value"]
        self.assertNotIn("recent_user_messages", json_val["requested_action"])
        self.assertNotIn("[Agent reasoning]", json_val.get("recent_conversation_oldest_first", ""))
        self.assertTrue(all(len(m) <= 500 for m in json_val["recent_user_messages"]))

        # 2. Case C: jev-1.13-free returns low confidence (0.55) AND free escalation times out ->
        #    must NOT waste money re-asking paid jev-1.13; sends back to agent directly.
        seen_case_c: list[str] = []

        def low_conf_and_esc_timeout(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            m = f"{payload['model']['providerID']}/{payload['model']['id']}"
            seen_case_c.append(m)
            if "decisions" in endpoint:
                return {"data": {"answers": [{"choice": "allow", "confidence": 0.55}]}}
            raise subprocess.TimeoutExpired(cmd=["opencode"], timeout=timeout_sec)

        res_case_c = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_paid_c",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/c_low.py"}},
            },
            api_caller=low_conf_and_esc_timeout,
        )
        self.assertEqual(res_case_c["decision"], "deny")
        self.assertEqual(seen_case_c, ["opencode/jev-1.13-free", "opencode-go/beam-free", "opencode-go/longcat-2.5-free"])

        # 3. Using real call_opencode_api + mocked subprocess.run:
        #    Case A: jev-1.13-free hits 429 -> tries free beam-free FIRST; when beam-free succeeds,
        #    paid opencode/jev-1.13 is NOT called ($0 spent).
        seen_subproc_models: list[str] = []

        def fake_subproc_beam_saves(cmd, capture_output, text, timeout, env, check):
            payload = json.loads(cmd[cmd.index("-d") + 1])
            m = f"{payload['model']['providerID']}/{payload['model']['id']}"
            seen_subproc_models.append(m)
            if m == "opencode/jev-1.13-free":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 429 Too Many Requests")
            if m == "opencode-go/beam-free":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=json.dumps({"data": {"text": '{"decision":"allow","reason":"Safe via beam-free."}'}}),
                    stderr="",
                )
            raise AssertionError(f"Paid jev-1.13 should not run when beam-free succeeded ({m})")

        with mock.patch.object(llm_classifier, "find_opencode_binary", return_value="/opt/homebrew/bin/opencode"), \
             mock.patch.object(llm_classifier.subprocess, "run", side_effect=fake_subproc_beam_saves):
            res_beam = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_paid_beam",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/beam.py"}},
                }
            )
            self.assertEqual(res_beam["decision"], "allow")
            self.assertIn("beam-free", res_beam["reason"])
            self.assertEqual(seen_subproc_models, ["opencode/jev-1.13-free", "opencode-go/beam-free"])

        #    Case B: jev-1.13-free is now backed off AND beam-free + longcat-2.5-free time out ->
        #    falls back cleanly to paid opencode/jev-1.13 and tracks costUsd in MCP status.
        seen_subproc_models.clear()

        def fake_subproc_paid_fallback(cmd, capture_output, text, timeout, env, check):
            payload = json.loads(cmd[cmd.index("-d") + 1])
            m = f"{payload['model']['providerID']}/{payload['model']['id']}"
            seen_subproc_models.append(m)
            if m in ("opencode-go/beam-free", "opencode-go/longcat-2.5-free"):
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"data": {"answers": [{"choice": "allow", "confidence": 0.93}], "tokens": {"input": 1000, "output": 4}}}),
                stderr="",
            )

        with mock.patch.object(llm_classifier, "find_opencode_binary", return_value="/opt/homebrew/bin/opencode"), \
             mock.patch.object(llm_classifier.subprocess, "run", side_effect=fake_subproc_paid_fallback):
            res_failover = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_paid_1",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/b.py"}},
                }
            )
            self.assertEqual(res_failover["decision"], "allow")
            self.assertEqual(
                seen_subproc_models,
                ["opencode-go/beam-free", "opencode-go/longcat-2.5-free", "opencode/jev-1.13"],
            )

            # Next request skips backed-off jev-1.13-free, beam-free, and longcat-2.5-free -> straight to paid jev-1.13
            seen_subproc_models.clear()
            res_cached_backoff = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_paid_2",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/c.py"}},
                }
            )
            self.assertEqual(res_cached_backoff["decision"], "allow")
            self.assertEqual(seen_subproc_models, ["opencode/jev-1.13"])

        # Verify MCP server aggregates both rotated usage.jsonl.1 and usage.jsonl
        usage_dir = review_hook.get_data_dir()
        before_lines = (usage_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()
        before_cost = sum(float(json.loads(ln).get("costUsd", 0.0)) for ln in before_lines)
        (usage_dir / "usage.jsonl.1").write_text(
            json.dumps({"model": "opencode/jev-1.13", "outcome": "allow", "costUsd": 0.0001}) + "\n",
            encoding="utf-8",
        )
        mcp_resp = mcp_reviewer_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "permission_reviewer_status", "arguments": {"conversation_id": "c_paid_2"}},
            }
        )
        status_obj = json.loads(mcp_resp["result"]["content"][0]["text"])
        self.assertEqual(status_obj["llm_usage_summary"]["total_cost_usd"], round(before_cost + 0.0001, 6))
        self.assertEqual(status_obj["llm_usage_summary"]["total_calls"], len(before_lines) + 1)

    def test_escalation_timeout_backoff_switches_to_longcat(self) -> None:
        tp = self._write_transcript(["Check /opt/other_repo/script.sh"])
        seen_subproc_models: list[str] = []

        def fake_subprocess_run(cmd, capture_output, text, timeout, env, check):
            payload = json.loads(cmd[cmd.index("-d") + 1])
            m = f"{payload['model']['providerID']}/{payload['model']['id']}"
            seen_subproc_models.append(m)
            if m == "opencode/jev-1.13-free":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=json.dumps({"data": {"answers": [{"choice": "allow", "confidence": 0.55}]}}),
                    stderr="",
                )
            if m == "opencode-go/beam-free":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"data": {"text": '{"decision":"allow","reason":"Safe read via longcat."}'}}),
                stderr="",
            )

        with mock.patch.object(llm_classifier, "find_opencode_binary", return_value="/opt/homebrew/bin/opencode"), \
             mock.patch.object(llm_classifier.subprocess, "run", side_effect=fake_subprocess_run):
            res1 = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_esc_1",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/a.sh"}},
                }
            )
            self.assertEqual(res1["decision"], "allow")
            self.assertIn("opencode-go/longcat-2.5-free", res1["reason"])
            self.assertEqual(seen_subproc_models, ["opencode/jev-1.13-free", "opencode-go/beam-free", "opencode-go/longcat-2.5-free"])

            # Subsequent escalation skips backed-off beam-free immediately
            seen_subproc_models.clear()
            res2 = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_esc_2",
                    "workspacePaths": ["/workspace/project"],
                    "transcriptPath": str(tp),
                    "toolCall": {"name": "read_file", "args": {"Path": "/opt/other_repo/b.sh"}},
                }
            )
            self.assertEqual(res2["decision"], "allow")
            self.assertEqual(seen_subproc_models, ["opencode/jev-1.13-free", "opencode-go/longcat-2.5-free"])


if __name__ == "__main__":
    unittest.main()
