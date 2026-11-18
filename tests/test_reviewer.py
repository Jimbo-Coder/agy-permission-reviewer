from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import intent_extractor
import llm_classifier
import review_hook


class TestPermissionReviewer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        review_hook.os.environ["PLUGIN_DATA"] = str(self.tmp_path / "plugin_data")
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "ledger_only"

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

    def test_tier0_in_workspace_and_hard_deny(self) -> None:
        # In-workspace write -> Tier 0 allow
        res = review_hook.evaluate_tool_call(
            {
                "conversationId": "c1",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "edit_file",
                    "args": {"Path": "/workspace/project/src/main.cpp"},
                },
            }
        )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("Tier 0", res["reason"])

        # Destructive command -> Tier 0 deny
        res_deny = review_hook.evaluate_tool_call(
            {
                "conversationId": "c1",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "rm -rf ~"},
                },
            }
        )
        self.assertEqual(res_deny["decision"], "deny")

    def test_tier0_local_config_read_only_repos_and_paths(self) -> None:
        local_cfg_path = self.tmp_path / "local_config.json"
        local_cfg_path.write_text(
            json.dumps(
                {
                    "read_only_repos": ["collaborator-org/reference-repo"],
                    "extra_read_only_paths": ["/opt/readonly_ref"],
                    "extra_denied_paths": ["/opt/private_vault"],
                }
            ),
            encoding="utf-8",
        )
        review_hook.os.environ["AGY_PERMISSION_REVIEWER_LOCAL_CONFIG"] = str(local_cfg_path)
        self.addCleanup(lambda: review_hook.os.environ.pop("AGY_PERMISSION_REVIEWER_LOCAL_CONFIG", None))

        # Push or gh issue/PR to read_only_repos -> hard-denied at Tier 0
        r_push = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_local",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "gh pr create --repo collaborator-org/reference-repo"},
                },
            }
        )
        self.assertEqual(r_push["decision"], "deny")
        self.assertIn("read-only repository", r_push["reason"])

        # extra_read_only_paths allows read, blocks write at Tier 0
        r_ro_read = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_local",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/readonly_ref/README.md"},
                },
            }
        )
        self.assertEqual(r_ro_read["decision"], "allow")

        r_ro_write = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_local",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "write_file",
                    "args": {"Path": "/opt/readonly_ref/README.md", "Content": "x"},
                },
            }
        )
        self.assertEqual(r_ro_write["decision"], "deny")

        # extra_denied_paths blocks read at Tier 0
        r_denied = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_local",
                "workspacePaths": ["/workspace/project"],
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/private_vault/secret.txt"},
                },
            }
        )
        self.assertEqual(r_denied["decision"], "deny")

    def test_tier1_verbal_grant_read_vs_write_and_prohibition(self) -> None:
        tp = self._write_transcript(
            [
                "Please inspect /opt/ref_code in read-only mode, and you can create and edit files in /opt/custom_plugins. Never touch /opt/secret_data. Note that /config and /settings are slash commands, not paths."
            ]
        )
        ledger = review_hook.collect_session_ledger(tp)
        grant_paths = [pg["path"] for pg in ledger["path_grants"]]
        self.assertNotIn("/config", grant_paths)
        self.assertNotIn("/settings", grant_paths)

        # 1. Read from /opt/ref_code -> allowed by Tier 1
        r1 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/ref_code/kernel.hpp"},
                },
            }
        )
        self.assertEqual(r1["decision"], "allow")
        self.assertIn("Tier 1", r1["reason"])

        # 2. Write to /opt/ref_code (granted read-only) -> denied by Tier 1!
        r2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "write_file",
                    "args": {"Path": "/opt/ref_code/kernel.hpp", "Content": "x"},
                },
            }
        )
        self.assertEqual(r2["decision"], "deny")
        self.assertIn("READ-ONLY", r2["reason"])

        # 3. Write to /opt/custom_plugins (granted write) -> allowed by Tier 1!
        r3 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "write_file",
                    "args": {"Path": "/opt/custom_plugins/plugin.json", "Content": "{}"},
                },
            }
        )
        self.assertEqual(r3["decision"], "allow")

        # 4. Read from /opt/secret_data (prohibited) -> denied by Tier 1!
        r4 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/secret_data/keys.txt"},
                },
            }
        )
        self.assertEqual(r4["decision"], "deny")

        # 5. Unmentioned external path -> falls through to Tier 2 fallback ('ask')
        r5 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/unmentioned/file.txt"},
                },
            }
        )
        self.assertEqual(r5["decision"], "ask")

    def test_tier2_compact_prompt_and_signature_cache(self) -> None:
        prompt = llm_classifier.build_compact_prompt(
            workspace_paths=["/ws"],
            recent_user_messages=["Please check the logs in the parent folder."],
            tool_name="read_file",
            tool_args={"Path": "/outside/run.log"},
        )
        self.assertLess(len(prompt), 500)
        sig1 = llm_classifier.compute_signature(
            "c3", 4, "read_file", {"Path": "/outside/a.log"}
        )
        sig2 = llm_classifier.compute_signature(
            "c3", 4, "read_file", {"Path": "/outside/b.log"}
        )
        # Same directory, same turn, same tool -> identical cache signature!
        self.assertEqual(sig1, sig2)

    def test_git_commit_message_with_git_push_or_slashes_does_not_force_ask(self) -> None:
        # A local git commit inside workspace whose commit message mentions "git push/registry"
        # must fast-allow at Tier 0 rather than matching FORCE_ASK (`git push`) or external paths (`push/registry`).
        res = review_hook.evaluate_tool_call(
            {
                "conversationId": "c4",
                "workspacePaths": ["/workspace/plugins/agy-permission-reviewer"],
                "cwd": "/workspace/project",
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": (
                            'git -C /workspace/plugins/agy-permission-reviewer '
                            'commit -am "Require force_ask on git push/registry publish and add persistent SSH rules" '
                            "# purpose: commit permission reviewer updates"
                        ),
                        "BypassSandbox": True,
                    },
                },
            }
        )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("Tier 0", res["reason"])

        # Whereas an actual `git push` command MUST return 'ask' (translated to 'force_ask')
        res_push = review_hook.evaluate_tool_call(
            {
                "conversationId": "c4",
                "workspacePaths": ["/workspace/plugins/agy-permission-reviewer"],
                "cwd": "/workspace/project",
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git -C /workspace/plugins/agy-permission-reviewer push origin main",
                        "BypassSandbox": True,
                    },
                },
            }
        )
        self.assertEqual(res_push["decision"], "ask")
        self.assertEqual(review_hook.to_antigravity_hook_output(res_push)["decision"], "force_ask")

    def test_cache_cap_and_missing_opencode_fallback(self) -> None:
        # 1. Verify per-conversation verdict cache is bounded by MAX_CACHE_ENTRIES
        for idx in range(llm_classifier.MAX_CACHE_ENTRIES + 20):
            llm_classifier.put_cached_verdict(
                "c_cap",
                f"k_{idx}",
                {"effect": "allow", "model": "test"},
            )
        cache_file = intent_extractor.get_data_dir() / "cache_c_cap.json"
        cached_dict = json.loads(cache_file.read_text(encoding="utf-8"))
        self.assertEqual(len(cached_dict), llm_classifier.MAX_CACHE_ENTRIES)
        self.assertNotIn("k_0", cached_dict)
        self.assertIn(f"k_{llm_classifier.MAX_CACHE_ENTRIES + 19}", cached_dict)

        # 2. Verify missing opencode binary falls back cleanly to native prompt ('ask') without 3-strike send-back loop
        res = llm_classifier.apply_unresolved_send_back(
            conversation_id="c_no_opencode",
            verdict={
                "effect": "ask",
                "reasonCode": "review_failed",
                "reason": "opencode/jev-1.13-free: opencode binary not found",
            },
        )
        self.assertEqual(res["decision"], "ask")
        self.assertEqual(res["tier"], "tier2_unavailable")

    def test_directory_delete_and_git_reset_hard_bypass_t0_and_t1_to_tier2(self) -> None:
        ws = str(self.tmp_path / "workspace")
        tp = self._write_transcript([f"Feel free to edit files in {ws} and /opt/custom_plugins"])
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"

        called_cmds: list[str] = []

        def fake_deny_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            called_cmds.append(endpoint)
            return {
                "data": {
                    "answers": [{"choice": "deny", "confidence": 0.96}],
                    "tokens": {"input": 180, "output": 8},
                }
            }

        # Even inside the workspace (or with a Tier 1 path write grant) and BypassSandbox=False,
        # rm -rf and git reset --hard must escalate to Tier 2 (confirmation via chat).
        for cmd in (f"rm -rf {ws}/old_dir", "rm -rf /opt/custom_plugins/old_dir", "git reset --hard HEAD~1"):
            called_cmds.clear()
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c5",
                    "workspacePaths": [ws],
                    "transcriptPath": str(tp),
                    "cwd": ws,
                    "toolCall": {
                        "name": "run_command",
                        "args": {
                            "CommandLine": cmd,
                            "BypassSandbox": False,
                        },
                    },
                },
                api_caller=fake_deny_api,
            )
            self.assertEqual(res["decision"], "deny")
            self.assertEqual(len(called_cmds), 1)
            act = review_hook.build_requested_action(
                "run_command",
                {"CommandLine": cmd, "BypassSandbox": False},
                workspace_paths=[ws],
                cwd=ws,
            )
            self.assertTrue(act["details"]["isDestructiveOrDelete"])

    def test_internal_non_permission_tools_fast_allow_and_hooks_matcher(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"LLM tier should not be invoked for internal tools ({endpoint})")

        for tool_name, args in (
            ("exit_plan_mode", {"plan": "# Big plan\nStep 1..."}),
            ("enter_plan_mode", {"reason": "Need design choice"}),
            ("manage_task", {"Action": "list"}),
            ("schedule", {"DurationSeconds": 10, "Prompt": "ping"}),
            ("search_web", {"query": "python unittest"}),
        ):
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_internal",
                    "workspacePaths": ["/workspace/project"],
                    "toolCall": {"name": tool_name, "args": args},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res["decision"], "allow", msg=f"{tool_name} failed to fast-allow")
            self.assertIn("Tier 0", res["reason"])

        hooks_cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "hooks.json").read_text(encoding="utf-8")
        )
        matcher = hooks_cfg["context-permission-reviewer"]["PreToolUse"][0]["matcher"]
        matcher_re = re.compile(matcher)
        for reviewed in ("run_command", "read_file", "edit_file", "write_file", "call_mcp_tool", "mcp_foo", "browser_click"):
            self.assertIsNotNone(matcher_re.fullmatch(reviewed), msg=f"matcher missed {reviewed}")
        for internal in ("exit_plan_mode", "enter_plan_mode", "manage_task", "schedule", "search_web", "swarm", "invoke_subagent"):
            self.assertIsNone(matcher_re.fullmatch(internal), msg=f"matcher unexpectedly matched {internal}")

    def test_std_bin_paths_env_path_redirects_multi_m_commit_and_gh_api(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run for routine Tier 0 command ({endpoint})")

        routine_cases = [
            ("/usr/bin/git add src/a.py", False),
            ('/usr/bin/git commit -m "title" -m "body mentioning git push and /etc/passwd"', False),
            ('export PATH="/opt/homebrew/bin:/usr/bin:/bin"; /opt/homebrew/bin/uv run pytest 2>&1', True),
            ("/opt/homebrew/bin/gh api repos/org/repo/pulls/1 --jq '.title'", True),
            ("/opt/homebrew/bin/gh pr checks 58 -R org/repo", True),
        ]
        for cmd, bypass in routine_cases:
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_std_bin",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": cmd, "BypassSandbox": bypass}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res["decision"], "allow", msg=f"Expected Tier 0 allow for: {cmd!r}, got {res}")

        # Mutating /opt/homebrew/bin/gh api -X PATCH must STILL require force_ask
        res_patch = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_std_bin",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": "/opt/homebrew/bin/gh api -X PATCH repos/org/repo/pulls/1 --input /tmp/p.json",
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_patch["decision"], "ask")
        self.assertEqual(review_hook.to_antigravity_hook_output(res_patch)["decision"], "force_ask")

        # Redirect or rm -f targeting /usr/local/bin must NOT fast-allow at Tier 0
        tier2_calls: list[str] = []

        def fake_deny(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            tier2_calls.append(endpoint)
            return {"data": {"answers": [{"choice": "deny", "confidence": 0.95}]}}

        for unsafe_cmd in ("echo hi > /usr/local/bin/evil", "rm -f /usr/local/bin/evil"):
            tier2_calls.clear()
            res_unsafe = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_std_bin",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": unsafe_cmd, "BypassSandbox": True}},
                },
                api_caller=fake_deny,
            )
            self.assertEqual(res_unsafe["decision"], "deny")
            self.assertEqual(len(tier2_calls), 1, msg=f"{unsafe_cmd} bypassed Tier 0!")

    def test_network_command_in_sandbox_denies_without_calling_llm_or_prompting(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run for sandbox network mismatch ({endpoint})")

        for net_cmd in (
            "git push origin main",
            "/opt/homebrew/bin/gh pr list --repo org/repo",
            'ssh dev-workstation "df -h"',
        ):
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_net_sb",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": net_cmd, "BypassSandbox": False}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res["decision"], "deny", msg=f"Expected deny for sandboxed {net_cmd}")
            self.assertIn("BypassSandbox=true", res["reason"])

        # Same git push with BypassSandbox=True -> single native force_ask
        res_ok = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_net_sb",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "run_command", "args": {"CommandLine": "git push origin main", "BypassSandbox": True}},
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_ok["decision"], "ask")
        self.assertEqual(review_hook.to_antigravity_hook_output(res_ok)["decision"], "force_ask")

        # Commit message mentioning git pull / gh pr must still fast-allow inside sandbox
        res_commit = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_net_sb",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": 'git commit -m "Document git pull and gh pr flow"', "BypassSandbox": False},
                },
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_commit["decision"], "allow")

    def test_sandboxed_inline_python_reuses_session_dir_cache_after_first_approval(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"
        calls: list[str] = []

        def fake_allow(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            calls.append(endpoint)
            return {"data": {"answers": [{"choice": "allow", "confidence": 0.96}]}}

        res1 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "python3 -c 'print(open(\"/opt/ext_proj/a.py\").read())'", "BypassSandbox": False},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res1["decision"], "allow")
        self.assertEqual(len(calls), 1)

        # Different inline python command on the same external dir (still sandboxed) hits session_dir_cache!
        res2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "python3 -c 'print(len(open(\"/opt/ext_proj/b.py\").read()))'", "BypassSandbox": False},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res2["decision"], "allow")
        self.assertEqual(len(calls), 1)

        # Unsandboxed inline python must NOT reuse session_dir_cache and must still go to Tier 2
        res3 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "python3 -c 'print(open(\"/opt/ext_proj/c.py\").read())'", "BypassSandbox": True},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res3["decision"], "allow")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
