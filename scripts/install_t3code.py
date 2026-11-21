#!/usr/bin/env python3
"""Install or remove the `context-permission-reviewer` PreToolUse hook inside T3 Code Antigravity ACP profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

HOOK_KEY = "context-permission-reviewer"
DEFAULT_MATCHER = (
    "^(run_command|read_file|write_file|edit_file|view_file|write_to_file|"
    "create_file|client_create_file|replace_file_content|multi_replace_file_content|"
    "delete_file|call_mcp_tool|list_resources|read_resource|mcp_.*|browser_.*)$"
)


def default_t3_home() -> Path:
    return Path(os.environ.get("T3CODE_HOME") or (Path.home() / ".t3"))


def discover_t3_antigravity_profiles(t3_home: Path | None = None) -> list[Path]:
    """Return all Antigravity ACP instance profile directories under T3 Code's state root."""
    root = (t3_home or default_t3_home()) / "userdata" / "providers" / "antigravity"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _quote_win_shlex(path_str: str) -> str:
    norm = path_str.replace("\\", "/")
    if not norm or any(ch.isspace() or ch in "\"'&|;<>()$`" for ch in norm):
        escaped = norm.replace('"', '\\"')
        return f'"{escaped}"'
    return norm


def build_acp_hook_command(
    repo_root: Path,
    python_exe: str | None = None,
    *,
    is_windows: bool | None = None,
) -> str:
    """Build a `shlex.split`-safe command string for `agy_acp_server`'s hook runner."""
    is_win = (os.name == "nt") if is_windows is None else is_windows
    py = python_exe or sys.executable
    hook_script = str((repo_root / "scripts" / "review_hook.py").resolve())
    if is_win:
        return f"{_quote_win_shlex(py)} -X utf8 {_quote_win_shlex(hook_script)}"
    return f"{shlex.quote(py)} {shlex.quote(hook_script)}"


def load_plugin_matcher(repo_root: Path) -> str:
    hooks_file = repo_root / "hooks.json"
    if hooks_file.is_file():
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        entries = (data.get(HOOK_KEY) or {}).get("PreToolUse") or []
        if entries and isinstance(entries[0], dict) and entries[0].get("matcher"):
            return str(entries[0]["matcher"])
    return DEFAULT_MATCHER


def _load_profile_hooks(hooks_path: Path) -> dict[str, Any]:
    if not hooks_path.is_file():
        return {}
    raw = hooks_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{hooks_path} must contain a JSON object")
    return data


def install_into_profile(
    profile_dir: Path,
    repo_root: Path,
    *,
    python_exe: str | None = None,
    timeout: int = 20,
    dry_run: bool = False,
) -> Path:
    """Merge `context-permission-reviewer` into `<profile_dir>/config/hooks.json`, preserving existing hooks."""
    hooks_path = profile_dir / "config" / "hooks.json"
    cfg = _load_profile_hooks(hooks_path)
    cfg[HOOK_KEY] = {
        "enabled": True,
        "PreToolUse": [
            {
                "matcher": load_plugin_matcher(repo_root),
                "hooks": [
                    {
                        "type": "command",
                        "command": build_acp_hook_command(repo_root, python_exe=python_exe),
                        "timeout": timeout,
                    }
                ],
            }
        ],
    }
    if not dry_run:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return hooks_path


def uninstall_from_profile(profile_dir: Path, *, dry_run: bool = False) -> bool:
    """Remove only `context-permission-reviewer` from `<profile_dir>/config/hooks.json`."""
    hooks_path = profile_dir / "config" / "hooks.json"
    if not hooks_path.is_file():
        return False
    cfg = _load_profile_hooks(hooks_path)
    if HOOK_KEY not in cfg:
        return False
    del cfg[HOOK_KEY]
    if not dry_run:
        hooks_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t3-home", type=Path, default=None, help="T3 Code state directory (default: ~/.t3)")
    parser.add_argument(
        "--profile",
        action="append",
        type=Path,
        default=[],
        help="Explicit T3 Antigravity profile directory (repeatable; defaults to auto-discovery)",
    )
    parser.add_argument("--python", default=None, help="Python executable path to embed in hooks.json")
    parser.add_argument("--uninstall", action="store_true", help="Remove context-permission-reviewer hook")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    profiles = args.profile or discover_t3_antigravity_profiles(args.t3_home)
    if not profiles:
        t3_root = args.t3_home or default_t3_home()
        sys.stderr.write(
            f"No T3 Code Antigravity profiles found under {t3_root / 'userdata' / 'providers' / 'antigravity'}.\n"
            "Enable Antigravity in T3 Code once first, then re-run this script.\n"
        )
        return 1

    for prof in profiles:
        if args.uninstall:
            changed = uninstall_from_profile(prof, dry_run=args.dry_run)
            action = "would remove" if args.dry_run else "removed"
            status = action if changed else "not present in"
            print(f"{status}: {prof / 'config' / 'hooks.json'}")
        else:
            out_path = install_into_profile(
                prof, repo_root, python_exe=args.python, dry_run=args.dry_run
            )
            verb = "would install" if args.dry_run else "installed"
            print(f"{verb}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
