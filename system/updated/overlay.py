import json
import os
from pathlib import Path
import subprocess
import sys


def git_output(basedir: str, *args: str) -> str:
  return subprocess.check_output(
    ["git", *args], cwd=basedir, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    stderr=subprocess.STDOUT, text=True,
  ).strip()


def base_state(basedir: str) -> dict[str, str | bool]:
  return {
    "commit": git_output(basedir, "rev-parse", "HEAD"),
    "branch": git_output(basedir, "rev-parse", "--abbrev-ref", "HEAD"),
    "clean": not git_output(basedir, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"),
  }


def write_base_state(basedir: str) -> None:
  state = base_state(basedir)
  Path(basedir, ".overlay_init").write_text(json.dumps(state), encoding="utf-8")


def overlay_skip_reason(basedir: str) -> str | None:
  marker = Path(basedir, ".overlay_init")
  try:
    original = json.loads(marker.read_text(encoding="utf-8"))
  except (FileNotFoundError, json.JSONDecodeError):
    return "missing or legacy update baseline; download the update again"
  if not isinstance(original, dict) or not isinstance(original.get("commit"), str) or not isinstance(original.get("branch"), str):
    return "invalid update baseline; download the update again"
  if original.get("clean") is not True:
    return "local files were modified when the update baseline was created"
  current = base_state(basedir)
  if current["commit"] != original["commit"] or current["branch"] != original["branch"]:
    return "local commit or branch changed since the update was prepared"
  if not current["clean"]:
    return "local files or submodules have uncommitted changes"
  return None


def main() -> int:
  try:
    reason = overlay_skip_reason(sys.argv[1])
  except (OSError, subprocess.CalledProcessError) as exc:
    print(f"Skipping update: unable to verify local changes: {exc}", file=sys.stderr)
    return 1
  if reason is not None:
    print(f"Skipping update: {reason}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
