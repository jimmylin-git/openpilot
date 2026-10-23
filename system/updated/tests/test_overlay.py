import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("update_overlay", ROOT / "system/updated/overlay.py")
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


def git(repo, *args):
  return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.STDOUT, text=True).strip()


@pytest.fixture
def repo(tmp_path):
  git(tmp_path, "init", "-q")
  git(tmp_path, "config", "user.name", "Updater test")
  git(tmp_path, "config", "user.email", "updater-test@example.invalid")
  (tmp_path / ".gitignore").write_text(".overlay_init\nbuild-output\n", encoding="utf-8")
  (tmp_path / "source").write_text("original\n", encoding="utf-8")
  git(tmp_path, "add", ".")
  git(tmp_path, "commit", "-qm", "baseline")
  overlay.write_base_state(str(tmp_path))
  return tmp_path


def test_metadata_changes_do_not_invalidate_clean_baseline(repo):
  marker = repo / ".overlay_init"
  old = marker.stat().st_mtime - 20
  os.utime(marker, (old, old))
  lock = repo / ".git/index.lock"
  lock.touch()
  lock.unlink()
  git(repo, "update-index", "--refresh")
  git(repo, "config", "updater.test", "metadata only")
  (repo / ".git/FETCH_HEAD").write_text("metadata only\n", encoding="utf-8")
  assert (repo / ".git").stat().st_mtime > marker.stat().st_mtime
  assert overlay.overlay_skip_reason(str(repo)) is None


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked", "deleted", "commit", "branch"])
def test_real_local_changes_block_activation(repo, change):
  if change == "untracked":
    (repo / "new-source").write_text("new\n", encoding="utf-8")
  elif change == "branch":
    git(repo, "checkout", "-qb", "other-branch")
  elif change == "deleted":
    (repo / "source").unlink()
  else:
    (repo / "source").write_text("modified\n", encoding="utf-8")
    if change in ("staged", "commit"):
      git(repo, "add", "source")
    if change == "commit":
      git(repo, "commit", "-qm", "local change")
  assert overlay.overlay_skip_reason(str(repo)) is not None


def test_initially_dirty_baseline_is_never_activated(repo):
  (repo / "source").write_text("local change\n", encoding="utf-8")
  overlay.write_base_state(str(repo))
  (repo / "source").write_text("original\n", encoding="utf-8")
  assert "baseline was created" in overlay.overlay_skip_reason(str(repo))


def test_ignored_build_artifacts_do_not_block_activation(repo):
  (repo / "build-output").write_text("generated\n", encoding="utf-8")
  assert overlay.overlay_skip_reason(str(repo)) is None


@pytest.mark.parametrize("change", ["tracked", "untracked", "commit"])
def test_submodule_changes_block_activation(repo, tmp_path, change):
  child = tmp_path / "child"
  child.mkdir()
  git(child, "init", "-q")
  git(child, "config", "user.name", "Updater test")
  git(child, "config", "user.email", "updater-test@example.invalid")
  (child / "source").write_text("original\n", encoding="utf-8")
  git(child, "add", ".")
  git(child, "commit", "-qm", "submodule baseline")
  # Ignore the local source repository, not the checked-out submodule.
  with open(repo / ".gitignore", "a", encoding="utf-8") as f:
    f.write("child/\n")
  git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(child), "module")
  git(repo, "commit", "-qam", "add submodule")
  overlay.write_base_state(str(repo))
  assert overlay.overlay_skip_reason(str(repo)) is None
  module = repo / "module"
  (module / ("new-file" if change == "untracked" else "source")).write_text("modified\n", encoding="utf-8")
  if change == "commit":
    git(module, "-c", "user.name=Updater test", "-c", "user.email=updater-test@example.invalid",
        "commit", "-qam", "local submodule commit")
  assert overlay.overlay_skip_reason(str(repo)) is not None


@pytest.mark.parametrize("content", [None, "", "{broken", "[]", "{}", '{"commit": 4, "branch": "main"}'])
def test_missing_legacy_or_invalid_markers_fail_closed(repo, content):
  marker = repo / ".overlay_init"
  if content is None:
    marker.unlink()
  else:
    marker.write_text(content, encoding="utf-8")
  assert overlay.overlay_skip_reason(str(repo)) is not None


def test_cli_reports_error_and_exit_status(repo):
  command = [sys.executable, str(ROOT / "system/updated/overlay.py"), str(repo)]
  assert subprocess.run(command, capture_output=True).returncode == 0
  (repo / "source").write_text("modified\n", encoding="utf-8")
  result = subprocess.run(command, capture_output=True, text=True)
  assert result.returncode == 1
  assert "uncommitted changes" in result.stderr


def test_git_errors_are_visible_and_block_activation(repo):
  command = [sys.executable, str(ROOT / "system/updated/overlay.py"), str(repo)]
  (repo / ".git/HEAD").write_text("invalid\n", encoding="utf-8")
  result = subprocess.run(command, capture_output=True, text=True)
  assert result.returncode == 1
  assert "unable to verify" in result.stderr


def test_launcher_and_updater_use_same_source_baseline():
  launch = (ROOT / "launch_chffrplus.sh").read_text(encoding="utf-8")
  assert 'python3 "$DIR/system/updated/overlay.py" "$DIR" || LOCAL_MODIFIED=1' in launch
  assert "find ${DIR}/.git -newer" not in launch
  tree = ast.parse((ROOT / "system/updated/updated.py").read_text(encoding="utf-8"))
  init = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_overlay")
  calls = [ast.unparse(n) for n in ast.walk(init) if isinstance(n, ast.Call)]
  assert "overlay_skip_reason(BASEDIR)" in calls
  assert "write_base_state(BASEDIR)" in calls


def test_init_overlay_reuses_clean_base_after_metadata_refresh(repo):
  tree = ast.parse((ROOT / "system/updated/updated.py").read_text(encoding="utf-8"))
  init = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_overlay")
  reuse = init.body[0]
  git(repo, "config", "updater.test", "metadata only")
  from types import SimpleNamespace
  namespace = {
    "OVERLAY_INIT": repo / ".overlay_init", "OVERLAY_MERGED": "mounted-overlay", "BASEDIR": str(repo),
    "os": SimpleNamespace(path=SimpleNamespace(ismount=lambda path: True)),
    "overlay_skip_reason": overlay.overlay_skip_reason,
  }
  function = ast.FunctionDef(name="reuse", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                            body=[reuse, ast.Return(value=ast.Constant(value="rebuild"))], decorator_list=[])
  exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "<overlay reuse>", "exec"), namespace)
  assert namespace["reuse"]() is None
