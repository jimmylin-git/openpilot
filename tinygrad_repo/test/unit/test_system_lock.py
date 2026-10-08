import ast
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[2] / "tinygrad" / "runtime" / "support" / "system.py"


def system_class(directory):
  tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_System")
  cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
              n.name in ("__init__", "flock_acquire", "_reset_flocks_after_fork")]
  namespace = {"os": os, "threading": threading, "temp": lambda name: str(Path(directory) / name)}
  exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
  return namespace["_System"]


class TestSystemLock(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.register = Mock()
    with patch.object(os, "register_at_fork", self.register, create=True):
      self.system = system_class(self.directory.name)()
    self.fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=Mock())
    self.modules = patch.dict("sys.modules", {"fcntl": self.fcntl})
    self.modules.start()
    self.addCleanup(self.modules.stop)
    self.addCleanup(self.close_fds)

  def close_fds(self):
    for fd in self.system._flock_fds.values():
      os.close(fd)
    self.system._flock_fds.clear()

  def acquire(self, name):
    with patch.object(os, "umask"):
      return self.system.flock_acquire(name)

  def test_cache_reuses_owned_fd_and_separates_devices(self):
    first = self.acquire("gpu1")
    self.assertEqual(self.acquire("gpu1"), first)
    second = self.acquire("gpu2")
    self.assertNotEqual(first, second)
    self.assertEqual(self.acquire("gpu1"), first)
    self.assertEqual(self.system.lock_fd, first)
    self.assertFalse(os.get_inheritable(first))
    self.assertEqual(self.fcntl.flock.call_count, 2)

  def test_concurrent_same_device_only_locks_once(self):
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []

    def flock(*args):
      entered.set()
      self.assertTrue(release.wait(2))

    def acquire():
      try:
        results.append(self.acquire("gpu"))
      except BaseException as error:
        errors.append(error)

    self.fcntl.flock.side_effect = flock
    threads = [threading.Thread(target=acquire) for _ in range(2)]
    try:
      threads[0].start()
      self.assertTrue(entered.wait(2))
      threads[1].start()
      time.sleep(0.02)
      self.assertEqual(self.fcntl.flock.call_count, 1)
    finally:
      release.set()
      for thread in threads:
        if thread.ident is not None:
          thread.join(2)
    self.assertFalse(errors)
    self.assertEqual(len(results), 2)
    self.assertEqual(results[0], results[1])
    self.assertEqual(self.fcntl.flock.call_count, 1)

  def test_failed_flock_closes_fd_and_is_not_cached(self):
    self.fcntl.flock.side_effect = BlockingIOError("busy")
    original = os.open
    opened = []

    def record(*args):
      fd = original(*args)
      opened.append(fd)
      return fd

    with patch.object(os, "open", side_effect=record):
      with self.assertRaisesRegex(RuntimeError, "Failed to acquire"):
        self.acquire("gpu")
    self.assertFalse(self.system._flock_fds)
    with self.assertRaises(OSError):
      os.fstat(opened[0])
    self.fcntl.flock.side_effect = None
    self.acquire("gpu")
    self.assertEqual(self.fcntl.flock.call_count, 2)

  def test_child_callback_closes_without_unlock_and_reacquires(self):
    fd = self.acquire("gpu")
    callbacks = self.register.call_args.kwargs
    callbacks["before"]()
    callbacks["after_in_child"]()
    self.assertFalse(self.system._flock_fds)
    self.assertFalse(hasattr(self.system, "lock_fd"))
    with self.assertRaises(OSError):
      os.fstat(fd)
    self.acquire("gpu")
    self.assertEqual(self.fcntl.flock.call_count, 2)
    # Hooks stay valid for a second fork in the child.
    callbacks["before"]()
    callbacks["after_in_parent"]()
    self.assertEqual(len(self.system._flock_fds), 1)

  def test_descriptor_setup_failure_closes_fd_before_flock(self):
    with patch.object(os, "set_inheritable", side_effect=OSError("cannot configure descriptor")), patch.object(os, "close", wraps=os.close) as close:
      with self.assertRaisesRegex(RuntimeError, "Failed to acquire") as failure:
        self.acquire("gpu")
      self.assertIsInstance(failure.exception.__cause__, OSError)
      close.assert_called_once()
    self.assertFalse(self.system._flock_fds)
    self.fcntl.flock.assert_not_called()

  @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork and flock")
  def test_real_child_cannot_inherit_parent_ownership(self):
    self.modules.stop()
    system = system_class(self.directory.name)()
    try:
      with patch.object(os, "umask"):
        fd = system.flock_acquire("real_gpu")
      pid = os.fork()
      if pid == 0:
        try:
          system.flock_acquire("real_gpu")
        except RuntimeError:
          os._exit(0)
        os._exit(1)
      _, status = os.waitpid(pid, 0)
      self.assertEqual(os.waitstatus_to_exitcode(status), 0)
      self.assertEqual(system.flock_acquire("real_gpu"), fd)
    finally:
      for fd in system._flock_fds.values():
        os.close(fd)
      system._flock_fds.clear()
      self.modules.start()


if __name__ == "__main__":
  unittest.main()
