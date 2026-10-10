import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from openpilot.sunnypilot.modeld_v2 import runtime_guard


class TestRuntimeGuard(unittest.TestCase):
  def make_guard(self, script, **kwargs):
    return runtime_guard.RuntimeGuard([sys.executable, "-c", script], Mock(),
                                      startup_timeout=2., run_timeout=.3, **kwargs)

  def test_native_stall_reaps_owner_before_small_only_restart(self):
    with tempfile.TemporaryDirectory() as directory:
      record = Path(directory) / "owners"
      script = f"""
import ctypes, os, signal
with open({str(record)!r}, "a") as f:
  f.write(str(os.getpid()) + ":" + os.getenv("MODELD_DISABLE_CHESTNUT", "0") + "\\n")
os.write(int(os.environ["MODELD_HEARTBEAT_FD"]), b"R")
if os.getenv("MODELD_DISABLE_CHESTNUT") != "1":
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  ctypes.CDLL(None).sleep(60)
"""
      guard = self.make_guard(script)
      def failed(reason):
        old_pid = int(record.read_text().split(":")[0])
        with self.assertRaises(ProcessLookupError):
          os.kill(old_pid, 0)
        self.assertIn("phase R", reason)
      guard.on_failure.side_effect = failed
      with patch.object(runtime_guard, "STOP_TIMEOUT", .1):
        self.assertEqual(guard.run(), 0)
      self.assertEqual([line.split(":")[1] for line in record.read_text().splitlines()], ["0", "1"])
      guard.on_failure.assert_called_once()
      self.assertIsNone(guard.child)

  def test_no_retry_loop_when_small_model_also_stalls(self):
    guard = self.make_guard('import os,time; os.write(int(os.environ["MODELD_HEARTBEAT_FD"]), b"R"); time.sleep(60)')
    self.assertEqual(guard.run(), 1)
    self.assertEqual(guard.on_failure.call_count, 2)

  def test_startup_stall_and_load_failure_use_small_only_restart(self):
    for first in ('import time; time.sleep(60)', 'raise RuntimeError("load failed")'):
      with self.subTest(first=first):
        script = 'import os\nif os.getenv("MODELD_DISABLE_CHESTNUT") != "1":\n  ' + first
        guard = self.make_guard(script)
        guard.startup_timeout = .3
        self.assertEqual(guard.run(), 0)
        guard.on_failure.assert_called_once()

  def test_loading_activity_does_not_extend_startup_deadline(self):
    script = """
import os,time
if os.getenv("MODELD_DISABLE_CHESTNUT") != "1":
  while True:
    os.write(int(os.environ["MODELD_HEARTBEAT_FD"]), b"L")
    time.sleep(.05)
"""
    guard = self.make_guard(script)
    guard.startup_timeout = .3
    self.assertEqual(guard.run(), 0)
    self.assertIn("phase L", guard.on_failure.call_args.args[0])

  def test_progress_keeps_healthy_model_alive_beyond_runtime_deadline(self):
    script = """
import os,time
fd = int(os.environ["MODELD_HEARTBEAT_FD"])
for _ in range(12):
  os.write(fd, b"R")
  time.sleep(.06)
  os.write(fd, b"PI")
"""
    guard = self.make_guard(script)
    self.assertEqual(guard.run(), 0)
    guard.on_failure.assert_not_called()

  def test_activity_without_publication_does_not_hide_stall(self):
    script = """
import os,time
fd = int(os.environ["MODELD_HEARTBEAT_FD"])
while True:
  os.write(fd, b"RI")
  time.sleep(.05)
"""
    guard = self.make_guard(script)
    with patch.dict(os.environ, {runtime_guard.DISABLE_CHESTNUT_ENV: "1"}):
      self.assertEqual(guard.run(), 1)
    guard.on_failure.assert_called_once()

  def test_shutdown_does_not_restart_child(self):
    guard = self.make_guard('import time; time.sleep(60)')
    def stop():
      while guard.child is None:
        time.sleep(.01)
      guard.stop(signal.SIGINT)
    thread = threading.Thread(target=stop)
    thread.start()
    try:
      self.assertEqual(guard.run(), 0)
    finally:
      thread.join(timeout=2)
    self.assertFalse(thread.is_alive())
    guard.on_failure.assert_not_called()
    self.assertIsNone(guard.child)

  def test_small_only_environment_never_retries_chestnut(self):
    guard = self.make_guard('raise RuntimeError("small failed")')
    with patch.dict(os.environ, {runtime_guard.DISABLE_CHESTNUT_ENV: "1"}):
      self.assertEqual(guard.run(), 1)
    guard.on_failure.assert_called_once()

  def test_closed_progress_channel_is_failure_not_success(self):
    script = """
import os,time
os.close(int(os.environ["MODELD_HEARTBEAT_FD"]))
time.sleep(60)
"""
    guard = self.make_guard(script)
    with patch.dict(os.environ, {runtime_guard.DISABLE_CHESTNUT_ENV: "1"}):
      self.assertEqual(guard.run(), 1)
    self.assertIn("closed progress channel", guard.on_failure.call_args.args[0])

  def test_progress_is_optional_outside_supervisor(self):
    with patch.dict(os.environ, {}, clear=True), patch.object(os, "write") as write:
      runtime_guard.report_progress(b"R")
    write.assert_not_called()

  def test_progress_uses_inherited_fd(self):
    with patch.dict(os.environ, {runtime_guard.HEARTBEAT_ENV: "42"}), patch.object(os, "write") as write:
      runtime_guard.report_progress(b"I")
    write.assert_called_once_with(42, b"I")


if __name__ == "__main__":
  unittest.main()
