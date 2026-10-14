import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.sunnypilot.modeld_v2.stall_diagnostics import ModelStallDiagnostics


class TestModelStallDiagnostics(unittest.TestCase):
  def setUp(self):
    root = tempfile.TemporaryDirectory()
    self.addCleanup(root.cleanup)
    for target, kwargs in (
      ("openpilot.sunnypilot.modeld_v2.stall_diagnostics.Paths.swaglog_root", {"return_value": root.name}),
      ("openpilot.sunnypilot.modeld_v2.stall_diagnostics.cloudlog", {}),
      ("openpilot.sunnypilot.modeld_v2.stall_diagnostics.faulthandler", {}),
    ):
      patcher = patch(target, **kwargs)
      mocked = patcher.start()
      self.addCleanup(patcher.stop)
      if target.endswith("cloudlog"):
        self.logs = mocked
      if target.endswith("faulthandler"):
        self.handler = mocked
    self.diagnostics = ModelStallDiagnostics()
    self.addCleanup(self.diagnostics.close)

  def test_snapshot_records_stage_frames_and_publications(self):
    self.diagnostics.mark("adapter_run_usb_enqueue")
    self.diagnostics.frame(100, 101)
    self.diagnostics.published()
    snapshot = self.diagnostics._snapshot(time.monotonic())
    self.assertIn("stage=adapter_run_usb_enqueue", snapshot)
    self.assertIn("frames=(100, 101)", snapshot)
    self.assertIn("published=1", snapshot)
    self.assertIn("no_publish_s=", snapshot)

  def test_stall_records_stacks_and_recovery(self):
    stalled, recovered = threading.Event(), threading.Event()

    def warning(message):
      if "publication stalled:" in message:
        stalled.set()
      if "stall recovered:" in message:
        recovered.set()

    self.logs.warning.side_effect = warning
    self.diagnostics.mark("chestnut_gpu_metrics")
    self.diagnostics.start()
    self.diagnostics.last_completed -= 4
    self.assertTrue(stalled.wait(2))
    self.diagnostics.published()
    self.assertTrue(recovered.wait(2))
    self.handler.dump_traceback.assert_called_once()
    self.assertGreaterEqual(self.handler.dump_traceback_later.call_count, 2)
    self.diagnostics.close()
    self.assertIn("stage=chestnut_gpu_metrics", Path(self.diagnostics.path).read_text())
    self.handler.cancel_dump_traceback_later.assert_called_once()

  def test_close_before_start(self):
    self.diagnostics.close()
    self.assertFalse(self.diagnostics.thread.is_alive())
    self.assertTrue(self.diagnostics.output.closed)


if __name__ == "__main__":
  unittest.main()
