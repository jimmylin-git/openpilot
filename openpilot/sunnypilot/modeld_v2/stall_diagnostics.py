"""Record model publication stalls without resetting or interrupting inference."""

import faulthandler
from datetime import UTC, datetime
import os
import threading
import time

from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


class ModelStallDiagnostics:
  def __init__(self) -> None:
    root = os.path.join(Paths.swaglog_root(), "modeld_stalls")
    os.makedirs(root, exist_ok=True)
    self.path = os.path.join(root, f"{time.time_ns()}-{os.getpid()}.log")
    self.output = open(self.path, "a", buffering=1)
    self.stage = "initializing"
    self.frames = (0, 0)
    self.completed = 0
    self.last_completed = time.monotonic()
    self.active = threading.Event()
    self.stopped = threading.Event()
    # Start before modeld enables realtime scheduling, so this observer does not inherit FIFO priority.
    self.thread = threading.Thread(target=self._monitor, name="modeld-stall-diagnostics", daemon=True)
    self.thread.start()
    cloudlog.warning(f"modeld stall diagnostics: {self.path}")

  def start(self) -> None:
    self.last_completed = time.monotonic()
    self.active.set()

  def mark(self, stage: str) -> None:
    self.stage = stage

  def frame(self, main_id: int, extra_id: int) -> None:
    self.frames = (main_id, extra_id)

  def published(self) -> None:
    self.completed += 1
    self.last_completed = time.monotonic()

  def _snapshot(self, now: float) -> str:
    return (f"wall={datetime.now(UTC).isoformat()} monotonic={now:.3f} stage={self.stage} " +
            f"frames={self.frames} published={self.completed} no_publish_s={now - self.last_completed:.3f}")

  def _monitor(self) -> None:
    self.active.wait()
    previous_completed = -1
    next_snapshot = 0.0
    stalled = False
    try:
      while not self.stopped.is_set():
        now = time.monotonic()
        if self.completed != previous_completed:
          previous_completed = self.completed
          # The C watchdog can dump stacks even when a native call holds the Python GIL.
          faulthandler.dump_traceback_later(5, file=self.output)
          if stalled:
            cloudlog.warning(f"modeld stall recovered: {self._snapshot(now)}")
          stalled = False
        if now >= next_snapshot:
          self.output.write(self._snapshot(now) + "\n")
          next_snapshot = now + 1
        if now - self.last_completed >= 3 and not stalled:
          stalled = True
          cloudlog.warning(f"modeld publication stalled: {self._snapshot(now)}; stacks={self.path}")
          faulthandler.dump_traceback(file=self.output, all_threads=True)
        self.stopped.wait(0.5)
    except Exception:
      cloudlog.exception("modeld stall diagnostics failed")
    finally:
      faulthandler.cancel_dump_traceback_later()
      self.output.close()

  def close(self) -> None:
    self.stopped.set()
    self.active.set()
    self.thread.join()
