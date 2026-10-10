import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

STARTUP_TIMEOUT = 120.
RUN_TIMEOUT = 5.
STOP_TIMEOUT = 2.
HEARTBEAT_ENV = "MODELD_HEARTBEAT_FD"
DISABLE_CHESTNUT_ENV = "MODELD_DISABLE_CHESTNUT"


def report_progress(phase: bytes) -> None:
  if (fd := os.getenv(HEARTBEAT_ENV)) is not None:
    os.write(int(fd), phase)


class RuntimeGuard:
  def __init__(self, command: Sequence[str], on_failure: Callable[[str], None],
               startup_timeout: float = STARTUP_TIMEOUT, run_timeout: float = RUN_TIMEOUT):
    self.command = list(command)
    self.on_failure = on_failure
    self.startup_timeout = startup_timeout
    self.run_timeout = run_timeout
    self.stopping = False
    self.child: subprocess.Popen[bytes] | None = None

  def stop(self, signum: int, _frame=None) -> None:
    self.stopping = True
    if self.child is not None and self.child.poll() is None:
      self.child.send_signal(signum)

  def _stop_child(self) -> None:
    assert self.child is not None
    if self.child.poll() is None:
      self.child.terminate()
      try:
        self.child.wait(timeout=STOP_TIMEOUT)
      except subprocess.TimeoutExpired:
        self.child.kill()
    # Never start a replacement until the OS has reaped the old GPU owner.
    self.child.wait()

  def _run_child(self, small_only: bool) -> tuple[int, str]:
    read_fd, write_fd = os.pipe()
    try:
      env = dict(os.environ)
      env[HEARTBEAT_ENV] = str(write_fd)
      if small_only:
        env[DISABLE_CHESTNUT_ENV] = "1"
      self.child = subprocess.Popen(self.command, env=env, pass_fds=(write_fd,))
    finally:
      os.close(write_fd)
      if self.child is None:
        os.close(read_fd)
    phase = b"L"
    started = False
    deadline = time.monotonic() + self.startup_timeout
    try:
      while not self.stopping:
        if (code := self.child.poll()) is not None:
          return code, f"modeld exited with code {code}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          return 1, f"modeld stalled in phase {phase.decode('ascii')} (pid={self.child.pid})"
        if select.select([read_fd], [], [], min(remaining, .1))[0]:
          data = os.read(read_fd, 4096)
          if not data:
            if self.stopping:
              return 0, "modeld supervisor stopped"
            try:
              code = self.child.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
              return 1, f"modeld closed progress channel while still running (pid={self.child.pid})"
            return code, f"modeld exited with code {code}"
          phase = data[-1:]
          if any(value not in b"LRIP" for value in data):
            raise RuntimeError(f"invalid modeld progress phase: {phase!r}")
          if not started and any(value != ord("L") for value in data):
            started = True
            deadline = time.monotonic() + self.run_timeout
          if b"P" in data:
            deadline = time.monotonic() + self.run_timeout
      return 0, "modeld supervisor stopped"
    finally:
      os.close(read_fd)
      self._stop_child()
      self.child = None

  def run(self) -> int:
    small_only = os.getenv(DISABLE_CHESTNUT_ENV) == "1"
    while not self.stopping:
      code, reason = self._run_child(small_only)
      if self.stopping or code == 0:
        return code
      self.on_failure(reason)
      if small_only:
        return code
      small_only = True
    return 0


def main() -> None:
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog

  cloudlog.bind(daemon="modeld_tinygrad_guard")
  params = Params()

  def failed(reason: str) -> None:
    params.put_bool("ChestnutActive", False)
    params.put_bool("ChestnutLoading", False)
    cloudlog.error(f"{reason}; old modeld reaped; Chestnut disabled for this supervisor session")

  command = [sys.executable, "-m", "openpilot.sunnypilot.modeld_v2.modeld", *sys.argv[1:]]
  guard = RuntimeGuard(command, failed)
  for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, guard.stop)
  sys.exit(guard.run())


if __name__ == "__main__":
  main()
