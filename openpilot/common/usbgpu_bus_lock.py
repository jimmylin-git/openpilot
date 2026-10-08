from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator

try:
  import fcntl
except ImportError:  # pragma: no cover - Windows development fallback
  fcntl = None  # type: ignore[assignment]


USBGPU_BUS_LOCK_PATH = "/tmp/carrot_usbgpu_bus.lock"
USBGPU_PRIORITY_PATH = USBGPU_BUS_LOCK_PATH + ".priority"

_process_lock = threading.RLock()
_priority_lock = threading.Lock()
_priority_requests = 0
_priority_fds: set[int] = set()
_thread_state = threading.local()
_lock_fd: int | None = None
_lock_pid: int | None = None


def _reset_after_fork() -> None:
  global _process_lock, _priority_lock, _priority_requests, _priority_fds, _thread_state, _lock_fd, _lock_pid

  if _lock_fd is not None:
    with contextlib.suppress(OSError):
      os.close(_lock_fd)
  for fd in _priority_fds:
    with contextlib.suppress(OSError):
      os.close(fd)
  _process_lock = threading.RLock()
  _priority_lock = threading.Lock()
  _priority_requests = 0
  _priority_fds = set()
  _thread_state = threading.local()
  _lock_fd = None
  _lock_pid = None


if hasattr(os, "register_at_fork"):
  os.register_at_fork(after_in_child=_reset_after_fork)


def _shared_lock_fd() -> int:
  global _lock_fd, _lock_pid

  pid = os.getpid()
  if _lock_fd is None or _lock_pid != pid:
    if _lock_fd is not None:
      with contextlib.suppress(OSError):
        os.close(_lock_fd)
    _lock_fd = os.open(USBGPU_BUS_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    _lock_pid = pid
  return _lock_fd


@contextlib.contextmanager
def _priority_request() -> Iterator[None]:
  global _priority_requests
  with _priority_lock:
    _priority_requests += 1
  fd = None
  try:
    if fcntl is not None:
      # A separate descriptor per request keeps concurrent waiters visible.
      with _priority_lock:
        fd = os.open(USBGPU_PRIORITY_PATH, os.O_CREAT | os.O_RDWR, 0o666)
        _priority_fds.add(fd)
      fcntl.flock(fd, fcntl.LOCK_SH)
    yield
  finally:
    if fd is not None:
      with _priority_lock:
        os.close(fd)
        _priority_fds.remove(fd)
    with _priority_lock:
      _priority_requests -= 1


def _try_low_priority_lock() -> bool:
  if not _process_lock.acquire(blocking=False):
    return False
  acquired = False
  fd = None
  try:
    with _priority_lock:
      if _priority_requests:
        return False
      if fcntl is not None:
        fd = os.open(USBGPU_PRIORITY_PATH, os.O_CREAT | os.O_RDWR, 0o666)
        _priority_fds.add(fd)
        try:
          fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
          fcntl.flock(_shared_lock_fd(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
          return False
      acquired = True
      return True
  finally:
    if fd is not None:
      with _priority_lock:
        os.close(fd)
        _priority_fds.remove(fd)
    if not acquired:
      _process_lock.release()


@contextlib.contextmanager
def usbgpu_bus_lock(*, low_priority: bool = False, blocking: bool = True) -> Iterator[bool]:
  """One transfer lock; waiting model requests exclude new low-priority transfers."""
  if not low_priority and not blocking:
    raise ValueError("nonblocking acquisition is only supported for low-priority transfers")
  depth = getattr(_thread_state, "depth", 0)
  if depth:
    _thread_state.depth = depth + 1
    try:
      yield True
    finally:
      _thread_state.depth = depth
    return

  with contextlib.ExitStack() as stack:
    if low_priority:
      while not _try_low_priority_lock():
        if not blocking:
          yield False
          return
        time.sleep(0.001)
      stack.callback(_process_lock.release)
    else:
      stack.enter_context(_priority_request())
      stack.enter_context(_process_lock)
      if fcntl is not None:
        fcntl.flock(_shared_lock_fd(), fcntl.LOCK_EX)
    _thread_state.depth = 1
    try:
      yield True
    finally:
      _thread_state.depth = 0
      if fcntl is not None:
        fcntl.flock(_shared_lock_fd(), fcntl.LOCK_UN)
