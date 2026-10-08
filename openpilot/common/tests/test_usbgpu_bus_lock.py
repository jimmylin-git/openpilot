import multiprocessing
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from openpilot.common import usbgpu_bus_lock as locks
from openpilot.common.usbgpu_bus_lock import fcntl, usbgpu_bus_lock


def _acquire_lock_in_child(entered):
  with usbgpu_bus_lock():
    entered.set()


def _hold_priority_in_child(entered, release):
  with locks._priority_request():
    entered.set()
    release.wait(3)


class TestUsbGpuBusLock(unittest.TestCase):
  def test_low_priority_admission_uses_nonblocking_intent_and_bus(self):
    calls = []
    fake = SimpleNamespace(LOCK_SH=1, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8,
                           flock=lambda fd, flags: calls.append((fd, flags)))
    with (
      patch.object(locks, "fcntl", fake),
      patch.object(locks.os, "open", return_value=456),
      patch.object(locks.os, "close") as close,
      patch.object(locks, "_shared_lock_fd", return_value=123),
    ):
      with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
        self.assertTrue(acquired)
        close.assert_called_once_with(456)
      self.assertEqual(calls, [(456, 6), (123, 6), (123, 8)])
    self.assertFalse(locks._priority_fds)

  def test_cross_process_busy_intent_releases_local_mutex(self):
    fake = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
    with (
      patch.object(locks, "fcntl", fake),
      patch.object(locks.os, "open", return_value=456),
      patch.object(locks.os, "close") as close,
      patch.object(fake, "flock", side_effect=BlockingIOError),
    ):
      with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
        self.assertFalse(acquired)
      close.assert_called_once_with(456)
    self.assertFalse(locks._priority_fds)
    # A different thread can now acquire, not just a reentrant caller.
    acquired_in_thread = threading.Event()

    def acquire():
      with usbgpu_bus_lock():
        acquired_in_thread.set()

    thread = threading.Thread(target=acquire)
    thread.start()
    thread.join(2)
    self.assertTrue(acquired_in_thread.is_set())

  def test_model_intent_is_announced_before_bus_acquisition(self):
    calls = []
    fake = SimpleNamespace(LOCK_SH=1, LOCK_EX=2, LOCK_UN=8,
                           flock=lambda fd, flags: calls.append((fd, flags)))
    with (
      patch.object(locks, "fcntl", fake),
      patch.object(locks.os, "open", return_value=456),
      patch.object(locks.os, "close") as close,
      patch.object(locks, "_shared_lock_fd", return_value=123),
    ):
      with usbgpu_bus_lock():
        self.assertEqual(locks._priority_requests, 1)
        close.assert_not_called()
      self.assertEqual(calls, [(456, 1), (123, 2), (123, 8)])
      close.assert_called_once_with(456)
    self.assertEqual(locks._priority_requests, 0)

  def test_blocking_display_wait_does_not_block_local_model(self):
    entered, release, display_entered = threading.Event(), threading.Event(), threading.Event()

    def model():
      with locks._priority_request():
        entered.set()
        release.wait(3)

    def display():
      with usbgpu_bus_lock(low_priority=True):
        display_entered.set()

    model_thread = threading.Thread(target=model)
    display_thread = threading.Thread(target=display)
    model_thread.start()
    try:
      self.assertTrue(entered.wait(2))
      display_thread.start()
      self.assertFalse(display_entered.wait(0.05))
      with usbgpu_bus_lock():
        pass
    finally:
      release.set()
      model_thread.join(2)
      if display_thread.ident is not None:
        display_thread.join(2)
    self.assertTrue(display_entered.is_set())

  def test_nonblocking_low_priority_skips_busy_thread(self):
    entered, release = threading.Event(), threading.Event()

    def holder():
      with usbgpu_bus_lock():
        entered.set()
        release.wait(3)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
      self.assertTrue(entered.wait(2))
      with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
        self.assertFalse(acquired)
    finally:
      release.set()
      thread.join(2)
    with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
      self.assertTrue(acquired)

  def test_pending_model_excludes_display_even_when_bus_is_free(self):
    with locks._priority_request():
      with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
        self.assertFalse(acquired)
    with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
      self.assertTrue(acquired)

  def test_priority_and_bus_errors_are_not_busy_skips(self):
    with patch.object(locks, "_try_low_priority_lock", side_effect=OSError("lock failed")):
      with self.assertRaisesRegex(OSError, "lock failed"):
        with usbgpu_bus_lock(low_priority=True, blocking=False):
          pass

  def test_nested_transfer_keeps_admitted_transaction_complete(self):
    with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
      self.assertTrue(acquired)
      with locks._priority_request():
        with usbgpu_bus_lock() as nested:
          self.assertTrue(nested)

  def test_invalid_nonblocking_model_request_is_explicit(self):
    with self.assertRaises(ValueError):
      with usbgpu_bus_lock(blocking=False):
        pass

  @unittest.skipIf(fcntl is None, "flock is only available on POSIX")
  def test_cross_process_model_intent_excludes_new_display(self):
    context = multiprocessing.get_context("fork")
    entered, release = context.Event(), context.Event()
    child = context.Process(target=_hold_priority_in_child, args=(entered, release))
    child.start()
    try:
      self.assertTrue(entered.wait(2))
      with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
        self.assertFalse(acquired)
    finally:
      release.set()
      child.join(3)
      if child.is_alive():
        child.terminate()
        child.join()
    self.assertEqual(child.exitcode, 0)
    with usbgpu_bus_lock(low_priority=True, blocking=False) as acquired:
      self.assertTrue(acquired)

  def test_is_reentrant(self):
    with usbgpu_bus_lock():
      with usbgpu_bus_lock():
        pass

  def test_serializes_threads(self):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first():
      with usbgpu_bus_lock():
        first_entered.set()
        release_first.wait(2)

    def second():
      first_entered.wait(2)
      with usbgpu_bus_lock():
        second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    self.assertTrue(first_entered.wait(2))
    time.sleep(0.05)
    self.assertFalse(second_entered.is_set())
    release_first.set()
    first_thread.join(2)
    second_thread.join(2)
    self.assertTrue(second_entered.is_set())

  @unittest.skipIf(fcntl is None, "flock is only available on POSIX")
  def test_serializes_processes(self):
    context = multiprocessing.get_context("fork")
    entered = context.Event()

    with usbgpu_bus_lock():
      child = context.Process(target=_acquire_lock_in_child, args=(entered,))
      child.start()
      time.sleep(0.05)
      self.assertFalse(entered.is_set())

    self.assertTrue(entered.wait(2))
    child.join(2)
    self.assertEqual(child.exitcode, 0)
