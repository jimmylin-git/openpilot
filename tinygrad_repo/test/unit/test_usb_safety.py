import ast
import contextlib
import ctypes
import itertools
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tinygrad.runtime.support import usb, system


def make_usb():
  device = usb.USB3.__new__(usb.USB3)
  device.handle = ctypes.POINTER(usb.libusb.struct_libusb_device_handle)()
  device.product = "custom test"
  device._ctrl_buf, device._ctrl_mv = usb.alloc_cbuffer(16)
  device._bulk_buf, device._bulk_mv = usb.alloc_cbuffer(16)
  device._transferred = ctypes.c_int()
  device._async_pending, device._async_locks, device._async_pool = {}, {}, []
  device._async_seq, device._async_err = itertools.count(1), 0
  device._interface_claimed = True
  device._async_cb = usb.libusb.libusb_transfer_cb_fn(device._on_bulk_done)
  return device


class TestUSBTransferSafety(unittest.TestCase):
  def test_short_bulk_read_is_rejected(self):
    device = make_usb()

    def transfer(*args):
      device._transferred.value = 3
      return 0

    with patch.object(usb.libusb, "libusb_bulk_transfer", side_effect=transfer):
      with self.assertRaisesRegex(RuntimeError, "short read"):
        device.bulk_read(4)

  def test_read_results_survive_subsequent_reads(self):
    for method in ("control", "bulk"):
      with self.subTest(method=method):
        device = make_usb()
        counter = itertools.count(1)

        def transfer(*args):
          value = next(counter)
          if method == "control":
            device._ctrl_mv[:4] = bytes([value]) * 4
            return 4
          device._bulk_mv[:4] = bytes([value]) * 4
          device._transferred.value = 4
          return 0

        fn = "libusb_control_transfer" if method == "control" else "libusb_bulk_transfer"
        with patch.object(usb.libusb, fn, side_effect=transfer):
          first = device.control_read(0xE4, 4) if method == "control" else device.bulk_read(4)
          second = device.control_read(0xE4, 4) if method == "control" else device.bulk_read(4)
        self.assertEqual(bytes(first), b"\x01" * 4)
        self.assertEqual(bytes(second), b"\x02" * 4)

  def test_write_buffer_preparation_is_inside_bus_lock(self):
    for method in ("control", "bulk"):
      with self.subTest(method=method):
        device = make_usb()
        entered = False

        @contextlib.contextmanager
        def lock():
          nonlocal entered
          if not entered:
            buffer = device._ctrl_mv if method == "control" else device._bulk_mv
            self.assertEqual(bytes(buffer[:4]), bytes(4))
            entered = True
          yield

        def transfer(*args):
          self.assertTrue(entered)
          if method == "control":
            self.assertEqual(bytes(device._ctrl_mv[:4]), b"data")
            return 4
          self.assertEqual(bytes(device._bulk_mv[:4]), b"data")
          device._transferred.value = 4
          return 0

        fn = "libusb_control_transfer" if method == "control" else "libusb_bulk_transfer"
        with patch.object(usb, "usbgpu_bus_lock", lock), patch.object(usb.libusb, fn, side_effect=transfer):
          if method == "control":
            device.control_write(0xE5, data=b"data")
          else:
            device.bulk_write(b"data")

  def test_short_writes_and_control_reads_raise_explicit_errors(self):
    device = make_usb()
    with patch.object(usb.libusb, "libusb_control_transfer", return_value=3):
      with self.assertRaisesRegex(RuntimeError, "short write"):
        device.control_write(0xE5, data=b"data")
      with self.assertRaisesRegex(RuntimeError, "short read"):
        device.control_read(0xE4, 4)
    with patch.object(usb.libusb, "libusb_bulk_transfer", return_value=0):
      device._transferred.value = 3
      with self.assertRaisesRegex(RuntimeError, "short write"):
        device.bulk_write(b"data")


class TestUSBInitializationSafety(unittest.TestCase):
  def test_failed_amd_interface_setup_closes_usb_handle(self):
    path = Path(__file__).resolve().parents[2] / "tinygrad" / "runtime" / "ops_amd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "USBIface")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    for failure in ("amdev", "properties"):
      with self.subTest(failure=failure):
        bridge = Mock()
        pci = Mock(usb=Mock(usb=bridge))
        namespace = {
          "USB3": Mock(list_devices=Mock(return_value=[(None, "usb:1-2")])),
          "hcq_filter_visible_devices": lambda values, name: values,
          "USBPCIDevice": Mock(return_value=pci),
          "AMDev": Mock(side_effect=RuntimeError("amdev failed") if failure == "amdev" else None),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        obj = Mock()
        if failure == "properties": obj._compute_props.side_effect = RuntimeError("properties failed")
        with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
          namespace["__init__"](obj, Mock(), 0)
        bridge.close.assert_called_once()

  def test_close_releases_claim_and_pooled_transfers_once(self):
    device = make_usb()
    device._async_pool = [Mock()]
    with (
      patch.object(usb.libusb, "libusb_release_interface", return_value=0) as release,
      patch.object(usb.libusb, "libusb_close") as close,
      patch.object(usb.libusb, "libusb_free_transfer") as free,
    ):
      device.close()
      device.close()
    release.assert_called_once()
    close.assert_called_once()
    free.assert_called_once()
    self.assertIsNone(device.handle)

  def test_close_releases_handle_even_if_release_fails(self):
    device = make_usb()
    with (
      patch.object(usb.libusb, "libusb_release_interface", side_effect=RuntimeError("release failed")),
      patch.object(usb.libusb, "libusb_close") as close,
    ):
      with self.assertRaisesRegex(RuntimeError, "release failed"):
        device.close()
    close.assert_called_once()
    self.assertIsNone(device.handle)

  def test_close_never_frees_pending_transfer(self):
    device = make_usb()
    device._async_pending[1] = Mock()
    with patch.object(usb.libusb, "libusb_close") as close:
      with self.assertRaisesRegex(RuntimeError, "pending"):
        device.close()
    close.assert_not_called()

  def test_failed_controller_or_bar_setup_closes_usb_handle(self):
    for failure in ("controller", "bars"):
      with self.subTest(failure=failure):
        device = Mock(product="custom test")
        with (
          patch.object(system.System, "flock_acquire", return_value=123),
          patch.object(system, "USB3", return_value=device),
          patch.object(system, "CustomASM24Controller",
                       side_effect=RuntimeError("controller failed") if failure == "controller" else None),
          patch.object(system.System, "pci_setup_usb_bars", side_effect=RuntimeError("bars failed")),
        ):
          with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
            system.USBPCIDevice("AM", None, "usb:1-2")
        device.close.assert_called_once()

  def test_optimized_python_still_executes_control_transfers_and_checks_lengths(self):
    code = """
from unittest.mock import patch
from test.unit.test_usb_safety import make_usb
from tinygrad.runtime.support import usb
device = make_usb()
with patch.object(usb.libusb, 'libusb_control_transfer', return_value=4) as transfer:
  device.control_write(0xE5, data=b'data')
  device.control_read(0xE4, 4)
  if transfer.call_count != 2: raise RuntimeError('control transfers were optimized away')
with patch.object(usb.libusb, 'libusb_control_transfer', return_value=3):
  try: device.control_read(0xE4, 4)
  except RuntimeError: pass
  else: raise RuntimeError('short read accepted')
print('optimized USB checks passed')
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("optimized USB checks passed", result.stdout)


class TestUSBAsyncSafety(unittest.TestCase):
  def test_delayed_async_cancellation_retains_lock_until_callback(self):
    device = make_usb()
    transfer = usb.libusb.struct_libusb_transfer()
    transfer.length, transfer.timeout, transfer.user_data = 4, 1000, 1
    transfer.type = usb.libusb.LIBUSB_TRANSFER_TYPE_BULK
    pointer = ctypes.pointer(transfer)
    device._async_pending[1] = (pointer, bytearray(4))
    lock = Mock(__exit__=Mock())
    device._async_locks[1] = lock
    polls = 0

    def poll(*args):
      nonlocal polls
      polls += 1
      lock.__exit__.assert_not_called()
      if polls == 2:
        transfer.status = usb.libusb.LIBUSB_TRANSFER_CANCELLED
        device._on_bulk_done(pointer)
      return 0

    with (
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.libusb, "libusb_cancel_transfer", return_value=0) as cancel,
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=poll),
      patch.object(usb.time, "monotonic", side_effect=[0, 3, 3, 5]),
      patch.object(usb.sys, "stderr") as stderr,
    ):
      with self.assertRaises(ExceptionGroup):
        device.bulk_wait(1)
    cancel.assert_called_once_with(pointer)
    lock.__exit__.assert_called_once()
    self.assertIn("retaining bus lock", "".join(call.args[0] for call in stderr.write.call_args_list))

  def test_unexpected_poll_exception_does_not_unlock_pending_transfer(self):
    device = make_usb()
    transfer = usb.libusb.struct_libusb_transfer()
    transfer.timeout = 1000
    device._async_pending[1] = (ctypes.pointer(transfer), bytearray(4))
    lock = Mock(__exit__=Mock())
    device._async_locks[1] = lock
    with (
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=RuntimeError("unexpected event failure")),
    ):
      with self.assertRaisesRegex(RuntimeError, "unexpected event failure"):
        device.bulk_wait(1)
    lock.__exit__.assert_not_called()
    self.assertIn(1, device._async_pending)
    self.assertIn(1, device._async_locks)

  def test_failed_submit_frees_transfer_and_releases_lock(self):
    for allocation_failure in (False, True):
      with self.subTest(allocation_failure=allocation_failure):
        device = make_usb()
        transfer = usb.libusb.struct_libusb_transfer()
        pointer = ctypes.pointer(transfer) if not allocation_failure else None
        lock = Mock(__enter__=Mock(), __exit__=Mock(return_value=False))
        with (
          patch.object(usb, "usbgpu_bus_lock", return_value=lock),
          patch.object(usb.libusb, "libusb_alloc_transfer", return_value=pointer),
          patch.object(usb.libusb, "libusb_submit_transfer", side_effect=RuntimeError("submit failed")),
          patch.object(usb.libusb, "libusb_free_transfer") as free,
        ):
          with self.assertRaises((MemoryError, RuntimeError)):
            device.bulk_write_async(memoryview(bytearray(4)))
        self.assertFalse(device._async_pending)
        self.assertFalse(device._async_locks)
        lock.__exit__.assert_called_once()
        if allocation_failure: free.assert_not_called()
        else: free.assert_called_once_with(pointer)

  def test_wait_uses_owned_context_and_drains_cancellation_before_unlock(self):
    for failure, timeout_ms in ((False, 1000), (True, 1000), (False, 0)):
      with self.subTest(failure=failure, timeout_ms=timeout_ms):
        device = make_usb()
        transfer = usb.libusb.struct_libusb_transfer()
        transfer.length, transfer.timeout, transfer.user_data = 4, timeout_ms, 1
        transfer.type = usb.libusb.LIBUSB_TRANSFER_TYPE_BULK
        pointer = ctypes.pointer(transfer)
        device._async_pending[1] = (pointer, bytearray(4))
        lock = Mock(__exit__=Mock())
        device._async_locks[1] = lock
        context = object()
        events = []

        def poll(ctx, timeout):
          self.assertIs(ctx, context)
          lock.__exit__.assert_not_called()
          events.append("poll")
          if failure and len(events) == 1: return usb.libusb.LIBUSB_ERROR_IO
          transfer.status = usb.libusb.LIBUSB_TRANSFER_CANCELLED if failure else 0
          transfer.actual_length = 0 if failure else 4
          device._on_bulk_done(pointer)
          return 0

        with (
          patch.object(usb.USB3, "ctx", return_value=context),
          patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=poll),
          patch.object(usb.libusb, "libusb_cancel_transfer", return_value=0) as cancel,
          patch.object(usb.time, "sleep"),
        ):
          if failure:
            with self.assertRaises(ExceptionGroup):
              device.bulk_wait(1)
            cancel.assert_called_once_with(pointer)
          else:
            device.bulk_wait(1)
            cancel.assert_not_called()
        lock.__exit__.assert_called_once()
        self.assertFalse(device._async_pending)
        self.assertFalse(device._async_locks)


if __name__ == "__main__":
  unittest.main()
