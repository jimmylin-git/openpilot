import contextlib
import ctypes
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common import usbgpu_bus_lock as shared_lock
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import CallInfo, Ops, UOp
from tinygrad.engine import realize
from tinygrad.runtime.support import usb


class TestUSBMMIOHostSynchronization(unittest.TestCase):
  def setUp(self):
    self.bridge = Mock()
    self.bridge.pcie_mem_read.return_value = memoryview(b"\x01\x00\x00\x00")
    self.mmio = usb.USBMMIOInterface(self.bridge, 0x1000, 4, "I")
    self.host = Mock()

  def test_host_timeline_poll_does_not_recursively_synchronize(self):
    self.host.synchronize.side_effect = lambda: self.mmio[0]
    with patch.object(usb, "Device", {usb.HCQ_RUNTIME_DEV.value: self.host}):
      self.assertEqual(self.mmio[0], 1)
    self.host.synchronize.assert_called_once()
    self.assertEqual(self.bridge.pcie_mem_read.call_count, 2)

  def test_write_still_waits_for_host_before_device_access(self):
    events = []
    self.host.synchronize.side_effect = lambda: events.append("sync")
    self.bridge.pcie_mem_write.side_effect = lambda *args: events.append("write")
    with patch.object(usb, "Device", {usb.HCQ_RUNTIME_DEV.value: self.host}):
      self.mmio[0] = 2
    self.assertEqual(events, ["sync", "write"])

  def test_failed_synchronization_is_propagated_and_guard_is_reset(self):
    self.host.synchronize.side_effect = [RuntimeError("timeline timeout"), None]
    with patch.object(usb, "Device", {usb.HCQ_RUNTIME_DEV.value: self.host}):
      with self.assertRaisesRegex(RuntimeError, "timeline timeout"):
        self.mmio[0]
      self.bridge.pcie_mem_read.assert_not_called()
      self.assertEqual(self.mmio[0], 1)
    self.assertEqual(self.host.synchronize.call_count, 2)


class TestUSBConfiguration(unittest.TestCase):
  def test_all_initialization_failures_close_open_handle(self):
    stages = ("descriptor", "product", "kernel", "detach", "reset", "configure", "unsupported_product", None)
    for failed_stage in stages:
      with self.subTest(stage=failed_stage):
        handle = ctypes.POINTER(usb.libusb.struct_libusb_device_handle)()

        def stage(name, result=0):
          if failed_stage == name:
            raise RuntimeError(f"failed {name}")
          return result

        def product(handle, index, buffer, size):
          value = b"other" if failed_stage == "unsupported_product" else b"custom GPU"
          buffer[:len(value)] = value
          return stage("product", len(value))

        with (
          patch.object(usb.c, "init_c_var", return_value=handle),
          patch.object(usb, "_usb_locked", side_effect=lambda fn, *args: fn(*args)),
          patch.object(usb.libusb, "libusb_get_device", return_value=None),
          patch.object(usb.libusb, "libusb_get_device_descriptor", side_effect=lambda *args: stage("descriptor")),
          patch.object(usb.libusb, "libusb_get_string_descriptor_ascii", side_effect=product),
          patch.object(usb.libusb, "libusb_kernel_driver_active", side_effect=lambda *args: stage("kernel", 1)),
          patch.object(usb.libusb, "libusb_detach_kernel_driver", side_effect=lambda *args: stage("detach")),
          patch.object(usb.libusb, "libusb_reset_device", side_effect=lambda *args: stage("reset")),
          patch.object(usb.USB3, "_configure_interface", side_effect=lambda: stage("configure")),
          patch.object(usb.libusb, "libusb_close") as close,
        ):
          if failed_stage == "unsupported_product":
            with self.assertRaisesRegex(RuntimeError, "Unsupported USB bridge"):
              usb.USB3(None)
          elif failed_stage is not None:
            with self.assertRaisesRegex(RuntimeError, f"failed {failed_stage}"):
              usb.USB3(None)
          else:
            usb.USB3(None)
          if failed_stage is None:
            close.assert_not_called()
          else:
            close.assert_called_once_with(handle)

  def configure(self, current=1, claim_error=None, alt_error=None):
    events = []

    @contextlib.contextmanager
    def lock():
      events.append("lock")
      try:
        yield
      finally:
        events.append("unlock")

    def get_configuration(handle, value):
      ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = current
      events.append("get")
      return 0

    def claim(*args):
      events.append("claim")
      if claim_error:
        raise claim_error
      return 0

    with (
      patch.object(usb, "usbgpu_bus_lock", lock),
      patch.object(usb.libusb, "libusb_get_configuration", side_effect=get_configuration),
      patch.object(usb.libusb, "libusb_set_configuration", side_effect=lambda *args: events.append("set") or 0),
      patch.object(usb.libusb, "libusb_claim_interface", side_effect=claim),
      patch.object(usb.libusb, "libusb_set_interface_alt_setting", side_effect=alt_error or (lambda *args: events.append("alt") or 0)),
      patch.object(usb.libusb, "libusb_release_interface", side_effect=lambda *args: events.append("release") or 0),
    ):
      try:
        usb.USB3._configure_interface(SimpleNamespace(handle=None))
      finally:
        self.events = events

  def test_active_configuration_is_not_reset(self):
    self.configure()
    self.assertEqual(self.events, ["lock", "get", "claim", "alt", "unlock"])

  def test_unconfigured_device_is_configured_before_claim(self):
    self.configure(current=0)
    self.assertEqual(self.events, ["lock", "get", "set", "claim", "alt", "unlock"])

  def test_busy_owner_is_not_ignored_or_reset(self):
    with self.assertRaisesRegex(RuntimeError, "Resource busy"):
      self.configure(claim_error=RuntimeError("Resource busy"))
    self.assertEqual(self.events, ["lock", "get", "claim", "unlock"])

  def test_failed_alt_setting_releases_claim(self):
    with self.assertRaisesRegex(RuntimeError, "No such device"):
      self.configure(alt_error=RuntimeError("No such device"))
    self.assertEqual(self.events, ["lock", "get", "claim", "release", "unlock"])


class TestUSBExecutionLock(unittest.TestCase):
  def setUp(self):
    self.dev = Mock(is_usb=True)
    self.transfer = usb.libusb.struct_libusb_transfer()
    self.transfer.timeout = 1000
    self.registry = patch.dict(usb._compiled_transfers, {self.dev: [self.transfer]}, clear=True)
    self.registry.start()
    self.addCleanup(self.registry.stop)
    self.call = SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(device=("AMD",))))
    self.ast = UOp.sink()
    self.events = []

  @contextlib.contextmanager
  def lock(self):
    self.events.append("lock")
    try:
      yield
    finally:
      self.events.append("unlock")

  def execute(self, run):
    with (
      patch.object(realize, "Device", {"AMD": self.dev}),
      patch.object(realize, "resolve_params", return_value=[]),
      patch.object(realize, "_exec_hcq", side_effect=run),
      patch.object(usb, "usbgpu_bus_lock", self.lock),
    ):
      return realize.exec_hcq(realize.ExecContext(), self.call, self.ast)

  def test_completion_before_unlock(self):
    def run(*args):
      self.events.append("submit")
      self.transfer.status, self.transfer.length = 0xFF, 4
      return [None]

    def complete(*args):
      self.events.append("complete")
      self.transfer.status, self.transfer.actual_length = 0, 4
      return 0

    with patch.object(usb.USB3, "ctx", return_value=None), patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=complete):
      self.assertEqual(self.execute(run), [None])
    self.assertEqual(self.events, ["lock", "submit", "complete", "unlock"])

  def test_acquisition_failure_prevents_execution(self):
    run = Mock()

    @contextlib.contextmanager
    def failed_lock():
      raise OSError("lock acquisition failed")
      yield

    with (
      patch.object(realize, "Device", {"AMD": self.dev}),
      patch.object(realize, "resolve_params", return_value=[]),
      patch.object(realize, "_exec_hcq", run),
      patch.object(usb, "usbgpu_bus_lock", failed_lock),
    ):
      with self.assertRaisesRegex(OSError, "lock acquisition failed"):
        realize.exec_hcq(realize.ExecContext(), self.call, self.ast)
    run.assert_not_called()

  def test_nested_shared_context_is_reentrant(self):
    flock = Mock()
    fake_fcntl = SimpleNamespace(flock=flock, LOCK_EX=2, LOCK_UN=8)
    with (
      patch.object(shared_lock, "fcntl", fake_fcntl),
      patch.object(shared_lock, "_shared_lock_fd", return_value=123),
      patch.object(shared_lock, "_priority_request", contextlib.nullcontext),
    ):
      with shared_lock.usbgpu_bus_lock():
        with usb.usb_execution([self.dev]):
          self.assertEqual(shared_lock._thread_state.depth, 2)
      self.assertEqual(flock.call_args_list, [unittest.mock.call(123, 2), unittest.mock.call(123, 8)])

  def test_runtime_error_cancels_and_completes_before_unlock(self):
    def run(*args):
      self.events.append("submit")
      self.transfer.status, self.transfer.length = 0xFF, 4
      raise RuntimeError("runtime failure")

    def cancel(*args):
      self.events.append("cancel")
      return 0

    def complete(*args):
      self.events.append("complete")
      self.transfer.status = usb.libusb.LIBUSB_TRANSFER_CANCELLED
      return 0

    with (
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.libusb, "libusb_cancel_transfer", side_effect=cancel),
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=complete),
    ):
      with self.assertRaises(ExceptionGroup) as caught:
        self.execute(run)
    self.assertIsInstance(caught.exception.__context__, RuntimeError)
    self.assertEqual(self.events, ["lock", "submit", "cancel", "complete", "unlock"])

  def test_non_usb_hcq_does_not_lock(self):
    self.dev.is_usb = False
    with (
      patch.object(realize, "Device", {"AMD": self.dev}),
      patch.object(realize, "_exec_hcq", return_value=[None]),
      patch.object(usb, "usbgpu_bus_lock") as lock,
    ):
      self.assertEqual(realize.exec_hcq(realize.ExecContext(), self.call, self.ast), [None])
    lock.assert_not_called()

  def test_restored_transfer_is_drained(self):
    restored = usb.libusb.struct_libusb_transfer()
    restored.status, restored.length, restored.timeout = 0xFF, 4, 1000

    def complete(*args):
      restored.status, restored.actual_length = 0, 4
      return 0

    with patch.object(usb.USB3, "ctx", return_value=None), patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=complete):
      with usb.usb_execution([self.dev], [restored]):
        pass
    self.assertEqual(restored.status, 0)
    self.assertEqual(restored.length, 0)

  def test_execution_finds_pickled_transfer_parameter(self):
    buffer = realize.Buffer("CPU", ctypes.sizeof(self.transfer), dtypes.uint8, preallocate=True)
    restored = usb.libusb.struct_libusb_transfer.from_address(buffer.host.addr)
    restored.timeout = 1000
    ast = UOp.param(0, dtypes.uint8, ctypes.sizeof(restored), "CPU", name="usb_xfer0_0").sink()

    def run(*args):
      restored.status, restored.length = 0xFF, 4
      return [None]

    def complete(*args):
      restored.status, restored.actual_length = 0, 4
      return 0

    with (
      patch.object(realize, "Device", {"AMD": self.dev}),
      patch.object(realize, "resolve_params", return_value=[UOp.from_buffer(buffer)]),
      patch.object(realize, "_exec_hcq", side_effect=run),
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=complete),
    ):
      self.assertEqual(realize.exec_hcq(realize.ExecContext(), self.call, ast), [None])
    self.assertEqual(restored.length, 0)

  def test_event_failure_cancels_then_reaps(self):
    self.transfer.status, self.transfer.length = 0xFF, 4

    def events(*args):
      if "cancel" not in self.events:
        return usb.libusb.LIBUSB_ERROR_IO
      self.transfer.status = usb.libusb.LIBUSB_TRANSFER_CANCELLED
      self.events.append("complete")
      return 0

    def cancel(*args):
      self.events.append("cancel")
      return 0

    with (
      patch.object(usb, "usbgpu_bus_lock", self.lock),
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.libusb, "libusb_cancel_transfer", side_effect=cancel),
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=events),
    ):
      with self.assertRaises(ExceptionGroup):
        with usb.usb_execution([self.dev]):
          pass
    self.assertEqual(self.events, ["lock", "cancel", "complete", "unlock"])

  def test_delayed_cancellation_retains_lock_until_completion(self):
    self.transfer.status, self.transfer.length = 0xFF, 4

    def cancel(*args):
      self.events.append("cancel")
      return 0

    def events(*args):
      self.events.append("poll")
      if self.events.count("poll") == 2:
        self.transfer.status = usb.libusb.LIBUSB_TRANSFER_CANCELLED
        self.events.append("complete")
      self.assertNotIn("unlock", self.events)
      return 0

    with (
      patch.object(usb, "usbgpu_bus_lock", self.lock),
      patch.object(usb.USB3, "ctx", return_value=None),
      patch.object(usb.time, "monotonic", side_effect=[0, 3, 3, 5]),
      patch.object(usb.sys, "stderr") as stderr,
      patch.object(usb.libusb, "libusb_cancel_transfer", side_effect=cancel),
      patch.object(usb.libusb, "libusb_handle_events_timeout", side_effect=events),
    ):
      with self.assertRaises(ExceptionGroup):
        with usb.usb_execution([self.dev]):
          pass
    self.assertEqual(self.events, ["lock", "cancel", "poll", "poll", "complete", "unlock"])
    self.assertIn("retaining bus lock", "".join(c.args[0] for c in stderr.write.call_args_list))

  def test_failed_submission_is_reported_without_waiting(self):
    self.transfer.status, self.transfer.length = 0xFF, 4
    with (
      patch.object(usb.libusb, "libusb_cancel_transfer", return_value=usb.libusb.LIBUSB_ERROR_NOT_FOUND),
      patch.object(usb.libusb, "libusb_handle_events_timeout") as events,
    ):
      with self.assertRaises(ExceptionGroup) as caught:
        with usb.usb_execution([self.dev]):
          raise RuntimeError("runtime failed")
    self.assertIn("not submitted", str(caught.exception.exceptions[0]))
    events.assert_not_called()
    self.assertEqual(self.transfer.length, 0)

  def test_stream_has_no_separate_fd_lock_calls(self):
    def fake_ccall(fn, *args):
      return UOp.custom_function(fn.__name__).call(*(a for a in args if isinstance(a, UOp)), name=fn.__name__)

    with patch.object(usb, "ccall", side_effect=fake_ccall):
      stream = usb.usb_stream(usb.usb_link(("MOCK",)), UOp.const(0, dtypes.uint64), UOp.placeholder((4,), dtypes.uint8, device="MOCK"), 4, True)
    names = [u.arg.name for u in stream.toposort() if u.op is Ops.CALL]
    self.assertEqual(names, ["libusb_control_transfer", "libusb_bulk_transfer"])

  def test_call_info_dtype_survives_pickle(self):
    self.assertEqual(pickle.loads(pickle.dumps(CallInfo(None, "test", dtype=dtypes.float32))).dtype, dtypes.float32)


if __name__ == "__main__":
  unittest.main()
