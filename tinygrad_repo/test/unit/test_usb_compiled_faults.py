"""CPU-only probes for compiled USB calls; expected failures document unfixed error handling."""

import ctypes
import unittest
from unittest.mock import patch

from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import lower_and_compile, run_linear
from tinygrad.runtime.support import usb
from tinygrad.runtime.support.hcq2 import ccall
from tinygrad.uop.ops import KernelInfo, Ops, UOp


# Windows aliases c_int to c_long; ccall needs the explicit signed-int format.
class CInt(ctypes._SimpleCData):
  _type_ = "i"


class TestCompiledUSBFaults(unittest.TestCase):
  def execute_call(self, operation, buffers):
    args, mapping = [], {}
    for i, param in enumerate(u for u in operation.toposort() if u.op is Ops.PARAM):
      self.assertIn(param, buffers)
      args.append(UOp.from_buffer(buffers[param]))
      mapping[param] = UOp.param(i, param.dtype, param.shape, "CPU")
    body = UOp.sink(operation.substitute(mapping, enter_calls=True), arg=KernelInfo("usb_fault_probe"))
    run_linear(lower_and_compile(UOp(Ops.LINEAR, src=(body.call(*args),))), update_stats=False)

  def run_bulk(self, status, received, guarded=False):
    calls = []
    callback_type = ctypes.CFUNCTYPE(CInt, ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_void_p,
                                    CInt, ctypes.c_void_p, ctypes.c_uint)

    @callback_type
    def transfer(handle, endpoint, address, length, actual, timeout):
      calls.append((endpoint, length, actual, timeout))
      if received:
        ctypes.memset(address, 0x11, received)
      if actual:
        ctypes.cast(actual, ctypes.POINTER(CInt)).contents.value = received
      return status

    transfer.__module__ = "tinygrad.runtime.autogen.libusb"
    transfer.__name__ = "libusb_bulk_transfer"
    function = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    function.host.view(fmt="Q")[0] = ctypes.cast(transfer, ctypes.c_void_p).value
    link = Buffer("CPU", 3, dtypes.uint64, preallocate=True)
    for i in range(3):
      link.host.view(fmt="Q")[i] = 0
    data = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    data.host.view(fmt="B")[:] = b"\xa5" * 16
    output = UOp.placeholder((16,), dtypes.uint8, device="CPU")
    result = Buffer("CPU", 2, dtypes.int32, preallocate=True)
    result.host.view(fmt="B")[:] = bytes(8)
    result_param = UOp.placeholder((2,), dtypes.int32, device="CPU")
    published = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    published.host.view(fmt="B")[:] = b"\x5a" * 16
    with patch.object(usb.libusb, "libusb_bulk_transfer", transfer):
      if guarded:
        # Candidate graph only: capture return code and byte count before consuming readback.
        ret = ccall(transfer, usb.usb_link(("CPU",)).index(0).load(), 0x81, output.index(0), 16,
                    result_param.index(1), 10000)
        bulk = result_param.index(0).store(ret)
      else:
        bulk = usb.usb_bulk(usb.usb_link(("CPU",)), 0x81, output.index(0), 16)

    buffers = {}
    for param in (u for u in bulk.toposort() if u.op is Ops.PARAM):
      if isinstance(param.tag, tuple) and param.tag[0] == "cfunc":
        buffer = function
      elif param.tag == "usb_host":
        buffer = link
      elif param is result_param:
        buffer = result
      else:
        buffer = data
      buffers[param] = buffer
    self.execute_call(bulk, buffers)
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][0:2], (0x81, 16))
    self.assertEqual(calls[0][3], 10000)
    if guarded:
      actual_status, actual_length = result.host.view(fmt="i")
      if actual_status != 0 or actual_length != 16:
        self.assertEqual(bytes(published.host.view(fmt="B")), b"\x5a" * 16)
        raise RuntimeError(f"compiled USB read failed: status={actual_status}, bytes={actual_length}/16")
      published.host.view(fmt="B")[:] = bytes(data.host.view(fmt="B"))
      self.assertEqual(bytes(published.host.view(fmt="B")), b"\x11" * 16)
    return bytes(data.host.view(fmt="B"))

  def run_control(self, returned, guarded=False):
    calls = []
    callback_type = ctypes.CFUNCTYPE(CInt, ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_ubyte,
                                    ctypes.c_ushort, ctypes.c_ushort, ctypes.c_void_p,
                                    ctypes.c_ushort, ctypes.c_uint)

    @callback_type
    def transfer(handle, request_type, request, value, index, address, length, timeout):
      calls.append((request_type, request, value, index, length, timeout))
      if returned > 0:
        ctypes.memset(address, 0x11, min(returned, length))
      return returned

    transfer.__module__ = "tinygrad.runtime.autogen.libusb"
    transfer.__name__ = "libusb_control_transfer"
    function = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    function.host.view(fmt="Q")[0] = ctypes.cast(transfer, ctypes.c_void_p).value
    link = Buffer("CPU", 3, dtypes.uint64, preallocate=True)
    link.host.view(fmt="B")[:] = bytes(24)
    data = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    data.host.view(fmt="B")[:] = b"\xa5" * 16
    output = UOp.placeholder((16,), dtypes.uint8, device="CPU")
    result = Buffer("CPU", 1, dtypes.int32, preallocate=True)
    result.host.view(fmt="B")[:] = bytes(4)
    result_param = UOp.placeholder((1,), dtypes.int32, device="CPU")
    with patch.object(usb.libusb, "libusb_control_transfer", transfer):
      call = usb.usb_ctrl(usb.usb_link(("CPU",)), 0xC0, 0xE4, 0xB450, 0, output.index(0), 16)
    operation = result_param.index(0).store(call) if guarded else call
    buffers = {}
    for param in (u for u in operation.toposort() if u.op is Ops.PARAM):
      if isinstance(param.tag, tuple) and param.tag[0] == "cfunc":
        buffer = function
      elif param.tag == "usb_host":
        buffer = link
      elif param is result_param:
        buffer = result
      else:
        buffer = data
      buffers[param] = buffer
    self.execute_call(operation, buffers)
    self.assertEqual(calls, [(0xC0, 0xE4, 0xB450, 0, 16, 1000)])
    if guarded and result.host.view(fmt="i")[0] != 16:
      raise RuntimeError(f"compiled USB control read failed: bytes={result.host.view(fmt='i')[0]}/16")
    return bytes(data.host.view(fmt="B"))

  def test_complete_read(self):
    self.assertEqual(self.run_bulk(0, 16), b"\x11" * 16)

  @unittest.expectedFailure
  def test_timeout_is_rejected(self):
    with self.assertRaises(RuntimeError):
      self.run_bulk(usb.libusb.LIBUSB_ERROR_TIMEOUT, 0)

  @unittest.expectedFailure
  def test_disconnect_is_rejected(self):
    with self.assertRaises(RuntimeError):
      self.run_bulk(usb.libusb.LIBUSB_ERROR_NO_DEVICE, 0)

  @unittest.expectedFailure
  def test_short_read_is_rejected(self):
    with self.assertRaises(RuntimeError):
      self.run_bulk(0, 8)

  def test_candidate_guard_accepts_complete_read(self):
    self.assertEqual(self.run_bulk(0, 16, guarded=True), b"\x11" * 16)

  def test_candidate_guard_rejects_errors_before_copying_readback(self):
    for status, received in ((usb.libusb.LIBUSB_ERROR_TIMEOUT, 0),
                             (usb.libusb.LIBUSB_ERROR_NO_DEVICE, 0),
                             (usb.libusb.LIBUSB_ERROR_TIMEOUT, 8), (0, 8), (0, 0)):
      with self.subTest(status=status, received=received):
        with self.assertRaisesRegex(RuntimeError, f"status={status}, bytes={received}/16"):
          self.run_bulk(status, received, guarded=True)

  def test_complete_control_read(self):
    self.assertEqual(self.run_control(16), b"\x11" * 16)

  @unittest.expectedFailure
  def test_control_timeout_is_rejected(self):
    with self.assertRaises(RuntimeError):
      self.run_control(usb.libusb.LIBUSB_ERROR_TIMEOUT)

  @unittest.expectedFailure
  def test_control_short_read_is_rejected(self):
    with self.assertRaises(RuntimeError):
      self.run_control(8)

  def test_candidate_control_guard(self):
    self.assertEqual(self.run_control(16, guarded=True), b"\x11" * 16)
    for returned in (usb.libusb.LIBUSB_ERROR_TIMEOUT, usb.libusb.LIBUSB_ERROR_NO_DEVICE, 8, 0):
      with self.subTest(returned=returned):
        with self.assertRaisesRegex(RuntimeError, f"bytes={returned}/16"):
          self.run_control(returned, guarded=True)


if __name__ == "__main__":
  unittest.main()
