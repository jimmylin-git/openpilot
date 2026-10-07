import ctypes
import inspect
import struct
import unittest
from unittest.mock import patch

from tinygrad import helpers
from tinygrad.runtime.autogen.am import smu_13_0_0
from tinygrad.runtime.support.am.ip import AM_SMU
from tinygrad.runtime.support.usb import USB3, alloc_cbuffer


class TestChestnutTelemetryCompatibility(unittest.TestCase):
  def test_ina_control_read_signature_and_payload(self):
    transport = USB3.__new__(USB3)
    transport.handle = None
    transport._ctrl_buf, transport._ctrl_mv = alloc_cbuffer(5)
    payload = struct.pack("<Hh?", 12000, -250, False)

    def read(handle, request_type, request, value, index, buffer, length, timeout):
      self.assertEqual((request_type, request, value, index, length, timeout), (0xC0, 0xC0, 0, 0, 5, 1000))
      ctypes.memmove(buffer, payload, len(payload))
      return len(payload)

    with patch("tinygrad.runtime.support.usb.libusb.libusb_control_transfer", side_effect=read):
      self.assertEqual(struct.unpack("<Hh?", bytes(transport.control_read(0xC0, 5))), (12000, -250, False))

  def test_legacy_smu_metrics_layout_and_message_signature(self):
    self.assertEqual(list(inspect.signature(AM_SMU._send_msg).parameters), ["self", "msg", "param", "read_back_arg", "timeout", "debug"])
    module = smu_13_0_0
    for name in ("PPSMC_MSG_GetPptLimit", "PPSMC_MSG_TransferTableSmu2Dram", "TABLE_SMU_METRICS"):
      self.assertIsInstance(getattr(module, name), int)
    metrics = module.SmuMetricsExternal_t.from_buffer(bytearray(ctypes.sizeof(module.SmuMetricsExternal_t))).SmuMetrics
    metrics.AvgTemperature[module.TEMP_HOTSPOT] = 42
    metrics.AvgTemperature[module.TEMP_MEM] = 38
    self.assertEqual(metrics.AvgTemperature[module.TEMP_HOTSPOT], 42)
    self.assertEqual(metrics.AvgTemperature[module.TEMP_MEM], 38)
    for name in ("AverageSocketPower", "AverageGfxActivity", "AverageGfxclkFrequencyPostDs", "AvgFanRpm"):
      self.assertEqual(getattr(metrics, name), 0)

  def test_legacy_firmware_fetch_signature(self):
    self.assertEqual(list(inspect.signature(helpers.fetch_fw).parameters), ["path", "name", "sha256"])


if __name__ == "__main__":
  unittest.main()
