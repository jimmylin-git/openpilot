import types
import unittest
from unittest import mock

import numpy as np

from openpilot.sunnypilot.modeld_v2 import verify_extracted_policy as verify


class TestVerificationChecks(unittest.TestCase):
  def test_exact_comparison_rejects_shape_dtype_nonfinite_and_values(self):
    expected = np.zeros((1, 4), dtype=np.float32)
    verify.check_equal('outputs', expected.copy(), expected)
    for actual in (expected.flatten(), expected.astype(np.float64),
                   np.full_like(expected, np.nan), np.full_like(expected, np.inf), expected + 1):
      with self.subTest(actual=actual):
        with self.assertRaises(RuntimeError):
          verify.check_equal('outputs', actual, expected)

  def test_timing_budget_is_strictly_over_50ms(self):
    result = verify.timing_summary([0, 50, 51, 100])
    self.assertEqual(result['samples'], 4)
    self.assertEqual(result['over_50ms'], 2)
    self.assertEqual(result['mean_ms'], 50.25)
    self.assertAlmostEqual(result['p95_ms'], 92.65)
    self.assertAlmostEqual(result['p99_ms'], 98.53)
    for invalid in ([], [-1], [np.nan], [np.inf]):
      with self.subTest(invalid=invalid):
        with self.assertRaises(ValueError):
          verify.timing_summary(invalid)

  def test_invalid_run_counts_rejected_before_device_access(self):
    for runs, warmup in ((0, 0), (1, -1)):
      with mock.patch.object(verify, 'require_parked') as parked:
        with self.assertRaises(ValueError):
          verify.verify(verify.Path('unused.pkl'), (1928, 1208), runs, warmup, False)
        parked.assert_not_called()
    with mock.patch.object(verify, 'require_parked') as parked:
      with self.assertRaisesRegex(ValueError, 'two total iterations'):
        verify.verify(verify.Path('unused.pkl'), (1928, 1208), 1, 0, True)
      parked.assert_not_called()


class TestParkedGuard(unittest.TestCase):
  def check_guard(self, *, offroad=True, present=True, cmdline=None):
    params = mock.Mock()
    params.get_bool.return_value = offroad
    modules = {
      'openpilot.common.params': types.SimpleNamespace(Params=lambda: params),
      'openpilot.selfdrive.modeld.helpers': types.SimpleNamespace(chestnut_present=lambda: present),
    }
    proc = mock.Mock()
    proc.is_dir.return_value = True
    entries = []
    if cmdline is not None:
      entry = mock.Mock()
      entry.parent.name = str(verify.os.getpid() + 1)
      entry.read_bytes.return_value = cmdline
      entries.append(entry)
    proc.glob.return_value = entries
    with mock.patch.dict('sys.modules', modules), mock.patch.object(verify.os, 'name', 'posix'), \
         mock.patch.object(verify, 'Path', return_value=proc):
      verify.require_parked()

  def test_offroad_idle_present_is_required(self):
    self.check_guard()
    with self.assertRaisesRegex(RuntimeError, 'onroad'):
      self.check_guard(offroad=False)
    with self.assertRaisesRegex(RuntimeError, 'absent'):
      self.check_guard(present=False)

  def test_running_modeld_is_rejected(self):
    for cmdline in (b'./modeld\0', b'python\0/data/openpilot/openpilot/sunnypilot/modeld_v2/modeld.py\0',
                    b'python\0-m\0openpilot.sunnypilot.modeld_v2.modeld\0'):
      with self.subTest(cmdline=cmdline):
        with self.assertRaisesRegex(RuntimeError, 'share GPU'):
          self.check_guard(cmdline=cmdline)
    self.check_guard(cmdline=b'python\0-m\0openpilot.sunnypilot.modeld_v2.verify_extracted_policy\0')
