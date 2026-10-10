"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import unittest
from unittest import mock
from pathlib import Path
import types

import numpy as np

from openpilot.sunnypilot.modeld_v2 import host_warp
from openpilot.sunnypilot.modeld_v2.host_warp import (
  CAMERA_INPUT_ABI, CompactInput, HostWarpRuntime, sampling_boundary_only,
  use_local_warp, validate_camera_input_abi, validate_chestnut_host_warp, warp_difference,
)

CAM_W, CAM_H = 1928, 1208


def host_warp_artifact():
  return {
    'metadata': {
      'camera_input_abi': CAMERA_INPUT_ABI,
      'frame_skip': 4,
      'warp_dev': 'AMD',
      'warp_source_hash': '0' * 64,
      'warped_shape': (2, 6, 128, 256),
      'model': {'input_shapes': {
        'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256), 'features_buffer': (1, 24, 512),
      }},
    },
    'input_devices': {'model': 'AMD'},
    'run_policy': object(),
    'run_policy_amd': object(),
    (CAM_W, CAM_H): object(),
  }


class TestChestnutHostWarp(unittest.TestCase):
  def test_compiler_requires_chestnut_supercombo_on_amd(self):
    validate_chestnut_host_warp('supercombo', True, True, 'AMD:LLVM')
    for model_type, chestnut, device, message in (
      ('vision_policy', True, 'AMD', 'supercombo and CHESTNUT'),
      ('supercombo', False, 'AMD', 'supercombo and CHESTNUT'),
      ('supercombo', True, 'QCOM', 'AMD inference'),
      ('supercombo', True, None, 'AMD inference'),
    ):
      with self.subTest(model_type=model_type, chestnut=chestnut, device=device):
        with self.assertRaisesRegex(ValueError, message):
          validate_chestnut_host_warp(model_type, True, chestnut, device)
    validate_chestnut_host_warp('supercombo', False, False, None)

  def test_runtime_accepts_matching_abi_and_preserves_legacy(self):
    artifact = host_warp_artifact()
    self.assertTrue(validate_camera_input_abi(artifact['metadata'], artifact, True, CAM_W, CAM_H))
    self.assertFalse(validate_camera_input_abi({}, {}, False, CAM_W, CAM_H))

  def test_rejects_old_and_incompatible_artifacts(self):
    cases = [
      (lambda artifact: artifact['metadata'].update(camera_input_abi='warped_yuv_v1'), True, 'Unsupported camera input ABI'),
      (lambda artifact: artifact['metadata'].pop('model'), True, 'Chestnut supercombo'),
      (lambda artifact: artifact['metadata'].update(warp_dev='QCOM'), True, 'retain an AMD warp'),
      (lambda artifact: artifact.pop((CAM_W, CAM_H)), True, 'no warp for camera size'),
      (lambda artifact: artifact['metadata'].update(frame_skip=1), True, 'invalid frame_skip'),
      (lambda artifact: artifact['metadata'].update(frame_skip=True), True, 'invalid frame_skip'),
      (lambda artifact: artifact.update(run_model=object()), True, 'Chestnut supercombo'),
      (lambda artifact: artifact.pop('run_policy_amd'), True, 'Chestnut supercombo'),
      (lambda artifact: artifact['metadata'].update(warped_shape=(2, 6, 64, 128)), True, 'camera input shapes'),
      (lambda artifact: artifact['metadata'].pop('warp_source_hash'), True, 'pinned warp source hash'),
      (lambda artifact: artifact.update(), False, 'Chestnut supercombo'),
    ]
    for change, chestnut, message in cases:
      with self.subTest(message=message):
        artifact = host_warp_artifact()
        change(artifact)
        with self.assertRaisesRegex(RuntimeError, message):
          validate_camera_input_abi(artifact['metadata'], artifact, chestnut, CAM_W, CAM_H)

  def test_device_gate_keeps_c4_and_pc_on_amd(self):
    for device in ('tici', 'tizi'):
      self.assertTrue(use_local_warp(device))
    for device in ('mici', 'pc', '', 'unknown'):
      self.assertFalse(use_local_warp(device))


class TestCompactInput(unittest.TestCase):
  def test_exact_image_payload_alignment_and_live_control_views(self):
    controls = {'desire': np.arange(8, dtype=np.float32), 'traffic_convention': np.zeros((1, 2), np.float32)}
    packed = CompactInput(controls, (2, 6, 128, 256))
    self.assertEqual(packed.control_bytes, 40)
    self.assertEqual(packed.image_offset, 512)
    self.assertEqual(packed.data.nbytes, 393728)
    packed.controls['desire'][3] = 42
    self.assertEqual(packed.data[:40].view(np.float32)[3], 42)
    packed.images[1, 5, 127, 255] = 255
    self.assertEqual(packed.data[-1], 255)
    np.testing.assert_array_equal(packed.data[40:512], 0)

  def test_recurrent_control_size_is_not_assumed_to_be_512_bytes(self):
    packed = CompactInput({'prev_feat': np.zeros((1, 16384), np.float32)}, (2, 6, 128, 256))
    self.assertEqual(packed.image_offset, 65536)
    self.assertEqual(packed.data.nbytes, 65536 + 393216)

  def test_rejects_non_float_controls(self):
    with self.assertRaisesRegex(ValueError, 'float32'):
      CompactInput({'bad': np.zeros(1, np.float64)}, (2, 6, 128, 256))


class TestWarpEquivalence(unittest.TestCase):
  def setUp(self):
    self.shape = (2, 6, 2, 2)
    self.frames = np.arange(2 * 24, dtype=np.uint8).reshape(2, 24)
    self.matrices = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    self.actual = np.zeros(self.shape, dtype=np.uint8)
    self.expected = self.actual.copy()

  def accepts(self):
    return sampling_boundary_only(self.actual, self.expected, self.frames, self.matrices, (4, 4), (4, 4, 2, 24), self.shape)

  def test_exact_and_shape_dtype_diagnostics(self):
    self.assertIsNone(warp_difference(self.actual, self.expected))
    self.assertIsNotNone(warp_difference(self.actual, self.expected.astype(np.float32)))
    self.assertFalse(sampling_boundary_only(self.actual, self.expected[:, :, :1], self.frames, self.matrices,
                                           (4, 4), (4, 4, 2, 24), self.shape))

  def test_only_correct_adjacent_y_source_pixels_pass(self):
    self.matrices[0, 0, 2] = .49998
    self.actual[0, 0, 0, 0] = self.frames[0, 1]
    self.expected[0, 0, 0, 0] = self.frames[0, 0]
    self.assertTrue(self.accepts())
    self.matrices[0, 0, 2] = .499
    self.assertFalse(self.accepts())

  def test_wrong_plane_or_camera_values_fail(self):
    self.matrices[0, 0, 2] = .49998
    self.expected[0, 0, 0, 0] = self.frames[0, 0]
    for wrong_value in (self.frames[0, 16], self.frames[1, 0], 255):
      self.actual[0, 0, 0, 0] = wrong_value
      self.assertFalse(self.accepts())

  def test_uv_offsets_are_checked_separately(self):
    self.matrices[1, 0, 2] = .99998
    for channel, offset in ((4, 16), (5, 17)):
      self.expected.fill(0)
      self.actual.fill(0)
      self.expected[1, channel, 0, 0] = self.frames[1, offset]
      self.actual[1, channel, 0, 0] = self.frames[1, offset + 2]
      self.assertTrue(self.accepts())
      self.actual[1, channel, 0, 0] = self.frames[1, offset + 1]
      self.assertFalse(self.accepts())

  def test_every_difference_must_be_explained(self):
    self.matrices[0, 0, 2] = .49998
    self.expected[0, 0, 0, 0] = self.frames[0, 0]
    self.actual[0, 0, 0, 0] = self.frames[0, 1]
    self.actual[1, 0, 1, 1] = 99
    self.assertFalse(self.accepts())

  def test_nonfinite_projection_is_not_accepted(self):
    self.matrices[0, 0, 2] = np.nan
    self.actual[0, 0, 0, 0] = 1
    self.assertFalse(self.accepts())


class FakeTensor:
  def __init__(self, data, **kwargs):
    self.data = np.asarray(data)

  def numpy(self):
    return self.data.copy()

  def realize(self):
    return self

  def _buffer(self):
    return self

  def copy_from(self, source):
    np.copyto(self.data, source.data)


class FakeWarp:
  def __init__(self, raw, frame_size, device, warp):
    self.raw = raw
    self.copies = []
    self.shape = (2, 6, 2, 2)

  def prepare(self):
    self.copies.append(self.raw.copy())
    return FakeTensor(np.full(self.shape, int(self.raw[72]), np.uint8))


class TestHostWarpRuntime(unittest.TestCase):
  def setUp(self):
    self.logger = mock.Mock()
    self.compact = CompactInput({'prev_feat': np.zeros((1, 2), np.float32)}, (2, 6, 2, 2))
    self.patches = [
      mock.patch.dict('sys.modules', {'tinygrad': types.SimpleNamespace(Tensor=FakeTensor)}),
      mock.patch.object(host_warp, 'WarpBackend', FakeWarp),
      mock.patch.object(host_warp, 'load_local_warp', side_effect=lambda raw, size, *args: FakeWarp(raw, size, 'QCOM', None)),
    ]
    for patch in self.patches:
      patch.start()
      self.addCleanup(patch.stop)

  def runtime(self, device_type='tizi'):
    return HostWarpRuntime(self.compact, 24, (4, 4), (4, 4, 2, 24), object(), 'AMD', device_type,
                           '0' * 64, Path('.'), self.logger)

  def test_probes_do_not_run_policy_and_clear_inputs(self):
    runtime = self.runtime()
    self.assertEqual(runtime.backend, 'qcom')
    self.assertEqual(len(runtime.amd.copies), 3)
    self.assertEqual(len(runtime.local.copies), 6)
    np.testing.assert_array_equal(runtime.raw, 0)
    np.testing.assert_array_equal(runtime.compact.data, 0)

  def test_compact_upload_has_latest_controls_and_latest_frame(self):
    runtime = self.runtime()
    policy = mock.Mock(return_value=FakeTensor([42]))
    fallback = mock.Mock()
    states = {'img_q': object(), 'feat_q': object(), 'desire_q': object()}
    for value in (7, 19):
      runtime.frames[:] = value
      runtime.compact.controls['prev_feat'][:] = value
      result = runtime.run(policy, fallback, states)
      self.assertIsInstance(result, FakeTensor)
      uploaded = policy.call_args.kwargs['packed_npy_inputs'].data
      np.testing.assert_array_equal(uploaded[:8].view(np.float32), value)
      np.testing.assert_array_equal(uploaded[512:], value)
      self.assertIs(policy.call_args.kwargs['feat_q'], states['feat_q'])
      self.assertEqual(runtime.last_timings['usb_input_bytes'], runtime.compact.data.nbytes)
    fallback.assert_not_called()

  def test_qcom_init_failure_preserves_same_model_and_state(self):
    with mock.patch.object(host_warp, 'load_local_warp', side_effect=RuntimeError('QCOM unavailable')):
      runtime = self.runtime()
    self.assertEqual(runtime.backend, 'amd')
    self.logger.exception.assert_called_once()
    policy, fallback = mock.Mock(), mock.Mock(return_value=FakeTensor([1]))
    states = {'feat_q': object()}
    runtime.run(policy, fallback, states)
    policy.assert_not_called()
    fallback.assert_called_once()
    self.assertIs(fallback.call_args.kwargs['feat_q'], states['feat_q'])

  def test_qcom_validation_failure_falls_back_and_clears_probes(self):
    bad = mock.Mock()
    bad.prepare.return_value = FakeTensor(np.zeros((2, 6, 2, 2), np.float32))
    with mock.patch.object(host_warp, 'load_local_warp', return_value=bad):
      runtime = self.runtime()
    self.assertEqual(runtime.backend, 'amd')
    self.logger.exception.assert_called_once()
    np.testing.assert_array_equal(runtime.raw, 0)

  def test_repeat_unstable_warp_is_rejected_even_if_first_matches(self):
    runtime = self.runtime()
    calls = iter((0, 1, 0, 1, 0, 1))
    runtime.local.prepare = lambda: FakeTensor(np.full((2, 6, 2, 2), next(calls), np.uint8))
    runtime.amd.prepare = lambda: FakeTensor(np.zeros((2, 6, 2, 2), np.uint8))
    with self.assertRaisesRegex(RuntimeError, 'repeat_matches_first'):
      runtime.validate_warp()
    np.testing.assert_array_equal(runtime.raw, 0)

  def test_prepare_failure_falls_back_before_any_recurrent_state_advances(self):
    runtime = self.runtime()
    runtime.local.prepare = mock.Mock(side_effect=RuntimeError('warp failed'))
    policy, fallback = mock.Mock(), mock.Mock(return_value=FakeTensor([1]))
    states = {'feat_q': object()}
    runtime.run(policy, fallback, states)
    self.assertEqual(runtime.backend, 'amd')
    policy.assert_not_called()
    fallback.assert_called_once()
    self.assertIs(fallback.call_args.kwargs['feat_q'], states['feat_q'])
    runtime.run(policy, fallback, states)
    self.assertEqual(fallback.call_count, 2)

  def test_policy_failure_is_not_retried_with_advanced_state(self):
    runtime = self.runtime()
    policy = mock.Mock(side_effect=RuntimeError('AMD inference failed'))
    fallback = mock.Mock()
    with self.assertRaisesRegex(RuntimeError, 'AMD inference failed'):
      runtime.run(policy, fallback, {})
    fallback.assert_not_called()
    policy.assert_called_once()

  def test_c4_does_not_initialize_or_validate_qcom(self):
    runtime = self.runtime('mici')
    host_warp.load_local_warp.assert_not_called()
    self.assertEqual(runtime.backend, 'amd')
    self.assertEqual(len(runtime.amd.copies), 0)
