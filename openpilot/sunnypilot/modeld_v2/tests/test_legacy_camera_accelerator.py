import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.sunnypilot.modeld_v2.model_adapters import LegacyModelAdapter
from openpilot.sunnypilot.modeld_v2.stock_dependencies import NV12Frame, make_run_model, make_warp
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.tensor import Tensor


class TestLegacyCameraAccelerator(unittest.TestCase):
  def make_adapter(self, enabled=True, smaller=True):
    shapes = {'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256),
              'desire_pulse': (1, 33, 8), 'features_buffer': (1, 32, 512)}
    full, small = Mock(), Mock()
    jits = {'metadata': {'model': {'input_shapes': shapes, 'output_slices': {}}}, (1928, 1208): full}
    if smaller:
      jits[(1344, 760)] = small

    def queues(input_shapes, frame_skip, device, frame_copy_size):
      prefix = 128
      packed = np.zeros(prefix + 2 * frame_copy_size, dtype=np.uint8)
      frames = packed[prefix:].reshape(2, frame_copy_size)
      return {'packed_npy_inputs': packed}, {'tfm': np.zeros((3, 3), np.float32),
                                           'big_tfm': np.zeros((3, 3), np.float32)}, dict(zip(('img', 'big_img'), frames, strict=True))

    with patch.dict(os.environ, ACCELERATE_CAMERA='1' if enabled else '0'), \
         patch('openpilot.sunnypilot.modeld_v2.model_adapters.stock_make_input_queues', side_effect=queues) as allocator:
      adapter = LegacyModelAdapter(jits, 1928, 1208, 'CPU', 'CPU', 'CPU', chestnut=True)
    if adapter._downscale_pool is not None:
      self.addCleanup(adapter._downscale_pool.shutdown)
    return adapter, full, small, allocator

  def test_uses_small_jit_and_matching_packed_frames(self):
    adapter, full, small, allocator = self.make_adapter()
    self.assertTrue(adapter.accelerate_camera)
    self.assertIs(adapter.run_policy, small)
    self.assertEqual(allocator.call_args.kwargs['frame_copy_size'], 1622016)
    self.assertEqual(adapter.frame_slots['img'].size, 1622016)
    dummy_frames, _, _ = adapter.get_dummy_inputs()
    stride, yh, uvh, _ = get_nv12_info(1928, 1208)
    self.assertEqual(dummy_frames['img'].size, stride * (yh + uvh))
    src = np.random.default_rng(1).integers(0, 256, dummy_frames['img'].size, dtype=np.uint8)
    adapter.copy_frames({'img': src, 'big_img': memoryview(src)})
    expected = np.zeros(adapter.frame_copy_size, np.uint8)
    adapter.downscaler.downsample_into(src, expected)
    np.testing.assert_array_equal(adapter.frame_slots['img'], expected)
    np.testing.assert_array_equal(adapter.frame_slots['big_img'], expected)
    adapter.run()
    small.assert_called_once()
    full.assert_not_called()
    adapter.reset_warmup_buffers()

  def test_disabled_preserves_original_jit_and_copy(self):
    adapter, full, small, allocator = self.make_adapter(enabled=False)
    self.assertFalse(adapter.accelerate_camera)
    self.assertIsNone(adapter.downscaler)
    self.assertIs(adapter.run_policy, full)
    stride, yh, uvh, _ = get_nv12_info(1928, 1208)
    self.assertEqual(allocator.call_args.kwargs['frame_copy_size'], stride * (yh + uvh))
    src = np.arange(adapter.frame_copy_size, dtype=np.uint8)
    adapter.copy_frames({'img': src, 'big_img': src})
    np.testing.assert_array_equal(adapter.frame_slots['img'], src)
    adapter.run()
    full.assert_called_once()
    small.assert_not_called()

  def test_missing_small_jit_is_explicit_and_preserves_original(self):
    with patch('openpilot.sunnypilot.modeld_v2.model_adapters.cloudlog') as logger:
      adapter, full, _, _ = self.make_adapter(smaller=False)
    self.assertFalse(adapter.accelerate_camera)
    self.assertIs(adapter.run_policy, full)
    logger.warning.assert_called_once()
    self.assertIn('no smaller camera JIT', logger.warning.call_args.args[0])

  def test_real_unified_jit_warp_and_scaled_coordinates(self):
    device = os.getenv('TEST_CAMERA_DEVICE', 'CPU')
    shapes = {'img': (1, 12, 16, 32), 'big_img': (1, 12, 16, 32),
              'desire_pulse': (1, 33, 8), 'features_buffer': (1, 32, 512)}
    metadata = {'input_shapes': shapes, 'output_slices': {}}
    nv12 = NV12Frame(1344, 760, *get_nv12_info(1344, 760))
    warp = make_warp(nv12, 64, 32)

    def policy(warped, img_q, big_img_q, feat_q, desire_q, policy_inputs):
      return warped

    with Context(DEV=device), patch.dict(os.environ, ACCELERATE_CAMERA='1'):
      jit = TinyJit(make_run_model(warp, policy, metadata, nv12.stride * (nv12.y_height + nv12.uv_height)), prune=True)
      adapter = LegacyModelAdapter({'metadata': {'model': metadata},
                                    (1928, 1208): Mock(), (1344, 760): jit},
                                   1928, 1208, device, device, device, chestnut=True)
      self.addCleanup(adapter._downscale_pool.shutdown)
      frames, _, _ = adapter.get_dummy_inputs()
      d = adapter.downscaler
      y = frames['img'][:d.src_y_size].reshape(d.src_yh, d.src_stride)
      r, c = np.mgrid[:1208, :1928]
      y[:1208, :1928] = ((np.sin(c / 37) + np.cos(r / 23)) * 60 + 128).astype(np.uint8)
      frames['img'][d.src_y_size:] = 128
      frames['big_img'][:] = frames['img']
      transform = np.array([[2.2, .05, 400], [.02, 2.2, 300], [0, 0, 1]], np.float32)
      adapter.copy_frames(frames)
      adapter.numpy_inputs['tfm'][:] = d.adjust_transform(transform)
      adapter.numpy_inputs['big_tfm'][:] = d.adjust_transform(transform)
      for _ in range(4):
        output = adapter.run().numpy()
      small_frame = Tensor(adapter.frame_slots['img'].copy(), device=device)
      scaled = Tensor(d.adjust_transform(transform), device=device)
      reference = warp(scaled, scaled, small_frame, Tensor(adapter.frame_slots['big_img'].copy(), device=device)).numpy()
      np.testing.assert_array_equal(output, reference)
      self.assertEqual(output.shape, (2, 6, 16, 32))
      full_warp = make_warp(NV12Frame(1928, 1208, *get_nv12_info(1928, 1208)), 64, 32)
      full_reference = full_warp(Tensor(transform, device=device), Tensor(transform, device=device),
                                 Tensor(frames['img'], device=device), Tensor(frames['big_img'], device=device)).numpy()
      self.assertLess(np.abs(output.astype(float) - full_reference).mean(), 3)


if __name__ == '__main__':
  unittest.main()
