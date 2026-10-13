#!/usr/bin/env python3
"""
Unit tests for Camera Accelerator (NV12 Downscaler and Transform Adjustment).
"""
import unittest
import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.sunnypilot.modeld_v2.camera_accelerator import (
  NV12Downscaler,
  get_downsampled_dims,
  is_accelerator_enabled
)


class TestCameraAccelerator(unittest.TestCase):
  def setUp(self):
    self.cam_w = 1928
    self.cam_h = 1208
    self.downscaler = NV12Downscaler(self.cam_w, self.cam_h)

  def test_dimensions(self):
    dst_w, dst_h = get_downsampled_dims(self.cam_w, self.cam_h)
    self.assertEqual(dst_w, 964)
    self.assertEqual(dst_h, 604)
    self.assertEqual(self.downscaler.dst_w, 964)
    self.assertEqual(self.downscaler.dst_h, 604)

    # Check stride and dimensions
    dst_stride, dst_yh, dst_uvh, _ = get_nv12_info(dst_w, dst_h)
    self.assertEqual(self.downscaler.dst_stride, dst_stride)
    self.assertEqual(self.downscaler.dst_yh, dst_yh)
    self.assertEqual(self.downscaler.dst_uvh, dst_uvh)

    # Destination frame size should be around 1/4 of source
    src_stride, src_yh, src_uvh, _ = get_nv12_info(self.cam_w, self.cam_h)
    src_size = src_stride * (src_yh + src_uvh)
    dst_size = dst_stride * (dst_yh + dst_uvh)
    ratio = dst_size / src_size
    self.assertAlmostEqual(ratio, 0.25, delta=0.01)

  def test_downsample_into(self):
    # Construct synthetic NV12 buffer
    src_stride = self.downscaler.src_stride
    src_yh = self.downscaler.src_yh
    src_uvh = self.downscaler.src_uvh
    src_total = src_stride * (src_yh + src_uvh)

    src_buf = np.zeros(src_total, dtype=np.uint8)

    # Fill Y plane with recognizable pattern (x + y * w) % 256
    y_pattern = np.fromfunction(
      lambda r, c: (c + r * 3) % 256,
      (self.cam_h, self.cam_w),
      dtype=int
    ).astype(np.uint8)
    src_buf[:self.downscaler.src_y_size].reshape(src_yh, src_stride)[:self.cam_h, :self.cam_w] = y_pattern

    # Fill UV plane with recognizable pattern
    uv_pattern = np.fromfunction(
      lambda r, c: ((c * 2) + r * 5) % 256,
      (self.cam_h // 2, self.cam_w),
      dtype=int
    ).astype(np.uint8)
    uv_offset = self.downscaler.src_y_size
    src_buf[uv_offset:uv_offset + src_stride * src_uvh].reshape(src_uvh, src_stride)[:self.cam_h // 2, :self.cam_w] = uv_pattern

    # Output buffer
    dst_buf = np.zeros(self.downscaler.dst_frame_size, dtype=np.uint8)
    self.downscaler.downsample_into(src_buf, dst_buf)

    # Validate downsampled Y matches [::2, ::2]
    dst_y = dst_buf[:self.downscaler.dst_y_size].reshape(self.downscaler.dst_yh, self.downscaler.dst_stride)
    expected_y = y_pattern[::2, ::2]
    np.testing.assert_array_equal(dst_y[:self.downscaler.dst_h, :self.downscaler.dst_w], expected_y)

    # Validate downsampled UV matches [::2, ::2, :]
    dst_uv = dst_buf[self.downscaler.dst_y_size:].reshape(self.downscaler.dst_uvh, self.downscaler.dst_stride)
    uv_pairs_expected = uv_pattern.reshape(self.cam_h // 2, self.cam_w // 2, 2)[::2, ::2, :]
    dst_uv_pairs = dst_uv[:self.downscaler.dst_h // 2, :self.downscaler.dst_w].reshape(
      self.downscaler.dst_h // 2, self.downscaler.dst_w // 2, 2
    )
    np.testing.assert_array_equal(dst_uv_pairs, uv_pairs_expected)

  def test_adjust_transform(self):
    # Homography matrix
    tfm = np.array([
      [1.2, 0.1, -10.0],
      [0.05, 1.1, -5.0],
      [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    adjusted = self.downscaler.adjust_transform(tfm)

    # Verify scaling on row 0 and row 1
    np.testing.assert_allclose(adjusted[0], tfm[0] * 0.5)
    np.testing.assert_allclose(adjusted[1], tfm[1] * 0.5)
    np.testing.assert_allclose(adjusted[2], tfm[2])

    # Mathematical test:
    # Any input coordinate p_dst in model space maps to p_src = tfm @ p_dst
    # With adjusted transform, p_src_down = adjusted @ p_dst
    # Verify p_src_down[:2] == 0.5 * p_src[:2]
    p_dst = np.array([256.0, 128.0, 1.0], dtype=np.float32)
    p_src = tfm @ p_dst
    p_src_down = adjusted @ p_dst

    np.testing.assert_allclose(p_src_down[0] / p_src_down[2], 0.5 * (p_src[0] / p_src[2]))
    np.testing.assert_allclose(p_src_down[1] / p_src_down[2], 0.5 * (p_src[1] / p_src[2]))

  def test_enabled_check(self):
    import os
    orig_env = os.environ.get('ACCELERATE_CAMERA')
    try:
      os.environ['ACCELERATE_CAMERA'] = '1'
      self.assertTrue(is_accelerator_enabled())
      os.environ['ACCELERATE_CAMERA'] = '0'
      self.assertFalse(is_accelerator_enabled())
      del os.environ['ACCELERATE_CAMERA']
      # Default is True without env or param
      self.assertTrue(is_accelerator_enabled())
    finally:
      if orig_env is not None:
        os.environ['ACCELERATE_CAMERA'] = orig_env
      elif 'ACCELERATE_CAMERA' in os.environ:
        del os.environ['ACCELERATE_CAMERA']


if __name__ == '__main__':
  unittest.main()
