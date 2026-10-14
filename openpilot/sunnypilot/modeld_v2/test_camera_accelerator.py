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
  is_accelerator_enabled,
  select_legacy_camera_dims,
)


class TestCameraAccelerator(unittest.TestCase):
  def setUp(self):
    self.cam_w = 1928
    self.cam_h = 1208
    self.downscaler = NV12Downscaler(self.cam_w, self.cam_h)

  def test_dimensions(self):
    dst_w, dst_h = get_downsampled_dims(self.cam_w, self.cam_h)
    self.assertEqual(dst_w, 1280)
    self.assertEqual(dst_h, 800)
    self.assertEqual(self.downscaler.dst_w, 1280)
    self.assertEqual(self.downscaler.dst_h, 800)

    # Check stride and dimensions
    dst_stride, dst_yh, dst_uvh, _ = get_nv12_info(dst_w, dst_h)
    self.assertEqual(self.downscaler.dst_stride, dst_stride)
    self.assertEqual(self.downscaler.dst_yh, dst_yh)
    self.assertEqual(self.downscaler.dst_uvh, dst_uvh)
    self.assertEqual(dst_stride % 128, 0)
    self.assertEqual(dst_yh % 32, 0)

    # The packed Y and UV planes should use approximately 4/9 of source pixels.
    src_size = self.downscaler.src_w * self.downscaler.src_h * 3 // 2
    dst_size = dst_w * dst_h * 3 // 2
    ratio = dst_size / src_size
    self.assertAlmostEqual(ratio, 4 / 9, delta=0.01)
    self.assertEqual(self.downscaler.dst_copy_size, dst_stride * (dst_yh + dst_uvh))
    self.assertGreaterEqual(self.downscaler.dst_frame_size, self.downscaler.dst_copy_size)

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

    # Validate Y uses nearest-neighbor coordinate mapping at the actual scale.
    dst_y = dst_buf[:self.downscaler.dst_y_size].reshape(self.downscaler.dst_yh, self.downscaler.dst_stride)
    expected_y = y_pattern[np.ix_(self.downscaler.dst_y_indices, self.downscaler.dst_x_indices)]
    np.testing.assert_array_equal(dst_y[:self.downscaler.dst_h, :self.downscaler.dst_w], expected_y)

    # Validate UV preserves interleaved U/V pairs.
    dst_uv = dst_buf[
      self.downscaler.dst_y_size:self.downscaler.dst_copy_size
    ].reshape(self.downscaler.dst_uvh, self.downscaler.dst_stride)
    uv_pairs_expected = uv_pattern.reshape(self.cam_h // 2, self.cam_w // 2, 2)[
      self.downscaler.dst_uv_y_indices[:, None], self.downscaler.dst_uv_x_indices, :
    ]
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

    scale_x = scale_y = 2.0 / 3.0
    np.testing.assert_allclose(adjusted[0], tfm[0] * scale_x)
    np.testing.assert_allclose(adjusted[1], tfm[1] * scale_y)
    np.testing.assert_allclose(adjusted[2], tfm[2])

    # Mathematical test:
    # Any input coordinate p_dst in model space maps to p_src = tfm @ p_dst
    # With adjusted transform, p_src_down = adjusted @ p_dst
    # Verify the adjusted transform maps into the downsampled frame coordinates.
    p_dst = np.array([256.0, 128.0, 1.0], dtype=np.float32)
    p_src = tfm @ p_dst
    p_src_down = adjusted @ p_dst

    np.testing.assert_allclose(p_src_down[0] / p_src_down[2], scale_x * (p_src[0] / p_src[2]))
    np.testing.assert_allclose(p_src_down[1] / p_src_down[2], scale_y * (p_src[1] / p_src[2]))

  def test_legacy_downsample_with_padding(self):
    d = NV12Downscaler(self.cam_w, self.cam_h, (1344, 760))
    src = np.random.default_rng(0).integers(0, 256, d.src_yuv_size, dtype=np.uint8)
    dst = np.zeros(d.dst_copy_size, dtype=np.uint8)
    d.downsample_into(memoryview(src), dst)
    y = src[:d.src_y_size].reshape(d.src_yh, d.src_stride)
    uv = src[d.src_y_size:d.src_y_size + d.src_stride * d.src_uvh].reshape(d.src_uvh, d.src_stride // 2, 2)
    expected_y = y[np.ix_(np.arange(760) * 1208 // 760, np.arange(1344) * 1928 // 1344)]
    expected_uv = uv[(np.arange(380) * 604 // 380)[:, None], np.arange(672) * 964 // 672]
    out_y = dst[:d.dst_y_size].reshape(d.dst_yh, d.dst_stride)
    out_uv = dst[d.dst_y_size:].reshape(d.dst_uvh, d.dst_stride // 2, 2)
    np.testing.assert_array_equal(out_y[:760, :1344], expected_y)
    np.testing.assert_array_equal(out_uv[:380, :672], expected_uv)
    self.assertFalse(out_y[760:].any())
    self.assertFalse(out_y[:, 1344:].any())
    self.assertFalse(out_uv[380:].any())
    np.testing.assert_allclose(d.adjust_transform(np.eye(3)), np.diag([1344 / 1928, 760 / 1208, 1]))
    self.assertEqual(d.dst_copy_size, 1622016)

  def test_select_existing_camera_jit(self):
    self.assertEqual(select_legacy_camera_dims({(1928, 1208): None, (1344, 760): None}, 1928, 1208), (1344, 760))
    self.assertIsNone(select_legacy_camera_dims({(1928, 1208): None, (2048, 1280): None}, 1928, 1208))
    self.assertIsNone(select_legacy_camera_dims({'run_policy': None, (1343, 760): None}, 1928, 1208))

  def test_invalid_destination(self):
    for dims in ((0, 760), (1343, 760), (2048, 760), (1344, 1209)):
      with self.subTest(dims=dims), self.assertRaises(ValueError):
        NV12Downscaler(self.cam_w, self.cam_h, dims)

  def test_short_buffers_raise(self):
    d = NV12Downscaler(self.cam_w, self.cam_h, (1344, 760))
    src_size = d.src_stride * (d.src_yh + d.src_uvh)
    with self.assertRaises(ValueError):
      d.downsample_into(np.zeros(src_size - 1, np.uint8), np.zeros(d.dst_copy_size, np.uint8))
    with self.assertRaises(ValueError):
      d.downsample_into(np.zeros(src_size, np.uint8), np.zeros(d.dst_copy_size - 1, np.uint8))

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
