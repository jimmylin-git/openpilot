"""
Camera Accelerator for openpilot / sunnypilot C3XL + USB eGPU.

Accelerates frame transmission over USB 3.0 by performing fast 2x downsampling
on the QCOM host CPU before USB transfer (reducing data from ~7.5MB to ~1.86MB),
and adjusting the homography matrix M_inv so that AMD Warp operates directly on
downsampled frames.
"""
import os
from typing import Tuple
import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def is_accelerator_enabled() -> bool:
  """
  Check if camera accelerator is enabled.
  Enabled by default (1) for C3XL / USB eGPU downscaled frame acceleration,
  unless explicitly disabled via ACCELERATE_CAMERA=0 or param AccelerateCamera=0.
  """
  if os.getenv('ACCELERATE_CAMERA') in ('0', 'false', 'False'):
    return False
  if os.getenv('ACCELERATE_CAMERA') in ('1', 'true', 'True'):
    return True
  try:
    from openpilot.common.params import Params
    param_val = Params().get("AccelerateCamera")
    if param_val is not None:
      return param_val.strip() in (b'1', b'true', b'True')
  except Exception:
    pass
  return True


def get_downsampled_dims(cam_w: int, cam_h: int) -> Tuple[int, int]:
  """Return downsampled width and height (must be even)."""
  return (cam_w // 2) & ~1, (cam_h // 2) & ~1


class NV12Downscaler:
  """
  Fast 2x downscaler for NV12 camera frames.
  Takes full-resolution NV12 buffer (e.g. 1928x1208 with stride 2048)
  and produces a downsampled NV12 buffer (e.g. 964x604 with stride 1024).
  """
  def __init__(self, src_w: int, src_h: int):
    self.src_w = src_w
    self.src_h = src_h
    self.src_stride, self.src_yh, self.src_uvh, self.src_yuv_size = get_nv12_info(src_w, src_h)
    self.src_y_size = self.src_stride * self.src_yh

    self.dst_w, self.dst_h = get_downsampled_dims(src_w, src_h)
    self.dst_stride, self.dst_yh, self.dst_uvh, self.dst_yuv_size = get_nv12_info(self.dst_w, self.dst_h)
    self.dst_y_size = self.dst_stride * self.dst_yh
    self.dst_frame_size = self.dst_stride * (self.dst_yh + self.dst_uvh)

    # Scaling matrix for inverse homography: M_inv_down = S @ M_inv
    # Since x_down = 0.5 * x_orig, y_down = 0.5 * y_orig
    self.scale_matrix = np.diag([0.5, 0.5, 1.0]).astype(np.float32)

  def downsample_into(self, src_buf: np.ndarray, dst_buf: np.ndarray) -> None:
    """
    Downsample src_buf (NV12) into preallocated dst_buf (NV12).
    Both buffers can be flat 1D uint8 arrays or memoryviews.
    """
    src_flat = np.frombuffer(src_buf, dtype=np.uint8)
    dst_flat = np.frombuffer(dst_buf, dtype=np.uint8)

    # 1. Y plane
    src_y = src_flat[:self.src_y_size].reshape(self.src_yh, self.src_stride)[:self.src_h, :self.src_w]
    dst_y = dst_flat[:self.dst_y_size].reshape(self.dst_yh, self.dst_stride)
    dst_y[:self.dst_h, :self.dst_w] = src_y[::2, ::2]

    # 2. UV plane (interleaved U, V pairs)
    src_uv = src_flat[self.src_y_size:self.src_y_size + self.src_stride * self.src_uvh].reshape(
      self.src_uvh, self.src_stride
    )[:self.src_h // 2, :self.src_w]
    dst_uv = dst_flat[self.dst_y_size:self.dst_y_size + self.dst_stride * self.dst_uvh].reshape(
      self.dst_uvh, self.dst_stride
    )

    src_uv_pairs = src_uv.reshape(self.src_h // 2, self.src_w // 2, 2)
    dst_uv_pairs = dst_uv[:self.dst_h // 2, :self.dst_w].reshape(self.dst_h // 2, self.dst_w // 2, 2)
    dst_uv_pairs[:, :, :] = src_uv_pairs[::2, ::2, :]

  def adjust_transform(self, tfm: np.ndarray) -> np.ndarray:
    """Adjust M_inv (3x3) so AMD Warp samples correctly from downscaled frame."""
    return self.scale_matrix @ tfm
