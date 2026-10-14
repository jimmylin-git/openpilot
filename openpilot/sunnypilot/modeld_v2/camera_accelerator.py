"""
Camera Accelerator for openpilot / sunnypilot C3XL + USB eGPU.

Reduces frame size before the Warp/model device transfer and adjusts the
homography matrix to match the downsampled frame dimensions.
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
  """Return approximately 1.5x downsampled dimensions aligned for NV12."""
  dst_w = (cam_w * 2 // 3 // 128) * 128
  dst_h = (cam_h * 2 // 3 // 32) * 32
  if dst_w < 2 or dst_h < 2:
    raise ValueError(f"Camera dimensions are too small to downsample: {cam_w}x{cam_h}")
  return dst_w, dst_h


class NV12Downscaler:
  """
  Approximately 1.5x downscaler for NV12 camera frames.
  For a 1928x1208 camera frame, produces a 1280x800 image with 128-byte
  stride alignment and a 32-line luma alignment.
  """
  def __init__(self, src_w: int, src_h: int):
    self.src_w = src_w
    self.src_h = src_h
    self.src_stride, self.src_yh, self.src_uvh, self.src_yuv_size = get_nv12_info(src_w, src_h)
    self.src_y_size = self.src_stride * self.src_yh

    self.dst_w, self.dst_h = get_downsampled_dims(src_w, src_h)
    self.dst_stride, self.dst_yh, self.dst_uvh, self.dst_yuv_size = get_nv12_info(self.dst_w, self.dst_h)
    self.dst_y_size = self.dst_stride * self.dst_yh
    self.dst_copy_size = self.dst_stride * (self.dst_yh + self.dst_uvh)
    self.dst_frame_size = self.dst_yuv_size
    self.dst_x_indices = np.arange(self.dst_w, dtype=np.intp) * src_w // self.dst_w
    self.dst_y_indices = np.arange(self.dst_h, dtype=np.intp) * src_h // self.dst_h
    self.dst_uv_x_indices = np.arange(self.dst_w // 2, dtype=np.intp) * (src_w // 2) // (self.dst_w // 2)
    self.dst_uv_y_indices = np.arange(self.dst_h // 2, dtype=np.intp) * (src_h // 2) // (self.dst_h // 2)
    self._y_scratch = np.empty((self.dst_h, self.src_w), dtype=np.uint8)
    self._uv_scratch = np.empty((self.dst_h // 2, self.src_w // 2, 2), dtype=np.uint8)

    self.scale_matrix = np.diag([self.dst_w / src_w, self.dst_h / src_h, 1.0]).astype(np.float32)

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
    np.take(src_y, self.dst_y_indices, axis=0, out=self._y_scratch)
    np.take(self._y_scratch, self.dst_x_indices, axis=1, out=dst_y[:self.dst_h, :self.dst_w])

    # 2. UV plane (interleaved U, V pairs)
    src_uv = src_flat[self.src_y_size:self.src_y_size + self.src_stride * self.src_uvh].reshape(
      self.src_uvh, self.src_stride
    )[:self.src_h // 2, :self.src_w]
    dst_uv = dst_flat[self.dst_y_size:self.dst_y_size + self.dst_stride * self.dst_uvh].reshape(
      self.dst_uvh, self.dst_stride
    )

    src_uv_pairs = src_uv.reshape(self.src_h // 2, self.src_w // 2, 2)
    dst_uv_pairs = dst_uv[:self.dst_h // 2, :self.dst_w].reshape(self.dst_h // 2, self.dst_w // 2, 2)
    np.take(src_uv_pairs, self.dst_uv_y_indices, axis=0, out=self._uv_scratch)
    np.take(self._uv_scratch, self.dst_uv_x_indices, axis=1, out=dst_uv_pairs)

  def adjust_transform(self, tfm: np.ndarray) -> np.ndarray:
    """Adjust M_inv (3x3) for the actual downsampled width and height."""
    return self.scale_matrix @ tfm
