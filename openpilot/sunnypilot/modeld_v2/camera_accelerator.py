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
  """Return exact 1.5x downsampled dimensions (of a cropped source) aligned for NV12."""
  dst_w = (cam_w * 2 // 3 // 128) * 128
  dst_h = (cam_h * 2 // 3 // 32) * 32
  if dst_w < 4 or dst_h < 4:
    raise ValueError(f"Camera dimensions are too small to downsample: {cam_w}x{cam_h}")
  return dst_w, dst_h


class NV12Downscaler:
  """
  Exact 1.5x nearest-neighbour downscaler for NV12 camera frames.
  Keeps 2 of every 3 pixels/rows, so a 1928x1208 frame (cropped to 1920x1200)
  becomes 1280x800 with a single strided copy per plane (no index gathers).
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
    # The warp only reads the Y and UV planes, so skip get_nv12_info's trailing padding to cut USB upload size.
    self.dst_frame_size = self.dst_copy_size
    # Nearest-neighbour source index for each output pixel: floor(1.5 * i) -> 0, 1, 3, 4, 6, 7, ...
    self.dst_x_indices = np.arange(self.dst_w, dtype=np.intp) * 3 // 2
    self.dst_y_indices = np.arange(self.dst_h, dtype=np.intp) * 3 // 2
    self.dst_uv_x_indices = np.arange(self.dst_w // 2, dtype=np.intp) * 3 // 2
    self.dst_uv_y_indices = np.arange(self.dst_h // 2, dtype=np.intp) * 3 // 2

    self.scale_matrix = np.diag([2.0 / 3.0, 2.0 / 3.0, 1.0]).astype(np.float32)

  def downsample_into(self, src_buf: np.ndarray, dst_buf: np.ndarray) -> None:
    """
    Downsample src_buf (NV12) into preallocated dst_buf (NV12).
    Both buffers can be flat 1D uint8 arrays or memoryviews.
    """
    src_flat = np.frombuffer(src_buf, dtype=np.uint8)
    dst_flat = np.frombuffer(dst_buf, dtype=np.uint8)
    dw, dh, ss, ds = self.dst_w, self.dst_h, self.src_stride, self.dst_stride

    # Keep rows 0,1 of every 3 and pixels 0,1 of every 3. Each kept pixel pair is read as one
    # (unaligned) uint16 at a 3-byte stride, which is ~5x faster on the device than per-byte gathers.
    np.copyto(np.ndarray((dh // 2, 2, dw // 2), np.uint16, buffer=dst_flat, offset=0, strides=(2 * ds, ds, 2)),
              np.ndarray((dh // 2, 2, dw // 2), np.uint16, buffer=src_flat, offset=0, strides=(3 * ss, ss, 3)))

    # UV plane: same pattern on interleaved (U, V) pairs, so two kept pairs form one uint32 at a 6-byte stride.
    np.copyto(np.ndarray((dh // 4, 2, dw // 4), np.uint32, buffer=dst_flat, offset=self.dst_y_size, strides=(2 * ds, ds, 4)),
              np.ndarray((dh // 4, 2, dw // 4), np.uint32, buffer=src_flat, offset=self.src_y_size, strides=(3 * ss, ss, 6)))

  def adjust_transform(self, tfm: np.ndarray) -> np.ndarray:
    """Adjust M_inv (3x3) for the actual downsampled width and height."""
    return self.scale_matrix @ tfm
