"""
Guarded C3 pre-upload warp, adapted from carrot-wip's MIT-licensed local warp.
Copyright (c) 2018, Comma.ai, Inc. See the repository's LICENSE.
"""

from collections.abc import Callable
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

CAMERA_INPUT_ABI = 'warped_yuv_v2'
CACHE_VERSION = 2
SAMPLING_BOUNDARY_TOLERANCE = 0.00025


def derive_frame_skip(vision_input_shapes: dict, policy_input_shapes: dict) -> int:
  features_buffer = policy_input_shapes.get('features_buffer')
  return 1 if not features_buffer or features_buffer[1] >= 99 else 4


def validate_camera_input_abi(metadata: dict, jits: dict, chestnut: bool, cam_w: int, cam_h: int) -> bool:
  abi = metadata.get('camera_input_abi')
  if abi is None:
    return False
  if abi != CAMERA_INPUT_ABI:
    raise RuntimeError(f"Unsupported camera input ABI: {abi}; rebuild the experimental artifact")
  if not chestnut or 'model' not in metadata or 'run_model' in jits or not {'run_policy', 'run_policy_amd'} <= jits.keys():
    raise RuntimeError("QCOM host-warp bundles require Chestnut supercombo policy artifacts")
  if metadata.get('warp_dev') != 'AMD' or jits.get('input_devices', {}).get('model', '').split(':')[0] != 'AMD':
    raise RuntimeError("QCOM host-warp bundle must retain an AMD warp and AMD inference")
  if (cam_w, cam_h) not in jits:
    raise RuntimeError(f"QCOM host-warp bundle has no warp for camera size {cam_w}x{cam_h}")
  shapes = metadata['model']['input_shapes']
  if type(metadata.get('frame_skip')) is not int or metadata['frame_skip'] != derive_frame_skip({}, shapes):
    raise RuntimeError("QCOM host-warp bundle has an invalid frame_skip")
  image_shape = tuple(metadata.get('warped_shape', ()))
  road, wide = [shapes.get(key, ()) for key in ('img', 'big_img')]
  if len(road) != 4 or road != wide or road[0] != 1 or road[1] % 6 or image_shape != (2, 6, *road[2:]):
    raise RuntimeError("QCOM host-warp bundle has incompatible camera input shapes")
  if not isinstance(metadata.get('warp_source_hash'), str) or len(metadata['warp_source_hash']) != 64:
    raise RuntimeError("QCOM host-warp bundle is missing its pinned warp source hash")
  return True


def use_local_warp(device_type: str | None) -> bool:
  return device_type in ('tici', 'tizi')


def warp_source_hash() -> str:
  from openpilot.selfdrive.modeld import compile_modeld as stock
  import tinygrad

  source = ''.join(inspect.getsource(getattr(stock, name)) for name in
                   ('make_warp', 'make_frame_prepare', 'frames_to_tensor', 'warp_perspective_tinygrad'))
  digest = hashlib.sha256(source.encode())
  runtime_root = Path(inspect.getfile(tinygrad)).parent
  for path in sorted(runtime_root.rglob('*.py')):
    digest.update(path.relative_to(runtime_root).as_posix().encode())
    digest.update(b'\0')
    digest.update(path.read_bytes())
    digest.update(b'\0')
  return digest.hexdigest()


class CompactInput:
  def __init__(self, controls: dict[str, np.ndarray], image_shape: tuple[int, ...]):
    self.control_bytes = sum(value.nbytes for value in controls.values())
    self.image_offset = (self.control_bytes + 511) // 512 * 512
    self.data = np.zeros(self.image_offset + math.prod(image_shape), dtype=np.uint8)
    self.images = self.data[self.image_offset:].reshape(image_shape)
    self.controls: dict[str, np.ndarray] = {}
    offset = 0
    for name, value in controls.items():
      if value.dtype != np.float32:
        raise ValueError(f"host-warp control {name} must be float32")
      view = self.data[offset:offset + value.nbytes].view(np.float32).reshape(value.shape)
      np.copyto(view, value)
      self.controls[name] = view
      offset += value.nbytes


def make_compact_policy(run_policy: Callable, control_bytes: int, image_shape: tuple[int, ...]) -> Callable:
  image_offset = (control_bytes + 511) // 512 * 512

  def run(img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    # This input is already on AMD; one explicit upload precedes the JIT.
    controls = packed_npy_inputs[:control_bytes].bitcast('float32')
    images = packed_npy_inputs[image_offset:].reshape(image_shape)
    return run_policy(warped=images, img_q=img_q, big_img_q=big_img_q, feat_q=feat_q,
                      desire_q=desire_q, packed_npy_inputs=controls)

  return run


def warp_difference(actual: np.ndarray, expected: np.ndarray) -> dict | None:
  detail = {'actual_shape': actual.shape, 'expected_shape': expected.shape,
            'actual_dtype': str(actual.dtype), 'expected_dtype': str(expected.dtype)}
  if actual.shape != expected.shape or actual.dtype != expected.dtype:
    return detail
  indices = np.argwhere(actual != expected)
  if not len(indices):
    return None
  detail.update(mismatched_pixels=len(indices), total_pixels=actual.size,
                max_abs_error=int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()),
                samples=[{'index': index.tolist(), 'qcom': int(actual[tuple(index)]),
                          'amd': int(expected[tuple(index)])} for index in indices[:8]])
  return detail


def sampling_boundary_only(actual: np.ndarray, expected: np.ndarray, frames: np.ndarray,
                           matrices: np.ndarray, camera_size: tuple[int, int],
                           frame_info: tuple, image_shape: tuple[int, ...]) -> bool:
  if actual.shape != image_shape or expected.shape != image_shape or actual.dtype != np.uint8 or expected.dtype != np.uint8:
    return False
  width, height = camera_size
  stride, y_height = frame_info[:2]
  for camera, channel, row, col in np.argwhere(actual != expected):
    uv = channel >= 4
    x, y = (col, row) if uv else (2 * col + channel // 2, 2 * row + channel % 2)
    matrix = matrices[camera].astype(np.float64)
    if uv:
      matrix *= np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]])
    projected = matrix @ [x, y, 1]
    if not np.isfinite(projected).all() or abs(projected[2]) < .5:
      return False
    source = projected[:2] / projected[2]
    candidates = []
    at_boundary = False
    for coordinate, limit in zip(source, (width // 2, height // 2) if uv else (width, height), strict=True):
      low = np.floor(coordinate)
      near = abs(coordinate - low - .5) <= SAMPLING_BOUNDARY_TOLERANCE
      at_boundary |= near
      choices = (int(low), int(low + 1)) if near else (int(np.rint(coordinate)),)
      candidates.append({min(max(value, 0), limit - 1) for value in choices})
    if not at_boundary:
      return False
    source_values = set()
    for sy in candidates[1]:
      for sx in candidates[0]:
        offset = stride * y_height + sy * stride + 2 * sx + channel - 4 if uv else sy * stride + sx
        source_values.add(int(frames[camera, offset]))
    if int(actual[camera, channel, row, col]) not in source_values or int(expected[camera, channel, row, col]) not in source_values:
      return False
  return True


class WarpBackend:
  def __init__(self, raw: np.ndarray, frame_size: int, device: str, warp):
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import UOp

    self.host = Tensor(raw, device='NPY')._buffer()
    self.buffer = Tensor(np.zeros_like(raw), device=device)._buffer()
    self.warp = warp

    def view(shape, dtype, offset):
      buffer = self.buffer.view(math.prod(shape), dtype, offset).ensure_allocated()
      return Tensor(UOp.from_buffer(buffer)).reshape(shape)

    self.inputs = {'tfm': view((3, 3), dtypes.float32, 0), 'big_tfm': view((3, 3), dtypes.float32, 36),
                   'frame': view((frame_size,), dtypes.uint8, 72),
                   'big_frame': view((frame_size,), dtypes.uint8, 72 + frame_size)}

  def prepare(self):
    # Mutable host mappings alone do not guarantee QCOM cache coherence.
    self.buffer.copy_from(self.host)
    return self.warp(**self.inputs)


def load_local_warp(raw: np.ndarray, frame_size: int, camera_size: tuple[int, int], frame_info: tuple,
                    image_shape: tuple[int, ...], source_hash: str, cache_dir: Path) -> WarpBackend:
  from tinygrad import Context
  from tinygrad.engine.jit import TinyJit
  from openpilot.selfdrive.modeld import compile_modeld as stock
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob

  if warp_source_hash() != source_hash:
    raise RuntimeError("QCOM warp source/runtime differs from the artifact; rebuild before enabling local warp")
  width, height = camera_size
  cache_dir.mkdir(parents=True, exist_ok=True)
  layout = '-'.join(str(value) for value in frame_info)
  cache = cache_dir / f'warp-qcom-v{CACHE_VERSION}-{source_hash}-{width}x{height}-{layout}-{image_shape[-1]}x{image_shape[-2]}.pkl'
  with Context(DEV='QCOM'):
    if cache.is_file():
      with cache.open('rb') as file:
        backend = WarpBackend(raw, frame_size, 'QCOM', load_oob(file))
    else:
      nv12 = stock.NV12Frame(width, height, *frame_info)
      warp = TinyJit(stock.make_warp(nv12, image_shape[-1] * 2, image_shape[-2] * 2), prune=True)
      backend = WarpBackend(raw, frame_size, 'QCOM', warp)
      try:
        raw[:72].view(np.float32).reshape(2, 3, 3)[:] = np.eye(3)
        for _ in range(3):
          backend.prepare().numpy()
        temporary = None
        try:
          with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as file:
            temporary = Path(file.name)
            dump_oob(warp, file)
          os.replace(temporary, cache)
        finally:
          if temporary is not None:
            temporary.unlink(missing_ok=True)
      finally:
        raw[:] = 0
  return backend


class HostWarpRuntime:
  def __init__(self, compact: CompactInput, frame_size: int, camera_size: tuple[int, int], frame_info: tuple,
               amd_warp, device: str, device_type: str | None, source_hash: str, cache_dir: Path, logger):
    from tinygrad import Tensor

    self.compact = compact
    self.raw = np.zeros(72 + 2 * frame_size, dtype=np.uint8)
    self.matrices = self.raw[:72].view(np.float32).reshape(2, 3, 3)
    self.frames = self.raw[72:].reshape(2, frame_size)
    self.camera_size, self.frame_info = camera_size, frame_info
    self.amd = WarpBackend(self.raw, frame_size, device, amd_warp)
    self.local: WarpBackend | None = None
    self.logger = logger
    self.compact_host = Tensor(compact.data, device='NPY')._buffer()
    self.compact_device = Tensor(np.zeros_like(compact.data), device=device).realize()
    self.last_timings: dict[str, float | int | str] = {}
    if use_local_warp(device_type):
      try:
        self.local = load_local_warp(self.raw, frame_size, camera_size, frame_info,
                                    compact.images.shape, source_hash, cache_dir)
        self.validate_warp()
      except Exception:
        logger.exception("QCOM pre-upload warp initialization/validation failed; retaining same-model AMD warp")
        self.local = None
      finally:
        self.clear_inputs()
    logger.info("Chestnut pre-upload warp: backend=%s compact_bytes=%d raw_bytes=%d",
                self.backend, compact.data.nbytes, self.raw.nbytes)

  @property
  def backend(self) -> str:
    return 'qcom' if self.local is not None else 'amd'

  def clear_inputs(self) -> None:
    self.raw[:] = 0
    self.compact.data[:] = 0

  def validate_warp(self) -> None:
    assert self.local is not None
    failures = []
    try:
      self.frames[:] = np.random.default_rng(0).integers(0, 256, self.frames.shape, dtype=np.uint8)
      for name, matrix in (
        ('identity', np.eye(3)),
        ('projective', [[2.3, .01, 20.2], [-.02, 2.1, 40.3], [.0001, -.0002, 1]]),
        ('border', [[1, 0, -200], [0, 1, -100], [0, 0, 1]]),
      ):
        self.matrices[:] = matrix
        expected = self.amd.prepare().numpy()
        if expected.shape != self.compact.images.shape or expected.dtype != np.uint8:
          raise RuntimeError("artifact AMD warp has an unexpected shape/dtype")
        actual = self.local.prepare().numpy()
        repeated = self.local.prepare().numpy()
        difference = warp_difference(actual, expected)
        if difference is not None or not np.array_equal(actual, repeated):
          detail: dict = difference if difference is not None else {'mismatched_pixels': 0}
          detail.update(probe=name, repeat_matches_first=bool(np.array_equal(actual, repeated)))
          explained = detail['repeat_matches_first'] and sampling_boundary_only(
            actual, expected, self.frames, self.matrices, self.camera_size, self.frame_info, self.compact.images.shape)
          detail['sampling_boundary_only'] = explained
          failures.append(detail)
      if failures:
        raise RuntimeError("QCOM pre-upload warp is not pixel-identical to artifact AMD warp: " + json.dumps(failures))
    finally:
      self.clear_inputs()

  def run(self, run_policy, run_policy_amd, policy_inputs: dict):
    started = time.perf_counter()
    images = None
    if self.local is not None:
      try:
        images = self.local.prepare().numpy()
        if images.shape != self.compact.images.shape or images.dtype != np.uint8:
          raise RuntimeError("QCOM warp changed its output shape/dtype")
        np.copyto(self.compact.images, images)
      except Exception:
        self.logger.exception("QCOM frame preparation failed; retaining same-model AMD warp")
        self.local = None
    prepared = time.perf_counter()
    if images is not None and self.local is not None:
      self.compact_device._buffer().copy_from(self.compact_host)
      uploaded = time.perf_counter()
      result = run_policy(**{**policy_inputs, 'packed_npy_inputs': self.compact_device})
      upload_bytes = self.compact.data.nbytes
    else:
      warped = self.amd.prepare()
      uploaded = time.perf_counter()
      result = run_policy_amd(**policy_inputs, warped=warped)
      upload_bytes = self.raw.nbytes + self.compact.control_bytes
    dispatched = time.perf_counter()
    self.last_timings = {
      'warp_backend': self.backend, 'usb_input_bytes': upload_bytes,
      'local_prepare_ms': (prepared - started) * 1000,
      'input_upload_ms': (uploaded - prepared) * 1000,
      'model_call_ms': (dispatched - uploaded) * 1000,
    }
    return result
