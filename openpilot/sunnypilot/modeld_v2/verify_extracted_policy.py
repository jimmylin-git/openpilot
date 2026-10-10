"""Explicit parked-only CTMV2 experiment. Never imported by modeld."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from openpilot.sunnypilot.modeld_v2.extract_policy import IMAGE_SHAPE, INPUT_NAMES, extract_ctmv2
from openpilot.sunnypilot.modeld_v2.host_warp import CompactInput, load_local_warp, warp_difference, warp_source_hash

SOURCE_SHA256 = "2ad8f827410cdc9ac4b06b1c1beea9ee05aa0e0403cba54f33e794912e236cf1"
STATE_NAMES = INPUT_NAMES[:4]


def require_parked() -> None:
  if os.name != 'posix' or not Path('/proc').is_dir():
    raise RuntimeError("This experiment requires the parked vehicle")
  from openpilot.common.params import Params
  from openpilot.selfdrive.modeld.helpers import chestnut_present

  if not Params().get_bool('IsOffroad'):
    raise RuntimeError("Refusing to test while onroad")
  for path in Path('/proc').glob('[0-9]*/cmdline'):
    if int(path.parent.name) == os.getpid():
      continue
    try:
      args = path.read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
      continue
    if any(arg.rsplit(b'/', 1)[-1] in (b'modeld', b'modeld_chestnut', b'modeld.py') or
           arg in (b'openpilot.sunnypilot.modeld_v2.modeld', b'openpilot.selfdrive.modeld.modeld') for arg in args):
      raise RuntimeError(f"Refusing to share GPU with modeld (PID {path.parent.name})")
  if not chestnut_present():
    raise RuntimeError("Chestnut USB device is absent; no power/settings will be changed")


def check_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
  if actual.shape != expected.shape or actual.dtype != expected.dtype:
    raise RuntimeError(f"{label}: shape/dtype mismatch")
  if not np.isfinite(actual).all() or not np.isfinite(expected).all():
    raise RuntimeError(f"{label}: non-finite values")
  if not np.array_equal(actual, expected):
    raise RuntimeError(f"{label}: {np.count_nonzero(actual != expected)} unequal values")


def timing_summary(values: list[float]) -> dict:
  if not values or not np.isfinite(values).all() or any(value < 0 for value in values):
    raise ValueError("Timing samples must be nonempty, finite and nonnegative")
  return {'samples': len(values), 'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95)),
          'p99_ms': float(np.percentile(values, 99)), 'over_50ms': sum(value > 50 for value in values)}


def verify(model_path: Path, camera_size: tuple[int, int], runs: int, warmup: int, roundtrip: bool) -> dict:
  if runs < 1 or warmup < 0:
    raise ValueError("runs must be positive and warmup nonnegative")
  if roundtrip and warmup + runs < 2:
    raise ValueError("Round-trip validation requires at least two total iterations")
  require_parked()
  from tinygrad import Tensor
  from openpilot.common.file_chunker import open_file_chunked
  from openpilot.common.hardware import HARDWARE
  from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
  from openpilot.sunnypilot.modeld_v2.compile_modeld import derive_frame_skip, stock
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  if HARDWARE.get_device_type() not in ('tici', 'tizi'):
    raise RuntimeError("Only the existing C3 QCOM warp backend is supported")
  with open_file_chunked(str(model_path)) as source:
    digest = hashlib.sha256()
    while block := source.read(1024 * 1024):
      digest.update(block)
  if digest.hexdigest() != SOURCE_SHA256:
    raise ValueError("Source SHA differs from the inspected CTMV2 v25 artifact")
  require_parked()
  with open_file_chunked(str(model_path)) as source:
    artifact = load_oob(source)
  frame_info = get_nv12_info(*camera_size)
  frame_bytes = stock.nv12_copy_size(*frame_info[:3])
  candidate = extract_ctmv2(artifact, camera_size, frame_bytes, resident_state=True)
  shapes = artifact['metadata']['model']['input_shapes']
  frame_skip = derive_frame_skip({}, shapes)
  original = artifact['run_model'][camera_size]
  reference_queues, reference_npy, reference_frames = stock.make_input_queues(shapes, frame_skip, 'AMD', frame_bytes)
  queues, _, _ = stock.make_input_queues(shapes, frame_skip, 'AMD', frame_bytes)
  compact = CompactInput({name: reference_npy[name] for name in
                          ('tfm', 'big_tfm', 'desire', 'traffic_convention', 'action_t')}, IMAGE_SHAPE)
  if (compact.control_bytes, compact.image_offset, compact.data.nbytes) != (
      candidate.control_bytes, candidate.image_offset, candidate.input_bytes):
    raise RuntimeError("Compact input layout differs from captured policy")
  queues['packed_npy_inputs'] = Tensor(compact.data, device='NPY').realize()
  if candidate.state is None:
    raise RuntimeError("Missing AMD-resident previous feature")
  candidate.state.assign(0).realize()
  raw = np.zeros(72 + 2 * frame_bytes, dtype=np.uint8)
  matrices = raw[:72].view(np.float32).reshape(2, 3, 3)
  frames = raw[72:].reshape(2, frame_bytes)
  source_hash = warp_source_hash()
  timings: dict[str, list[float]] = {'original': [], 'qcom_compact_resident': []}
  rng = np.random.default_rng(42)
  roundtrip_checked = False
  # All cache and round-trip files are isolated from the installed model/cache.
  with tempfile.TemporaryDirectory(prefix='ctmv2-policy-') as temporary:
    directory = Path(temporary)
    local_warp = load_local_warp(raw, frame_bytes, camera_size, frame_info, IMAGE_SHAPE, source_hash, directory)
    for step in range(warmup + runs):
      require_parked()
      for name in ('desire', 'traffic_convention', 'action_t'):
        reference_npy[name][:] = rng.standard_normal(reference_npy[name].shape).astype(np.float32)
      transform = (
        np.eye(3, dtype=np.float32),
        np.array([[2.3, .01, 20.2], [-.02, 2.1, 40.3], [.0001, -.0002, 1]], dtype=np.float32),
        np.array([[1, 0, -200], [0, 1, -100], [0, 0, 1]], dtype=np.float32),
      )[step % 3]
      for name in ('tfm', 'big_tfm'):
        reference_npy[name][:] = transform
      for i, name in enumerate(('img', 'big_img')):
        reference_frames[name][:] = rng.integers(0, 256, frame_bytes, dtype=np.uint8)
        frames[i][:] = reference_frames[name]
        matrices[i][:] = reference_npy[('tfm', 'big_tfm')[i]]
      for name, view in compact.controls.items():
        np.copyto(view, reference_npy[name])
      reference_inputs = {name: reference_queues[name] for name in INPUT_NAMES}
      expected_pixels = candidate.reference_warp(**reference_inputs).numpy().copy()

      start = time.perf_counter()
      actual_pixels = local_warp.prepare().numpy()
      np.copyto(compact.images, actual_pixels)
      actual = candidate.run_policy(**{name: queues[name] for name in INPUT_NAMES})[0].numpy().copy()
      candidate_ms = (time.perf_counter() - start) * 1000
      difference = warp_difference(actual_pixels, expected_pixels)
      if difference is not None:
        raise RuntimeError("QCOM/AMD pixels must match exactly for this qualification experiment: " + json.dumps(difference))

      start = time.perf_counter()
      expected = original(**reference_inputs)[0].numpy().copy()
      original_ms = (time.perf_counter() - start) * 1000
      check_equal(f"step {step} public outputs", actual, expected[:, :2066])
      check_equal(f"step {step} resident feature", candidate.state.numpy().view(np.float32).reshape(1, -1), expected[:, 2066:18450])
      for name in STATE_NAMES:
        check_equal(f"step {step} {name}", queues[name].numpy(), reference_queues[name].numpy())
      reference_npy['prev_feat'][:] = expected[:, 2066:18450]
      if step >= warmup:
        timings['original'].append(original_ms)
        timings['qcom_compact_resident'].append(candidate_ms)
      if roundtrip and step == warmup + runs - 2:
        require_parked()
        with (directory / 'candidate.pkl').open('wb') as stream:
          dump_oob(candidate, stream)
        with (directory / 'candidate.pkl').open('rb') as stream:
          candidate = load_oob(stream)
        roundtrip_checked = True
    require_parked()
  return {'qualified': False, 'source_sha256': SOURCE_SHA256, 'warp_source_hash': source_hash,
          'camera_size': camera_size, 'original_input_bytes': 65656 + 2 * frame_bytes,
          'candidate_input_bytes': compact.data.nbytes, 'resident_state_bytes': 65536,
          'readback_output_bytes': 2066 * 4, 'exact_output_and_history_match': True,
          'roundtrip_checked': roundtrip_checked, 'timings': {name: timing_summary(values) for name, values in timings.items()},
          'scope': 'synthetic parked inputs only; no live camera frame-ID/drop measurement or production selection'}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', type=Path, required=True, help='Trusted installed CTMV2 v25 PKL or chunk manifest base path')
  parser.add_argument('--camera', choices=('1928x1208', '1344x760'), default='1928x1208')
  parser.add_argument('--runs', type=int, default=100)
  parser.add_argument('--warmup', type=int, default=10)
  parser.add_argument('--roundtrip', action='store_true', help='Also serialize/reload candidate; needs additional RAM/VRAM/disk')
  args = parser.parse_args()
  width, height = (int(value) for value in args.camera.split('x'))
  print(json.dumps(verify(args.model, (width, height), args.runs, args.warmup, args.roundtrip), indent=2))


if __name__ == '__main__':
  main()
