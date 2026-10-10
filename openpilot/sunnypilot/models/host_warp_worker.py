"""Isolated offroad compiler and same-model hardware qualification worker."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import requests

from openpilot.common.basedir import BASEDIR
from openpilot.common.file_chunker import open_file_chunked
from openpilot.common.params import Params
from openpilot.sunnypilot.models.host_warp import atomic_json, bundle_identity, file_records, read_json
from openpilot.sunnypilot.models.helpers import get_selected_bundle


def assert_current(request: dict) -> None:
  params = Params()
  if not params.get_bool('IsOffroad') or params.get_bool('IsDriverViewEnabled') or params.get_bool('IsLiveStreaming'):
    raise RuntimeError("Host-warp qualification requires offroad with cameras stopped")
  if bundle_identity(get_selected_bundle(params, 'chestnut')) != request['bundle']:
    raise RuntimeError("Selected Chestnut model changed during preparation")


def compare_outputs(actual, expected) -> None:
  if isinstance(expected, dict):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
      raise ValueError("Host-warp output fields differ from baseline")
    for key in expected:
      compare_outputs(actual[key], expected[key])
  elif isinstance(expected, (np.ndarray, float, int, np.number)):
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4, equal_nan=False)
    if not np.all(np.isfinite(actual)):
      raise ValueError("Host-warp output is not finite")
  elif actual != expected:
    raise ValueError("Host-warp output differs from baseline")


def qualify(request: dict, path: Path, baseline_path: Path) -> dict:
  from openpilot.sunnypilot.modeld_v2.modeld import ModelState
  cam_w, cam_h = request['source']['camera_size']
  os.environ['CHESTNUT_COMBINED_MODEL_PKL'] = str(baseline_path)
  baseline = ModelState(cam_w, cam_h, chestnut=True)
  baseline.warmup()
  os.environ['CHESTNUT_COMBINED_MODEL_PKL'] = str(path)
  optimized = ModelState(cam_w, cam_h, chestnut=True)
  optimized.warmup()
  runtime = optimized.host_warp_runtime
  if runtime is None or runtime.local is None:
    raise ValueError("QCOM warp hardware validation failed; original model retained")
  if baseline.input_shapes != optimized.input_shapes or baseline.vision_output_slices != optimized.vision_output_slices:
    raise ValueError("Host-warp model ABI does not match the selected baseline")
  from tinygrad import Tensor
  for state in (baseline, optimized):
    for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q'):
      if key in state.input_queues:
        queue = state.input_queues[key]
        queue.assign(Tensor.zeros(*queue.shape, dtype=queue.dtype, device=queue.device)).realize()
  rng = np.random.default_rng(42)
  transforms = {key: np.eye(3, dtype=np.float32) for key in baseline.vision_input_names}
  timings = []
  for step in range(30):
    assert_current(request)
    frames = {key: rng.integers(0, 256, baseline.frame_buf_params[key][3], dtype=np.uint8)
              for key in baseline.vision_input_names}
    inputs = {key: np.zeros_like(value) for key, value in baseline.numpy_inputs.items()
              if key not in ('tfm', 'big_tfm', 'prev_feat')}
    inputs[baseline.desire_key].flat[step % inputs[baseline.desire_key].size] = 1
    expected = baseline.run(frames, transforms, {key: value.copy() for key, value in inputs.items()})
    start = time.monotonic()
    actual = optimized.run(frames, transforms, {key: value.copy() for key, value in inputs.items()})
    timings.append((time.monotonic() - start) * 1000)
    compare_outputs(actual, expected)
    if 'prev_feat' in baseline.numpy_inputs:
      compare_outputs(optimized.numpy_inputs['prev_feat'], baseline.numpy_inputs['prev_feat'])
    if runtime.local is None:
      raise ValueError("QCOM preparation fell back during qualification")
  p95 = float(np.percentile(timings[5:], 95))
  p99 = float(np.percentile(timings[5:], 99))
  if p99 >= 50:
    raise ValueError(f"Host-warp qualification exceeds 50 ms frame budget: P99={p99:.2f} ms")
  return {'runs': len(timings), 'p95_ms': p95, 'p99_ms': p99}


def main() -> None:
  from openpilot.common.hardware.hw import Paths
  from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
  os.environ['HOST_WARP_PREPARING'] = '1'
  os.environ['GMMU'] = '0'
  os.environ['DEV'] = 'USB+AMD:LLVM'
  work = Path(sys.argv[1]).resolve()
  request = read_json(work / 'request.json')
  assert_current(request)
  from openpilot.sunnypilot.models.host_warp import runtime_identity
  if runtime_identity() != request['runtime']:
    raise ValueError("Runtime changed before compilation")
  source = request['source']
  baseline_path = Path(Paths.model_root()) / request['bundle']['artifacts'][0]['fileName']
  with open_file_chunked(str(baseline_path)) as file:
    if hashlib.file_digest(file, 'sha256').hexdigest() != source['baseline_sha256']:
      raise ValueError("Selected baseline does not match source provenance")
  onnx = work / 'source.onnx'
  digest = hashlib.sha256()
  with requests.get(source['onnx']['url'], stream=True, timeout=(10, 30)) as response:
    response.raise_for_status()
    with onnx.open('wb') as file:
      for chunk in response.iter_content(128 * 1024):
        assert_current(request)
        file.write(chunk)
        digest.update(chunk)
  if digest.hexdigest() != source['onnx']['sha256']:
    raise ValueError("Downloaded ONNX SHA-256 mismatch")
  metadata = make_metadata_dict(onnx)
  shapes = metadata['input_shapes']
  model_w, model_h = source['model_size']
  if tuple(shapes.get('img', ())[2:]) != (model_h // 2, model_w // 2) or shapes.get('img') != shapes.get('big_img'):
    raise ValueError("ONNX image dimensions do not match source declaration")
  cam_w, cam_h = source['camera_size']
  output = work / 'model.pkl'
  env = {**os.environ, 'CHESTNUT': '1', 'FLOAT16': '1', 'JIT_BATCH_SIZE': '0'}
  # Require source checkpoint/layout agreement, not just a model name or image size.
  from openpilot.selfdrive.modeld.helpers import load_oob
  with open_file_chunked(str(baseline_path)) as file:
    baseline_jits = load_oob(file)
  baseline_metadata = baseline_jits['metadata'].get('model', baseline_jits['metadata'])
  for key in ('model_checkpoint', 'input_shapes', 'output_slices', 'output_shapes'):
    if not metadata.get(key) or metadata[key] != baseline_metadata.get(key):
      raise ValueError(f"Source ONNX {key} does not match the selected model")
  del baseline_jits
  subprocess.run([sys.executable, '-m', 'openpilot.sunnypilot.modeld_v2.compile_modeld',
                  '--model-type', 'supercombo', '--supercombo-onnx', str(onnx),
                  '--model-size', f'{model_w}x{model_h}', '--camera-resolutions', f'{cam_w}x{cam_h}',
                  '--chestnut-host-warp', '--benchmark-runs', '3', '--output', str(output)],
                 cwd=BASEDIR, env=env, check=True)
  assert_current(request)
  qualification = qualify(request, output, baseline_path)
  onnx.unlink()
  assert_current(request)
  runtime_identity.cache_clear()
  if runtime_identity() != request['runtime']:
    raise ValueError("Runtime changed during qualification")
  atomic_json(work / 'ready.json', {'request': request, 'files': file_records(output, hashes=True),
                                   'qualification': qualification})


if __name__ == '__main__':
  main()
