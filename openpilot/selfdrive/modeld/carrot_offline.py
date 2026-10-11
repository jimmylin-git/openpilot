"""Automatic offroad preparation and local-only testing; never enables driving."""
import json
import os
from pathlib import Path
import time

from openpilot.selfdrive.modeld.carrot_model import BigModelManifest, DEFAULT_MANIFEST_URL, fetch_manifest, model_cache_dir


def offline_usb_present() -> bool:
  from openpilot.common.hardware.usb import get_usb_state, is_chestnut_usb_id
  return any(is_chestnut_usb_id(device['vendorId'], device['productId']) for device in get_usb_state())


def write_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
  os.replace(temporary, path)


def read_status(cache: Path | None = None) -> dict:
  path = (cache or model_cache_dir()) / 'offline-status.json'
  if not path.is_file():
    return {'state': 'waiting', 'detail': 'Waiting for automatic preparation; original model remains selected.'}
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not isinstance(value.get('detail'), str):
      raise ValueError('Invalid status format')
    return value
  except (OSError, ValueError) as error:
    return {'state': 'error', 'detail': f'Cannot read local test status: {error}'}


def prepare(cache: Path, report) -> Path:
  from openpilot.selfdrive.modeld.precompiled_model import installed, ensure_precompiled, sha256
  selected = cache / 'offline-model.json'
  if selected.is_file():
    model = BigModelManifest.from_dict(json.loads(selected.read_text()), DEFAULT_MANIFEST_URL)
    path = installed(model, cache, allow_rejected=True)
    if path is not None:
      catalog = json.loads((path.parent / 'installed.json').read_text())
      if (catalog['format'] == 'comma-generic-onnx' and
          sha256(path) == catalog['pickle']['sha256'] and
          sha256(path.parent / 'runtime.tar.gz') == catalog['runtime']['sha256']):
        path = ensure_precompiled(model, cache)
        if path is None:
          failure = json.loads((cache / 'precompiled' / model.sha256 / 'last_failure.json').read_text())
          raise RuntimeError(f"Model remains rejected: {failure['error'].rstrip().splitlines()[-1]}")
        return path  # No manifest request or network call when locally complete.
      raise ValueError('Local generic model/runtime verification failed')
  report('downloading', 'Preparing model/runtime. Initial download needs Wi-Fi; later tests are offline.')
  model = fetch_manifest()

  def progress(current, total):
    report('downloading', f'Downloading carrot model: {100 * current / total:.0f}%')

  path = ensure_precompiled(model, cache, progress)
  if path is None:
    raise RuntimeError('Model rejected; inspect its last_failure.json')
  if json.loads((path.parent / 'installed.json').read_text())['format'] != 'comma-generic-onnx':
    raise ValueError('Offline test requires a generic artifact')
  write_json(selected, vars(model))
  return path


def test_model(path: Path, report, *, sampling_boundary_diagnostic=False) -> dict:
  import numpy as np
  from types import SimpleNamespace
  from openpilot.selfdrive.modeld.carrot_model import require_parked
  from openpilot.selfdrive.modeld.precompiled_runner import PrecompiledModelState

  for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
    if cmdline.parent.name == str(os.getpid()):
      continue
    try:
      args = cmdline.read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
      continue
    if any(arg.rsplit(b'/', 1)[-1] in (b'modeld', b'modeld.py', b'modeld_tinygrad', b'precompiled_worker.py') or
           arg in (b'openpilot.sunnypilot.modeld_v2.modeld', b'openpilot.selfdrive.modeld.modeld') for arg in args):
      raise RuntimeError('Refusing to share GPU with modeld or another worker')
  results = []
  for width, height in ((1928, 1208), (1344, 760)):
    require_parked()
    report('testing', f'Parked synthetic test {width}x{height}; do not start driving until it stops.')
    options = {'sampling_boundary_diagnostic': True} if sampling_boundary_diagnostic else {}
    runtime = PrecompiledModelState(width, height, path, **options)
    try:
      if sampling_boundary_diagnostic and runtime.backend != 'qcom':
        raise RuntimeError('Parked diagnostic did not select QCOM')
      rng = np.random.default_rng(42)
      frames = {name: SimpleNamespace(data=np.zeros(runtime.frame_size, np.uint8)) for name in ('img', 'big_img')}
      transforms = {name: np.eye(3, dtype=np.float32) for name in frames}
      inputs = {'desire_pulse': np.zeros(8, np.float32), 'traffic_convention': np.array([[1, 0]], np.float32),
                'action_t': np.zeros((1, 2), np.float32)}
      timings = []
      for step in range(110):
        require_parked()
        for frame in frames.values():
          frame.data[:] = rng.integers(0, 256, runtime.frame_size, dtype=np.uint8)
        started = time.monotonic()
        runtime.run(frames, transforms, inputs, False)
        elapsed = (time.monotonic() - started) * 1000
        if step >= 10:
          timings.append(elapsed)
        if step % 10 == 0:
          report('testing', f'Parked synthetic test {width}x{height}: {step + 1}/110 frames')
      runtime.reset()
      runtime.run(frames, transforms, inputs, False)
      results.append({'camera': [width, height], 'backend': runtime.backend, 'checkpoint': runtime.checkpoint,
                      'mean_ms': float(np.mean(timings)), 'p95_ms': float(np.percentile(timings, 95)),
                      'p99_ms': float(np.percentile(timings, 99)), 'over_50ms': sum(value > 50 for value in timings),
                      'samples': len(timings), 'reset_checked': True})
      if sampling_boundary_diagnostic:
        results[-1].update(usb_input_bytes=runtime.usb_input_bytes, warp_differences=runtime.warp_differences)
    finally:
      runtime.close()
  return {'qualified': False, 'sampling_boundary_diagnostic': sampling_boundary_diagnostic,
          'scope': 'synthetic parked smoke test, not equivalence or driving qualification',
          'results': results}


def diagnostic_main():
  from openpilot.selfdrive.modeld.carrot_model import require_parked
  from openpilot.selfdrive.modeld.egpu_worker_progress import boot_identity
  from openpilot.common.swaglog import cloudlog
  import traceback

  require_parked()
  cache, boot = model_cache_dir(), boot_identity()
  def report(state, detail):
    write_json(cache / 'qcom-diagnostic-status.json',
               {'state': state, 'detail': detail, 'boot_id': boot, 'qualified': False})
  try:
    path = prepare(cache, report)
    if not offline_usb_present():
      raise RuntimeError('No supported eGPU USB detected')
    result = test_model(path, report, sampling_boundary_diagnostic=True)
    write_json(cache / 'qcom-diagnostic-result.json', {**result, 'boot_id': boot, 'model': str(path)})
    report('complete', 'Parked QCOM diagnostic complete; NOT equivalent or driving-qualified.')
  except Exception:
    cloudlog.exception('Parked QCOM diagnostic failed; no automatic retry')
    write_json(cache / 'qcom-diagnostic-error.json',
               {'boot_id': boot, 'error': traceback.format_exc(), 'qualified': False})
    report('error', 'Parked QCOM diagnostic failed; no automatic retry. Inspect diagnostic error.')
    raise


def main():
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog
  from openpilot.selfdrive.modeld.egpu_worker_progress import boot_identity

  cache = model_cache_dir()
  params = Params()
  boot = boot_identity()

  def report(state, detail):
    write_json(cache / 'offline-status.json', {'state': state, 'detail': detail, 'boot_id': boot,
                                             'updated_monotonic': time.monotonic(), 'qualified': False})

  try:
    report('preparing', 'Preparing local files. Original model remains selected.')
    path = None
    tested = False
    while params.get_bool('IsOffroad'):
      if tested:
        time.sleep(1)
        continue
      try:
        if path is None:
          path = prepare(cache, report)
        if not offline_usb_present():
          report('waiting_chestnut', 'Offline files ready. No supported eGPU USB detected while offroad; power status unknown.')
          time.sleep(5)
          continue
        result = test_model(path, report)
        write_json(cache / 'offline-result.json', {**result, 'boot_id': boot, 'model': str(path)})
        report('complete', 'Parked smoke test complete; results saved locally. NOT driving-qualified; original model remains selected.')
        tested = True
      except Exception:
        cloudlog.exception('Carrot automatic parked preparation/test failed')
        import traceback
        error = traceback.format_exc()[-16384:]
        write_json(cache / 'offline-error.json', {'boot_id': boot, 'error': error, 'qualified': False})
        retry = 'No GPU retry until the next offroad session.' if path is not None else 'Retrying preparation in 60 seconds.'
        report('error', error.splitlines()[-1] + ' - original model unchanged. ' + retry)
        if path is not None:
          tested = True  # Do not repeatedly reload a failed GPU within this offroad session.
        time.sleep(60)
  finally:
    status = read_status(cache)
    if status.get('state') not in ('complete', 'error'):
      report('stopped', 'Parked test stopped. No carrot driving enablement; original model remains selected.')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sampling-boundary-diagnostic', action='store_true',
                      help='One-shot parked QCOM diagnostic; never changes the default exact guard')
  args = parser.parse_args()
  if args.sampling_boundary_diagnostic:
    diagnostic_main()
  else:
    main()
