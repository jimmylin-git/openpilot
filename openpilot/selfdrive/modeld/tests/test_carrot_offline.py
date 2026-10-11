import json
import importlib.util
from pathlib import Path
import sys

import pytest

from openpilot.selfdrive.modeld import carrot_offline as offline


@pytest.fixture
def usb_module(monkeypatch):
  # Load the real USB helper without importing platform-specific hardware.
  name = 'openpilot.common.hardware.usb'
  spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[3] / 'common/hardware/usb.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  monkeypatch.setitem(sys.modules, name, module)
  return module


@pytest.mark.parametrize('vendor,product,name,expected', [
  (0xADD1, 0x0001, 'custom other-firmware', True),
  (0x3801, 0x0001, '', True),
  (0x174C, 0x2464, 'bootloader', False),
  (0x174C, 0x2463, 'bootloader', False),
  (0x1D6B, 0x0003, 'USB hub', False),
])
def test_offline_usb_uses_supported_ids_not_product_name(monkeypatch, usb_module, vendor, product, name, expected):
  monkeypatch.setattr(usb_module, 'get_usb_state', lambda: [
    {'vendorId': vendor, 'productId': product, 'product': name},
  ])
  assert offline.offline_usb_present() is expected


def test_offline_usb_waits_when_no_device_is_enumerated(monkeypatch, usb_module):
  monkeypatch.setattr(usb_module, 'get_usb_state', list)
  assert not offline.offline_usb_present()


def test_complete_cache_never_uses_network(tmp_path, monkeypatch):
  from openpilot.selfdrive.modeld import precompiled_model
  value = {'model_id': 'test', 'filename': 'model.pkl', 'size': 1, 'sha256': 'a' * 64,
           'url': offline.DEFAULT_MANIFEST_URL.replace('manifest.json', 'model.pkl')}
  offline.write_json(tmp_path / 'offline-model.json', value)
  path = tmp_path / 'model.pkl'
  path.write_bytes(b'x')
  (tmp_path / 'installed.json').write_text(json.dumps(
    {'format': 'comma-generic-onnx', 'pickle': {'sha256': 'a' * 64}, 'runtime': {'sha256': 'b' * 64}}))
  monkeypatch.setattr(precompiled_model, 'installed', lambda *args, **kwargs: path)
  monkeypatch.setattr(precompiled_model, 'sha256', lambda path: ('a' if path.name == 'model.pkl' else 'b') * 64)
  monkeypatch.setattr(offline, 'fetch_manifest', lambda: pytest.fail('offline cache must not fetch manifest'))
  monkeypatch.setattr(precompiled_model, 'ensure_precompiled', lambda *args: path)
  assert offline.prepare(tmp_path, lambda *args: None) == path


def test_corrupt_cache_reports_error_without_silent_download(tmp_path, monkeypatch):
  from openpilot.selfdrive.modeld import precompiled_model
  offline.write_json(tmp_path / 'offline-model.json', {
    'model_id': 'test', 'filename': 'model.pkl', 'size': 1, 'sha256': 'a' * 64, 'url': 'model.pkl'})
  path = tmp_path / 'model.pkl'
  (tmp_path / 'installed.json').write_text(json.dumps(
    {'format': 'comma-generic-onnx', 'pickle': {'sha256': 'a' * 64}, 'runtime': {'sha256': 'b' * 64}}))
  monkeypatch.setattr(precompiled_model, 'installed', lambda *args, **kwargs: path)
  monkeypatch.setattr(precompiled_model, 'sha256', lambda path: 'c' * 64)
  with pytest.raises(ValueError, match='verification failed'):
    offline.prepare(tmp_path, lambda *args: None)


def test_status_is_persistent_and_invalid_status_is_visible(tmp_path):
  assert offline.read_status(tmp_path)['state'] == 'waiting'
  offline.write_json(tmp_path / 'offline-status.json', {'state': 'complete', 'detail': 'Not driving-qualified', 'qualified': False})
  assert offline.read_status(tmp_path)['qualified'] is False
  (tmp_path / 'offline-status.json').write_text('invalid')
  assert offline.read_status(tmp_path)['state'] == 'error'


def test_manager_only_runs_offroad_and_ui_exposes_local_status():
  import ast
  from pathlib import Path
  root = Path(__file__).parents[3]
  source = ast.parse((root / 'system/manager/process_config.py').read_text())
  call = next(node for node in ast.walk(source) if isinstance(node, ast.Call) and node.args and
              isinstance(node.args[0], ast.Constant) and node.args[0].value == 'carrot_offline')
  assert call.args[2].id == 'only_offroad'
  assert call.keywords[0].value.id == 'COMMA_HARDWARE'
  ui = (root / 'selfdrive/ui/sunnypilot/layouts/settings/models.py').read_text(encoding='utf-8')
  assert 'carrot_status_item' in ui and 'read_status' in ui


@pytest.mark.parametrize('diagnostic', [False, True])
def test_automatic_smoke_test_saves_both_layouts_and_closes_workers(tmp_path, monkeypatch, diagnostic):
  from openpilot.selfdrive.modeld import carrot_model, precompiled_runner
  instances = []
  class Runtime:
    frame_size = 12
    backend = 'qcom'
    checkpoint = 'checked'
    usb_input_bytes = 393728
    warp_differences = []
    last_timings = {'local_prepare_ms': 1.0, 'input_upload_ms': 2.0, 'model_call_ms': 3.0, 'output_read_ms': 4.0}
    def __init__(self, width, height, path, **options):
      assert options == ({'sampling_boundary_diagnostic': True} if diagnostic else {})
      self.runs = 0
      self.closed = False
      self.reset_called = False
      instances.append(self)
    def run(self, frames, transforms, inputs, prepare_only):
      assert frames['img'].data.shape == (12,)
      self.runs += 1
    def reset(self):
      self.reset_called = True
    def close(self):
      self.closed = True
  monkeypatch.setattr(carrot_model, 'require_parked', lambda: None)
  monkeypatch.setattr(precompiled_runner, 'PrecompiledModelState', Runtime)
  result = offline._test_model(tmp_path / 'model.pkl', lambda *args: None, sampling_boundary_diagnostic=diagnostic)
  assert result['qualified'] is False
  assert [entry['samples'] for entry in result['results']] == [100, 100]
  assert [entry['camera'] for entry in result['results']] == [[1928, 1208], [1344, 760]]
  assert all(runtime.runs == 111 and runtime.reset_called and runtime.closed for runtime in instances)
  assert result['sampling_boundary_diagnostic'] is diagnostic
  if diagnostic:
    assert all(entry['usb_input_bytes'] == 393728 for entry in result['results'])
    assert all(len(entry['timing_samples']) == 100 and len(entry['slowest_samples']) == 10 for entry in result['results'])
    assert result['results'][0]['stage_summary']['input_upload_ms']['mean_ms'] == 2.0


@pytest.mark.parametrize('fails', [False, True])
def test_parked_session_lock_released_on_success_and_failure(tmp_path, monkeypatch, fails):
  from types import SimpleNamespace
  from openpilot.selfdrive.modeld import carrot_model
  calls = []
  monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
                                                          flock=lambda fd, flags: calls.append(flags)))
  monkeypatch.setattr(carrot_model, 'require_parked', lambda: None)
  def run(*args, **kwargs):
    assert calls == [3]
    assert kwargs == {'sampling_boundary_diagnostic': True}
    if fails:
      raise RuntimeError('GPU failed')
    return {'qualified': False}
  monkeypatch.setattr(offline, '_test_model', run)
  path = tmp_path / 'artifact' / 'model.pkl'
  path.parent.mkdir()
  if fails:
    with pytest.raises(RuntimeError, match='GPU failed'):
      offline.test_model(path, lambda *a: None, sampling_boundary_diagnostic=True)
  else:
    assert offline.test_model(path, lambda *a: None, sampling_boundary_diagnostic=True) == {'qualified': False}
  assert calls == [3, 4]
  assert (tmp_path / 'parked-test.lock').exists()


def test_busy_parked_session_never_starts_gpu(tmp_path, monkeypatch):
  from types import SimpleNamespace
  from openpilot.selfdrive.modeld import carrot_model
  def busy(fd, flags):
    raise BlockingIOError('locked')
  monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=busy))
  monkeypatch.setattr(carrot_model, 'require_parked', lambda: None)
  monkeypatch.setattr(offline, '_test_model', lambda *a, **kw: pytest.fail('must not start a GPU worker'))
  path = tmp_path / 'artifact' / 'model.pkl'
  path.parent.mkdir()
  with pytest.raises(offline.ParkedTestBusy):
    offline.test_model(path, lambda *a: None)


@pytest.mark.skipif(sys.platform == 'win32', reason='Requires Linux flock')
def test_real_parked_session_excludes_another_process(tmp_path, monkeypatch):
  import subprocess
  from openpilot.selfdrive.modeld import carrot_model
  monkeypatch.setattr(carrot_model, 'require_parked', lambda: None)
  path = tmp_path / 'artifact' / 'model.pkl'
  path.parent.mkdir()
  def run(*args, **kwargs):
    probe = subprocess.run([sys.executable, '-c',
                            'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                            str(tmp_path / 'parked-test.lock')], capture_output=True, text=True)
    assert probe.returncode != 0 and 'BlockingIOError' in probe.stderr
    return {'qualified': False}
  monkeypatch.setattr(offline, '_test_model', run)
  assert offline.test_model(path, lambda *a: None) == {'qualified': False}
  import fcntl
  with (tmp_path / 'parked-test.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
