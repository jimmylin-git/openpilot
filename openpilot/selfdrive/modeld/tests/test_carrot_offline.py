import json

import pytest

from openpilot.selfdrive.modeld import carrot_offline as offline


def test_complete_cache_never_uses_network(tmp_path, monkeypatch):
  from openpilot.selfdrive.modeld import precompiled_model
  value = {'model_id': 'test', 'filename': 'model.pkl', 'size': 1, 'sha256': 'a' * 64,
           'url': offline.DEFAULT_MANIFEST_URL.replace('manifest.json', 'model.pkl')}
  offline.write_json(tmp_path / 'offline-model.json', value)
  path = tmp_path / 'model.pkl'
  path.write_bytes(b'x')
  (tmp_path / 'installed.json').write_text(json.dumps(
    {'format': 'comma-generic-onnx', 'pickle': {'sha256': 'a' * 64}, 'runtime': {'sha256': 'b' * 64}}))
  monkeypatch.setattr(precompiled_model, 'installed', lambda *args: path)
  monkeypatch.setattr(precompiled_model, 'sha256', lambda path: ('a' if path.name == 'model.pkl' else 'b') * 64)
  monkeypatch.setattr(offline, 'fetch_manifest', lambda: pytest.fail('offline cache must not fetch manifest'))
  monkeypatch.setattr(precompiled_model, 'ensure_precompiled', lambda *args: pytest.fail('offline cache must not download'))
  assert offline.prepare(tmp_path, lambda *args: None) == path


def test_corrupt_cache_reports_error_without_silent_download(tmp_path, monkeypatch):
  from openpilot.selfdrive.modeld import precompiled_model
  offline.write_json(tmp_path / 'offline-model.json', {
    'model_id': 'test', 'filename': 'model.pkl', 'size': 1, 'sha256': 'a' * 64, 'url': 'model.pkl'})
  path = tmp_path / 'model.pkl'
  (tmp_path / 'installed.json').write_text(json.dumps(
    {'format': 'comma-generic-onnx', 'pickle': {'sha256': 'a' * 64}, 'runtime': {'sha256': 'b' * 64}}))
  monkeypatch.setattr(precompiled_model, 'installed', lambda *args: path)
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


def test_automatic_smoke_test_saves_both_layouts_and_closes_workers(tmp_path, monkeypatch):
  from openpilot.selfdrive.modeld import carrot_model, precompiled_runner
  instances = []
  class Runtime:
    frame_size = 12
    backend = 'qcom'
    checkpoint = 'checked'
    def __init__(self, width, height, path):
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
  result = offline.test_model(tmp_path / 'model.pkl', lambda *args: None)
  assert result['qualified'] is False
  assert [entry['samples'] for entry in result['results']] == [100, 100]
  assert [entry['camera'] for entry in result['results']] == [[1928, 1208], [1344, 760]]
  assert all(runtime.runs == 111 and runtime.reset_called and runtime.closed for runtime in instances)
