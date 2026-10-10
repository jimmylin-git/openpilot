import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.sunnypilot.models import host_warp as hw
from openpilot.sunnypilot.models import host_warp_worker as worker


def selected():
  return {'ref': 'ctmv2', 'generation': 12, 'runner': 'tinygrad', 'is20hz': True, 'minimumSelectorVersion': 19,
          'overrides': [{'key': 'lat', 'value': '.1'}],
          'models': [{'artifact': {'fileName': 'baseline.pkl', 'downloadUri': {'sha256': 'a' * 64}, 'chunks': []}}]}


def source():
  return {'baseline_sha256': 'a' * 64, 'model_type': 'supercombo',
          'onnx': {'url': 'https://example.com/source.onnx', 'sha256': 'b' * 64},
          'model_size': [512, 256], 'camera_size': [1928, 1208]}


class Params:
  def __init__(self):
    self.values = {'IsOffroad': True, 'ModelManager_ModelsCache_Chestnut': {'bundles': [{'ref': 'ctmv2', 'host_warp': source()}]}}

  def get(self, key):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.get(key))


@pytest.fixture
def setup(tmp_path, monkeypatch):
  params = Params()
  monkeypatch.setattr(hw, 'cache_root', lambda: tmp_path)
  monkeypatch.setattr(hw, 'runtime_identity', lambda: 'runtime-v1')
  monkeypatch.setattr(hw, 'boot_id', lambda: 'test-boot')
  monkeypatch.setattr(hw.os, 'killpg', Mock())
  return params, tmp_path


def ready(params, root):
  request = hw.request_for_bundle(params, selected(), 'tici')
  directory = root / hw.request_key(request)
  directory.mkdir()
  artifact = directory / 'model.pkl'
  artifact.write_bytes(b'qualified-model')
  hw.atomic_json(directory / 'ready.json', {'request': request, 'files': hw.file_records(artifact, hashes=True)})
  return request, artifact


def test_status_missing_source_does_not_start_compiler(setup):
  params, root = setup
  params.values['ModelManager_ModelsCache_Chestnut']['bundles'][0].pop('host_warp')
  preparation = hw.HostWarpPreparation(params, 'tici')
  preparation.tick(selected())
  assert preparation.child is None
  state = hw.read_json(root / 'status.json')
  assert state['state'] == 'waiting_source'
  assert 'original model retained' in hw.status_for_bundle(selected())


@pytest.mark.parametrize('change', [
  lambda s: s.update(baseline_sha256='c' * 64),
  lambda s: s.update(model_type='split'),
  lambda s: s['onnx'].update(url='http://example.com/model.onnx'),
  lambda s: s['onnx'].update(sha256='not-a-hash'),
  lambda s: s.update(camera_size=[1928, 1209]),
])
def test_unverified_or_invalid_sources_are_rejected(setup, change):
  params, _ = setup
  change(params.values['ModelManager_ModelsCache_Chestnut']['bundles'][0]['host_warp'])
  with pytest.raises(ValueError):
    hw.request_for_bundle(params, selected(), 'tici')


def test_ready_cache_is_model_runtime_source_and_device_specific(setup):
  params, root = setup
  request, path = ready(params, root)
  assert hw.ready_artifact(request, verify_hashes=True) == path
  for key, value in [('runtime', 'runtime-v2'), ('device', 'tizi'), ('version', 2)]:
    changed = {**request, key: value}
    assert hw.ready_artifact(changed) is None
  other_bundle = selected()
  other_bundle['generation'] += 1
  assert hw.ready_artifact(hw.request_for_bundle(params, other_bundle, 'tici')) is None
  params.values['ModelManager_ModelsCache_Chestnut']['bundles'][0]['host_warp']['onnx']['sha256'] = 'c' * 64
  assert hw.ready_artifact(hw.request_for_bundle(params, selected(), 'tici')) is None


def test_cache_corruption_and_incomplete_outputs_are_rejected(setup):
  params, root = setup
  request, path = ready(params, root)
  path.write_bytes(b'corrupted')
  with pytest.raises(ValueError, match='changed after'):
    hw.ready_artifact(request)
  path.unlink()
  with pytest.raises(FileNotFoundError):
    hw.ready_artifact(request)


def test_chunked_receipt_requires_all_valid_files(setup):
  params, root = setup
  request, path = ready(params, root)
  path.unlink()
  from openpilot.common.file_chunker import get_chunk_name, get_manifest_path
  manifest = path.with_name(path.name + '.chunkmanifest')
  assert str(manifest) == get_manifest_path(str(path))
  manifest.write_text('2')
  chunks = [path.with_name(get_chunk_name(path.name, i, 2)) for i in range(2)]
  for chunk in chunks:
    chunk.write_bytes(b'chunk')
  hw.atomic_json(path.parent / 'ready.json', {'request': request, 'files': hw.file_records(path, hashes=True)})
  assert hw.ready_artifact(request, verify_hashes=True) == path
  chunks[0].unlink()
  with pytest.raises(FileNotFoundError):
    hw.ready_artifact(request)


@pytest.mark.parametrize('flag', ['IsDriverViewEnabled', 'IsLiveStreaming', 'IsOffroad'])
def test_driving_or_camera_use_cancels_process_group(setup, monkeypatch, flag):
  params, root = setup
  params.values[flag] = flag != 'IsOffroad'
  preparation = hw.HostWarpPreparation(params, 'tici')
  child = Mock(pid=12345)
  preparation.child = child
  preparation.work = root / 'attempt'
  preparation.work.mkdir()
  (preparation.work / 'partial.pkl').write_bytes(b'partial')
  signal_group = Mock()
  monkeypatch.setattr(hw.os, 'killpg', signal_group)
  preparation.tick(selected())
  signal_group.assert_called_once_with(12345, hw.signal.SIGTERM)
  child.wait.assert_called_once_with(timeout=1)
  assert not (root / 'attempt').exists()
  assert preparation.child is None
  assert hw.read_json(root / 'status.json')['state'] == 'paused'


def test_model_change_cancels_inflight_preparation(setup, monkeypatch):
  params, _ = setup
  preparation = hw.HostWarpPreparation(params, 'tici')
  preparation.child = Mock(pid=12345)
  preparation.request = hw.request_for_bundle(params, selected(), 'tici')
  preparation.started_at = time.monotonic()
  stop_group = Mock()
  monkeypatch.setattr(hw.os, 'killpg', stop_group)
  changed = selected()
  changed['overrides'][0]['value'] = '.2'
  preparation.tick(changed)
  stop_group.assert_called_once()
  assert preparation.child is None


def test_worker_failure_keeps_original_and_does_not_retry_immediately(setup):
  params, root = setup
  preparation = hw.HostWarpPreparation(params, 'tici')
  preparation.request = hw.request_for_bundle(params, selected(), 'tici')
  preparation.child = Mock(pid=12345)
  preparation.child.poll.return_value = 1
  preparation.started_at = time.monotonic()
  preparation.stop = Mock(side_effect=lambda: setattr(preparation, 'child', None))
  preparation.tick(selected())
  preparation.tick(selected())
  assert hw.read_json(root / 'status.json')['state'] == 'failed'
  assert preparation.child is None
  assert hw.ready_artifact(preparation.request) is None


def test_only_verified_worker_results_are_published(setup):
  params, root = setup
  preparation = hw.HostWarpPreparation(params, 'tici')
  request = hw.request_for_bundle(params, selected(), 'tici')
  preparation.work = root / 'attempt'
  preparation.work.mkdir()
  artifact = preparation.work / 'model.pkl'
  artifact.write_bytes(b'qualified')
  hw.atomic_json(preparation.work / 'ready.json', {'request': request, 'files': hw.file_records(artifact, hashes=True)})
  preparation.request = request
  preparation.child = Mock(pid=12345)
  preparation.child.poll.return_value = 0
  preparation.started_at = time.monotonic()
  preparation.tick(selected())
  assert hw.ready_artifact(request, verify_hashes=True) is not None
  assert hw.read_json(root / 'status.json')['state'] == 'ready'


def test_missing_worker_receipt_is_not_success(setup):
  params, root = setup
  preparation = hw.HostWarpPreparation(params, 'tici')
  preparation.work = root / 'attempt'
  preparation.work.mkdir()
  preparation.request = hw.request_for_bundle(params, selected(), 'tici')
  preparation.child = Mock(pid=12345)
  preparation.child.poll.return_value = 0
  preparation.started_at = time.monotonic()
  with pytest.raises(ValueError, match='did not validate'):
    preparation.tick(selected())
  assert hw.ready_artifact(preparation.request) is None


def test_rejected_runtime_artifact_cannot_be_selected_again(setup):
  params, root = setup
  request, path = ready(params, root)
  hw.reject_artifact(path, RuntimeError('GPU unavailable'))
  assert hw.ready_artifact(request) is None
  assert hw.read_json(root / 'status.json')['state'] == 'failed'


def test_output_qualification_rejects_nonfinite_or_wrong_results():
  worker.compare_outputs({'prediction': np.array([1., 2.])}, {'prediction': np.array([1., 2.])})
  for value in (np.array([1., np.nan]), np.array([1., 3.])):
    with pytest.raises(AssertionError):
      worker.compare_outputs({'prediction': value}, {'prediction': np.array([1., 2.])})
  with pytest.raises(ValueError):
    worker.compare_outputs({}, {'prediction': np.array([1.])})


@pytest.mark.parametrize('mode', ['ready', 'no_qcom', 'wrong_output', 'too_slow'])
def test_hardware_qualification_requires_qcom_equivalence_and_frame_budget(setup, monkeypatch, mode):
  import openpilot.sunnypilot.modeld_v2.modeld as modeld
  params, root = setup
  request = hw.request_for_bundle(params, selected(), 'tici')
  monkeypatch.setenv('CHESTNUT_COMBINED_MODEL_PKL', 'test-initial.pkl')
  monkeypatch.setattr(worker, 'assert_current', lambda _: None)
  monkeypatch.setattr(worker.time, 'monotonic', Mock(side_effect=[i * (.1 if mode == 'too_slow' else .001) for i in range(60)]))
  calls = []

  class State:
    def __init__(self, *args, **kwargs):
      self.optimized = bool(calls)
      calls.append(self)
      self.numpy_inputs = {'desire': np.zeros((1, 8), dtype=np.float32), 'prev_feat': np.zeros(2, dtype=np.float32)}
      self.desire_key = 'desire'
      self.input_shapes = {'img': (1, 6, 2, 2)}
      self.input_queues = {}
      self.vision_output_slices = {'hidden_state': slice(0, 2)}
      self.vision_input_names = ['img', 'big_img']
      self.frame_buf_params = dict.fromkeys(self.vision_input_names, (4, 4, 2, 24))
      self.host_warp_runtime = SimpleNamespace(local=None if mode == 'no_qcom' else object())
      self.runs = 0

    def warmup(self):
      pass

    def run(self, frames, transforms, inputs):
      self.runs += 1
      self.numpy_inputs['prev_feat'][:] = self.runs
      return {'prediction': np.array([self.runs + int(self.optimized and mode == 'wrong_output')], dtype=np.float32)}

  monkeypatch.setattr(modeld, 'ModelState', State)
  if mode == 'ready':
    result = worker.qualify(request, root / 'optimized.pkl', root / 'baseline.pkl')
    assert result['runs'] == 30
    assert result['p99_ms'] < 50
    assert [state.runs for state in calls] == [30, 30]
  else:
    with pytest.raises((ValueError, AssertionError)):
      worker.qualify(request, root / 'optimized.pkl', root / 'baseline.pkl')


def test_changed_ready_files_are_revoked_and_original_kept(setup):
  params, root = setup
  request, path = ready(params, root)
  path.write_bytes(b'wrong')
  preparation = hw.HostWarpPreparation(params, 'tici')
  preparation.tick(selected())
  assert hw.ready_artifact(request) is None
  assert hw.read_json(root / 'status.json')['state'] == 'failed'
