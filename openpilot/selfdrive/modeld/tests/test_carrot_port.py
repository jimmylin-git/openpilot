import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld import carrot_model
from openpilot.selfdrive.modeld.generic_model_runtime import GenericModelRuntime
from openpilot.selfdrive.modeld.precompiled_runner import PrecompiledModelState


def manifest():
  return {'model_id': 'cinque-v3', 'filename': 'model.pkl', 'size': 100, 'sha256': 'a' * 64, 'url': 'model.pkl'}


@pytest.mark.parametrize('change', [
  {'filename': '../model.pkl'}, {'filename': 'model.onnx'}, {'size': True}, {'size': 0},
  {'sha256': 'wrong'}, {'url': 'http://nas.example/model.pkl'}, {'url': 'https://other.example/model.pkl'},
])
def test_manifest_rejects_invalid_identity_or_origin(change):
  with pytest.raises(ValueError):
    carrot_model.BigModelManifest.from_dict(manifest() | change, 'https://nas.example/manifest.json')


def test_manifest_and_cache_do_not_change_selected_model(monkeypatch, tmp_path):
  model = carrot_model.BigModelManifest.from_dict(manifest(), 'https://nas.example/manifest.json')
  assert model.url == 'https://nas.example/model.pkl'
  assert carrot_model.model_path(model, tmp_path) == tmp_path / ('model-' + 'a' * 16 + '.pkl')
  assert carrot_model.active_manifest() is None


def test_parked_guard_rejects_onroad_before_model_loading(monkeypatch):
  import sys
  monkeypatch.setattr(carrot_model, 'os', SimpleNamespace(name='posix'))
  monkeypatch.setattr(carrot_model, 'Path', lambda path: SimpleNamespace(is_file=lambda: True))
  monkeypatch.setitem(sys.modules, 'openpilot.common.params', SimpleNamespace(Params=lambda: SimpleNamespace(get_bool=lambda key: False)))
  with pytest.raises(RuntimeError, match='parked testing only'):
    carrot_model.require_parked()


def load_model_method(name):
  source = Path(__file__).parents[3] / 'sunnypilot/modeld_v2/modeld.py'
  tree = ast.parse(source.read_text(encoding='utf-8'))
  model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelState')
  method = copy.deepcopy(next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == name))
  method.returns = None
  for arg in method.args.args:
    arg.annotation = None
  namespace = {'os': os, 'Path': Path, 'ModelStateBase': SimpleNamespace(__init__=lambda self: None)}
  exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
  return namespace[name], namespace


def test_carrot_selection_is_chestnut_only_and_default_loader_preserved(monkeypatch):
  constructor, namespace = load_model_method('__init__')
  selected = SimpleNamespace(generation=12, overrides=[SimpleNamespace(key='lat', value='.1')], models=[])
  namespace['get_active_bundle'] = lambda **kwargs: selected
  namespace['model_file_exists'] = lambda path: False
  namespace['_find_driving_pkl'] = lambda *args, **kwargs: 'working-tt.pkl'
  loaded = []
  state = SimpleNamespace(_init_carrot=lambda *args: loaded.append(('carrot', args)),
                          _init_combined=lambda *args: loaded.append(('legacy', args)))
  monkeypatch.delenv('COMBINED_MODEL_PKL', raising=False)
  monkeypatch.delenv('CARROT_CHESTNUT_PKL', raising=False)
  constructor(state, 1928, 1208, True)
  assert loaded[-1][0] == 'legacy' and loaded[-1][1][0] == 'working-tt.pkl'
  monkeypatch.setenv('CARROT_CHESTNUT_PKL', '/verified/carrot/model.pkl')
  constructor(state, 1928, 1208, False)
  assert loaded[-1][0] == 'legacy'
  constructor(state, 1928, 1208, True)
  assert loaded[-1][0] == 'carrot'


def test_resident_state_reset_copies_zeros_without_replacing_alias(monkeypatch):
  import sys
  targets = []
  state = SimpleNamespace(shape=(2,), _buffer=lambda: SimpleNamespace(copy_from=lambda value: targets.append(value)))
  runtime = object.__new__(GenericModelRuntime)
  runtime.state_pairs = {'state': 'next_state'}
  runtime.specs = {'state': ((2,), '<f4', 'AMD')}
  runtime.queues = {'state': state}
  monkeypatch.setitem(sys.modules, 'tinygrad', SimpleNamespace(Tensor=lambda value, device: SimpleNamespace(_buffer=lambda: value)))
  runtime.reset()
  assert runtime.queues['state'] is state
  np.testing.assert_array_equal(targets[0], np.zeros(2, np.float32))


def test_worker_reset_protocol_preserves_state_and_resets_desire(monkeypatch):
  import io
  model = object.__new__(PrecompiledModelState)
  model.process = SimpleNamespace(stdin=io.BytesIO())
  model.prev_desire = np.ones(8, np.float32)
  model.first_run = False
  monkeypatch.setattr(PrecompiledModelState, '_receive', lambda self, timeout: b'1\n')
  model.reset()
  assert model.process.stdin.getvalue() == b'z'
  np.testing.assert_array_equal(model.prev_desire, 0)
  assert model.first_run
  model.process = None


def test_fixture_parses_with_existing_car_control_parser():
  from openpilot.selfdrive.modeld.generic_model_runtime import model_metadata
  from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser
  artifact = json.loads((Path(__file__).parent / 'fixtures/cinque_v3_metadata.json').read_text())
  _, slices, _, count = model_metadata(artifact)
  outputs = Parser().parse_outputs({name: np.zeros((1, section.stop - section.start), np.float32) for name, section in slices.items()})
  assert count == 18452
  assert outputs['plan'].shape == (1, 33, 15)
  assert outputs['action'].shape == (1, 2)


@pytest.mark.parametrize('name,value', [
  ('desire', [[1, 8], '<f4', 'AMD']),
  ('outputs', [[1, 2580], '<f4', 'AMD']),
  ('outputs', [[1, 18452], '<f2', 'AMD']),
])
def test_port_rejects_unverified_control_or_output_contract(name, value):
  from openpilot.selfdrive.modeld.generic_model_runtime import model_metadata
  artifact = json.loads((Path(__file__).parent / 'fixtures/cinque_v3_metadata.json').read_text())
  artifact['output_specs' if name == 'outputs' else 'input_specs'][name] = value
  with pytest.raises(ValueError, match='contract|control input'):
    model_metadata(artifact)
