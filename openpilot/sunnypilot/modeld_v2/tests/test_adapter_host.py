import ast
import codecs
import importlib
import io
import os
import pickle
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes

from openpilot.sunnypilot.modeld_v2.helpers import dump_oob, load_oob
from openpilot.sunnypilot.modeld_v2.compile_modeld import compile_jit
from openpilot.common.file_chunker import chunk_file, open_file_chunked

MODEL_DIR = Path(__file__).resolve().parents[1]


def load_adapters():
  return importlib.import_module('openpilot.sunnypilot.modeld_v2.model_adapters')


def noop(**kwargs):
  return None


def load_model_state_methods():
  tree = ast.parse((MODEL_DIR / 'modeld.py').read_text(encoding='utf-8'))
  state = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelState')
  methods = [node for node in state.body if isinstance(node, ast.FunctionDef) and node.name in ('run', 'warmup')]
  selected = ast.Module(
    body=[
      ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
      ast.ClassDef(name='HostModelState', bases=[], keywords=[], body=methods, decorator_list=[]),
    ],
    type_ignores=[],
  )
  namespace = {'np': np}
  exec(compile(ast.fix_missing_locations(selected), str(MODEL_DIR / 'modeld.py'), 'exec'), namespace)
  return namespace['HostModelState']


class AdapterHostTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.adapters = load_adapters()

  def legacy_data(self):
    shapes = {
      'img': (1, 12, 2, 2),
      'big_img': (1, 12, 2, 2),
      'features_buffer': (1, 2, 4),
      'desire_pulse': (1, 3, 8),
      'traffic_convention': (1, 2),
      'action_t': (1, 2),
    }
    return {'metadata': {'model': {'input_shapes': shapes, 'output_slices': {'hidden_state': slice(0, 4)}}}, 'run_policy': noop, (32, 32): noop}

  def test_v25_chestnut_keeps_separate_warp_and_policy(self):
    data = self.legacy_data()
    calls = []
    data['run_policy'] = lambda **kwargs: calls.append(kwargs) or Tensor([1.0], device='PYTHON')
    data[(32, 32)] = lambda **kwargs: Tensor([2.0], device='PYTHON')
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    self.assertIsInstance(adapter, self.adapters.LegacyModelAdapter)
    self.assertEqual(adapter.WARP_DEV, 'PYTHON')
    frame = np.zeros(adapter.nv12_info[3], dtype=np.uint8)
    adapter.copy_frames({'img': frame, 'big_img': frame})
    self.assertEqual(adapter.run().item(), 1.0)
    self.assertIn('warped', calls[0])

  def test_unified_artifact_dispatch_and_copy(self):
    data = self.legacy_data()
    data['run_model'] = {(32, 32): lambda **kwargs: (Tensor([3.0], device='PYTHON'),)}
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    self.assertIsInstance(adapter, self.adapters.UnifiedModelAdapter)
    frame = np.full(adapter.frame_copy_size, 7, dtype=np.uint8)
    adapter.copy_frames({'img': frame, 'big_img': frame})
    np.testing.assert_array_equal(adapter.frame_slots['img'], frame)
    self.assertEqual(adapter.run().item(), 3.0)
    adapter.input_queues['feat_q'].assign(1).realize()
    adapter.reset_warmup_buffers()
    self.assertTrue(np.all(adapter.input_queues['feat_q'].numpy() == 0))

  def test_upstream_unified_tuple_dispatch(self):
    data = self.legacy_data()
    del data['run_policy']
    data['input_devices'] = {'model': 'PYTHON'}
    data[(32, 32)] = lambda **kwargs: (Tensor([3.0], device='PYTHON'),)
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    self.assertIsInstance(adapter, self.adapters.UnifiedModelAdapter)
    self.assertEqual(adapter.run().item(), 3.0)

  def test_chestnut_controls_remain_live_host_views(self):
    with patch.dict(os.environ, {'CHESTNUT': '1'}):
      adapter = self.adapters.get_model_adapter(self.legacy_data(), 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    adapter.numpy_inputs['desire'][:] = 1
    self.assertTrue(np.all(adapter.input_queues['packed_npy_inputs'].numpy()[:8] == 1))
    adapter.numpy_inputs['desire'][:] = 2
    self.assertTrue(np.all(adapter.input_queues['packed_npy_inputs'].numpy()[:8] == 2))

  def native_data(self):
    slices = codecs.encode(pickle.dumps({'hidden_state': slice(0, 1)}), 'base64').decode()
    specs = {
      'new_img': ((2, 6, 2, 2), 'uint8', 'PYTHON'),
      'desire': ((8,), 'float32', 'PYTHON'),
      'traffic_convention': ((2,), 'float32', 'PYTHON'),
      'counter': ((1,), 'int32', 'PYTHON'),
      'state': ((1,), 'float32', 'PYTHON'),
    }

    def run(output_buffers, state, counter, **kwargs):
      output_buffers['outputs'].assign(state + counter.cast(dtypes.float32)).realize()
      output_buffers['next_state'].assign(state + 1).realize()

    return {
      'input_specs': specs,
      'output_specs': {'outputs': ((1,), 'float32', 'PYTHON'), 'next_state': ((1,), 'float32', 'PYTHON')},
      'metadata': {
        'input_shapes': {'img': (1, 12, 2, 2), 'big_img': (1, 12, 2, 2)},
        'output_shapes': {'outputs': (1,), 'next_state': (1,)},
        'metadata': {'output_slices': slices},
      },
      'run': run,
      (32, 32): lambda **kwargs: Tensor.zeros(2, 6, 2, 2, dtype=dtypes.uint8, device='PYTHON'),
    }

  def test_native_state_alias_and_reset(self):
    adapter = self.adapters.get_model_adapter(self.native_data(), 32, 32, 'PYTHON', 'PYTHON', 'PYTHON')
    self.assertIsInstance(adapter, self.adapters.NativeTinygradAdapter)
    self.assertEqual(adapter.numpy_inputs['counter'].dtype, np.dtype('int32'))
    adapter.numpy_inputs['counter'][:] = 4
    self.assertEqual(adapter.run().item(), 4.0)
    self.assertEqual(adapter.run().item(), 5.0)
    adapter.reset_warmup_buffers()
    self.assertEqual(adapter.run().item(), 0.0)

  def test_legacy_warmup_clears_history(self):
    adapter = self.adapters.get_model_adapter(self.legacy_data(), 32, 32, 'PYTHON', 'PYTHON', 'PYTHON')
    for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q'):
      adapter.input_queues[key].assign(1).realize()
    adapter.reset_warmup_buffers()
    for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q'):
      self.assertTrue(np.all(adapter.input_queues[key].numpy() == 0))

  def test_native_rejects_mismatched_recurrent_state(self):
    data = self.native_data()
    data['output_specs']['next_state'] = ((2,), 'float32', 'PYTHON')
    with self.assertRaisesRegex(ValueError, 'Incompatible recurrent state specs'):
      self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON')

  def test_native_jit_replay_uses_new_controls(self):
    data = self.native_data()
    data['run'] = TinyJit(data['run'])
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON')
    for value in range(5):
      adapter.numpy_inputs['counter'][:] = value * 2
      self.assertEqual(adapter.run().item(), value * 3)
    adapter.reset_warmup_buffers()
    self.assertEqual(adapter.run().item(), 0.0)

  def test_native_compiler_pickle_replay_with_new_inputs(self):
    def run(inputs, output_buffers):
      output_buffers['outputs'].assign(inputs * 2).realize()

    def make_inputs(seed):
      array = np.random.default_rng(seed).standard_normal(4).astype(np.float32)
      return (), {'inputs': Tensor(array, device='PYTHON').realize(), 'output_buffers': {'outputs': Tensor.zeros(4, device='PYTHON').realize()}}

    loaded = compile_jit(run, make_inputs, benchmark_runs=3)
    args, kwargs = make_inputs(44)
    loaded(*args, **kwargs)
    np.testing.assert_array_equal(kwargs['output_buffers']['outputs'].numpy(), kwargs['inputs'].numpy() * 2)

  def test_legacy_compiler_pickle_replay_with_live_host_inputs(self):
    def make_queues(device):
      array = np.zeros(4, dtype=np.float32)
      return {'inputs': Tensor(array, device='NPY').realize()}, {'inputs': array}

    jit = TinyJit(lambda inputs: (inputs.to('PYTHON') * 2).realize())
    loaded = compile_jit(jit, ['inputs'], make_queues, benchmark_runs=3)
    queues, views = make_queues('PYTHON')
    views['inputs'][:] = 7
    np.testing.assert_array_equal(loaded(**queues).numpy(), np.full(4, 14, dtype=np.float32))

  def test_model_state_preserves_native_nonfinite_guard(self):
    data = self.native_data()
    data['run'] = lambda output_buffers, **kwargs: output_buffers['outputs'].assign(float('nan')).realize()
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    state = load_model_state_methods()()
    state.adapter = adapter
    state.numpy_inputs = adapter.numpy_inputs
    state.desire_key = 'desire'
    state.prev_desire = np.zeros(8, dtype=np.float32)
    state._vision_input_names = ['img', 'big_img']
    state._combined_model_type = 'supercombo'
    state.chestnut = True
    with self.assertRaisesRegex(RuntimeError, 'model output not finite'):
      state.run({}, {'img': np.eye(3), 'big_img': np.eye(3)}, {'desire': np.zeros(8)})

  def test_multi_policy_keeps_on_policy_precedence(self):
    shapes = self.legacy_data()['metadata']['model']['input_shapes']
    data = {
      'metadata': {
        'vision': {'input_shapes': {key: shapes[key] for key in ('img', 'big_img')}, 'output_slices': {'hidden_state': slice(0, 4)}},
        'onPolicy': {'input_shapes': {key: value for key, value in shapes.items() if 'img' not in key}, 'output_slices': {'plan': slice(0, 1)}},
        'offPolicy': {'input_shapes': {key: value for key, value in shapes.items() if 'img' not in key}, 'output_slices': {'plan': slice(0, 1)}},
      },
      'run_policy': lambda **kwargs: (Tensor([1.0, 1.0, 1.0, 1.0], device='PYTHON'), Tensor([2.0], device='PYTHON'), Tensor([3.0], device='PYTHON')),
      (32, 32): lambda **kwargs: Tensor([0.0], device='PYTHON'),
    }
    adapter = self.adapters.get_model_adapter(data, 32, 32, 'PYTHON', 'PYTHON', 'PYTHON', chestnut=True)
    state = load_model_state_methods()()
    state.adapter = adapter
    for key in (
      'numpy_inputs',
      '_vision_input_names',
      '_combined_model_type',
      'vision_output_slices',
      '_policy_slices_list',
      '_policy_keys',
      '_has_on_policy',
      '_road_key',
      '_wide_key',
    ):
      setattr(state, key, getattr(adapter, key))
    state.desire_key = 'desire'
    state.prev_desire = np.zeros(8, dtype=np.float32)
    state.chestnut = True
    state.parser = SimpleNamespace(parse_vision_outputs=lambda outputs: outputs, parse_policy_outputs=lambda outputs: outputs)
    frame = np.zeros(adapter.nv12_info[3], dtype=np.uint8)
    outputs = state.run({'img': frame, 'big_img': frame}, {'img': np.eye(3), 'big_img': np.eye(3)}, {'desire': np.zeros(8)})
    np.testing.assert_array_equal(outputs['plan'], [[2.0]])

  def test_oob_round_trip_preserves_values(self):
    data = {'array': np.arange(10, dtype=np.float32), 'slice': slice(1, 8)}
    stream = io.BytesIO()
    dump_oob(data, stream)
    stream.seek(0)
    loaded = load_oob(stream)
    np.testing.assert_array_equal(loaded['array'], data['array'])
    self.assertEqual(loaded['slice'], data['slice'])

  def test_oob_rejects_truncated_buffers(self):
    stream = io.BytesIO()
    dump_oob(np.arange(10, dtype=np.float32), stream)
    with self.assertRaisesRegex(EOFError, 'Truncated'):
      load_oob(io.BytesIO(stream.getvalue()[:-1]))

  def test_unchunked_rebuild_takes_precedence_over_old_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'model.pkl'
      path.write_bytes(b'new')
      Path(str(path) + '.chunkmanifest').write_text('1')
      chunk = Path(str(path) + '.chunk01of01')
      chunk.write_bytes(b'old')
      with open_file_chunked(str(path)) as stream:
        self.assertEqual(stream.read(), b'new')
      chunk_file(str(path), [str(path) + '.chunkmanifest', str(chunk)])
      with open_file_chunked(str(path)) as stream:
        self.assertEqual(stream.read(), b'new')

  def test_catalog_migration_clears_only_changed_active_chestnut(self):
    source = MODEL_DIR.parent / 'models' / 'fetcher.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fetcher = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelFetcher')
    fetcher.body = [node for node in fetcher.body if isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef) and node.name == '__init__']
    selected = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fetcher], type_ignores=[])
    namespace = {'ModelParser': lambda: None, 'ModelCache': lambda *args, **kwargs: None}
    exec(compile(ast.fix_missing_locations(selected), str(source), 'exec'), namespace)
    fetcher_class = namespace['ModelFetcher']
    values = {
      'ModelManager_ActiveJson': {'qcom': 'v22', 'chestnut': 'v25'},
      'ModelManager_ActiveBundleChestnut': 'old-big',
      'ModelManager_ActiveBundle': 'keep-small',
      'ModelManager_ModelsCache': 'old-cache',
      'ModelManager_ModelsCache_Chestnut': 'old-big-cache',
    }

    class Params:
      def get(self, key):
        return values.get(key)

      def remove(self, key):
        values.pop(key, None)

      def put(self, key, value, block=False):
        values[key] = value

    fetcher_class(Params())
    self.assertNotIn('ModelManager_ActiveBundleChestnut', values)
    self.assertNotIn('ModelManager_ModelsCache', values)
    self.assertNotIn('ModelManager_ModelsCache_Chestnut', values)
    self.assertEqual(values['ModelManager_ActiveBundle'], 'keep-small')
    values['ModelManager_ActiveBundleChestnut'] = 'new-big'
    fetcher_class(Params())
    self.assertEqual(values['ModelManager_ActiveBundleChestnut'], 'new-big')

  def test_bundled_warp_artifacts_have_native_contract(self):
    original = type(Device).__getitem__

    def host_device(instance, key):
      return original(instance, 'PYTHON' if key.split(':')[0] in ('AMD', 'QCOM') else key)

    paths = list((MODEL_DIR / 'models').glob('*driving_warp_*_tinygrad.pkl'))
    self.assertEqual(len(paths), 4)
    with patch.object(type(Device), '__getitem__', host_device):
      for path in paths:
        with self.subTest(path=path.name), path.open('rb') as stream:
          data = pickle.load(stream)
          self.assertIn('run', data)
          self.assertIn('input_specs', data)
          self.assertIn('input_frame', data['input_specs'])


if __name__ == '__main__':
  unittest.main()
