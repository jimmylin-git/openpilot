import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="HCQ2 tinygrad requires Python 3.11+")

SETUP = """
import ast
import base64
import importlib.util
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import onnx
from onnx import TensorProto, helper
import sys
root = Path(sys.argv[1])
import openpilot
openpilot.__path__ = [str(root)]
sys.path.insert(0, str(root / 'dragonpilot/egpu'))
import tg_env
tg_env.check()
from tinygrad import Tensor, Device
from tinygrad.device import Buffer
from tinygrad.dtype import _from_np_dtype
from tinygrad.helpers import round_up
from tinygrad.uop.ops import UOp
from examples.openpilot.helpers import allocate_inputs, load_pickle, dump_pickle
spec = importlib.util.spec_from_file_location('egpu_compile', root / 'dragonpilot/egpu/compile.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)

def save_model(layout, queued_frames=2):
  def value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)
  if layout == 'stateful':
    inputs = [value('new_img', [2, 6, 2, 2], TensorProto.UINT8), value('state_hidden', [1]),
              value('desire', [8]), value('traffic_convention', [1, 2]), value('action_t', [1, 2])]
    nodes = [helper.make_node('Cast', ['new_img'], ['pixels'], to=TensorProto.FLOAT),
             helper.make_node('ReduceMean', ['pixels'], ['mean'], keepdims=0),
             helper.make_node('Add', ['mean', 'state_hidden'], ['next_state_hidden']),
             helper.make_node('Identity', ['next_state_hidden'], ['outputs'])]
    outputs = [value('outputs', [1]), value('next_state_hidden', [1])]
  else:
    inputs = [value('img', [1, 12, 2, 2]), value('big_img', [1, 12, 2, 2]),
              value('features_buffer', [1, queued_frames, 1]), value('desire_pulse', [1, queued_frames, 8]),
              value('traffic_convention', [1, 2]), value('action_t', [1, 2])]
    nodes = [helper.make_node('ReduceMean', [name], [name + '_mean'], keepdims=0) for name in
             ['img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t']]
    names = [name + '_mean' for name in ['img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t']]
    nodes += [helper.make_node('Sum', names, ['total']), helper.make_node('Unsqueeze', ['total'], ['outputs'], axes=[0])]
    outputs = [value('outputs', [1])]
  model = helper.make_model(helper.make_graph(nodes, layout, inputs, outputs), opset_imports=[helper.make_opsetid('', 11)])
  helper.set_model_props(model, {'output_slices': base64.b64encode(pickle.dumps({'hidden_state': slice(0, 1)})).decode()})
  onnx.checker.check_model(model)
  path = Path(layout + '.onnx')
  onnx.save(model, path)
  return str(path)
"""


def run_backend(tmp_path, code, flags=None):
  env = {**os.environ, "DEV": "CPU", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", **(flags or {})}
  result = subprocess.run([sys.executable, "-c", SETUP + textwrap.dedent(code), str(ROOT)], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=180)
  assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("layout", ["stateful", "queued"])
def test_compile_serialize_and_execute_recurrent_model(tmp_path, layout):
  run_backend(tmp_path, f"""
    compiler.compile_model_cmd(save_model({layout!r}), 'model.pkl', 1)
    artifact = load_pickle('model.pkl', out_of_band=True)
    assert artifact['layout'] == {layout!r}
    inputs = allocate_inputs(artifact['input_specs'])
    outputs = allocate_inputs(artifact['output_specs'])
    if {layout!r} == 'stateful':
      outputs.update({{'next_' + name: inputs[name] for name in artifact['state_inputs']}})
    artifact['run'](output_buffers=outputs, **inputs)
    assert np.isfinite(outputs['outputs'].numpy()).all()
    if {layout!r} == 'stateful':
      state = inputs['state_hidden']
      assert outputs['next_state_hidden'] is state
      inputs['new_img'].assign(Tensor.full(inputs['new_img'].shape, 4, dtype='uint8')).realize()
      for expected in (4, 8, 12):
        artifact['run'](output_buffers=outputs, **inputs)
        np.testing.assert_array_equal(outputs['outputs'].numpy(), [expected])
        np.testing.assert_array_equal(state.numpy(), [expected])
    else:
      inputs['new_img'].assign(Tensor.full(inputs['new_img'].shape, 4, dtype='uint8')).realize()
      for _ in range(6):
        artifact['run'](output_buffers=outputs, **inputs)
      assert outputs['outputs'].numpy()[0] == 8
    assert not Path('model.pkl.tmp').exists()
  """)


def test_byte_views_share_storage_and_preserve_dtype(tmp_path):
  run_backend(tmp_path, """
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'input_view']
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<input_view>', 'exec'))
    packed = np.zeros(512, dtype=np.uint8)
    buffer = Tensor(packed, device='NPY')._buffer()
    view = input_view(buffer, (2, 3), _from_np_dtype(np.dtype('float32')), 128)
    np.ndarray((2, 3), dtype=np.float32, buffer=packed, offset=128)[:] = 7
    np.testing.assert_array_equal(view.numpy(), np.full((2, 3), 7))
    assert view._buffer().nbytes == 24
    assert view._buffer().offset == 128
  """)


def test_compile_validation_uses_model_partition_for_temporary_weights(tmp_path):
  run_backend(tmp_path, """
    temporary_directory = compiler.tempfile.TemporaryDirectory
    paths = []
    def record_directory(*args, **kwargs):
      directory = temporary_directory(*args, **kwargs)
      paths.append(Path(directory.name).resolve())
      return directory
    compiler.tempfile.TemporaryDirectory = record_directory
    def make_inputs(seed):
      return (), {'value': Tensor([float(seed)]).realize()}
    compiler.compile_jit(lambda value: (value + 1).realize(), make_inputs, 1, out_of_band=True)
    assert paths and all(path.parent == Path.cwd() for path in paths), paths
    assert all(not path.exists() for path in paths)
  """)


def test_cross_state_feedback_preserves_old_values(tmp_path):
  run_backend(tmp_path, """
    from tinygrad import Context
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    inputs = [helper.make_tensor_value_info('new_img', TensorProto.UINT8, [2, 6, 2, 2]),
              helper.make_tensor_value_info('state_a', TensorProto.FLOAT, [1]),
              helper.make_tensor_value_info('state_b', TensorProto.FLOAT, [1]),
              helper.make_tensor_value_info('desire', TensorProto.FLOAT, [8]),
              helper.make_tensor_value_info('traffic_convention', TensorProto.FLOAT, [1, 2]),
              helper.make_tensor_value_info('action_t', TensorProto.FLOAT, [1, 2])]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
               for name in ('outputs', 'next_state_a', 'next_state_b')]
    nodes = [helper.make_node('Add', ['state_a', 'state_b'], ['outputs']),
             helper.make_node('Identity', ['state_b'], ['next_state_a']),
             helper.make_node('Identity', ['state_a'], ['next_state_b'])]
    model = helper.make_model(helper.make_graph(nodes, 'swap', inputs, outputs), opset_imports=[helper.make_opsetid('', 11)])
    helper.set_model_props(model, {'output_slices': base64.b64encode(pickle.dumps({})).decode()})
    onnx.save(model, 'swap.onnx')
    compiler.compile_model_cmd('swap.onnx', 'model.pkl', 1)
    compiler.compile_warp_cmd(1928, 1208, 'warp.pkl', 1, (4, 4))
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    class RawParser:
      def parse_vision_outputs(self, values): return values
      def parse_policy_outputs(self, values): return values
    namespace = dict(globals(), cloudlog=SimpleNamespace(warning=lambda msg: None, exception=lambda msg: None),
                     Parser=RawParser, ModelConstants=SimpleNamespace(DESIRE_LEN=8),
                     log=SimpleNamespace(ModelDataV2=SimpleNamespace(Action=object)),
                     STATEFUL_FRAME='new_img', DRIVING_OUTPUT='outputs', VISION_INPUTS=('img', 'big_img'),
                     TG_FLAGS_AMD={'DEV': 'CPU'})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<EgpuModelState>', 'exec'), namespace)
    model = namespace['EgpuModelState'](1928, 1208, {}, 'model.pkl', 'warp.pkl')
    model.state['state_a'].assign(Tensor([3.])).realize()
    model.state['state_b'].assign(Tensor([7.])).realize()
    frame = np.zeros(get_nv12_info(1928, 1208)[3], dtype=np.uint8)
    cameras = dict.fromkeys(('img', 'big_img'), frame)
    transforms = dict.fromkeys(('img', 'big_img'), np.eye(3, dtype=np.float32))
    inputs = {'desire_pulse': np.zeros(8), 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)}
    original_buffers = {name: model.inputs[name]._buffer() for name in model.state_names}
    for a, b in ((7, 3), (3, 7), (7, 3), (3, 7)):
      model.run(cameras, transforms, inputs)
      assert all(model.inputs[name]._buffer() is original_buffers[name] for name in model.state_names)
      assert all(model.outputs['next_' + name] is model.inputs[name] for name in model.state_names)
      np.testing.assert_array_equal(model.outputs['outputs'].numpy(), [10])
      np.testing.assert_array_equal(model.inputs['state_a'].numpy(), [a])
      np.testing.assert_array_equal(model.inputs['state_b'].numpy(), [b])
    assert all(model.inputs[name]._buffer() is original_buffers[name] for name in model.state_names)
    model.run(cameras, transforms, inputs)
    model.reset()
    model.run(cameras, transforms, inputs)
    assert all(not tensor.numpy().any() for tensor in model.state.values())
  """)


@pytest.mark.parametrize("image_optimizations", [False, True])
def test_out_of_band_weights_survive_reload(tmp_path, image_optimizations):
  run_backend(tmp_path, """
    from tinygrad import Context
    from tinygrad.helpers import IMAGE
    columns = (np.arange(1024, dtype=np.float32) % 32) * .5
    weights = Tensor(np.tile(columns, (1024, 1))).realize()
    specs = {'value': ((1, 1024), 'float32', Device.DEFAULT)}
    def make_inputs(seed):
      return (), allocate_inputs(specs, lambda values: values['value'].fill(seed)) | {'output': allocate_inputs(specs)['value']}
    def run(value, output):
      output.assign(value @ weights).realize()
    jit = compiler.compile_jit(run, make_inputs, 1, out_of_band=True)
    compiler._atomic_dump({'run': jit}, 'weights.pkl', dump_pickle, True)
    assert Path('weights.pkl').stat().st_size > weights.numel() * (2 if IMAGE else weights.dtype.itemsize)
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EgpuModelState')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    load = next(node for node in init.body if isinstance(node, ast.With) and
                any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'load_pickle'
                    for call in ast.walk(node)))
    namespace = dict(globals(), model_pkl='weights.pkl', TG_FLAGS_AMD={'DEV': 'CPU'})
    with Context(DEV='NPY'):
      exec(compile(ast.Module(body=[load], type_ignores=[]), '<model weight placement>', 'exec'), namespace)
      assert Device.DEFAULT == 'NPY'
    loaded = namespace['art']
    _, inputs = make_inputs(17)
    loaded['run'](**inputs)
    np.testing.assert_array_equal(inputs['output'].numpy(), columns.reshape(1, 1024) * (17 * 1024))
  """, {"IMAGE": str(int(image_optimizations)), "OPENPILOT_HACKS": str(int(image_optimizations)), "FLOAT16": "1"})


@pytest.mark.parametrize("dependency", ["chain", "three_cycle", "dense", "reverse"])
def test_in_place_state_dependency_graphs(tmp_path, dependency):
  run_backend(tmp_path, f"""
    from onnx import numpy_helper
    inputs = [helper.make_tensor_value_info('new_img', TensorProto.UINT8, [2, 6, 2, 2])]
    inputs += [helper.make_tensor_value_info('state_' + name, TensorProto.FLOAT, [4]) for name in 'abc']
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [4])
               for name in ('outputs', 'next_state_a', 'next_state_b', 'next_state_c')]
    dependency = {dependency!r}
    nodes = [helper.make_node('Add', ['state_a', 'state_b'], ['outputs'])]
    if dependency == 'dense':
      for target, sources in zip('abc', [('state_b', 'state_c'), ('state_a', 'state_c'), ('state_a', 'state_b')]):
        nodes.append(helper.make_node('Add', list(sources), ['next_state_' + target]))
    elif dependency == 'reverse':
      nodes += [helper.make_node('Gather', ['state_' + name, 'indices'], ['next_state_' + name], axis=0) for name in 'abc']
    else:
      sources = 'bca' if dependency == 'three_cycle' else 'bcc'
      nodes += [helper.make_node('Identity', ['state_' + source], ['next_state_' + target])
                for target, source in zip('abc', sources)]
    graph = helper.make_graph(nodes, 'dependencies', inputs, outputs,
                              initializer=[numpy_helper.from_array(np.arange(3, -1, -1, dtype=np.int64), 'indices')]
                              if dependency == 'reverse' else [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    helper.set_model_props(model, {{'output_slices': base64.b64encode(pickle.dumps({{}})).decode()}})
    onnx.save(model, 'dependencies.onnx')
    compiler.compile_model_cmd('dependencies.onnx', 'model.pkl', 1)
    artifact = load_pickle('model.pkl', out_of_band=True)
    inputs = allocate_inputs(artifact['input_specs'])
    output_specs = {{name: spec for name, spec in artifact['output_specs'].items() if name == 'outputs'}}
    outputs = allocate_inputs(output_specs)
    outputs.update({{'next_' + name: inputs[name] for name in artifact['state_inputs']}})
    expected = [np.arange(4, dtype=np.float32) + offset for offset in (1, 10, 100)]
    for name, value in zip('abc', expected):
      inputs['state_' + name].assign(Tensor(value)).realize()
    for _ in range(5):
      old_a, old_b, old_c = [value.copy() for value in expected]
      artifact['run'](output_buffers=outputs, **inputs)
      np.testing.assert_array_equal(outputs['outputs'].numpy(), old_a + old_b)
      if dependency == 'dense': expected = [old_b + old_c, old_a + old_c, old_a + old_b]
      elif dependency == 'reverse': expected = [value[::-1] for value in (old_a, old_b, old_c)]
      elif dependency == 'three_cycle': expected = [old_b, old_c, old_a]
      else: expected = [old_b, old_c, old_c]
      for name, value in zip('abc', expected):
        np.testing.assert_array_equal(inputs['state_' + name].numpy(), value)
  """)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_in_place_state_requires_matching_output_spec(tmp_path, mismatch):
  run_backend(tmp_path, f"""
    inputs = [helper.make_tensor_value_info('new_img', TensorProto.UINT8, [2, 6, 2, 2]),
              helper.make_tensor_value_info('state_hidden', TensorProto.FLOAT, [2])]
    mismatch = {mismatch!r}
    node = (helper.make_node('ReduceSum', ['state_hidden'], ['next_state_hidden'], keepdims=0)
            if mismatch == 'shape' else helper.make_node('Cast', ['state_hidden'], ['next_state_hidden'], to=TensorProto.INT32))
    outputs = [helper.make_tensor_value_info('next_state_hidden', TensorProto.FLOAT if mismatch == 'shape' else TensorProto.INT32,
                                            [] if mismatch == 'shape' else [2])]
    model = helper.make_model(helper.make_graph([node], 'invalid_state', inputs, outputs), opset_imports=[helper.make_opsetid('', 11)])
    helper.set_model_props(model, {{'output_slices': base64.b64encode(pickle.dumps({{}})).decode()}})
    onnx.save(model, 'invalid.onnx')
    try:
      compiler.compile_model_cmd('invalid.onnx', 'model.pkl', 1)
    except ValueError as error:
      assert 'state input/output specification mismatch: state_hidden' in str(error)
    else:
      raise AssertionError('unsafe state alias was accepted')
    assert not Path('model.pkl').exists()
  """)


def test_queued_full_rate_history(tmp_path):
  run_backend(tmp_path, """
    compiler.compile_model_cmd(save_model('queued', 99), 'model.pkl', 1)
    artifact = load_pickle('model.pkl', out_of_band=True)
    assert artifact['frame_skip'] == 1
    assert artifact['input_specs']['feat_q'][0] == (99, 1, 1)
    inputs = allocate_inputs(artifact['input_specs'])
    outputs = allocate_inputs(artifact['output_specs'])
    inputs['new_img'].assign(Tensor.full(inputs['new_img'].shape, 4, dtype='uint8')).realize()
    for expected in (4, 8, 8):
      artifact['run'](output_buffers=outputs, **inputs)
      np.testing.assert_array_equal(outputs['outputs'].numpy(), [expected])
  """)


@pytest.mark.parametrize("camera", [(1928, 1208), (1344, 760)])
def test_warp_roundtrip_and_external_camera_mapping(tmp_path, camera):
  run_backend(tmp_path, f"""
    compiler.compile_warp_cmd(*{camera!r}, 'warp.pkl', 1)
    with open('warp.pkl', 'rb') as f:
      artifact = pickle.load(f)
    shape, dtype, device = artifact['input_specs']['input_frame']
    frame = np.full(shape, 73, dtype=np.uint8)
    camera = Tensor.from_blob(frame.ctypes.data, shape, dtype=np.dtype(dtype).name, device=device)
    transform = Tensor(np.eye(3, dtype=np.float32), device='NPY')
    for expected in (73, 19):
      frame[:] = expected
      warped = artifact['run'](input_frame=camera, M_inv=transform)
      np.testing.assert_array_equal(warped.numpy(), np.full((6, 128, 256), expected, dtype=np.uint8))
    assert artifact['metadata']['camera'] == {camera!r}
    assert artifact['input_specs']['M_inv'][2] == 'NPY'
  """)


@pytest.mark.parametrize("layout", ["stateful", "queued"])
@pytest.mark.parametrize("default_device", ["CPU", "NPY"])
def test_runner_warmup_camera_frames_feedback_and_reset(tmp_path, layout, default_device):
  run_backend(tmp_path, f"""
    from tinygrad import Context
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    compiler.compile_model_cmd(save_model({layout!r}), 'model.pkl', 1)
    compiler.compile_warp_cmd(1928, 1208, 'warp.pkl', 1, (4, 4))
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    class RawParser:
      def parse_vision_outputs(self, values): return values
      def parse_policy_outputs(self, values): return values
    def load_model(*args, **kwargs):
      assert Device.DEFAULT == 'CPU', 'model weights must not load onto the default warp device'
      return load_pickle(*args, **kwargs)
    namespace = dict(globals(), cloudlog=SimpleNamespace(warning=lambda msg: None, exception=lambda msg: None),
                     Parser=RawParser, ModelConstants=SimpleNamespace(DESIRE_LEN=8),
                     log=SimpleNamespace(ModelDataV2=SimpleNamespace(Action=object)),
                     STATEFUL_FRAME='new_img', DRIVING_OUTPUT='outputs', VISION_INPUTS=('img', 'big_img'),
                     TG_FLAGS_AMD={{'DEV': 'CPU'}}, load_pickle=load_model)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<EgpuModelState>', 'exec'), namespace)
    with Context(DEV={default_device!r}):
      model = namespace['EgpuModelState'](1928, 1208, {{'generation': 11}}, 'model.pkl', 'warp.pkl')
    model.warmup()
    assert not model._warming_up and model._first_frame
    frame = np.full(get_nv12_info(1928, 1208)[3], 4, dtype=np.uint8)
    cameras = dict.fromkeys(('img', 'big_img'), frame)
    transforms = dict.fromkeys(('img', 'big_img'), np.eye(3, dtype=np.float32))
    inputs = {{'desire_pulse': np.zeros(8, dtype=np.float32), 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)}}
    results = [model.run(cameras, transforms, inputs)['hidden_state'][0, 0] for _ in range(10)]
    if {layout!r} == 'stateful':
      expected = np.arange(1, 11) * 4
    else:
      image_history, feature_history = np.zeros(5), np.zeros(8)
      previous, expected = 0, []
      for _ in results:
        image_history = np.append(image_history[1:], 4)
        feature_history = np.append(feature_history[1:], previous)
        previous = 2 * image_history[::4].mean() + feature_history[::4].mean()
        expected.append(previous)
    np.testing.assert_array_equal(results, expected)
    assert not model._first_frame
    assert len(model._blob_cache) == 2
    model.reset()
    assert not model._blob_cache
    assert not model.packed.any()
    assert all(not tensor.numpy().any() for tensor in model.state.values())
    assert model.run(cameras, transforms, inputs)['hidden_state'][0, 0] == results[0]
  """)


def test_runtime_target_is_configured_before_tinygrad_import(tmp_path):
  code = """
import ast
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'dragonpilot/egpu'))
tree = ast.parse((root / 'dragonpilot/egpu/modeld_egpu.py').read_text())
header = []
for node in tree.body:
  if isinstance(node, ast.ImportFrom) and node.module == 'setproctitle':
    break
  header.append(node)
exec(compile(ast.Module(body=header, type_ignores=[]), '<runtime setup>', 'exec'))
from tinygrad.helpers import DEV
assert DEV.device == 'QCOM', repr(DEV)
assert DEV.target('AMD').interface == 'USB', repr(DEV)
assert DEV.target('AMD').renderer == 'LLVM', repr(DEV)
"""
  env = {k: v for k, v in os.environ.items() if k != "DEV"}
  result = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
  assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_device_error_propagates_before_readback(tmp_path):
  run_backend(tmp_path, """
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EgpuModelState')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_run')
    events = []
    class Devices:
      def __getitem__(self, name):
        return SimpleNamespace(error_state=SimpleNamespace(host=SimpleNamespace(view=lambda **kwargs: [-7])))
    namespace = {'np': np, 'VISION_INPUTS': (), 'Device': Devices()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<runtime error>', 'exec'), namespace)
    model = SimpleNamespace(prev_desire=np.zeros(8), npy={'desire': np.zeros(8), 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)},
                            _set_stage=events.append, input_host=None, input_device=SimpleNamespace(copy_from=lambda host: None),
                            inputs={}, outputs={}, model_device='AMD', run_model=lambda **kwargs: None)
    try:
      namespace['_run'](model, {}, {}, {'desire_pulse': np.zeros(8), 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)})
    except RuntimeError as error:
      assert 'failed with -7' in str(error)
    else:
      raise AssertionError('device failure was swallowed')
    assert events == ['AMD input upload', 'AMD inference']
  """)


def test_vendor_usb_ids_and_cache_version(tmp_path):
  run_backend(tmp_path, """
    from dragonpilot.egpu.constants import CACHE_TAG, VENDORED_TINYGRAD_REF
    assert CACHE_TAG == 'tg02e69816v4'
    assert VENDORED_TINYGRAD_REF in (root / 'dragonpilot/egpu/tinygrad_spm/VERSION').read_text()
    tree = ast.parse((root / 'dragonpilot/egpu/tinygrad_spm/tinygrad/runtime/ops_amd.py').read_text())
    iface = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'USBIface')
    init = next(node for node in iface.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    guard = next(node for node in init.body if isinstance(node, ast.If))
    ids = [(0xADD1, 0x0001), (0x3801, 0x0001), (0xADD1, 0x0002)]
    for supported in ids:
      namespace = {'dev_id': 0, 'USB3': SimpleNamespace(list_devices=lambda vendor, product: ['gpu'] if (vendor, product) == supported else []),
                   'filter_visible_devices': lambda devices, name: devices, 'pluralize': lambda *args: 'devices'}
      exec(compile(ast.Module(body=[guard], type_ignores=[]), '<USB enumeration>', 'exec'), namespace)
      assert namespace['visible'] == ['gpu']
  """)
