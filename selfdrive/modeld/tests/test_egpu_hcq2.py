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


def run_backend(tmp_path, code):
  env = {**os.environ, "DEV": "CPU", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
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
    artifact['run'](output_buffers=outputs, **inputs)
    assert np.isfinite(outputs['outputs'].numpy()).all()
    if {layout!r} == 'stateful':
      state = inputs['state_hidden']
      outputs['next_state_hidden'] = Tensor(UOp.from_buffer(state._buffer(), state.dtype)).reshape(state.shape)
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


def test_out_of_band_weights_survive_reload(tmp_path):
  run_backend(tmp_path, """
    weights = Tensor(np.full((1024, 1024), .5, dtype=np.float32)).realize()
    specs = {'value': ((1, 1024), 'float32', Device.DEFAULT)}
    def make_inputs(seed):
      return (), allocate_inputs(specs, lambda values: values['value'].fill(seed)) | {'output': allocate_inputs(specs)['value']}
    def run(value, output):
      output.assign(value @ weights).realize()
    jit = compiler.compile_jit(run, make_inputs, 1, out_of_band=True)
    compiler._atomic_dump({'run': jit}, 'weights.pkl', dump_pickle, True)
    assert Path('weights.pkl').stat().st_size > weights.numel() * weights.dtype.itemsize
    loaded = load_pickle('weights.pkl', out_of_band=True)
    _, inputs = make_inputs(17)
    loaded['run'](**inputs)
    np.testing.assert_array_equal(inputs['output'].numpy(), np.full((1, 1024), 17 * 1024 * .5))
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


def test_warp_roundtrip_and_external_camera_mapping(tmp_path):
  run_backend(tmp_path, """
    compiler.compile_warp_cmd(1928, 1208, 'warp.pkl', 1, (8, 8))
    with open('warp.pkl', 'rb') as f:
      artifact = pickle.load(f)
    shape, dtype, device = artifact['input_specs']['input_frame']
    frame = np.full(shape, 73, dtype=np.uint8)
    camera = Tensor.from_blob(frame.ctypes.data, shape, dtype=np.dtype(dtype).name, device=device)
    transform = Tensor(np.eye(3, dtype=np.float32), device='NPY')
    for expected in (73, 19):
      frame[:] = expected
      warped = artifact['run'](input_frame=camera, M_inv=transform)
      np.testing.assert_array_equal(warped.numpy(), np.full((6, 4, 4), expected, dtype=np.uint8))
    assert artifact['metadata']['camera'] == (1928, 1208)
    assert artifact['input_specs']['M_inv'][2] == 'NPY'
  """)


@pytest.mark.parametrize("layout", ["stateful", "queued"])
def test_runner_warmup_camera_frames_feedback_and_reset(tmp_path, layout):
  run_backend(tmp_path, f"""
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    compiler.compile_model_cmd(save_model({layout!r}), 'model.pkl', 1)
    compiler.compile_warp_cmd(1928, 1208, 'warp.pkl', 1, (4, 4))
    tree = ast.parse((root / 'dragonpilot/egpu/runner.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    class RawParser:
      def parse_vision_outputs(self, values): return values
      def parse_policy_outputs(self, values): return values
    namespace = dict(globals(), cloudlog=SimpleNamespace(warning=lambda msg: None, exception=lambda msg: None),
                     Parser=RawParser, ModelConstants=SimpleNamespace(DESIRE_LEN=8),
                     log=SimpleNamespace(ModelDataV2=SimpleNamespace(Action=object)),
                     STATEFUL_FRAME='new_img', DRIVING_OUTPUT='outputs', VISION_INPUTS=('img', 'big_img'))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<EgpuModelState>', 'exec'), namespace)
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
    assert CACHE_TAG == 'tg02e69816v2'
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
