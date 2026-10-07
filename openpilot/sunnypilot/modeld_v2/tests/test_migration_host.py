import ast
import codecs
import contextlib
import enum
import os
import pickle
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock, skipIf

import numpy as np

try:
  import onnx
except ModuleNotFoundError as exc:
  if exc.name != 'onnx':
    raise
  onnx = None
from tinygrad import Device, Tensor, TinyJit

from openpilot.selfdrive.modeld.compile_native import compile_model, compile_warp, make_warp, NV12Frame
from openpilot.sunnypilot.modeld_v2.helpers import _dynamic_factory, _patch_system_flock_acquire, load_oob, load_pickle
from openpilot.sunnypilot.modeld_v2.model_adapters import get_model_adapter


ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, namespace):
  tree = ast.parse(path.read_text(encoding='utf-8'))
  cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
  module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
  return namespace[name]


def stock_state(artifact, directory):
  namespace = {
    'np': np,
    'load_oob': load_oob,
    'load_pickle': load_pickle,
    'open_file_chunked': lambda path: open(path, 'rb'),
    'modeld_pkl_path': lambda _: artifact,
    'MODELS_DIR': directory,
    'get_model_adapter': get_model_adapter,
    'ModelStateBase': object,
    'ModelConstants': SimpleNamespace(DESIRE_LEN=8),
    'Parser': lambda: SimpleNamespace(parse_outputs=lambda v: v),
    'SEND_RAW_PRED': False,
  }
  return load_class(ROOT / 'selfdrive' / 'modeld' / 'modeld.py', 'ModelState', namespace)(8, 8, True)


def driver_state(artifact, directory):
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  logger = mock.Mock()
  namespace = {
    'np': np,
    'Tensor': Tensor,
    'time': __import__('time'),
    'pickle': pickle,
    'base64': __import__('base64'),
    'json': __import__('json'),
    'load_oob': load_oob,
    'load_pickle': load_pickle,
    'model_file_exists': lambda path: path.is_file(),
    'open_file_chunked': lambda path: open(path, 'rb'),
    'MODEL_PKL_PATH': artifact,
    'MODELS_DIR': directory,
    'get_nv12_info': get_nv12_info,
    'cloudlog': logger,
  }
  cls = load_class(ROOT / 'selfdrive' / 'modeld' / 'dmonitoringmodeld.py', 'ModelState', namespace)
  return cls(8, 8), logger


class PickleCompatibilityTests(TestCase):
  def test_extra_and_missing_constructor_arguments_logged(self):
    class Value:
      def __init__(self, value, optional=7):
        self.value, self.optional = value, optional

    proxy = _dynamic_factory(Value)
    with self.assertLogs('openpilot.sunnypilot.modeld_v2.helpers', level='WARNING'):
      value = proxy(1, 2, 3)
    self.assertEqual((value.value, value.optional), (1, 2))
    with self.assertLogs('openpilot.sunnypilot.modeld_v2.helpers', level='WARNING'):
      value = proxy()
    self.assertEqual((value.value, value.optional), (None, 7))

  def test_internal_type_error_not_masked(self):
    class Value:
      def __init__(self, value):
        raise TypeError('internal constructor error')

    with self.assertRaisesRegex(TypeError, 'internal constructor error'):
      _dynamic_factory(Value)(1)

  def test_keyword_only_padding_and_nonclass_symbols(self):
    class Value:
      def __init__(self, value, *, required):
        self.value, self.required = value, required

    with self.assertLogs('openpilot.sunnypilot.modeld_v2.helpers', level='WARNING'):
      value = _dynamic_factory(Value)(1)
    self.assertEqual((value.value, value.required), (1, None))

    class Kind(enum.Enum):
      ONE = 1

    self.assertIs(_dynamic_factory(Kind), Kind)
    self.assertIs(_dynamic_factory(stock_state), stock_state)

  def test_system_lock_reuses_acquisition_not_owned_descriptors(self):
    from tinygrad.runtime.support.system import System

    with tempfile.TemporaryFile() as stream:
      root = os.dup(stream.fileno())
      original = mock.Mock(return_value=root)
      original._model_pickle_compat = False
      with mock.patch.object(System, 'flock_acquire', original):
        _patch_system_flock_acquire()
        one, two = System.flock_acquire('fixture'), System.flock_acquire('fixture')
        try:
          original.assert_called_once_with('fixture')
          self.assertNotEqual(one, two)
          os.close(one)
          os.fstat(two)
        finally:
          os.close(two)
          os.close(root)

  def test_lock_acquisition_failure_is_not_cached(self):
    from tinygrad.runtime.support.system import System

    with tempfile.TemporaryFile() as stream:
      root = os.dup(stream.fileno())
      original = mock.Mock(side_effect=[OSError('lock acquisition failed'), root])
      original._model_pickle_compat = False
      with mock.patch.object(System, 'flock_acquire', original):
        _patch_system_flock_acquire()
        with self.assertRaisesRegex(OSError, 'lock acquisition failed'):
          System.flock_acquire('fixture')
        descriptor = System.flock_acquire('fixture')
        os.close(descriptor)
        descriptor = System.flock_acquire('fixture')
        self.assertEqual(original.call_count, 2)
        os.fstat(descriptor)
        os.close(descriptor)
        os.close(root)


@skipIf(onnx is None, "Synthetic ONNX fixtures require the optional onnx package")
class NativeCompilerTests(TestCase):
  def model(self, path, image=False):
    shapes = {'input_img': [1, 64], 'calib': [1, 3]} if image else {'new_img': [1, 4], 'state': [1, 4]}
    inputs = [
      onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UINT8 if image and name == 'input_img' else onnx.TensorProto.FLOAT, shape)
      for name, shape in shapes.items()
    ]
    if image:
      nodes = [
        onnx.helper.make_node('Cast', ['input_img'], ['pixels'], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node('ReduceSum', ['pixels'], ['pixel_sum'], keepdims=1),
        onnx.helper.make_node('ReduceSum', ['calib'], ['calib_sum'], keepdims=1),
        onnx.helper.make_node('Add', ['pixel_sum', 'calib_sum'], ['outputs']),
      ]
      outputs = [onnx.helper.make_tensor_value_info('outputs', onnx.TensorProto.FLOAT, [1, 1])]
    else:
      nodes = [onnx.helper.make_node('Add', ['new_img', 'state'], ['outputs']), onnx.helper.make_node('Identity', ['outputs'], ['next_state'])]
      outputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 4]) for name in ('outputs', 'next_state')]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, 'fixture', inputs, outputs), opset_imports=[onnx.helper.make_opsetid('', 13)])
    model.metadata_props.add(key='output_slices', value=codecs.encode(pickle.dumps({'hidden_state': slice(0, 4)}), 'base64').decode())
    onnx.save(model, path)

  def test_actual_onnx_compilation_reloaded_output_buffers(self):
    with tempfile.TemporaryDirectory() as directory:
      model, artifact = Path(directory) / 'fixture.onnx', Path(directory) / 'fixture.pkl'
      self.model(model)
      compile_model(str(model), str(artifact))
      with artifact.open('rb') as stream:
        data = load_oob(stream)
      self.assertEqual(set(data), {'run', 'metadata', 'input_specs', 'output_specs'})
      outputs = {name: Tensor.zeros(*shape, device=device, dtype=dtype).realize() for name, (shape, dtype, device) in data['output_specs'].items()}
      state = Tensor.zeros(1, 4, device=Device.DEFAULT).realize()
      for value in (2, 4, 8):
        data['run'](new_img=Tensor.full((1, 4), value, dtype='float32', device=Device.DEFAULT).realize(), state=state, output_buffers=outputs)
        np.testing.assert_array_equal(outputs['outputs'].numpy(), np.full((1, 4), value, dtype=np.float32))

  def test_actual_luma_and_yuv_warp_artifact_contracts(self):
    with tempfile.TemporaryDirectory() as directory:
      for layout, frames in (('luma', 1), ('yuv420', 2)):
        artifact = Path(directory) / f'{layout}.pkl'
        compile_warp((8, 8, 8, 8, 4, 96), (8, 8), layout, frames, 'NPY', 16, str(artifact))
        with artifact.open('rb') as stream:
          data = load_pickle(stream)
        shape = data['input_specs']['input_frame'][0]
        tfm_shape = data['input_specs']['M_inv'][0]
        result = data['run'](
          input_frame=Tensor(np.full(shape, 7, dtype=np.uint8), device=Device.DEFAULT).realize(),
          M_inv=Tensor(np.broadcast_to(np.eye(3, dtype=np.float32), tfm_shape).copy(), device='NPY').realize(),
        )
        self.assertEqual(result.shape, (1, 64) if layout == 'luma' else (2, 6, 4, 4))
        np.testing.assert_array_equal(result.numpy(), np.full(result.shape, 7, dtype=np.uint8))

  def test_driver_loader_updates_image_and_calibration_after_reload(self):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      model, artifact = directory / 'dm.onnx', directory / 'dm.pkl'
      self.model(model, image=True)
      compile_model(str(model), str(artifact))
      compile_warp((8, 8, *get_nv12_info(8, 8)), (8, 8), 'luma', 1, 'NPY', 16, str(directory / 'dm_warp_8x8_native.pkl'))
      state, logger = driver_state(artifact, directory)
      logger.warning.assert_not_called()
      frame = np.full(get_nv12_info(8, 8)[3], 2, dtype=np.uint8)
      for value in (1, 3, 5):
        output, _ = state.run(SimpleNamespace(data=frame.data), np.full(3, value, dtype=np.float32), np.eye(3, dtype=np.float32))
        np.testing.assert_array_equal(output, [128 + 3 * value])

  def test_bundled_legacy_driver_artifacts_remain_loadable(self):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    import json

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)

      def run(input_img, calib):
        return (input_img.cast('float32').sum() + calib.to(Device.DEFAULT).sum()).reshape(1, 1).realize()

      model = TinyJit(run, prune=True)
      native_warp = make_warp(NV12Frame(8, 8, *get_nv12_info(8, 8)), (8, 8), 'luma', 1, 16)
      warp = TinyJit(lambda frame, tfm: native_warp(frame, tfm), prune=True)
      frame = Tensor(np.full(get_nv12_info(8, 8)[3], 2, dtype=np.uint8), device=Device.DEFAULT).realize()
      transform = Tensor(np.eye(3, dtype=np.float32), device='NPY').realize()
      calib = Tensor(np.ones((1, 3), dtype=np.float32), device='NPY').realize()
      for _ in range(3):
        image = warp(frame, transform)
        model(input_img=image, calib=calib)
      metadata = {'input_shapes': {'input_img': (1, 64), 'calib': (1, 3)}, 'output_slices': {'hidden_state': slice(0, 1)}}
      for name, value in [('dmonitoring_model_tinygrad.pkl', model), ('dmonitoring_model_metadata.pkl', metadata), ('dm_warp_8x8_tinygrad.pkl', warp)]:
        with (directory / name).open('wb') as stream:
          pickle.dump(value, stream, protocol=5)
      (directory / 'tg_input_devices.json').write_text(json.dumps({'openpilot.selfdrive.modeld.dmonitoringmodeld': {'default': {'DEV': Device.DEFAULT}}}))
      state, logger = driver_state(directory / 'missing_native.pkl', directory)
      self.assertFalse(state.native)
      logger.warning.assert_called_once()
      frame = np.full(get_nv12_info(8, 8)[3], 2, dtype=np.uint8)
      for value in (1, 3, 5):
        output, _ = state.run(SimpleNamespace(data=frame.data), np.full(3, value, dtype=np.float32), np.eye(3, dtype=np.float32))
        np.testing.assert_array_equal(output, [128 + 3 * value])

  def test_malformed_native_driver_artifact_does_not_fall_back(self):
    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      artifact = directory / 'native.pkl'
      artifact.write_bytes(b'bad')
      with self.assertRaisesRegex(EOFError, 'Truncated'):
        driver_state(artifact, directory)

  def test_stock_legacy_compile_reload_and_temporal_controls(self):
    with tempfile.TemporaryDirectory() as directory:
      model, artifact = Path(directory) / 'stock.onnx', Path(directory) / 'stock.pkl'
      shapes = {
        'img': [1, 12, 2, 2],
        'big_img': [1, 12, 2, 2],
        'features_buffer': [1, 2, 4],
        'desire_pulse': [1, 3, 8],
        'traffic_convention': [1, 2],
        'action_t': [1, 2],
      }
      inputs = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UINT8 if 'img' in name else onnx.TensorProto.FLOAT, shape) for name, shape in shapes.items()
      ]
      nodes = [
        onnx.helper.make_node('Add', ['traffic_convention', 'action_t'], ['controls']),
        onnx.helper.make_node('Concat', ['controls', 'controls'], ['outputs'], axis=1),
      ]
      output = onnx.helper.make_tensor_value_info('outputs', onnx.TensorProto.FLOAT, [1, 4])
      fixture = onnx.helper.make_model(onnx.helper.make_graph(nodes, 'fixture', inputs, [output]), opset_imports=[onnx.helper.make_opsetid('', 13)])
      fixture.metadata_props.add(key='output_slices', value=codecs.encode(pickle.dumps({'hidden_state': slice(0, 4)}), 'base64').decode())
      onnx.save(fixture, model)
      compile_model(str(model), str(artifact), camera_resolutions=[(8, 8)])
      with artifact.open('rb') as stream:
        data = load_oob(stream)
      adapter = get_model_adapter(data, 8, 8, Device.DEFAULT, Device.DEFAULT, Device.DEFAULT)
      frame = np.zeros(adapter.frame_copy_size, dtype=np.uint8)
      adapter.copy_frames({'img': frame, 'big_img': frame})
      adapter.numpy_inputs['tfm'][:] = np.eye(3)
      adapter.numpy_inputs['big_tfm'][:] = np.eye(3)
      for value in (2, 4, 8):
        adapter.numpy_inputs['traffic_convention'][:] = value
        adapter.numpy_inputs['action_t'][:] = 1
        np.testing.assert_array_equal(adapter.run().numpy(), np.full((1, 4), value + 1, dtype=np.float32))
      state = stock_state(artifact, Path(directory))
      state.warmup()
      frames = dict.fromkeys(state.vision_input_names, frame)
      transforms = dict.fromkeys(state.vision_input_names, np.eye(3, dtype=np.float32))
      inputs = {'desire_pulse': np.zeros(8, dtype=np.float32), 'traffic_convention': np.ones(2, dtype=np.float32), 'action_t': np.full(2, 2, dtype=np.float32)}
      result = state.run(frames, transforms, inputs)
      np.testing.assert_array_equal(result['hidden_state'], np.full((1, 4), 3, dtype=np.float32))
      np.testing.assert_array_equal(state.npy['prev_feat'], np.full((1, 4), 3, dtype=np.float32))
      inputs['action_t'][:] = float('nan')
      with self.assertRaisesRegex(RuntimeError, 'not finite'):
        state.run(frames, transforms, inputs)

  def test_stock_native_cameras_recurrent_state_and_warmup(self):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      model, artifact = directory / 'native.onnx', directory / 'native.pkl'
      shapes = {'new_img': [2, 6, 2, 2], 'state': [1, 4], 'desire': [8], 'traffic_convention': [2], 'action_t': [2]}
      inputs = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.UINT8 if name == 'new_img' else onnx.TensorProto.FLOAT, shape)
        for name, shape in shapes.items()
      ]
      nodes = [
        onnx.helper.make_node('Cast', ['new_img'], ['pixels'], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node('ReduceSum', ['pixels'], ['pixel_sum'], keepdims=0),
        onnx.helper.make_node('ReduceSum', ['action_t'], ['action_sum'], keepdims=0),
        onnx.helper.make_node('Add', ['pixel_sum', 'action_sum'], ['signal']),
        onnx.helper.make_node('Add', ['state', 'signal'], ['outputs']),
        onnx.helper.make_node('Add', ['state', 'one'], ['next_state']),
      ]
      outputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 4]) for name in ('outputs', 'next_state')]
      fixture = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, 'fixture', inputs, outputs, initializer=[onnx.numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), 'one')]),
        opset_imports=[onnx.helper.make_opsetid('', 13)],
      )
      fixture.metadata_props.add(key='output_slices', value=codecs.encode(pickle.dumps({'hidden_state': slice(0, 4)}), 'base64').decode())
      onnx.save(fixture, model)
      compile_model(str(model), str(artifact))
      nv12 = get_nv12_info(8, 8)
      frame_size = nv12[0] * (nv12[1] + nv12[2])
      compile_warp((8, 8, *nv12[:3], frame_size), (4, 4), 'yuv420', 2, Device.DEFAULT, 16, str(directory / 'big_driving_warp_8x8_tinygrad.pkl'))
      state = stock_state(artifact, directory)
      self.assertEqual(state.vision_input_names, ('img', 'big_img'))
      self.assertEqual(state.frame_copy_size, frame_size)
      state.warmup()
      transforms = dict.fromkeys(state.vision_input_names, np.eye(3, dtype=np.float32))
      controls = {'desire_pulse': np.zeros(8, dtype=np.float32), 'traffic_convention': np.zeros(2, dtype=np.float32), 'action_t': np.ones(2, dtype=np.float32)}
      for index, value in enumerate((2, 4, 8)):
        frames = {'img': np.full(frame_size, value, dtype=np.uint8), 'big_img': np.full(frame_size, value + 1, dtype=np.uint8)}
        result = state.run(frames, transforms, controls)
        np.testing.assert_array_equal(result['hidden_state'], np.full((1, 4), 24 * (2 * value + 1) + 2 + index, dtype=np.float32))
      state.warmup()
      np.testing.assert_array_equal(state.adapter.input_queues['state'].numpy(), np.zeros((1, 4), dtype=np.float32))


class ChestnutMonitoringTests(TestCase):
  def test_gpu_telemetry_marks_failures_invalid_and_recovers(self):
    messages = []
    smu = SimpleNamespace(
      _send_msg=mock.Mock(side_effect=RuntimeError('GPU read failed')),
      smu_mod=SimpleNamespace(TABLE_SMU_METRICS=0, PPSMC_MSG_TransferTableSmu2Dram=1, PPSMC_MSG_GetPptLimit=2, TEMP_HOTSPOT=0, TEMP_MEM=1),
      driver_table_paddr=0,
      adev=SimpleNamespace(vram=SimpleNamespace(view=lambda *args: bytes(1))),
    )
    metrics = SimpleNamespace(AvgTemperature=[50, 40], AverageSocketPower=10, AverageGfxActivity=20, AverageGfxclkFrequencyPostDs=1000, AvgFanRpm=1500)
    smu.smu_mod.SmuMetricsExternal_t = SimpleNamespace(from_buffer=lambda _: SimpleNamespace(SmuMetrics=metrics))

    class Devices(dict):
      _opened_devices = {'AMD'}

    devices = Devices(AMD=SimpleNamespace(iface=SimpleNamespace(dev_impl=SimpleNamespace(smu=smu))))
    namespace = {
      'Device': devices,
      'ctypes': SimpleNamespace(sizeof=lambda _: 1),
      'cached_property': __import__('functools').cached_property,
      'messaging': SimpleNamespace(new_message=lambda _: SimpleNamespace(valid=True, chestnutGpuState=SimpleNamespace())),
      'cloudlog': mock.Mock(),
    }
    cls = load_class(ROOT / 'selfdrive' / 'modeld' / 'modeld.py', 'ChestnutGpuState', namespace)
    telemetry = cls(SimpleNamespace(send=lambda name, msg: messages.append((name, msg))), True)
    telemetry.send()
    telemetry.send()
    self.assertEqual([msg.valid for _, msg in messages], [False, False])
    namespace['cloudlog'].exception.assert_called_once()
    smu._send_msg.side_effect = None
    smu._send_msg.return_value = 25
    telemetry.sends = 100
    telemetry.send()
    self.assertTrue(messages[-1][1].valid)
    self.assertEqual(messages[-1][0], 'chestnutGpuState')
    self.assertEqual(messages[-1][1].chestnutGpuState.tempC, 50)
    self.assertEqual(messages[-1][1].chestnutGpuState.powerLimitW, 25)

  def test_supply_and_pcie_reads_share_lock_and_keep_gpu_fields(self):
    path = ROOT / 'system' / 'hardware' / 'chestnut' / 'monitoring.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'read_chestnut_state')
    events = []

    @contextlib.contextmanager
    def lock():
      events.append('lock')
      try:
        yield
      finally:
        events.append('unlock')

    message = SimpleNamespace(chestnutState=SimpleNamespace(), valid=False)
    namespace = {
      'messaging': SimpleNamespace(new_message=lambda _: message),
      'struct': struct,
      'usb1': SimpleNamespace(USBError=RuntimeError),
      'cloudlog': mock.Mock(),
      'usbgpu_bus_lock': lock,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)

    def read(*args, **kwargs):
      self.assertEqual(events[0], 'lock')
      self.assertNotIn('unlock', events)
      return struct.pack('<Hh?', 12000, 250, False) if args[1] == 0xC0 else bytes([0x78])

    gpu = SimpleNamespace(tempC=45)
    result = namespace['read_chestnut_state'](SimpleNamespace(controlRead=read), gpu)
    self.assertTrue(result.valid)
    self.assertEqual(result.chestnutState.tempC, 45)
    self.assertEqual(result.chestnutState.supplyVoltage, 12000)
    self.assertEqual(result.chestnutState.pcieLtssm, 0x78)
    self.assertEqual(events, ['lock', 'unlock'])
    events.clear()
    result = namespace['read_chestnut_state'](SimpleNamespace(controlRead=mock.Mock(side_effect=RuntimeError)), None)
    self.assertFalse(result.valid)
    namespace['cloudlog'].exception.assert_called_once()

  def test_monitor_reconnects_and_marks_stale_gpu_invalid(self):
    path = ROOT / 'system' / 'hardware' / 'chestnut' / 'monitoring.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    held = []

    @contextlib.contextmanager
    def lock():
      held.append(True)
      try:
        yield
      finally:
        held.pop()

    def read(*args, **kwargs):
      self.assertTrue(held)
      return struct.pack('<Hh?', 12000, 250, False) if args[1] == 0xC0 else bytes([0x78])

    def close():
      self.assertTrue(held)

    handle = SimpleNamespace(controlRead=read, close=mock.Mock(side_effect=close))
    context = mock.MagicMock()
    context.__enter__.return_value = context
    context.openByVendorIDAndProductID.side_effect = [RuntimeError('unavailable'), handle]
    subscriber = SimpleNamespace(alive={'chestnutGpuState': True}, valid={'chestnutGpuState': True}, seen={'chestnutGpuState': True})
    messages = []
    iterations = []

    def update(_):
      subscriber.alive['chestnutGpuState'] = len(iterations) != 2

    subscriber.update = update

    class SubMaster:
      def __getattr__(self, name):
        return getattr(subscriber, name)

      def __getitem__(self, name):
        return SimpleNamespace(tempC=45)

    def pubmaster(services):
      self.assertEqual(services, ['chestnutState'])
      return SimpleNamespace(send=lambda service, msg: messages.append((service, msg)))

    namespace = {
      'messaging': SimpleNamespace(
        PubMaster=pubmaster, SubMaster=lambda _: SubMaster(), new_message=lambda _: SimpleNamespace(valid=False, chestnutState=SimpleNamespace(tempC=0))
      ),
      'struct': struct,
      'usb1': SimpleNamespace(USBError=RuntimeError, USBContext=lambda: context),
      'usbgpu_bus_lock': lock,
      'cloudlog': mock.Mock(),
      'CHESTNUT_USB_PRODUCT': 'Chestnut',
      'get_usb_state': lambda: [{'vendorId': 1, 'productId': 2, 'product': 'Chestnut'}],
      'is_chestnut_usb_id': lambda *args: True,
      'SERVICE_LIST': {'chestnutState': SimpleNamespace(frequency=10)},
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['chestnut_state_thread'](SimpleNamespace(is_set=lambda: len(iterations) >= 4, wait=lambda delay: iterations.append(delay)))
    self.assertEqual([msg.valid for _, msg in messages], [True, False, True])
    self.assertEqual([msg.chestnutState.tempC for _, msg in messages], [45, 0, 45])
    self.assertEqual(iterations, [1.0, 0.1, 0.1, 0.1])
    context.openByVendorIDAndProductID.assert_called_with(1, 2, skip_on_error=False)
    handle.close.assert_called_once()
    namespace['cloudlog'].exception.assert_called_once()


class StockBuildTests(TestCase):
  def test_existing_stock_queue_equivalence_without_linux_runtime_imports(self):
    cls = load_class(Path(__file__).with_name('test_compile_modeld.py'), 'TestStockCompileModeldEquivalence', {'OpenpilotTestCase': TestCase})
    cls().test_get_policy_npy_shapes_matches_stock()
    cls().test_make_input_queues_full_stock_equivalence()

  def recipe(self, skip=False, stock_source=True):
    path = ROOT / 'selfdrive' / 'modeld' / 'SConscript'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [
      node
      for node in tree.body
      if not isinstance(node, (ast.Import, ast.ImportFrom))
      and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Import')
    ]

    def file(name):
      value = Path(name[1:]) if name.startswith('#') else Path(name)
      if not value.is_absolute():
        value = (ROOT.parent if name.startswith('#') else path.parent) / value
      return SimpleNamespace(abspath=str(value), relpath=str(value.relative_to(ROOT.parent)))

    commands, effects = [], []
    env = SimpleNamespace(
      Clone=lambda: env,
      Dir=file,
      PrependENVPath=mock.Mock(),
      Execute=mock.Mock(return_value=23),
      Command=lambda target, sources, action: commands.append((target, sources, action)) or target,
      SideEffect=lambda lock, node: effects.append((lock, node)),
    )
    namespace = {
      'env': env,
      'arch': 'comma_arm64',
      'File': file,
      'Dir': file,
      'Action': lambda command, label: SimpleNamespace(command=command, label=label),
      'Value': lambda v: v,
      'glob': SimpleNamespace(glob=lambda *args, **kwargs: []),
      'time': SimpleNamespace(sleep=mock.Mock()),
      'os': SimpleNamespace(
        getenv=lambda name: '1' if skip and name == 'SKIP_TINYGRAD_COMPILE' else None,
        path=SimpleNamespace(expanduser=lambda _: '', isfile=lambda name: stock_source or 'big_driving_supercombo' not in name, join=os.path.join),
      ),
      '_ar_ox_fisheye': SimpleNamespace(width=1344, height=760),
      '_os_fisheye': SimpleNamespace(width=1928, height=1208),
      'MEDMODEL_INPUT_SIZE': (512, 256),
      'DM_INPUT_SIZE': (1440, 960),
      'chestnut_present': lambda: True,
      'get_nv12_info': lambda w, h: (2048, h, h // 2, 2048 * (h + h // 2)),
    }
    with mock.patch('openpilot.common.file_chunker.get_existing_chunks', side_effect=lambda name: [name]):
      exec(compile(tree, str(path), 'exec'), namespace)
    return namespace, commands, effects

  def test_build_commands_use_existing_compiler_and_shared_dependencies(self):
    namespace, commands, effects = self.recipe()
    self.assertTrue(Path(namespace['compiler']).is_file())
    self.assertEqual(len(commands), 9)
    self.assertEqual(len(effects), 3)
    for _, sources, action in commands:
      command = action if isinstance(action, str) else action.command
      if isinstance(command, str):
        self.assertIn(namespace['compiler'], command)
      self.assertIn(namespace['compiler'], [getattr(source, 'abspath', source) for source in sources])
    _, skipped, effects = self.recipe(skip=True)
    self.assertEqual(len(skipped), 3)
    self.assertEqual(effects, [])
    _, missing_source, effects = self.recipe(stock_source=False)
    self.assertEqual(len(missing_source), 6)
    self.assertEqual(effects, [])

  def test_unready_chestnut_fails_build_and_ready_propagates_exit_status(self):
    from types import ModuleType

    namespace, _, _ = self.recipe()
    flash = ModuleType('openpilot.system.hardware.chestnut.flash')
    flash.link_up = mock.Mock(return_value=False)
    with mock.patch.dict('sys.modules', {'openpilot.system.hardware.chestnut.flash': flash}):
      action = namespace['chestnut_action']('compile fixture')
      self.assertEqual(action.command([], [], namespace['env']), 1)
      namespace['env'].Execute.assert_not_called()
      flash.link_up.side_effect = [False, True]
      self.assertEqual(action.command([], [], namespace['env']), 23)
      namespace['env'].Execute.assert_called_once_with('compile fixture')
