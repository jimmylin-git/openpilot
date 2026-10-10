import unittest

from tinygrad import dtypes
from tinygrad.engine.jit import CapturedJit, _TinyJit
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

from openpilot.sunnypilot.modeld_v2.extract_policy import CHECKPOINT, INPUT_NAMES, _check_control_access, extract_ctmv2


def control_call(index, *, writes=False, globals_map=(0,)):
  packed = UOp.new_buffer('AMD', 7536760, dtypes.uint8)
  param = UOp.param(0, dtypes.uint8, 7536760, 'AMD')
  address = UOp(Ops.INDEX, src=(param, UOp.const(index, dtypes.int)))
  sink = UOp(Ops.SINK, src=(address,))
  program = UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo(
    globals=globals_map, outs=(0,) if writes else (), ins=(0,)))
  return program.call(packed), packed


def synthetic_artifact():
  raw_bytes = 7536760
  packed = UOp.new_buffer('AMD', raw_bytes, dtypes.uint8)
  host = UOp.param(4, dtypes.uint8, raw_bytes, 'NPY')
  planes = [UOp.new_buffer('AMD', size, dtypes.uint8) for size in (131072, 32768, 32768) * 2]
  warped = UOp.new_buffer('AMD', 393216, dtypes.uint8)

  def program_call(*args, ins):
    return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),), arg=ProgramInfo(
      globals=tuple(range(len(args))), outs=(0,), ins=ins)).call(*args)

  prefix = [program_call(plane, packed, ins=(1,)) for plane in planes]
  prefix.append(program_call(warped, *planes, ins=tuple(range(1, 7))))
  policy = [program_call(UOp.new_buffer('AMD', 983040, dtypes.uint8), UOp.param(slot, dtypes.uint8, 983040, 'AMD'),
                         warped, ins=(1, 2)) for slot in (0, 3)]
  control, _ = control_call(120)
  policy.append(control.replace(src=(control.src[0], packed)))
  graph = UOp.custom_function('graph', UOp(Ops.LINEAR, src=tuple(prefix + policy)))
  linear = UOp(Ops.LINEAR, src=(host.copy_to_device('AMD').call(packed, host), graph.call()))
  info = [(UOp(Ops.NOOP), (), dtypes.uint8, 'AMD')] * 4 + [(UOp(Ops.NOOP), (), dtypes.uint8, 'NPY')]
  capture = CapturedJit(None, linear, list(INPUT_NAMES), info)
  artifact = {
    'metadata': {'model': {
      'model_checkpoint': CHECKPOINT,
      'input_shapes': {'img': [1, 12, 128, 256], 'big_img': [1, 12, 128, 256],
                       'features_buffer': [1, 32, 32, 512], 'desire_pulse': [1, 33, 8],
                       'traffic_convention': [1, 2], 'action_t': [1, 2]},
    }},
    'input_devices': {'model': 'AMD'},
    'run_model': {(1928, 1208): _TinyJit(None, capture)},
  }
  return artifact, capture


class TestExtractPolicyGuards(unittest.TestCase):
  def test_candidate_preserves_original_graph_and_programs(self):
    artifact, original = synthetic_artifact()
    linear = original._linear
    result = extract_ctmv2(artifact, (1928, 1208), 3735552)
    self.assertIs(original._linear, linear)
    self.assertFalse(result.qualified)
    self.assertEqual((result.control_bytes, result.image_offset, result.input_bytes), (65656, 66048, 459264))
    self.assertEqual((result.removed_calls, result.retained_calls), (7, 3))
    before = linear.src[1].src[0].src[0].src[7:]
    after = result.run_policy.captured._linear.src[2].src[0].src[0].src
    for left, right in zip(before, after, strict=True):
      self.assertIs(left.src[0], right.src[0])

  def test_candidate_rejects_recurrent_input_in_warp(self):
    artifact, capture = synthetic_artifact()
    graph_call = capture._linear.src[1]
    graph = graph_call.src[0]
    calls = list(graph.src[0].src)
    calls[0] = calls[0].replace(src=(*calls[0].src, UOp.param(0, dtypes.uint8, 983040, 'AMD')))
    capture._linear = capture._linear.replace(src=(
      capture._linear.src[0], graph_call.replace(src=(graph.replace(src=(graph.src[0].replace(src=tuple(calls)),)),))))
    with self.assertRaisesRegex(ValueError, "recurrent inputs"):
      extract_ctmv2(artifact, (1928, 1208), 3735552)

  def test_control_access_accepts_last_control_byte(self):
    for offset in (0, 72, 120, 65655):
      with self.subTest(offset=offset):
        call, packed = control_call(offset)
        _check_control_access(call, packed, 65656)

  def test_raw_image_access_is_rejected(self):
    for offset in (-1, 65656, 7536759):
      with self.subTest(offset=offset):
        call, packed = control_call(offset)
        with self.assertRaisesRegex(ValueError, "raw images or unbounded"):
          _check_control_access(call, packed, 65656)

  def test_control_writes_are_rejected(self):
    call, packed = control_call(120, writes=True)
    with self.assertRaisesRegex(ValueError, "must not write"):
      _check_control_access(call, packed, 65656)

  def test_noncontiguous_argument_mapping_is_rejected(self):
    call, packed = control_call(120, globals_map=(1,))
    with self.assertRaisesRegex(ValueError, "argument mapping"):
      _check_control_access(call, packed, 65656)

  def test_missing_access_proof_is_rejected(self):
    packed = UOp.new_buffer('AMD', 7536760, dtypes.uint8)
    program = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),), arg=ProgramInfo(globals=(0,), ins=(0,)))
    with self.assertRaisesRegex(ValueError, "Cannot prove"):
      _check_control_access(program.call(packed), packed, 65656)

  def test_models_other_than_ctmv2_are_rejected_without_touching_jit(self):
    for checkpoint in ("TT", "IDM", "LM", "CTM", ""):
      with self.subTest(checkpoint=checkpoint):
        artifact = {'metadata': {'model': {'model_checkpoint': checkpoint}}}
        with self.assertRaisesRegex(ValueError, "Only the inspected CTMV2"):
          extract_ctmv2(artifact, (1928, 1208), 3735552)

  def test_wrong_device_is_rejected(self):
    artifact = {'metadata': {'model': {'model_checkpoint': CHECKPOINT}}, 'input_devices': {'model': 'QCOM'}}
    with self.assertRaisesRegex(ValueError, "original AMD"):
      extract_ctmv2(artifact, (1928, 1208), 3735552)

  def test_unknown_camera_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "camera layout"):
      extract_ctmv2({}, (4, 4), 24)
