"""Parked-only CTMV2 graph extraction experiment; never selected by modeld."""
from dataclasses import dataclass
import math

from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import CapturedJit, _TinyJit
from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins
from tinygrad.uop.ops import Ops, UOp

CHECKPOINT = "b9facbcc-4d47-410e-b3ce-dfcbad12ba92/56320/1a421175-db71-4e3d-9d62-e2166421b02b/12864"
INPUT_NAMES = ['big_img_q', 'desire_q', 'feat_q', 'img_q', 'packed_npy_inputs']
IMAGE_SHAPE = (2, 6, 128, 256)


@dataclass
class ExtractedPolicy:
  run_policy: _TinyJit
  reference_warp: _TinyJit
  control_bytes: int
  image_offset: int
  input_bytes: int
  removed_calls: int
  retained_calls: int
  qualified: bool = False
  state: Tensor | None = None


def _resident_feedback(ret, control_bytes: int) -> tuple[Tensor, UOp, UOp]:
  if (not isinstance(ret, tuple) or len(ret) != 1 or not isinstance(ret[0], Tensor) or
      ret[0].shape != (1, 18452) or ret[0].dtype != dtypes.float32 or ret[0].device != 'AMD'):
    raise ValueError("Resident state requires the inspected CTMV2 float32 output")
  output = ret[0].uop.reshape((18452,))
  hidden = output.shrink(((2066, 18450),)).bitcast(dtypes.uint8)
  state = UOp.new_buffer('AMD', 65536, dtypes.uint8)
  if control_bytes - math.prod(state.shape) != 120:
    raise ValueError("Unexpected CTMV2 previous-feature offset")
  return Tensor(state), hidden, output.shrink(((0, 2066),))


def _check_control_access(call: UOp, packed: UOp, limit: int) -> None:
  args = get_call_arg_uops(call)
  slots = [i for i, arg in enumerate(args) if arg is packed]
  if not slots:
    return
  program = call.src[0]
  if program.op is not Ops.PROGRAM:
    raise ValueError("Policy packed input must be accessed by an inspectable program")
  if program.arg.globals != tuple(range(len(args))):
    raise ValueError("Unsupported policy program argument mapping")
  outs, _ = get_call_outs_ins(call)
  if any(slot in outs for slot in slots):
    raise ValueError("Policy must not write packed controls")
  accessed = set()
  for node in program.src[0].toposort():
    if node.op is not Ops.INDEX:
      continue
    params = [p for p in node.src[0].toposort() if p.op is Ops.PARAM and p.arg.slot in slots]
    if not params:
      continue
    if len(params) != 1 or node.src[0].op is not Ops.PARAM:
      raise ValueError("Unsupported packed input pointer conversion")
    low, high = node.src[1].vmin, node.src[1].vmax
    itemsize = node.src[0].dtype.itemsize
    if not isinstance(low, int) or not isinstance(high, int) or low < 0 or (high + 1) * itemsize > limit:
      raise ValueError("Policy still depends on raw images or unbounded packed input")
    accessed.add(params[0].arg.slot)
  if accessed != set(slots):
    raise ValueError("Cannot prove policy control access bounds")


def extract_ctmv2(artifact: dict, camera_size: tuple[int, int], frame_bytes: int, *, resident_state: bool = False) -> ExtractedPolicy:
  """Construct an unqualified candidate from a trusted artifact, without changing it.

  The kernel binaries and scratch arena are retained. Compact images are copied
  into the original warp output view before replaying the policy kernels.
  This must not be serialized or selected for driving without hardware validation.
  """
  if camera_size not in ((1928, 1208), (1344, 760)) or frame_bytes <= 0:
    raise ValueError("Unsupported camera layout")
  metadata = artifact.get('metadata', {}).get('model', {})
  if metadata.get('model_checkpoint') != CHECKPOINT:
    raise ValueError("Only the inspected CTMV2 checkpoint is supported")
  if artifact.get('input_devices') != {'model': 'AMD'}:
    raise ValueError("Extraction requires the original AMD artifact")
  shapes = metadata['input_shapes']
  if (tuple(shapes['img']) != (1, 12, 128, 256) or shapes['big_img'] != shapes['img'] or
      tuple(shapes['features_buffer']) != (1, 32, 32, 512)):
    raise ValueError("Unsupported CTMV2 image/history layout")
  capture = artifact['run_model'][camera_size].captured
  if capture.expected_names != INPUT_NAMES or len(capture.expected_input_info) != 5:
    raise ValueError("Unexpected captured input contract")
  control_bytes = (18 + shapes['desire_pulse'][2] + math.prod(shapes['traffic_convention']) +
                   math.prod(shapes['action_t']) + math.prod(shapes['features_buffer'][2:])) * 4
  raw_bytes = control_bytes + 2 * frame_bytes
  linear = capture._linear
  if len(linear.src) != 2:
    raise ValueError("Expected one input upload followed by one AMD graph")
  upload, graph_call = linear.src
  if upload.src[0].op is not Ops.COPY:
    raise ValueError("Missing explicit raw input upload")
  packed, host = get_call_arg_uops(upload)
  if packed.dtype != dtypes.uint8 or packed.shape != (raw_bytes,) or host.op is not Ops.PARAM or host.arg.slot != 4:
    raise ValueError("Unexpected raw packed input layout")
  graph = graph_call.src[0]
  if graph.op is not Ops.CUSTOM_FUNCTION or graph.arg != 'graph' or graph.src[0].op is not Ops.LINEAR:
    raise ValueError("Unsupported graph wrapper")
  calls = graph.src[0].src
  if len(calls) < 10 or any(c.src[0].op is not Ops.PROGRAM for c in calls):
    raise ValueError("Expected inspectable AMD kernels")
  prefix, policy = calls[:7], calls[7:]
  planes = []
  for call in prefix[:6]:
    args = get_call_arg_uops(call)
    outs, ins = get_call_outs_ins(call)
    if outs != (0,) or packed not in [args[i] for i in ins]:
      raise ValueError("Unexpected camera plane kernel")
    if any(p.op is Ops.PARAM for b in args for p in b.toposort()):
      raise ValueError("Warp prefix unexpectedly depends on recurrent inputs")
    planes.append(args[0])
  combine = get_call_arg_uops(prefix[6])
  outs, ins = get_call_outs_ins(prefix[6])
  warped = combine[0]
  if (outs != (0,) or {combine[i] for i in ins} != set(planes) or
      warped.dtype != dtypes.uint8 or warped.shape != (math.prod(IMAGE_SHAPE),) or
      sum(math.prod(p.shape) for p in planes) != math.prod(IMAGE_SHAPE)):
    raise ValueError("Cannot identify the isolated two-camera warp output")
  if any(p.op is Ops.PARAM for p in warped.toposort()):
    raise ValueError("Warp output aliases a recurrent input")
  if len(planes) != len(set(planes)):
    raise ValueError("Warp plane outputs must be distinct")
  for call in prefix:
    for b in get_call_arg_uops(call):
      if any(p.op is Ops.PARAM for p in b.toposort()):
        raise ValueError("Warp prefix unexpectedly depends on recurrent inputs")
  for call in policy:
    for b in get_call_arg_uops(call):
      if b is not packed and packed in b.toposort():
        raise ValueError("Unsupported policy view of raw packed input")
    _check_control_access(call, packed, control_bytes)
  if not all(warped in get_call_arg_uops(call) for call in policy[:2]):
    raise ValueError("Image history kernels do not consume the identified warp output")

  state, hidden, visible = _resident_feedback(capture.ret, control_bytes) if resident_state else (None, None, None)
  upload_controls = 120 if resident_state else control_bytes
  image_offset = (upload_controls + 511) // 512 * 512
  input_bytes = image_offset + math.prod(IMAGE_SHAPE)
  compact = UOp.new_buffer('AMD', input_bytes, dtypes.uint8)
  compact_host = UOp.param(4, dtypes.uint8, input_bytes, 'NPY')
  controls = UOp.new_buffer('AMD', control_bytes, dtypes.uint8) if resident_state else compact.shrink(((0, control_bytes),))
  images = compact.shrink(((image_offset, input_bytes),))
  rewritten = tuple(c.replace(src=(c.src[0], *(b.substitute({packed: controls}) for b in c.src[1:]))) for c in policy)
  policy_graph = graph.replace(src=(graph.src[0].replace(src=rewritten),))
  calls_before = [compact_host.copy_to_device('AMD').call(compact, compact_host)]
  calls_after = []
  if state is not None:
    header = compact.shrink(((0, upload_controls),))
    calls_before.extend((
      header.copy_to_device('AMD').call(controls.shrink(((0, upload_controls),)), header),
      state.uop.copy_to_device('AMD').call(controls.shrink(((upload_controls, control_bytes),)), state.uop),
    ))
    calls_after.append(hidden.copy_to_device('AMD').call(state.uop, hidden))
  new_linear = linear.replace(src=(
    *calls_before,
    images.copy_to_device('AMD').call(warped, images),
    graph_call.replace(src=(policy_graph, *graph_call.src[1:])),
    *calls_after,
  ))
  info = list(capture.expected_input_info)
  old_view, variables, dtype, device = info[4]
  if variables or dtype != dtypes.uint8 or device != 'NPY' or old_view.op is not Ops.NOOP:
    raise ValueError("Unsupported host input specification")
  ret = (Tensor(visible).reshape((1, 2066)),) if resident_state else capture.ret
  run_policy = _TinyJit(None, CapturedJit(ret, new_linear, list(capture.expected_names), info))
  reference_linear = linear.replace(src=(upload, graph_call.replace(src=(
    graph.replace(src=(graph.src[0].replace(src=prefix),)), *graph_call.src[1:]))))
  reference_warp = _TinyJit(None, CapturedJit(Tensor(warped).reshape(IMAGE_SHAPE), reference_linear,
                                          list(capture.expected_names), list(capture.expected_input_info)))
  return ExtractedPolicy(run_policy, reference_warp, upload_controls, image_offset, input_bytes, len(prefix), len(policy), state=state)
