#!/usr/bin/env python3
"""
dp - egpu: compile a big driving model for this device's eGPU, on this device, with the vendored
pinned HCQ2 tinygrad. Run as its own process by egpu_manager (offroad only, eGPU plugged in):

  compile.py warp  --cam 1928x1208 --out warp.pkl      (DEV=QCOM ...)      the camera warp, once per camera
  compile.py model --onnx x.onnx   --out model.pkl     (DEV=USB+AMD:LLVM)  one per ONNX

Why two JITs (the 3X fix): the warp reads the full NV12 frames, so it runs on the comma's own GPU
(QCOM); only its output, the 2x6x128x256 warped crop, crosses USB to the eGPU. sunnypilot-master
compiles the same two JITs but puts the big warp on the eGPU too (FRAME_DEV=CPU, whole frames over
USB); AmyJeanes/sunnypilot@173d90b (port of commaai#38862) moved it to QCOM inside one fused JIT.
Here it is a separate QCOM warp JIT, as sunnypilot-master splits them and as jetlink's comma side does.

Both graph layouts in the wild are handled (jetlink spec.py):
  stateful (openpilot #38916, Cinque Terre V3 onwards): new_img + state_* in, next_state_* out.
    compiled as-is with the pinned ONNX runner (every input on the eGPU)
  queued (up to Cinque Terre V2): img/big_img/features_buffer/desire_pulse history stacks in.
    wrapped in a JIT that keeps the history queues on the eGPU and takes new_img, as AmyJeanes'
    make_run_policy / upstream selfdrive/modeld/compile_modeld.py (get_policy_npy_shapes,
    make_input_queues, shift_and_sample) did; prev_feat (the hidden state) is fed back by the runner.

Both produce one artifact shape: {'metadata', 'run', 'input_specs', 'output_specs', 'layout', 'state_inputs'}.
"""
import argparse
import hashlib
from graphlib import CycleError, TopologicalSorter
import math
import os
import pickle
import tempfile
import time
from pathlib import Path

import tg_env  # (this script's dir is sys.path[0]; tg_env puts the vendored tinygrad first and drops it)

import numpy as np

STATEFUL_FRAME = 'new_img'
STATE_OUTPUT_PREFIX = 'next_'
DRIVING_OUTPUT = 'outputs'
QUEUED_STATE = ('img_q', 'big_img_q', 'feat_q', 'desire_q')


def log(msg: str) -> None:
  print(f"[egpu compile] {msg}", flush=True)


def compile_jit(function, make_inputs, benchmark_runs: int, *, out_of_band: bool = False):
  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.nn.state import get_parameters
  from examples.openpilot.helpers import dump_pickle, load_pickle

  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")
  jit = TinyJit(function, prune=True)

  def run(fn, seed, count):
    args, kwargs = make_inputs(seed)
    results = []
    for i in range(count):
      Device.default.synchronize()
      start = time.perf_counter()
      output = fn(*args, **kwargs)
      if tensors := get_parameters(output):
        Tensor.realize(*tensors)
      Device.default.synchronize()
      log(f"validation {seed} [{i + 1}/{count}]: {(time.perf_counter() - start) * 1e3:.2f} ms")
      # Recurrent state can be large; retain exact-content digests rather than several full copies.
      results.append([(t.shape, t.dtype, hashlib.sha256(memoryview(t.numpy()).cast('B')).digest())
                      for t in get_parameters((output, args, kwargs))])
    return results

  run(jit, 42, 3)
  # The manager runs in the model directory on /data; avoid system temp space for full-size weights.
  with tempfile.TemporaryDirectory(dir=".") as directory:
    path = str(Path(directory) / "jit.pkl")
    if out_of_band:
      dump_pickle(jit, path)
    else:
      with open(path, 'wb') as f:
        pickle.dump(jit, f)
    loaded = load_pickle(path, out_of_band=out_of_band)
    for seed in (42, 43):
      expected = run(function, seed, max(3, benchmark_runs))
      actual = run(loaded, seed, max(3, benchmark_runs))
      for ref_step, actual_step in zip(expected, actual, strict=True):
        for ref, value in zip(ref_step, actual_step, strict=True):
          if ref != value:
            raise AssertionError(f"serialized JIT differs from reference (seed={seed})")
  return jit


# ---------------------------------------------------------------------------------------------- warp
def compile_warp_cmd(cam_w: int, cam_h: int, out: str, benchmark_runs: int, model_size=None) -> None:
  from tinygrad import Device
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from examples.openpilot.compile_warp import NV12Frame, make_frame_prepare
  from examples.openpilot.helpers import dump_pickle
  from dragonpilot.egpu.constants import MODEL_INPUT_SIZE
  model_size = tuple(model_size or MODEL_INPUT_SIZE)
  frame = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  log(f"warp {cam_w}x{cam_h} -> {model_size} on {os.environ.get('DEV')}")
  # one frame per call: the runner hands each camera buffer in place (Tensor.from_blob), no 2x3 MB copy;
  # transforms on the host (NPY), as the DM warp does
  specs = {'input_frame': ((frame.size,), np.dtype(np.uint8).str, Device.DEFAULT),
           'M_inv': ((3, 3), np.dtype(np.float32).str, 'NPY')}

  def make_inputs(seed):
    from examples.openpilot.helpers import allocate_inputs
    rng = np.random.default_rng(seed)

    def initialize(arrays):
      arrays['input_frame'][:] = rng.integers(0, 256, (frame.size,), dtype=np.uint8)
      arrays['M_inv'][:] = np.eye(3, dtype=np.float32) + rng.standard_normal((3, 3)) * .01
    return (), allocate_inputs(specs, initialize)

  jit = compile_jit(make_frame_prepare(frame, *model_size), make_inputs, benchmark_runs)
  artifact = {'run': jit, 'input_specs': specs, 'metadata': {'camera': (cam_w, cam_h), 'model_size': model_size}}
  _atomic_dump(artifact, out, dump_pickle, out_of_band=False)


# ---------------------------------------------------------------------------------------------- model
def _shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def _sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def _sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def queued_shapes(input_shapes: dict, frame_skip: int) -> dict:
  """every JIT input of a wrapped queued graph: (shape, numpy dtype str). Mirrors upstream compile_modeld
  make_input_queues / get_policy_npy_shapes (feat_dim = prod(fb[2:]), jetlink spec.py)."""
  img = input_shapes['img']                 # (1, 12, H, W)
  fb = input_shapes['features_buffer']      # (1, T, ...)
  dp = input_shapes['desire_pulse']         # (1, T, 8)
  feat_dim = math.prod(fb[2:])
  n_frames = img[1] // 6
  img_buf = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])
  u8, f32 = np.dtype(np.uint8).str, np.dtype(np.float32).str
  return {
    STATEFUL_FRAME: ((2, 6, img[2], img[3]), u8),
    'img_q': (img_buf, u8), 'big_img_q': (img_buf, u8),
    'feat_q': ((frame_skip * fb[1], fb[0], feat_dim), f32),
    'desire_q': ((frame_skip * dp[1], dp[0], dp[2]), f32),
    'desire': ((dp[2],), f32),
    'traffic_convention': (tuple(input_shapes['traffic_convention']), f32),
    'action_t': (tuple(input_shapes['action_t']), f32),
    'prev_feat': ((fb[0], feat_dim), f32),
  }


def compile_model_cmd(onnx: str, out: str, benchmark_runs: int) -> None:
  from tinygrad import Tensor, Device
  from tinygrad.nn.onnx import OnnxRunner
  from tinygrad.dtype import _to_np_dtype
  from examples.openpilot.compile_onnx import onnx_metadata
  from examples.openpilot.helpers import allocate_inputs, dump_pickle

  properties, output_shapes = onnx_metadata(onnx)
  if 'output_slices' not in properties:
    raise ValueError("not an openpilot driving model: no output_slices metadata")
  runner = OnnxRunner(onnx)
  graph_inputs = dict(runner.graph_inputs)

  if STATEFUL_FRAME in graph_inputs:
    log(f"stateful graph (#38916) on {os.environ.get('DEV')}: every input on the device")
    specs = {name: (tuple(spec.shape), np.dtype(_to_np_dtype(spec.dtype)).name, Device.DEFAULT)
             for name, spec in graph_inputs.items()}
    outputs = runner(allocate_inputs(specs))
    output_specs = {name: (tuple(value.shape), np.dtype(_to_np_dtype(value.dtype)).name, Device.DEFAULT)
                    for name, value in outputs.items()}
    state_names = [name for name in specs if STATE_OUTPUT_PREFIX + name in output_specs]
    for name in state_names:
      if specs[name] != output_specs[STATE_OUTPUT_PREFIX + name]:
        raise ValueError(f"state input/output specification mismatch: {name}")

    def make_inputs(seed):
      rng = np.random.default_rng(seed)

      def initialize(arrays):
        for value in arrays.values():
          value[...] = rng.integers(0, 256, value.shape, dtype=np.uint8) if value.dtype == np.uint8 else rng.standard_normal(value.shape)
      inputs = allocate_inputs(specs, initialize)
      buffers = allocate_inputs({name: spec for name, spec in output_specs.items()
                                 if name not in {STATE_OUTPUT_PREFIX + state for state in state_names}})
      buffers.update({STATE_OUTPUT_PREFIX + name: inputs[name] for name in state_names})
      return (), inputs | {'output_buffers': buffers}

    def run(output_buffers, **inputs):
      outputs = {name: Tensor(value.uop) for name, value in runner(inputs).items()}
      # Break only cyclic state dependencies, not every state: the scheduler orders the remaining reads before writes.
      while True:
        dependencies = {name: {other for other in state_names if other != name and
                               inputs[other].uop.storage_base in outputs[STATE_OUTPUT_PREFIX + name].uop.toposort()}
                        for name in state_names}
        try:
          tuple(TopologicalSorter(dependencies).static_order())
          break
        except CycleError as error:
          name = min(error.args[1][:-1], key=lambda n: inputs[n].numel() * inputs[n].dtype.itemsize)
          snapshot = inputs[name].clone().realize()
          replacement = {inputs[name].uop.storage_base: snapshot.uop}
          outputs = {key: Tensor(value.uop.substitute(replacement, walk=True)) for key, value in outputs.items()}
      Tensor.realize(*(output_buffers[name].assign(value) for name, value in outputs.items()))

    jit = compile_jit(run, make_inputs, benchmark_runs, out_of_band=True)
    artifact = {'metadata': {'metadata': properties, 'input_shapes': {n: s[0] for n, s in specs.items()},
                             'output_shapes': output_shapes},
                'run': jit, 'input_specs': specs, 'output_specs': output_specs, 'layout': 'stateful',
                'state_inputs': state_names}
    _atomic_dump(artifact, out, dump_pickle, out_of_band=True)
    return

  # ------------------------------------------------------------------ queued graph, wrapped
  input_shapes = {name: tuple(d if isinstance(d, int) else 1 for d in spec.shape) for name, spec in graph_inputs.items()}
  missing = {'img', 'big_img', 'features_buffer', 'desire_pulse', 'traffic_convention', 'action_t'} - set(input_shapes)
  if missing:
    raise ValueError(f"unsupported queued graph, missing inputs {missing}")
  fb = input_shapes['features_buffer']
  frame_skip = 1 if fb[1] >= 99 else 4   # sunnypilot modeld_v2 derive_frame_skip
  shapes = queued_shapes(input_shapes, frame_skip)
  log(f"queued graph on {os.environ.get('DEV')}: frame_skip {frame_skip}, history queues on the device")
  graph_dtypes = {name: spec.dtype for name, spec in graph_inputs.items()}
  skip = lambda b: _sample_skip(b, frame_skip)      # noqa: E731
  desire_sample = lambda b: _sample_desire(b, frame_skip)  # noqa: E731

  def model(new_img, img_q, big_img_q, feat_q, desire_q, desire, traffic_convention, action_t, prev_feat):
    img = _shift_and_sample(img_q, new_img[0:1], skip)
    big_img = _shift_and_sample(big_img_q, new_img[1:2], skip)
    desire_buf = _shift_and_sample(desire_q, desire.reshape(1, 1, -1), desire_sample)
    feat_buf = _shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), skip)
    inputs = {'img': img, 'big_img': big_img, 'features_buffer': feat_buf.reshape(input_shapes['features_buffer']),
              'desire_pulse': desire_buf, 'traffic_convention': traffic_convention, 'action_t': action_t}
    inputs = {name: value.cast(graph_dtypes[name]) for name, value in inputs.items()}
    outs = runner(inputs)
    return next(iter(outs.values())).cast('float32').contiguous()

  specs = {name: (shape, dtype, Device.DEFAULT) for name, (shape, dtype) in shapes.items()}
  out_shape = tuple(model(**allocate_inputs(specs)).shape)
  output_specs = {DRIVING_OUTPUT: (out_shape, np.dtype(np.float32).name, Device.DEFAULT)}

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    def initialize(arrays):
      for value in arrays.values():
        value[...] = rng.integers(0, 256, value.shape, dtype=np.uint8) if value.dtype == np.uint8 else rng.standard_normal(value.shape)
    return (), allocate_inputs(specs, initialize) | {'output_buffers': allocate_inputs(output_specs)}

  def run(output_buffers, **inputs):
    out = model(**inputs)
    Tensor.realize(out)
    Tensor.realize(output_buffers[DRIVING_OUTPUT].assign(out))

  jit = compile_jit(run, make_inputs, benchmark_runs, out_of_band=True)
  metadata = {'metadata': properties,
              'input_shapes': {n: s for n, (s, _) in shapes.items()},
              'output_shapes': {DRIVING_OUTPUT: out_shape},
              'graph_input_shapes': input_shapes}
  artifact = {'metadata': metadata, 'run': jit, 'input_specs': specs, 'output_specs': output_specs,
              'layout': 'queued', 'state_inputs': list(QUEUED_STATE), 'frame_skip': frame_skip}
  _atomic_dump(artifact, out, dump_pickle, out_of_band=True)


def _atomic_dump(artifact, out: str, dump, out_of_band: bool) -> None:
  tmp = out + ".tmp"
  if out_of_band:
    dump(artifact, tmp)
  else:
    with open(tmp, 'wb') as f:
      pickle.dump(artifact, f)
  os.replace(tmp, out)
  log(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  w = sub.add_parser('warp')
  w.add_argument('--cam', required=True, help='WxH')
  w.add_argument('--out', required=True)
  w.add_argument('--model-size', default=None, help='WxH, default MEDMODEL 512x256 (tests use a tiny one)')
  m = sub.add_parser('model')
  m.add_argument('--onnx', required=True)
  m.add_argument('--out', required=True)
  for s in (w, m):
    s.add_argument('--benchmark-runs', type=int, default=3)
  args = p.parse_args(argv)

  tg_env.check()
  tg_env.patch_fetch_fw()
  if os.path.isfile('/TICI') and (os.cpu_count() or 0) >= 8:
    # zoompilot SConscript: "CPU 7 is isolated with isolcpus on AGNOS, so explicitly pin the compiler to it"
    # (taskset -c 7): a compile of many minutes then leaves cores 0-6 to the UI, wifi and the rest
    try:
      os.sched_setaffinity(0, {7})
    except OSError:
      pass
  if args.cmd == 'warp':
    cam_w, cam_h = (int(x) for x in args.cam.lower().split('x'))
    ms = tuple(int(x) for x in args.model_size.lower().split('x')) if args.model_size else None
    compile_warp_cmd(cam_w, cam_h, args.out, args.benchmark_runs, ms)
  else:
    compile_model_cmd(args.onnx, args.out, args.benchmark_runs)


if __name__ == "__main__":
  main()
