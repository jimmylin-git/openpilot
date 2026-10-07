#!/usr/bin/env python3
import argparse
import codecs
import io
import pickle
from functools import partial

import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.nn.onnx import OnnxRunner

from openpilot.sunnypilot.modeld_v2.compile_modeld import (
  MetadataOnnxPBParser,
  compile_jit,
  read_file_chunked_to_disk,
  _parse_size,
  derive_frame_skip,
  make_metadata_dict,
  make_run_policy,
)
from openpilot.sunnypilot.modeld_v2.helpers import dump_oob
from openpilot.sunnypilot.modeld_v2.stock_dependencies import NV12Frame, make_frame_prepare, warp_perspective_tinygrad
from openpilot.sunnypilot.modeld_v2 import stock_dependencies as stock
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def model_metadata(path):
  model = MetadataOnnxPBParser(path).parse()
  metadata = {prop['key']: prop['value'] for prop in model['metadata_props']}
  if 'output_slices' not in metadata:
    raise ValueError("Model metadata is missing output_slices")
  pickle.loads(codecs.decode(metadata['output_slices'].encode(), 'base64'))

  def specs(values):
    return {
      value['name']: (tuple(dim if isinstance(dim, int) else 1 for dim in value['parsed_type'].shape), np.dtype(value['parsed_type'].dtype.fmt).name)
      for value in values
    }

  return metadata, specs(model['graph']['input']), specs(model['graph']['output'])


def make_native_run(runner, device):
  def run(output_buffers, **inputs):
    outputs = runner({name: value.to(device) for name, value in inputs.items()})
    if outputs.keys() != output_buffers.keys():
      raise ValueError(f"Model output names differ: {outputs.keys()} != {output_buffers.keys()}")
    Tensor.realize(*(output_buffers[name].assign(value.cast(output_buffers[name].dtype)) for name, value in outputs.items()))

  return run


def make_legacy_run(run_model):
  def run(**kwargs):
    return (run_model(**kwargs),)

  return run


def compile_model(path, output, benchmark_runs=1, camera_resolutions=()):
  path = read_file_chunked_to_disk(path)
  metadata, inputs, outputs = model_metadata(path)
  device = Device.DEFAULT
  # Preserve the float32 public interface used by the stock loaders.
  inputs = {name: (shape, 'float32' if dtype == 'float16' else dtype, device) for name, (shape, dtype) in inputs.items()}
  outputs = {name: (shape, 'float32' if dtype == 'float16' else dtype, device) for name, (shape, dtype) in outputs.items()}
  runner = OnnxRunner(path)
  if 'img' in inputs and 'new_img' not in inputs:
    if not camera_resolutions:
      raise ValueError("Legacy stock model compilation requires camera resolutions")
    legacy_metadata = make_metadata_dict(path)
    frame_skip = derive_frame_skip({}, legacy_metadata['input_shapes'])
    run_policy = make_run_policy(None, [runner], legacy_metadata['output_slices']['hidden_state'], frame_skip, legacy_metadata['input_shapes'])
    run_models = {}
    for width, height in camera_resolutions:
      nv12 = NV12Frame(width, height, *get_nv12_info(width, height))
      copy_size = stock.nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
      img_shape = legacy_metadata['input_shapes']['img']
      warp = stock.make_warp(nv12, img_shape[3] * 2, img_shape[2] * 2)
      run_model = stock.make_run_model(warp, run_policy, legacy_metadata, copy_size)
      jit = TinyJit(make_legacy_run(run_model), prune=True)
      queues = partial(stock.make_input_queues, legacy_metadata['input_shapes'], frame_skip, frame_copy_size=copy_size)
      run_models[(width, height)] = compile_jit(jit, stock.MODELD_INPUTS, queues, benchmark_runs=benchmark_runs)
    data = {'metadata': {'model': legacy_metadata, 'warp_dev': device}, 'input_devices': {'model': device}, 'run_model': run_models}
    with open(output, 'wb') as stream:
      dump_oob(data, stream)
    return

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, (shape, dtype, dev) in inputs.items():
      values = rng.integers(0, 16, size=shape) if np.issubdtype(np.dtype(dtype), np.integer) else rng.standard_normal(shape)
      tensors[name] = Tensor(values.astype(dtype), device=dev).realize()
    tensors['output_buffers'] = {name: Tensor(np.zeros(shape, dtype=dtype), device=dev).realize() for name, (shape, dtype, dev) in outputs.items()}
    return (), tensors

  run = compile_jit(make_native_run(runner, device), make_inputs, benchmark_runs=benchmark_runs)
  data = {
    'input_specs': inputs,
    'output_specs': outputs,
    'run': run,
    'metadata': {
      'input_shapes': {name: shape for name, (shape, _, _) in inputs.items()},
      'output_shapes': {name: shape for name, (shape, _, _) in outputs.items()},
      'metadata': metadata,
    },
  }
  with open(output, 'wb') as stream:
    dump_oob(data, stream)


def make_warp(nv12, size, layout, frames, border_fill):
  width, height = size
  prepare = make_frame_prepare(nv12, width, height)

  def run(input_frame, M_inv):
    transform = M_inv.to(Device.DEFAULT).realize()
    frame = input_frame.to(Device.DEFAULT).realize()
    if layout == 'luma':
      return (
        warp_perspective_tinygrad(
          frame[: nv12.height * nv12.stride], transform, size, (nv12.height, nv12.width), nv12.stride - nv12.width, border_fill_val=border_fill
        )
        .reshape(1, height * width)
        .realize()
      )
    return Tensor.stack(*(prepare(frame[i], transform[i]) for i in range(frames))).realize()

  return run


def compile_warp(frame, size, layout, frames, transform_device, border_fill, output):
  nv12 = NV12Frame(*frame)
  frame_shape = (frames, nv12.size) if layout == 'yuv420' else (nv12.size,)
  transform_shape = (frames, 3, 3) if layout == 'yuv420' else (3, 3)
  transform_device = transform_device or Device.DEFAULT

  def make_inputs(seed):
    rng = np.random.default_rng(seed)
    return (), {
      'input_frame': Tensor(rng.integers(0, 256, size=frame_shape, dtype=np.uint8), device=Device.DEFAULT).realize(),
      'M_inv': Tensor(np.broadcast_to(np.eye(3, dtype=np.float32), transform_shape).copy(), device=transform_device).realize(),
    }

  jit = TinyJit(make_warp(nv12, size, layout, frames, border_fill), prune=True)
  reference = None
  for seed in (42, 43):
    _, inputs = make_inputs(seed)
    for _ in range(2):
      jit(**inputs)
    result = jit(**inputs).numpy().copy()
    with io.BytesIO() as stream:
      pickle.dump(jit, stream, protocol=5)
      stream.seek(0)
      loaded = pickle.load(stream)
    actual = loaded(**inputs).numpy()
    np.testing.assert_array_equal(result, actual)
    if reference is not None and np.array_equal(reference, actual):
      raise RuntimeError("Warp output did not change with new frame input")
    reference = actual.copy()
  data = {'run': loaded, 'input_specs': {'input_frame': (frame_shape, 'uint8', Device.DEFAULT), 'M_inv': (transform_shape, 'float32', transform_device)}}
  with open(output, 'wb') as stream:
    pickle.dump(data, stream, protocol=5)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='kind', required=True)
  model = subparsers.add_parser('model')
  model.add_argument('onnx')
  model.add_argument('output')
  model.add_argument('--benchmark-runs', type=int, default=1)
  model.add_argument('--camera-resolutions', type=_parse_size, nargs='+', default=[])
  warp = subparsers.add_parser('warp')
  warp.add_argument('--frame', type=lambda value: tuple(int(part) for part in value.split(',')), required=True)
  warp.add_argument('--warp-to', type=_parse_size, required=True)
  warp.add_argument('--layout', choices=('yuv420', 'luma'), required=True)
  warp.add_argument('--frames', type=int, default=1)
  warp.add_argument('--transform-device')
  warp.add_argument('--border-fill', type=int, default=16)
  warp.add_argument('--output', required=True)
  args = parser.parse_args()
  if args.kind == 'model':
    if args.benchmark_runs < 1:
      parser.error('--benchmark-runs must be positive')
    compile_model(args.onnx, args.output, args.benchmark_runs, args.camera_resolutions)
  else:
    if len(args.frame) != 6 or any(part <= 0 for part in args.frame) or args.frames < 1 or min(args.warp_to) <= 0:
      parser.error('frame dimensions, frame count and warp size must be positive')
    if args.layout == 'luma' and args.frames != 1:
      parser.error('luma supports exactly one frame')
    compile_warp(args.frame, args.warp_to, args.layout, args.frames, args.transform_device, args.border_fill, args.output)
