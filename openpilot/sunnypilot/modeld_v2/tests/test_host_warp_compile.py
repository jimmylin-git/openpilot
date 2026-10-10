import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tinygrad import Context, Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from openpilot.selfdrive.modeld import compile_modeld as stock
from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.sunnypilot.modeld_v2.compile_modeld import (
  POLICY_INPUTS, make_host_warp_input_queues, make_run_policy,
)
from openpilot.sunnypilot.modeld_v2.host_warp import (
  HostWarpRuntime, WarpBackend, load_local_warp, make_compact_policy, warp_source_hash,
)

SHAPES = {
  'img': (1, 12, 2, 2), 'big_img': (1, 12, 2, 2), 'features_buffer': (1, 2, 2),
  'desire_pulse': (1, 3, 2), 'traffic_convention': (1, 2), 'action_t': (1, 2),
}


class EchoModel:
  graph_inputs = {key: SimpleNamespace(dtype=dtypes.float32) for key in SHAPES}

  def __init__(self, shapes=None):
    self.shapes = shapes if shapes is not None else SHAPES

  def __call__(self, inputs):
    if any(value.dtype != dtypes.float32 for value in inputs.values()):
      raise AssertionError("model inputs were not cast to graph dtypes")
    if any(inputs[key].shape != tuple(shape) for key, shape in self.shapes.items()):
      raise AssertionError("model input shapes changed")
    return {'out': Tensor.cat(*(inputs[key].flatten() for key in SHAPES))}


class TestHostWarpCompile(unittest.TestCase):
  def test_policy_compact_and_amd_match_and_preserve_recurrent_buffers(self):
    with Context(DEV='PYTHON'):
      policy = make_run_policy(None, [EchoModel()], slice(0, 2), 4, SHAPES)
      direct = TinyJit(policy, prune=True)
      direct_queues, direct_npy, direct_input = make_host_warp_input_queues(SHAPES, 4, 'PYTHON')
      compact_queues, compact_npy, compact_input = make_host_warp_input_queues(SHAPES, 4, 'PYTHON')
      compact_jit = TinyJit(make_compact_policy(policy, compact_input.control_bytes, compact_input.images.shape), prune=True)
      device_input = Tensor(np.zeros_like(compact_input.data), device='PYTHON').realize()
      host_input = Tensor(compact_input.data, device='NPY').realize()
      states = {key: compact_queues[key] for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q')}
      rng = np.random.default_rng(8)
      for _ in range(5):
        for name in direct_input.controls:
          values = rng.random(direct_npy[name].shape).astype(np.float32)
          direct_npy[name][:] = compact_npy[name][:] = values
        images = rng.integers(0, 256, compact_input.images.shape, dtype=np.uint8)
        compact_input.images[:] = images
        device_input._buffer().copy_from(host_input._buffer())
        expected = direct(**{key: direct_queues[key] for key in POLICY_INPUTS},
                          warped=Tensor(images, device='PYTHON').realize())
        actual = compact_jit(**{**{key: compact_queues[key] for key in POLICY_INPUTS},
                                'packed_npy_inputs': device_input})
        self.assertIsInstance(actual, Tensor)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        for key, state in states.items():
          self.assertIs(state, compact_queues[key])
          np.testing.assert_array_equal(state.numpy(), direct_queues[key].numpy())
      serialized = io.BytesIO()
      dump_oob(compact_jit, serialized)
      serialized.seek(0)
      loaded = load_oob(serialized)
      result = loaded(**{**{key: compact_queues[key] for key in POLICY_INPUTS},
                         'packed_npy_inputs': device_input})
      self.assertIsInstance(result, Tensor)
      self.assertTrue(np.isfinite(result.numpy()).all())

  def test_spatial_hidden_features_match_on_compact_and_amd_paths(self):
    shapes = {**SHAPES, 'features_buffer': (1, 2, 2, 2)}
    with Context(DEV='PYTHON'):
      policy = make_run_policy(None, [EchoModel(shapes)], slice(0, 4), 4, shapes)
      direct = TinyJit(policy, prune=True)
      direct_queues, direct_npy, _ = make_host_warp_input_queues(shapes, 4, 'PYTHON')
      compact_queues, compact_npy, compact = make_host_warp_input_queues(shapes, 4, 'PYTHON')
      compact_jit = TinyJit(make_compact_policy(policy, compact.control_bytes, compact.images.shape), prune=True)
      packed = Tensor(np.zeros_like(compact.data), device='PYTHON').realize()
      host = Tensor(compact.data, device='NPY').realize()
      for step in range(3):
        direct_npy['prev_feat'][:] = compact_npy['prev_feat'][:] = np.arange(4) + step
        compact.images[:] = step
        packed._buffer().copy_from(host._buffer())
        expected = direct(**{key: direct_queues[key] for key in POLICY_INPUTS},
                          warped=Tensor(compact.images, device='PYTHON').realize())
        actual = compact_jit(**{**{key: compact_queues[key] for key in POLICY_INPUTS}, 'packed_npy_inputs': packed})
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(compact_queues['feat_q'].numpy(), direct_queues['feat_q'].numpy())

  def test_captured_policy_switch_preserves_live_history_and_output(self):
    with Context(DEV='PYTHON'):
      policy = make_run_policy(None, [EchoModel()], slice(0, 2), 4, SHAPES)
      reference = TinyJit(policy, prune=True)
      fallback = TinyJit(policy, prune=True)
      reference_queues, reference_npy, _ = make_host_warp_input_queues(SHAPES, 4, 'PYTHON')
      queues, npy, compact = make_host_warp_input_queues(SHAPES, 4, 'PYTHON')
      optimized = TinyJit(make_compact_policy(policy, compact.control_bytes, compact.images.shape), prune=True)
      # Capture both entry points before testing a switch on shared live state.
      for _ in range(3):
        fallback(**{key: queues[key] for key in POLICY_INPUTS},
                 warped=Tensor(compact.images, device='PYTHON').realize()).numpy()
      for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q'):
        queues[key].assign(Tensor.zeros(*queues[key].shape, dtype=queues[key].dtype, device='PYTHON')).realize()
      packed = Tensor(np.zeros_like(compact.data), device='PYTHON').realize()
      host = Tensor(compact.data, device='NPY').realize()
      states = {key: queues[key] for key in ('img_q', 'big_img_q', 'feat_q', 'desire_q')}
      for step in range(7):
        for name in compact.controls:
          npy[name][:] = reference_npy[name][:] = step + 1
        compact.images[:] = step + 11
        images = Tensor(compact.images, device='PYTHON').realize()
        expected = reference(**{key: reference_queues[key] for key in POLICY_INPUTS}, warped=images)
        if step < 4:
          packed._buffer().copy_from(host._buffer())
          actual = optimized(**{**{key: queues[key] for key in POLICY_INPUTS}, 'packed_npy_inputs': packed})
        else:
          actual = fallback(**{key: queues[key] for key in POLICY_INPUTS}, warped=images)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        for key, state in states.items():
          self.assertIs(state, queues[key])
          np.testing.assert_array_equal(state.numpy(), reference_queues[key].numpy())

  def test_real_warp_jit_accepts_buffer_views_and_sees_each_new_frame(self):
    with Context(DEV='PYTHON'):
      raw = np.zeros(72 + 2 * 24, np.uint8)
      matrices = raw[:72].view(np.float32).reshape(2, 3, 3)
      matrices[:] = np.eye(3)
      jit = TinyJit(stock.make_warp(stock.NV12Frame(4, 4, 4, 4, 2, 24), 4, 4), prune=True)
      backend = WarpBackend(raw, 24, 'PYTHON', jit)
      for value in (7, 19, 33, 42):
        raw[72:] = value
        output = backend.prepare().numpy()
        self.assertEqual(output.shape, (2, 6, 2, 2))
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(output, value)

  def test_amd_fallback_runs_without_qcom_or_host_image_readback(self):
    with Context(DEV='PYTHON'):
      queues, npy, compact = make_host_warp_input_queues(SHAPES, 4, 'PYTHON')
      warp = TinyJit(stock.make_warp(stock.NV12Frame(4, 4, 4, 4, 2, 24), 4, 4), prune=True)
      direct = TinyJit(make_run_policy(None, [EchoModel()], slice(0, 2), 4, SHAPES), prune=True)
      with tempfile.TemporaryDirectory() as cache:
        runtime = HostWarpRuntime(compact, 24, (4, 4), (4, 4, 2, 24), warp, 'PYTHON', 'mici',
                                  warp_source_hash(), Path(cache), mock.Mock())
      npy['desire'][:] = 1
      runtime.matrices[:] = np.eye(3)
      runtime.frames[:] = 17
      result = runtime.run(mock.Mock(side_effect=AssertionError("must not run compact policy")), direct,
                           {key: queues[key] for key in POLICY_INPUTS})
      self.assertIsInstance(result, Tensor)
      self.assertEqual(runtime.backend, 'amd')
      self.assertEqual(runtime.last_timings['usb_input_bytes'], runtime.raw.nbytes + compact.control_bytes)

  def test_source_mismatch_is_rejected_before_qcom_or_cache_access(self):
    with Context(DEV='PYTHON'):
      with tempfile.TemporaryDirectory() as cache:
        with self.assertRaisesRegex(RuntimeError, 'differs from the artifact'):
          load_local_warp(np.zeros(120, np.uint8), 24, (4, 4), (4, 4, 2, 24),
                          (2, 6, 2, 2), '0' * 64, Path(cache))
        self.assertEqual(list(Path(cache).iterdir()), [])

  def test_local_warp_cache_round_trip_and_cleanup(self):
    real_context, real_backend = Context, WarpBackend
    with mock.patch('tinygrad.Context', side_effect=lambda **kwargs: real_context(DEV='PYTHON')):
      with mock.patch('openpilot.sunnypilot.modeld_v2.host_warp.WarpBackend',
                      side_effect=lambda raw, size, device, warp: real_backend(raw, size, 'PYTHON', warp)):
        with tempfile.TemporaryDirectory() as cache:
          raw = np.zeros(120, np.uint8)
          args = (raw, 24, (4, 4), (4, 4, 2, 24), (2, 6, 2, 2), warp_source_hash(), Path(cache))
          load_local_warp(*args)
          np.testing.assert_array_equal(raw, 0)
          self.assertEqual(len(list(Path(cache).iterdir())), 1)
          cached = load_local_warp(*args)
          raw[:72].view(np.float32).reshape(2, 3, 3)[:] = np.eye(3)
          raw[72:] = 77
          with Context(DEV='PYTHON'):
            np.testing.assert_array_equal(cached.prepare().numpy(), 77)
