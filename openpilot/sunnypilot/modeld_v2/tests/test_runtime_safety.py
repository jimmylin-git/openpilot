import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_method(filename, cls, method):
  tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
  definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
  function = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method)
  namespace = {"np": np}
  module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
                      type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
  return namespace[method]


class TestRuntimeSafety(unittest.TestCase):
  def test_legacy_reset_real_cpu_tensors_preserves_queue_identity(self):
    from tinygrad import Tensor
    reset = load_method("model_adapters.py", "LegacyModelAdapter", "reset_warmup_buffers")
    queues = {key: Tensor(np.full((2, 2), np.nan, dtype=np.float32), device="CPU").realize()
              for key in ("feat_q", "desire_q")}
    original = dict(queues)
    host = {"prev_feat": np.full(2, np.nan)}
    obj = SimpleNamespace(numpy_inputs=host, input_queues=queues, chestnut=False,
                          full_frames={"img": object()}, _blob_cache={"img": object()})
    reset(obj)
    for key, tensor in queues.items():
      self.assertIs(tensor, original[key])
      np.testing.assert_array_equal(tensor.numpy(), np.zeros((2, 2)))
    self.assertFalse(obj.full_frames)
    self.assertFalse(obj._blob_cache)
    np.testing.assert_array_equal(host["prev_feat"], [0., 0.])

  def test_legacy_warmup_clears_device_history_and_host_frames(self):
    reset = load_method("model_adapters.py", "LegacyModelAdapter", "reset_warmup_buffers")
    host = np.full(4, np.nan, dtype=np.float32)
    frame = np.full(4, 255, dtype=np.uint8)
    queues = {}
    arrays = {}
    for key in ("img_q", "big_img_q", "feat_q", "desire_q"):
      array = np.full(4, np.nan)
      tensor = Mock()
      tensor.assign.side_effect = lambda value, a=array, t=tensor: (a.fill(value), t)[1]
      queues[key] = tensor
      arrays[key] = array
    packed = Mock()
    queues["packed_npy_inputs"] = packed
    obj = SimpleNamespace(numpy_inputs={"prev_feat": host}, input_queues=queues, chestnut=True,
                          frame_slots={"img": frame})
    reset(obj)
    self.assertTrue(np.all(host == 0))
    self.assertTrue(np.all(frame == 0))
    for key, tensor in queues.items():
      if key == "packed_npy_inputs":
        tensor.assign.assert_not_called()
      else:
        self.assertTrue(np.all(arrays[key] == 0))
        tensor.realize.assert_called_once()

  def state(self, outputs, model_type="supercombo"):
    host = {"desire": np.zeros(2), "tfm": np.eye(3), "big_tfm": np.eye(3), "prev_feat": np.zeros(2)}
    return SimpleNamespace(
      adapter=SimpleNamespace(copy_frames=Mock(), run=lambda: outputs, is_native=False),
      desire_key="desire", numpy_inputs=host, prev_desire=np.zeros(2), _vision_input_names=["img", "big_img"],
      _road_key="img", _wide_key="big_img", _combined_model_type=model_type, chestnut=True,
      vision_output_slices={"hidden_state": slice(0, 2)}, parser=Mock(), _policy_keys=["policy"],
      _policy_slices_list=[{}], _has_on_policy=False, mlsim=False,
    )

  def run_state(self, state):
    run = load_method("modeld.py", "ModelState", "run")
    return run(state, {}, {"img": np.eye(3), "big_img": np.eye(3)}, {"desire": np.zeros(2)})

  def test_nonfinite_supercombo_is_rejected_on_small_and_big(self):
    for chestnut in (False, True):
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(chestnut=chestnut, value=value):
          state = self.state(SimpleNamespace(numpy=lambda v=value: np.array([v, 1.])))
          state.chestnut = chestnut
          with self.assertRaisesRegex(RuntimeError, "stage=supercombo"):
            self.run_state(state)
          state.parser.parse_outputs.assert_not_called()
          self.assertTrue(np.all(state.numpy_inputs["prev_feat"] == 0))

  def test_split_policy_failure_is_rejected_before_parser_or_feedback(self):
    finite = SimpleNamespace(numpy=lambda: np.array([1., 2.]))
    bad = SimpleNamespace(numpy=lambda: np.array([np.nan]))
    state = self.state([finite, bad], "split")
    with self.assertRaisesRegex(RuntimeError, "stage=policy"):
      self.run_state(state)
    state.parser.parse_vision_outputs.assert_not_called()
    self.assertTrue(np.all(state.numpy_inputs["prev_feat"] == 0))

  def test_finite_output_preserves_existing_feedback(self):
    state = self.state(SimpleNamespace(numpy=lambda: np.array([1., 2.])))
    state.parser.parse_outputs.return_value = {"plan": np.array([3.])}
    result = self.run_state(state)
    self.assertEqual(result["plan"][0], 3.)
    np.testing.assert_array_equal(state.numpy_inputs["prev_feat"], [1., 2.])


if __name__ == "__main__":
  unittest.main()
