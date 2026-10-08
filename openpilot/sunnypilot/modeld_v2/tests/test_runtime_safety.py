import ast
import enum
import inspect
import subprocess
import sys
import threading
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

def load_functions(filename, names, namespace):
  tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
  functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
  module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
                      type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
  return namespace


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

  def test_parsed_nonfinite_outputs_do_not_update_feedback(self):
    for model_type in ("supercombo", "split"):
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(model_type=model_type, value=value):
          finite = SimpleNamespace(numpy=lambda: np.array([1., 2.]))
          state = self.state(finite if model_type == "supercombo" else [finite, finite], model_type)
          state.parser.parse_outputs.return_value = {"pose": np.array([value])}
          state.parser.parse_vision_outputs.return_value = {"hidden_state": np.array([1., 2.])}
          state.parser.parse_policy_outputs.return_value = {"pose": np.array([value])}
          with self.assertRaisesRegex(RuntimeError, "parsed model output not finite: stage=pose"):
            self.run_state(state)
          np.testing.assert_array_equal(state.numpy_inputs["prev_feat"], [0., 0.])

  def test_pose_validity_checks_all_published_vectors(self):
    logger = Mock()
    fill = load_functions("fill_model_msg.py", {"fill_pose_msg"}, {"np": np, "cloudlog": logger})["fill_pose_msg"]
    shapes = {"pose": 6, "pose_stds": 6, "wide_from_device_euler": 3, "road_transform": 6,
              "wide_from_device_euler_stds": 3, "road_transform_stds": 6}
    for key in shapes:
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(key=key, value=value):
          outputs = {k: np.zeros((1, size)) for k, size in shapes.items()}
          outputs[key][0, -1] = value
          msg = SimpleNamespace(cameraOdometry=SimpleNamespace())
          fill(msg, outputs, 123, 0, 456, True)
          self.assertFalse(msg.valid)
    outputs = {k: np.zeros((1, size)) for k, size in shapes.items()}
    for seen, dropped, valid in ((True, 0, True), (False, 0, False), (True, 1, False)):
      msg = SimpleNamespace(cameraOdometry=SimpleNamespace())
      fill(msg, outputs, 123, dropped, 456, seen)
      self.assertEqual(msg.valid, valid)
      self.assertEqual(msg.cameraOdometry.frameId, 123)
      self.assertEqual(msg.cameraOdometry.timestampEof, 456)
    outputs["pose_stds"][0, 0] = -1
    fill(msg, outputs, 123, 0, 456, True)
    self.assertFalse(msg.valid)

  def test_negative_parsed_uncertainty_is_rejected_before_feedback(self):
    state = self.state(SimpleNamespace(numpy=lambda: np.array([1., 2.])))
    state.parser.parse_outputs.return_value = {"pose_stds": np.array([-1.])}
    with self.assertRaisesRegex(RuntimeError, "negative uncertainty: stage=pose_stds"):
      self.run_state(state)
    np.testing.assert_array_equal(state.numpy_inputs["prev_feat"], [0., 0.])

  def test_unsupported_legacy_chestnut_split_fails_before_allocating(self):
    init = load_method("model_adapters.py", "LegacyModelAdapter", "__init__")
    # Supply the base initialization because the extracted method has no __class__ closure.
    init.__globals__["super"] = lambda: SimpleNamespace(__init__=Mock())
    init.__globals__["nv12_copy_size"] = lambda *_: 4
    obj = SimpleNamespace(jits={"metadata": {"vision": {}, "policy": {}}}, chestnut=True, nv12_info=(1, 1, 1))
    with self.assertRaisesRegex(RuntimeError, "packed-camera ABI"):
      init(obj)


class TestModelCompatibility(unittest.TestCase):
  def factory(self, cls):
    namespace = load_functions("helpers.py", {"_pad_args", "_dynamic_factory"}, {"inspect": inspect, "enum": enum})
    return namespace["_dynamic_factory"](cls)

  def test_constructor_internal_typeerror_is_not_retried(self):
    calls = []

    class Broken:
      def __init__(self, value):
        calls.append(value)
        raise TypeError("internal allocation failure")

    with self.assertRaisesRegex(TypeError, "internal allocation failure"):
      self.factory(Broken)(3)
    self.assertEqual(calls, [3])

  def test_compatibility_preserves_keywords_and_trims_old_positional_fields(self):
    class Current:
      def __init__(self, value, *, mode="safe"):
        self.value, self.mode = value, mode

    result = self.factory(Current)(3, "obsolete", mode="custom")
    self.assertEqual((result.value, result.mode), (3, "custom"))
    result = self.factory(Current)(value=4)
    self.assertEqual((result.value, result.mode), (4, "safe"))

  def test_missing_required_fields_fail_before_constructor(self):
    calls = []

    class Current:
      def __init__(self, value, *, required):
        calls.append(value)

    with self.assertRaises(TypeError):
      self.factory(Current)(3)
    self.assertEqual(calls, [])

  def test_positional_defaults_and_constructor_failure_after_adaptation(self):
    class Current:
      def __init__(self, value=1, /, mode="safe"):
        self.values = value, mode

    self.assertEqual(self.factory(Current)().values, (1, "safe"))
    self.assertEqual(self.factory(Current)(2, "custom", "obsolete").values, (2, "custom"))
    calls = []

    class Broken:
      def __init__(self, value, *, mode="safe"):
        calls.append((value, mode))
        raise TypeError("internal failure")

    with self.assertRaisesRegex(TypeError, "internal failure"):
      self.factory(Broken)(3, "obsolete", mode="custom")
    self.assertEqual(calls, [(3, "custom")])

  def test_variadic_fields_and_enums_are_preserved(self):
    class Current:
      def __init__(self, value, *args, mode="safe", **kwargs):
        self.values = value, args, mode, kwargs

    self.assertEqual(self.factory(Current)(1, 2, 3, mode="custom", extra=4).values, (1, (2, 3), "custom", {"extra": 4}))

    class Kind(enum.Enum):
      ONE = 1

    self.assertIs(self.factory(Kind), Kind)


class TestCalibrationInputSafety(unittest.TestCase):
  def test_saved_state_bounds_and_shapes(self):
    path = ROOT.parents[1] / "selfdrive" / "locationd" / "calibrationd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Calibrator")
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    namespace = {"np": np, "cloudlog": Mock(), "RPY_INIT": np.zeros(3), "HEIGHT_INIT": np.array([1.22]),
                 "WIDE_FROM_DEVICE_EULER_INIT": np.zeros(3), "INPUTS_WANTED": 50}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), reset],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    reset = namespace["reset"]
    valid = [np.array([0., 0.03, 0.01]), 5, np.array([0., 0.01, 0.02]), np.array([1.3])]
    for index, invalid in ((0, np.array([0., 0.])), (0, np.array([[0., 0., 0.]])),
                           (0, np.array([0., np.nan, 0.])), (1, 51), (1, -1), (1, 1.5), (1, np.inf),
                           (2, np.zeros(2)), (3, np.zeros(2)), (3, np.array([np.inf]))):
      with self.subTest(index=index, invalid=invalid):
        args = valid.copy()
        args[index] = invalid
        obj = SimpleNamespace()
        reset(obj, *args)
        self.assertEqual(obj.valid_blocks, 0)
        np.testing.assert_array_equal(obj.rpy, np.zeros(3))
        self.assertEqual(obj.rpys.shape, (50, 3))
        self.assertEqual(obj.heights.shape, (50, 1))
    for blocks in (0, 5, 50):
      obj = SimpleNamespace()
      reset(obj, valid[0], blocks, valid[2], valid[3], smooth_from=np.array([0., 0.02, 0.]))
      self.assertEqual(obj.valid_blocks, blocks)
      np.testing.assert_array_equal(obj.rpy, valid[0])
      self.assertEqual(obj.old_rpy_weight, 1.)
    reset(obj, *valid, smooth_from=np.array([0., np.nan, 0.]))
    self.assertEqual(obj.old_rpy_weight, 0.)

  def test_production_input_guard_rejects_bad_samples_without_mutation(self):
    path = ROOT.parents[1] / "selfdrive" / "locationd" / "calibrationd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Calibrator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "handle_cam_odom")
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
               and n.targets[0].attr == "old_rpy_weight")
    fn.body = [*fn.body[:end], ast.Return(value=ast.Constant(value=True))]
    logger = Mock()
    namespace = {"np": np, "cloudlog": logger}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    guard = namespace["handle_cam_odom"]
    obj = SimpleNamespace(v_ego=20., _invalid_cam_odom=False, old_rpy_weight=0.5, idx=10, valid_blocks=5)
    vectors = [[20., 0., 0.], [0.] * 3, [0.] * 3, [0.001] * 3, [0., 0., 1.22], [0.001] * 3]
    for index in range(len(vectors)):
      for value in (np.nan, np.inf, -np.inf):
        bad = [list(v) for v in vectors]
        bad[index][1] = value
        self.assertIsNone(guard(obj, *bad))
    logger.error.assert_called_once()
    self.assertEqual((obj.old_rpy_weight, obj.idx, obj.valid_blocks), (0.5, 10, 5))
    self.assertTrue(guard(obj, *vectors))
    self.assertFalse(obj._invalid_cam_odom)
    for index in range(len(vectors)):
      bad = [list(v) for v in vectors]
      bad[index] = [0., 0.]
      self.assertIsNone(guard(obj, *bad))
    for index in (3, 5):
      bad = [list(v) for v in vectors]
      bad[index][1] = -1.
      self.assertIsNone(guard(obj, *bad))
    obj.v_ego = np.nan
    self.assertIsNone(guard(obj, *vectors))
    obj.v_ego = 20.
    vectors[2] = vectors[4] = vectors[5] = []
    self.assertTrue(guard(obj, *vectors))

  def test_invalid_camera_odometry_message_is_not_consumed(self):
    path = ROOT.parents[1] / "selfdrive" / "locationd" / "calibrationd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.While))
    gate = next(n for n in loop.body if isinstance(n, ast.If))
    condition = compile(ast.Expression(body=gate.test), str(path), "eval")
    for updated, valid, accepted in ((True, True, True), (True, False, False), (False, True, False)):
      sm = SimpleNamespace(updated={"cameraOdometry": updated}, valid={"cameraOdometry": valid})
      self.assertEqual(eval(condition, {"sm": sm}), accepted)


class TestBigModelLoader(unittest.TestCase):
  def load(self, factory):
    namespace = load_functions("modeld.py", {"load_big_model"}, {
      "threading": threading, "cloudlog": Mock(), "ModelState": factory, "BIG_MODEL_TIMEOUT": 60,
    })
    return namespace["load_big_model"], namespace["cloudlog"]

  def test_success_returns_warmed_model(self):
    model = Mock()
    factory = Mock(return_value=model)
    load, _ = self.load(factory)
    self.assertIs(load(1928, 1208), model)
    model.warmup.assert_called_once()
    factory.assert_called_once_with(cam_w=1928, cam_h=1208, chestnut=True)

  def test_completed_load_failure_keeps_small_fallback(self):
    for phase in ("init", "warmup"):
      with self.subTest(phase=phase):
        model = Mock()
        factory = Mock(return_value=model)
        if phase == "init":
          factory.side_effect = RuntimeError("USB disconnected")
        else:
          model.warmup.side_effect = RuntimeError("non-finite warmup")
        load, logger = self.load(factory)
        self.assertIsNone(load(1928, 1208))
        logger.exception.assert_called_once()

  def test_timeout_raises_instead_of_returning_small_fallback(self):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def warmup():
      entered.set()
      release.wait(5)
      finished.set()

    model = Mock()
    model.warmup.side_effect = warmup
    load, _ = self.load(Mock(return_value=model))
    try:
      with self.assertRaisesRegex(TimeoutError, "restarting modeld"):
        load(1928, 1208, timeout=0.02)
      self.assertTrue(entered.is_set())
      self.assertFalse(finished.is_set())
    finally:
      release.set()
      self.assertTrue(finished.wait(2))

  def test_main_timeout_clears_flags_and_propagates_before_small_model(self):
    tree = ast.parse((ROOT / "modeld.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in main.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "CHESTNUT"
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "load_big_model"
                         for call in ast.walk(n)))
    params = Mock()
    namespace = {"CHESTNUT": True, "params": params, "cloudlog": Mock(),
                 "vipc_client_main": SimpleNamespace(width=1928, height=1208),
                 "load_big_model": Mock(side_effect=TimeoutError("timed out"))}
    with self.assertRaises(TimeoutError):
      exec(compile(ast.Module(body=[block], type_ignores=[]), "modeld.py", "exec"), namespace)
    self.assertEqual(params.put_bool.call_args_list[0].args, ("ChestnutActive", False))
    self.assertEqual(params.put_bool.call_args_list[1].args, ("ChestnutLoading", False))

  def test_timeout_process_exits_with_daemon_loader_still_blocked(self):
    tree = ast.parse((ROOT / "modeld.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_big_model")
    script = """import threading
from types import SimpleNamespace
BIG_MODEL_TIMEOUT = 60
cloudlog = SimpleNamespace(exception=lambda message: print(message))
class ModelState:
  def __init__(self, **kwargs): pass
  def warmup(self): threading.Event().wait()
"""
    script += ast.unparse(function) + "\nload_big_model(1928, 1208, timeout=0.02)\nprint('small model started')\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("TimeoutError", result.stderr)
    self.assertNotIn("small model started", result.stdout)


if __name__ == "__main__":
  unittest.main()
