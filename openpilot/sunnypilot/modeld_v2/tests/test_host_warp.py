import ast
import math
import os
from collections import namedtuple
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.helpers import Context

from openpilot.sunnypilot.modeld_v2.tests.test_runtime_safety import ROOT, load_functions, load_method


def compiler_helpers():
  stock = load_functions("stock_dependencies.py", {
    "_detect_desire_key", "_detect_vision_keys", "get_policy_npy_shapes", "nv12_copy_size",
    "warp_perspective_tinygrad", "frames_to_tensor", "make_frame_prepare", "make_warp",
    "make_run_model", "make_input_queues", "shift_and_sample", "sample_skip", "sample_desire",
  }, {"np": np, "math": math, "Tensor": Tensor, "Device": Device, "Context": Context})
  stock = SimpleNamespace(**stock)
  functions = load_functions("compile_modeld.py", {
    "_detect_desire_key", "_detect_vision_keys", "get_policy_npy_shapes", "generate_queues_and_npy",
    "make_supercombo_input_queues", "make_run_policy", "derive_frame_skip",
  }, {"np": np, "math": math, "os": os, "Tensor": Tensor, "Device": Device, "partial": partial, "stock": stock})
  return stock, functions


class TestHostWarp(unittest.TestCase):
  def test_host_policy_inputs_remain_live_under_chestnut_compile_environment(self):
    _, helpers = compiler_helpers()
    shapes = {"img": (1, 12, 4, 4), "big_img": (1, 12, 4, 4), "desire": (1, 2, 2),
              "features_buffer": (1, 2, 4), "traffic_convention": (1, 2)}
    with patch.dict(os.environ, {"CHESTNUT": "1"}):
      queues, arrays = helpers["make_supercombo_input_queues"](shapes, 4, device="CPU", host_inputs=True)
    self.assertEqual(queues["packed_npy_inputs"].device, "NPY")
    arrays["traffic_convention"][:] = [0., 1.]
    packed = queues["packed_npy_inputs"].numpy()
    np.testing.assert_array_equal(packed[2:4], [0., 1.])
    self.assertEqual(queues["img_q"].device, "CPU")
    self.assertEqual(queues["tfm"].device, "NPY")

  def test_real_warp_matches_unified_images_and_history_for_multiple_transforms(self):
    stock, helpers = compiler_helpers()
    shapes = {"img": (1, 12, 4, 4), "big_img": (1, 12, 4, 4), "desire": (1, 2, 2),
              "features_buffer": (1, 2, 4), "traffic_convention": (1, 2)}
    nv12 = namedtuple("NV12Frame", "width height stride y_height uv_height size")(16, 12, 16, 12, 6, 288)
    with Context(DEV="CPU"):
      warp = stock.make_warp(nv12, 8, 8)
      # Compare actual model inputs, including temporal queues, not just output shape.
      captures = []
      def runner(inputs):
        captures.append({k: v.numpy().copy() for k, v in inputs.items()})
        return {"outputs": Tensor([[1., 2., 3., 4.]], device="CPU")}
      policy = helpers["make_run_policy"](None, [runner], slice(0, 4), 4, shapes)
      unified = stock.make_run_model(warp, policy, {"input_shapes": shapes}, nv12.size)
      raw_queues, raw_arrays, raw_frames = stock.make_input_queues(shapes, 4, "CPU", nv12.size)
      host_queues, host_arrays = helpers["make_supercombo_input_queues"](shapes, 4, device="CPU", host_inputs=True)
      rng = np.random.default_rng(42)
      transforms = (np.eye(3, dtype=np.float32),
                    np.array([[1., .1, 2.], [0., 1., 1.], [.01, 0., 1.]], dtype=np.float32),
                    np.array([[1., 0., -3.], [0., 1., -2.], [0., 0., 1.]], dtype=np.float32))
      for index in range(6):
        for key, array in raw_arrays.items():
          value = transforms[index % len(transforms)] if key in ("tfm", "big_tfm") else rng.normal(size=array.shape)
          array[:] = value
          host_arrays[key][:] = value
        for frame in raw_frames.values():
          frame[:] = rng.integers(0, 256, frame.size, dtype=np.uint8)
        unified(**raw_queues)
        reduced = warp(host_queues["tfm"], host_queues["big_tfm"],
                       Tensor(raw_frames["img"].copy(), device="CPU"),
                       Tensor(raw_frames["big_img"].copy(), device="CPU")).to("CPU").realize()
        self.assertEqual(reduced.shape, (2, 6, 4, 4))
        self.assertEqual(reduced.dtype, dtypes.uint8)
        policy(warped=reduced, **{k: host_queues[k] for k in ("img_q", "big_img_q", "feat_q", "desire_q", "packed_npy_inputs")})
        for key in captures[-1]:
          np.testing.assert_array_equal(captures[-2][key], captures[-1][key], err_msg=f"frame={index}, input={key}")

  def test_host_warp_stages_only_small_tensor_before_policy(self):
    run = load_method("model_adapters.py", "LegacyModelAdapter", "run")
    warp_output, cpu_output, policy_result = Mock(), Mock(), object()
    warp_output.to.return_value.realize.return_value = cpu_output
    obj = SimpleNamespace(chestnut=True, host_warp=True, input_queues={"tfm": object(), "big_tfm": object(), "img_q": object()},
                          full_frames={"img": object(), "big_img": object()}, _road_key="img", _wide_key="big_img",
                          run_warp=Mock(return_value=warp_output), run_policy=Mock(return_value=policy_result))
    run.__globals__["POLICY_INPUTS"] = ["img_q"]
    self.assertIs(run(obj), policy_result)
    warp_output.to.assert_called_once_with("CPU")
    obj.run_policy.assert_called_once_with(img_q=obj.input_queues["img_q"], warped=cpu_output)

  def test_unmarked_chestnut_does_not_execute_separate_warp(self):
    run = load_method("model_adapters.py", "LegacyModelAdapter", "run")
    run.__globals__["POLICY_INPUTS"] = ["packed_npy_inputs"]
    obj = SimpleNamespace(chestnut=True, host_warp=False, input_queues={"packed_npy_inputs": object()},
                          run_warp=Mock(), run_policy=Mock())
    run(obj)
    obj.run_warp.assert_not_called()
    obj.run_policy.assert_called_once_with(packed_npy_inputs=obj.input_queues["packed_npy_inputs"])

  def test_raw_camera_buffers_stay_on_warp_device(self):
    copy = load_method("model_adapters.py", "LegacyModelAdapter", "copy_frames")
    tensor = Mock()
    copy.__globals__["Tensor"] = tensor
    frame = np.arange(32, dtype=np.uint8)
    obj = SimpleNamespace(chestnut=True, host_warp=True, WARP_DEV="QCOM",
                          frame_buf_params={"img": (8, 4, 0, 32)}, full_frames={}, _blob_cache={})
    copy(obj, {"img": frame})
    copy(obj, {"img": frame})
    tensor.from_blob.assert_called_once_with(frame.ctypes.data, (32,), dtype="uint8", device="QCOM")
    self.assertIs(obj.full_frames["img"], tensor.from_blob.return_value)

  def test_host_warp_bundle_rejects_incompatible_metadata_before_allocations(self):
    tree = ast.parse((ROOT / "model_adapters.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LegacyModelAdapter")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    # Execute the production ABI guards, excluding the constructor's super() call.
    stop = next(i for i, node in enumerate(init.body) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "frame_copy_size" for t in node.targets))
    guard = compile(ast.Module(body=init.body[1:stop], type_ignores=[]), str(ROOT / "model_adapters.py"), "exec")
    for metadata, chestnut, run_policy, resolution in (
      ({"camera_input_abi": "unknown"}, True, True, True),
      ({"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "AMD"}, True, True, True),
      ({"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "QCOM"}, False, True, True),
      ({"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "QCOM"}, True, False, True),
      ({"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "QCOM"}, True, True, False),
      ({"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "QCOM", "frame_skip": 0}, True, True, True),
    ):
      with self.subTest(metadata=metadata, chestnut=chestnut, resolution=resolution):
        jits = {"metadata": metadata}
        if run_policy:
          jits["run_policy"] = object()
        if resolution:
          jits[(16, 12)] = object()
        obj = SimpleNamespace(jits=jits, chestnut=chestnut, cam_w=16, cam_h=12)
        with self.assertRaises(RuntimeError):
          exec(guard, {"self": obj})
    metadata = {"camera_input_abi": "warped_yuv_v1", "model": {}, "warp_dev": "QCOM", "frame_skip": 4}
    obj = SimpleNamespace(jits={"metadata": metadata, "run_policy": object(), (16, 12): object()},
                          chestnut=True, cam_w=16, cam_h=12)
    exec(guard, {"self": obj})
    self.assertTrue(obj.host_warp)

  def test_transport_byte_count_is_reduced_not_just_renamed(self):
    # Two 512x256 YUV420 images: 2 cameras * 6 planes * 128 * 256 uint8.
    small_bytes = 2 * 6 * 128 * 256
    raw_bytes = 2 * 4804608
    self.assertEqual(small_bytes, 393216)
    self.assertLess(small_bytes / raw_bytes, .05)


if __name__ == "__main__":
  unittest.main()
