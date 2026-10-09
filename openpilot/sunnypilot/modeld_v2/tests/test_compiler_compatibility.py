import ast
import base64
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

from examples.openpilot.compile_onnx import compile_onnx
from examples.openpilot.compile_warp import NV12Frame, compile_warp
from examples.openpilot.helpers import allocate_inputs, dump_pickle, load_pickle
from openpilot.common.file_chunker import chunk_file, get_chunk_targets, open_file_chunked
from openpilot.common.model_pickle import load_oob as common_load_oob
from openpilot.sunnypilot.modeld_v2.helpers import load_oob
from openpilot.sunnypilot.modeld_v2.model_adapters import NativeTinygradAdapter
from tinygrad import Tensor


ROOT = Path(__file__).resolve().parents[4]
FRAME = NV12Frame(4, 4, 4, 4, 2, 24)

def stock_load_oob(file):
  # Execute the stock loader without importing its unrelated hardware/capnp dependencies.
  path = ROOT / "openpilot" / "selfdrive" / "modeld" / "helpers.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_oob")
  namespace = {"_load_oob": common_load_oob}
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
  return namespace["load_oob"](file)


class TestCompilerCompatibility(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name)

  def model(self):
    specs = [helper.make_tensor_value_info("new_img", TensorProto.UINT8, [2, 6, 2, 2]),
             helper.make_tensor_value_info("history", TensorProto.FLOAT, [1, 2]),
             helper.make_tensor_value_info("desire", TensorProto.FLOAT, [1, 2])]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2]) for name in ("outputs", "next_history")]
    graph = helper.make_graph([
      helper.make_node("Cast", ["new_img"], ["float_img"], to=TensorProto.FLOAT),
      helper.make_node("ReduceSum", ["float_img"], ["sum"], keepdims=0),
      helper.make_node("Add", ["history", "sum"], ["next_history"]),
      helper.make_node("Add", ["next_history", "desire"], ["outputs"]),
    ], "compiler_compatibility", specs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    slices = base64.b64encode(pickle.dumps({"hidden_state": slice(0, 2)})).decode()
    helper.set_model_props(model, {"output_slices": slices})
    onnx.checker.check_model(model)
    path = self.path / "model.onnx"
    onnx.save(model, path)
    return path, slices

  def roundtrip(self, artifact, out_of_band):
    stream = io.BytesIO()
    dump_pickle(artifact, stream, out_of_band=out_of_band)
    stream.seek(0)
    return load_oob(stream) if out_of_band else load_pickle(stream)

  def test_onnx_artifact_metadata_and_runtime_with_both_serializations(self):
    path, slices = self.model()
    for out_of_band in (False, True):
      with self.subTest(out_of_band=out_of_band):
        artifact = compile_onnx(path, device_inputs=("*",), benchmark_runs=1, out_of_band=out_of_band)
        loaded = self.roundtrip(artifact, out_of_band)
        self.assertEqual(set(loaded), {"metadata", "run", "input_specs", "output_specs"})
        self.assertEqual(loaded["metadata"]["metadata"]["output_slices"], slices)
        self.assertEqual(loaded["metadata"]["output_shapes"]["next_history"], (1, 2))
        for name, (_, _, device) in loaded["input_specs"].items():
          self.assertEqual(device, "CPU", name)
        for value in (1., 2.):
          inputs = allocate_inputs(loaded["input_specs"], lambda arrays, v=value: [a.fill(v) for a in arrays.values()])
          outputs = allocate_inputs(loaded["output_specs"])
          loaded["run"](output_buffers=outputs, **inputs)
          np.testing.assert_array_equal(outputs["next_history"].numpy(), np.full((1, 2), value * 49))
          np.testing.assert_array_equal(outputs["outputs"].numpy(), np.full((1, 2), value * 50))

  def test_luma_warp_identity_border_fill_and_host_transform(self):
    artifact = compile_warp(FRAME, (4, 4), layout="luma", border_fill=16, transform_device="NPY", benchmark_runs=1)
    loaded = self.roundtrip(artifact, False)
    self.assertEqual(loaded["input_specs"]["M_inv"], ((3, 3), "<f4", "NPY"))
    pixels = np.arange(24, dtype=np.uint8)
    transform = np.eye(3, dtype=np.float32)
    output = loaded["run"](input_frame=Tensor(pixels, device="CPU"), M_inv=Tensor(transform, device="NPY"))
    np.testing.assert_array_equal(output.numpy(), pixels[:16].reshape(1, 16))
    transform[0, 2] = -1
    output = loaded["run"](input_frame=Tensor(pixels, device="CPU"), M_inv=Tensor(transform, device="NPY")).numpy()
    expected = np.full((4, 4), 16, dtype=np.uint8)
    expected[:, 1:] = pixels[:16].reshape(4, 4)[:, :3]
    np.testing.assert_array_equal(output, expected.reshape(1, 16))

  def test_yuv_warp_two_camera_shape_and_plane_order(self):
    artifact = compile_warp(FRAME, (4, 4), layout="yuv420", frames=2, benchmark_runs=1)
    loaded = self.roundtrip(artifact, False)
    pixels = np.arange(24, dtype=np.uint8)
    data = np.stack((pixels, pixels + 30))
    matrices = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    output = loaded["run"](input_frame=Tensor(data, device="CPU"), M_inv=Tensor(matrices, device="CPU")).numpy()
    self.assertEqual(output.shape, (2, 6, 2, 2))
    for camera in range(2):
      y = data[camera, :16].reshape(4, 4)
      uv = data[camera, 16:].reshape(2, 4)
      expected = np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], uv[:, ::2], uv[:, 1::2]))
      np.testing.assert_array_equal(output[camera], expected)

  def test_nv12_stride_and_plane_height_padding(self):
    frame = NV12Frame(4, 4, 8, 8, 4, 96)
    artifact = compile_warp(frame, (4, 4), layout="yuv420", frames=2, benchmark_runs=1)
    loaded = self.roundtrip(artifact, False)
    pixels = np.full((2, 96), 255, dtype=np.uint8)
    y = np.arange(16, dtype=np.uint8).reshape(4, 4)
    uv = np.arange(16, 24, dtype=np.uint8).reshape(2, 4)
    for camera in range(2):
      pixels[camera, :64].reshape(8, 8)[:4, :4] = y + camera * 30
      pixels[camera, 64:].reshape(4, 8)[:2, :4] = uv + camera * 30
    transforms = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    output = loaded["run"](input_frame=Tensor(pixels, device="CPU"), M_inv=Tensor(transforms, device="CPU")).numpy()
    expected = np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2], uv[:, ::2], uv[:, 1::2]))
    np.testing.assert_array_equal(output, np.stack((expected, expected + 30)))

  def test_native_adapter_packed_upload_state_alias_and_reset(self):
    path, _ = self.model()
    model = self.roundtrip(compile_onnx(path, device_inputs=("*",), benchmark_runs=1, out_of_band=True), True)
    warp = self.roundtrip(compile_warp(FRAME, (4, 4), layout="yuv420", frames=2, benchmark_runs=1), False)
    model[(4, 4)] = warp["run"]
    with patch("openpilot.sunnypilot.modeld_v2.model_adapters.get_nv12_info", return_value=(4, 4, 2, 24)):
      adapter = NativeTinygradAdapter(model, 4, 4, "CPU", "CPU", "CPU", chestnut=True)
    self.assertEqual(adapter._vision_input_names, ["img", "big_img"])
    for name in ("history",):
      self.assertEqual(adapter.input_queues[name]._buffer().get_buf("CPU"),
                       adapter.outputs[f"next_{name}"]._buffer().get_buf("CPU"))
    frames = {"img": np.ones(24, dtype=np.uint8).tobytes(), "big_img": np.ones(24, dtype=np.uint8).tobytes()}
    adapter.copy_frames(frames)
    adapter.numpy_inputs["tfm"][:] = np.eye(3, dtype=np.float32)
    adapter.numpy_inputs["desire"][:] = 2
    np.testing.assert_array_equal(adapter.run().numpy(), [[50., 50.]])
    np.testing.assert_array_equal(adapter.run().numpy(), [[98., 98.]])
    adapter.reset_warmup_buffers()
    np.testing.assert_array_equal(adapter.input_queues["history"].numpy(), [[0., 0.]])
    self.assertFalse(adapter.packed_input.any())
    adapter.copy_frames(frames)
    adapter.numpy_inputs["tfm"][:] = np.eye(3, dtype=np.float32)
    adapter.numpy_inputs["desire"][:] = 2
    np.testing.assert_array_equal(adapter.run().numpy(), [[50., 50.]])

  def test_build_cli_commands_produce_loadable_artifacts(self):
    path, _ = self.model()
    scripts = ROOT / "tinygrad_repo" / "examples" / "openpilot"
    model_path, warp_path, dm_warp_path = self.path / "model.pkl", self.path / "warp.pkl", self.path / "dm_warp.pkl"
    commands = [
      [str(scripts / "compile_onnx.py"), str(path), str(model_path), "--device-input", "*", "--out-of-band", "--benchmark-runs", "1"],
      [str(scripts / "compile_warp.py"), "--frame", "4,4,4,4,2,24", "--warp-to", "4x4", "--layout", "yuv420",
       "--frames", "2", "--output", str(warp_path), "--benchmark-runs", "1"],
      [str(scripts / "compile_warp.py"), "--frame", "4,4,4,4,2,24", "--warp-to", "4x4", "--layout", "luma",
       "--border-fill", "16", "--transform-device", "NPY", "--output", str(dm_warp_path), "--benchmark-runs", "1"],
    ]
    for command in commands:
      result = subprocess.run([sys.executable, *command], env=os.environ.copy(), capture_output=True, text=True, timeout=120)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    with model_path.open("rb") as file:
      self.assertIn("output_specs", load_oob(file))
    with model_path.open("rb") as file:
      self.assertIn("output_specs", stock_load_oob(file))
    with warp_path.open("rb") as file:
      self.assertEqual(pickle.load(file)["input_specs"]["input_frame"][0], (2, 24))
    with dm_warp_path.open("rb") as file:
      self.assertEqual(pickle.load(file)["input_specs"]["M_inv"][2], "NPY")
    targets = get_chunk_targets(str(model_path), model_path.stat().st_size)
    chunk_file(str(model_path), targets)
    self.assertFalse(model_path.exists())
    with open_file_chunked(str(model_path)) as file:
      self.assertIn("output_specs", load_oob(file))
    wrapper = ROOT / "openpilot" / "sunnypilot" / "modeld_v2" / "compile_modeld_v2.py"
    wrapper_path = self.path / "wrapper.pkl"
    result = subprocess.run([sys.executable, str(wrapper), str(path), str(wrapper_path), "--device-input", "*",
                             "--out-of-band", "--benchmark-runs", "1"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    with wrapper_path.open("rb") as file:
      self.assertIn("output_specs", load_oob(file))

  def test_bad_compiler_options_fail_explicitly(self):
    path, _ = self.model()
    with self.assertRaisesRegex(ValueError, "Unknown inputs"):
      compile_onnx(path, device_inputs=("missing",), benchmark_runs=1)
    with self.assertRaisesRegex(ValueError, "Unknown warp layout"):
      compile_warp(FRAME, (4, 4), layout="missing")
    with self.assertRaisesRegex(ValueError, "benchmark_runs"):
      compile_warp(FRAME, (4, 4), benchmark_runs=0)


if __name__ == "__main__":
  unittest.main()
