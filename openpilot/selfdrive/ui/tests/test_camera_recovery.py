import ast
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
CAMERAS = [ROOT / "selfdrive" / "ui" / "onroad" / "cameraview.py",
           ROOT / "selfdrive" / "ui" / "mici" / "onroad" / "cameraview.py"]


def load_camera(path):
  tree = ast.parse(path.read_text(encoding="utf-8"))
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CameraView")
  names = {"_render_egl", "_reset_connection", "_clear_textures", "_ensure_connection", "close"}
  methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
  namespace = {"COMMA_HARDWARE": True, "cloudlog": Mock(), "rl": Mock(), "VisionIpcClient": Mock(),
               "create_egl_image": Mock(), "destroy_egl_image": Mock(), "bind_egl_image_to_texture": Mock(),
               "CONNECTION_RETRY_INTERVAL": .2}
  namespace["rl"].get_time.return_value = 10.
  module = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    ast.ClassDef(name="CameraView", bases=[], keywords=[], body=methods, decorator_list=[]),
  ], type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
  view = namespace["CameraView"]()
  view._name, view._stream_type = "camerad", 0
  view.frame = SimpleNamespace(idx=0, width=1928, height=1208, stride=1984, uv_offset=0, fd=7)
  view.egl_images = {1: object()}
  view.texture_y = view.texture_uv = None
  view.egl_texture = SimpleNamespace(id=1)
  view.shader = SimpleNamespace(id=2)
  view.available_streams = [0]
  view.client = Mock()
  view._target_client = Mock()
  view._target_stream_type = 1
  view._switching = True
  view.last_connection_attempt = 0.
  view._update_texture_color_filtering = Mock()
  return view, namespace


class TestCameraRecovery(unittest.TestCase):
  def test_failed_import_discards_frame_cache_and_reconnects_once(self):
    for path in CAMERAS:
      with self.subTest(path=path):
        view, ns = load_camera(path)
        ns["create_egl_image"].return_value = None
        old_image = view.egl_images[1]
        view._render_egl(None, None)
        self.assertIsNone(view.frame)
        self.assertEqual(view.egl_images, {})
        self.assertEqual(view.available_streams, [])
        ns["destroy_egl_image"].assert_called_once_with(old_image)
        ns["VisionIpcClient"].assert_called_once_with("camerad", 0, conflate=True)
        self.assertIs(view.client, ns["VisionIpcClient"].return_value)
        self.assertEqual(view.last_connection_attempt, 10.)
        ns["bind_egl_image_to_texture"].assert_not_called()
        ns["cloudlog"].error.assert_called_once()
        view._render_egl(None, None)
        ns["create_egl_image"].assert_called_once()

  def test_recovery_throttles_reconnect_and_accepts_new_frame(self):
    for path in CAMERAS:
      with self.subTest(path=path):
        view, ns = load_camera(path)
        view._reset_connection()
        view.client.is_connected.return_value = False
        ns["rl"].get_time.return_value = 10.1
        self.assertFalse(view._ensure_connection())
        view.client.connect.assert_not_called()
        ns["rl"].get_time.return_value = 10.3
        view._initialize_textures = Mock()
        view.client.num_buffers = 4
        view.client.connect.return_value = True
        self.assertTrue(view._ensure_connection())
        view.client.connect.assert_called_once_with(False)
        frame = SimpleNamespace(idx=0, width=1928, height=1208, stride=1984, uv_offset=0, fd=9)
        view.frame = frame
        image = object()
        ns["create_egl_image"].return_value = image
        view._render_egl(None, None)
        self.assertIs(view.egl_images[0], image)
        ns["bind_egl_image_to_texture"].assert_called_once_with(1, image)

  def test_disconnect_clears_images_before_reimporting_buffers(self):
    for path in CAMERAS:
      with self.subTest(path=path):
        view, ns = load_camera(path)
        view.client.is_connected.return_value = False
        view.client.num_buffers = 4
        def connect(block, view=view):
          self.assertIsNone(view.frame)
          self.assertEqual(view.egl_images, {})
          return False
        view.client.connect.side_effect = connect
        self.assertFalse(view._ensure_connection())
        ns["destroy_egl_image"].assert_called_once()

  def test_close_releases_pending_stream_switch(self):
    for path in CAMERAS:
      with self.subTest(path=path):
        view, ns = load_camera(path)
        view.close()
        self.assertIsNone(view.frame)
        self.assertIsNone(view.client)
        self.assertIsNone(view._target_client)
        self.assertIsNone(view._target_stream_type)
        self.assertFalse(view._switching)
        view.close()
        ns["rl"].unload_shader.assert_called_once()
        ns["rl"].unload_texture.assert_called_once()


class TestEGLResources(unittest.TestCase):
  def setUp(self):
    path = ROOT / "system" / "ui" / "lib" / "egl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {"create_egl_image", "destroy_egl_image"})
             or (isinstance(n, ast.ClassDef) and n.name == "EGLImage")
             or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper())]
    self.egl = Mock(initialized=True, NO_IMAGE_KHR=None)
    self.logger = Mock()
    ns = {"os": os, "dataclass": dataclass, "Any": Any, "_egl": self.egl, "cloudlog": self.logger}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    self.create = ns["create_egl_image"]
    self.destroy = ns["destroy_egl_image"]
    self.file = tempfile.TemporaryFile()
    self.addCleanup(self.file.close)

  def test_invalid_fd_is_logged_without_import(self):
    self.assertIsNone(self.create(1, 1, 1, -1, 1))
    self.logger.exception.assert_called_once()
    self.egl.create_image_khr.assert_not_called()

  def test_success_owns_duplicate_and_destroy_is_idempotent(self):
    image = self.create(1, 1, 1, self.file.fileno(), 1)
    fd = image.fd
    os.fstat(fd)
    self.assertNotEqual(fd, self.file.fileno())
    self.destroy(image)
    self.assertEqual(image.fd, -1)
    with self.assertRaises(OSError):
      os.fstat(fd)
    replacement = os.dup(self.file.fileno())
    try:
      self.destroy(image)
      os.fstat(replacement)
      self.egl.destroy_image_khr.assert_called_once()
    finally:
      os.close(replacement)
    os.fstat(self.file.fileno())

  def test_failed_import_closes_duplicate(self):
    duplicated = []
    original_dup = os.dup
    def dup(fd):
      result = original_dup(fd)
      duplicated.append(result)
      return result
    for failure in ("null", "exception"):
      with self.subTest(failure=failure), patch.object(os, "dup", side_effect=dup):
        self.egl.create_image_khr.side_effect = RuntimeError("import failed") if failure == "exception" else None
        self.egl.create_image_khr.return_value = None
        if failure == "exception":
          with self.assertRaisesRegex(RuntimeError, "import failed"):
            self.create(1, 1, 1, self.file.fileno(), 1)
        else:
          self.assertIsNone(self.create(1, 1, 1, self.file.fileno(), 1))
        with self.assertRaises(OSError):
          os.fstat(duplicated[-1])
        os.fstat(self.file.fileno())

  def test_destroy_exception_still_closes_duplicate(self):
    image = self.create(1, 1, 1, self.file.fileno(), 1)
    fd = image.fd
    self.egl.destroy_image_khr.side_effect = RuntimeError("destroy failed")
    with self.assertRaisesRegex(RuntimeError, "destroy failed"):
      self.destroy(image)
    with self.assertRaises(OSError):
      os.fstat(fd)
    self.assertEqual(image.fd, -1)


if __name__ == "__main__":
  unittest.main()
