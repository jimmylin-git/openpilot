import ast
import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[4]


def load_method(path, cls, method, namespace):
  tree = ast.parse(path.read_text(encoding="utf-8"))
  definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
  function = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method)
  module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
                      type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
  return namespace[method]


class TestSetupDownload(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.destination = Path(self.temp.name) / "installer"
    self.url_path = Path(self.temp.name) / "url"
    self.gui = Mock()
    namespace = {"os": os, "time": time, "urllib": urllib, "USER_AGENT": "test", "HARDWARE": Mock(),
                 "INSTALLER_DESTINATION_PATH": str(self.destination), "INSTALLER_URL_PATH": str(self.url_path),
                 "gui_app": self.gui}
    self.download = load_method(ROOT / "openpilot" / "system" / "ui" / "tici_setup.py", "Setup", "_download_thread", namespace)
    self.setup = SimpleNamespace(download_url="https://example.invalid/installer", download_failed=Mock(), download_progress=0)
    self.created = []
    original = tempfile.mkstemp

    def create(**kwargs):
      result = original(dir=self.temp.name, **kwargs)
      self.created.append(result)
      return result

    self.mkstemp = patch("tempfile.mkstemp", side_effect=create)
    self.mkstemp.start()
    self.addCleanup(self.mkstemp.stop)

  def assert_temporary_resources_closed(self):
    for fd, path in self.created:
      self.assertFalse(Path(path).exists())
      with self.assertRaises(OSError):
        os.fstat(fd)

  def test_all_http_errors_report_failure_and_clean_up(self):
    for status in (403, 404, 409, 500):
      with self.subTest(status=status):
        self.setup.download_failed.reset_mock()
        error = urllib.error.HTTPError(self.setup.download_url, status, "failed", {}, io.BytesIO(b"branch unavailable"))
        with patch("urllib.request.urlopen", side_effect=error):
          self.download(self.setup)
        self.setup.download_failed.assert_called_once()
        message = self.setup.download_failed.call_args.args[1]
        if status == 409:
          self.assertEqual(message, "branch unavailable")
        else:
          self.assertIn(str(status), message)
        self.assertFalse(self.destination.exists())
        self.assert_temporary_resources_closed()
    self.gui.request_close.assert_not_called()

  def test_timeout_reports_failure(self):
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
      self.download(self.setup)
    self.setup.download_failed.assert_called_once()
    self.assert_temporary_resources_closed()

  def test_non_elf_reports_failure(self):
    response = io.BytesIO(b"not an installer")
    response.headers = {"content-length": "16"}
    with patch("urllib.request.urlopen", return_value=response):
      self.download(self.setup)
    self.setup.download_failed.assert_called_once_with(self.setup.download_url, "No custom software found at this URL.")
    self.assert_temporary_resources_closed()
    self.gui.request_close.assert_not_called()

  def test_success_preserves_installer_and_url(self):
    payload = b"\x7fELFtest"
    response = io.BytesIO(payload)
    response.headers = {"content-length": str(len(payload))}
    with patch("urllib.request.urlopen", return_value=response), patch("time.sleep"):
      self.download(self.setup)
    self.assertEqual(self.destination.read_bytes(), payload)
    self.assertEqual(self.url_path.read_text(), self.setup.download_url)
    self.assertEqual(self.setup.download_progress, 100)
    self.setup.download_failed.assert_not_called()
    self.gui.request_close.assert_called_once()
    self.assert_temporary_resources_closed()


class TestAlertTranslation(unittest.TestCase):
  def test_mici_alert_translation_uses_imported_helper(self):
    path = ROOT / "openpilot" / "selfdrive" / "ui" / "mici" / "onroad" / "alert_renderer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "openpilot.system.ui.lib.multilang" and
                        any(alias.name == "tr" for alias in n.names) for n in tree.body))
    def translate(text):
      return {"Cruise Fault": "Translated Fault"}.get(text, text)
    method = load_method(path, "AlertRenderer", "_tr_alert_text", {"tr": translate})
    self.assertEqual(method(None, "Cruise Fault"), "Translated Fault")
    self.assertEqual(method(None, "Cruise Fault: code 1"), "Translated Fault: code 1")
    self.assertEqual(method(None, "Unknown message"), "Unknown message")


if __name__ == "__main__":
  unittest.main()
