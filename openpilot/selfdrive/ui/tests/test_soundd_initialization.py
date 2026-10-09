import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TestSounddInitialization(unittest.TestCase):
  def setUp(self):
    source = Path(__file__).resolve().parents[1] / "soundd.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Soundd")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_stream")
    self.logger, self.sleep = Mock(), Mock()
    namespace = {"cloudlog": self.logger, "time": SimpleNamespace(sleep=self.sleep),
                 "SAMPLE_RATE": 48000, "SAMPLE_BUFFER": 4096}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    self.get_stream = namespace["get_stream"]
    self.sound = SimpleNamespace(callback=Mock())

    class PortAudioError(Exception):
      pass

    self.error_type = PortAudioError
    self.sd = Mock()
    self.sd.PortAudioError = PortAudioError

  def test_success_preserves_stream_configuration(self):
    stream = self.get_stream(self.sound, self.sd)
    self.assertIs(stream, self.sd.OutputStream.return_value)
    self.sd.OutputStream.assert_called_once_with(channels=1, samplerate=48000, callback=self.sound.callback, blocksize=4096)
    self.sleep.assert_not_called()

  def test_transient_failure_reinitializes_and_reports_cause(self):
    stream = object()
    self.sd.OutputStream.side_effect = [self.error_type("ALSA unavailable"), stream]
    self.assertIs(self.get_stream(self.sound, self.sd), stream)
    self.assertEqual(self.sd._initialize.call_count, 2)
    self.logger.exception.assert_called_once()
    self.sleep.assert_called_once_with(3)

  def test_final_failure_preserves_original_exception(self):
    failure = self.error_type("Invalid output device")
    self.sd.OutputStream.side_effect = failure
    with self.assertRaises(self.error_type) as raised:
      self.get_stream(self.sound, self.sd)
    self.assertIs(raised.exception, failure)
    self.assertEqual(self.logger.exception.call_count, 10)
    self.assertEqual(self.sleep.call_count, 9)

  def test_programming_error_is_not_retried_as_audio_failure(self):
    self.sd.OutputStream.side_effect = TypeError("bad stream argument")
    with self.assertRaises(TypeError):
      self.get_stream(self.sound, self.sd)
    self.sd.OutputStream.assert_called_once()
    self.sleep.assert_not_called()


if __name__ == "__main__":
  unittest.main()
