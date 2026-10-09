import ast
import io
from pathlib import Path
import pickle
import struct
import tempfile
from types import SimpleNamespace
import unittest

from openpilot.common.file_chunker import ChunkStream, get_chunk_name, get_manifest_path, open_file_chunked
from openpilot.common.model_pickle import load_oob
from openpilot.sunnypilot.modeld_v2.helpers import dump_oob, load_oob as dynamic_load_oob


ROOT = Path(__file__).resolve().parents[3]


class ShortReadStream(io.BytesIO):
  def readinto(self, target):
    return super().readinto(memoryview(target)[:3])


class TestModelLoading(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = Path(self.directory.name) / "model.pkl"
    stream = io.BytesIO()
    dump_oob({"buffer": pickle.PickleBuffer(b"12345678")}, stream)
    self.payload = stream.getvalue()

  def test_complete_artifact_and_short_reads_with_both_loaders(self):
    for loader in (load_oob, dynamic_load_oob):
      for stream_type in (io.BytesIO, ShortReadStream):
        with self.subTest(loader=loader, stream=stream_type):
          self.assertEqual(bytes(loader(stream_type(self.payload))["buffer"]), b"12345678")

  def test_every_truncation_is_rejected_by_both_loaders(self):
    for loader in (load_oob, dynamic_load_oob):
      for length in range(len(self.payload)):
        with self.subTest(loader=loader, length=length):
          with self.assertRaises((EOFError, pickle.UnpicklingError)):
            loader(io.BytesIO(self.payload[:length]))

  def test_negative_lengths_rejected_by_both_loaders(self):
    opcode_size = struct.unpack('<q', self.payload[:8])[0]
    buffer_header = 8 + opcode_size
    corrupt = (struct.pack('<q', -1),
               self.payload[:buffer_header] + struct.pack('<q', -1) + self.payload[buffer_header + 8:])
    for loader in (load_oob, dynamic_load_oob):
      for payload in corrupt:
        with self.subTest(loader=loader, payload=payload):
          with self.assertRaisesRegex(ValueError, "negative model"):
            loader(io.BytesIO(payload))

  def test_empty_buffer_is_valid(self):
    stream = io.BytesIO()
    dump_oob(pickle.PickleBuffer(b""), stream)
    for loader in (load_oob, dynamic_load_oob):
      self.assertEqual(bytes(loader(io.BytesIO(stream.getvalue()))), b"")

  def test_chunked_artifact_crosses_headers_and_buffer_boundaries(self):
    paths = []
    for offset in range(0, len(self.payload), 3):
      path = self.path.with_name(f"chunk{offset}")
      path.write_bytes(self.payload[offset:offset + 3])
      paths.append(path)
    with io.BufferedReader(ChunkStream(paths)) as file:
      self.assertEqual(bytes(dynamic_load_oob(file)["buffer"]), b"12345678")

  def test_reader_close_releases_current_file_and_is_idempotent(self):
    self.path.write_bytes(b"x" * 32768)
    reader = open_file_chunked(self.path)
    self.addCleanup(reader.close)
    reader.read(1)
    raw = reader.raw
    underlying = raw._f
    self.assertFalse(underlying.closed)
    reader.close()
    reader.close()
    self.assertTrue(underlying.closed)
    self.assertTrue(raw.closed)
    with self.assertRaises(ValueError):
      raw.readinto(bytearray(1))

  def test_reader_exception_closes_current_chunk(self):
    count = 2
    for index in range(count):
      Path(get_chunk_name(self.path, index, count)).write_bytes(b"x" * 32768)
    Path(get_manifest_path(self.path)).write_text(str(count))
    with self.assertRaisesRegex(RuntimeError, "load failed"):
      with open_file_chunked(self.path) as reader:
        reader.read(32769)
        underlying = reader.raw._f
        raise RuntimeError("load failed")
    self.assertTrue(reader.closed)
    self.assertTrue(underlying.closed)

  def test_chunk_advance_closes_previous_file(self):
    first, second = self.path.with_name("first"), self.path.with_name("second")
    first.write_bytes(b"abc")
    second.write_bytes(b"def")
    with ChunkStream([first, second]) as raw:
      self.assertEqual(raw.read(1), b"a")
      previous = raw._f
      self.assertEqual(raw.read(3), b"bcd")
      current = raw._f
      self.assertTrue(previous.closed)
      self.assertFalse(current.closed)
    self.assertTrue(current.closed)

  def test_all_model_entry_points_close_reader_on_success_and_failure(self):
    cases = (
      ("openpilot/sunnypilot/modeld_v2/modeld.py", "_init_combined", (self.path, 4, 4, None), 2),
      ("openpilot/selfdrive/modeld/modeld.py", "__init__", (4, 4, False), 2),
      ("openpilot/selfdrive/modeld/dmonitoringmodeld.py", "__init__", (4, 4), 1),
    )
    self.path.write_bytes(b"x" * 32768)
    for path, name, args, statements in cases:
      tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
      cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelState")
      function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
      function.body = function.body[:statements] + [ast.Return(value=ast.Name(id="jits", ctx=ast.Load()))]
      for failure in (False, True):
        with self.subTest(path=path, failure=failure):
          opened = []
          handles = []

          def open_reader(*_, opened=opened):
            reader = open_file_chunked(self.path)
            opened.append(reader)
            return reader

          def load(reader, handles=handles, failure=failure):
            reader.read(1)
            handles.append(reader.raw._f)
            if failure:
              raise EOFError("incomplete model buffer")
            return {"metadata": {}}

          namespace = {"open_file_chunked": open_reader, "load_oob": load,
                       "ModelStateBase": SimpleNamespace(__init__=lambda _: None),
                       "modeld_pkl_path": lambda _: self.path, "MODEL_PKL_PATH": self.path,
                       "cloudlog": SimpleNamespace(warning=lambda *_: None)}
          module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                    function], type_ignores=[])
          exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
          if failure:
            with self.assertRaisesRegex(EOFError, "incomplete model buffer"):
              namespace[name](SimpleNamespace(), *args)
          else:
            self.assertEqual(namespace[name](SimpleNamespace(), *args), {"metadata": {}})
          self.assertTrue(opened[0].closed)
          self.assertTrue(handles[0].closed)


if __name__ == "__main__":
  unittest.main()
