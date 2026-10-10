import os
import threading
import logging
import json
import io
import unittest
from unittest import mock
import requests
from pathlib import Path
from openpilot.common.hardware.hw import Paths

from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.uploader import clear_locks, main, Uploader, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE

from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase
from openpilot.system.loggerd import uploader as uploader_module


class TestUploadUrlResponse(unittest.TestCase):
  def setUp(self):
    self.uploader = Uploader.__new__(Uploader)
    self.uploader.api = mock.Mock()
    self.uploader.dongle_id = "test-device"
    self.uploader.last_filename = ""
    self.stream = io.BytesIO(b"file data")
    stream_patch = mock.patch.object(uploader_module, "get_upload_stream", return_value=(self.stream, 9))
    put_patch = mock.patch.object(uploader_module.requests, "put")
    fake_patch = mock.patch.object(uploader_module, "fake_upload", False)
    for patcher in (stream_patch, put_patch, fake_patch):
      self.addCleanup(patcher.stop)
    self.stream_mock = stream_patch.start()
    self.put = put_patch.start()
    fake_patch.start()

  def response(self, status: int, body: str):
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.headers["Content-Type"] = "application/json" if body.startswith("{") else "text/html"
    self.uploader.api.get.return_value = response
    return response

  def test_http_errors_are_not_parsed_or_uploaded(self):
    for status in (401, 403, 404, 429, 500):
      with self.subTest(status=status):
        self.response(status, "<html>error</html>")
        with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
          self.uploader.do_upload("boot/test", "unused")
    self.put.assert_not_called()
    self.stream_mock.assert_not_called()

  def test_invalid_json_and_shape_are_rejected(self):
    for body in ("", "<html>not JSON</html>", "[]", "null", "{}", '{"url": 1, "headers": {}}',
                 '{"url": "https://storage.test/file", "headers": []}',
                 '{"url": "https://storage.test/file", "headers": {"key": 1}}',
                 '{"url": "file:///tmp/test", "headers": {}}'):
      with self.subTest(body=body):
        self.response(200, body)
        with self.assertRaises(ValueError):
          self.uploader.do_upload("boot/test", "unused")
    self.put.assert_not_called()
    self.stream_mock.assert_not_called()

  def test_valid_url_uploads_and_closes_stream(self):
    self.response(200, '{"url": "https://storage.test/file", "headers": {"x-test": "value"}}')
    result = self.uploader.do_upload("boot/test", "unused")
    self.assertIs(result, self.put.return_value)
    self.put.assert_called_once_with("https://storage.test/file", data=self.stream, headers={"x-test": "value"}, timeout=10)
    self.assertTrue(self.stream.closed)

  def test_ignored_response_does_not_open_file(self):
    response = self.response(412, "")
    self.assertIs(self.uploader.do_upload("boot/test", "unused"), response)
    self.put.assert_not_called()
    self.stream_mock.assert_not_called()

  def test_put_failure_closes_stream(self):
    self.response(200, '{"url": "https://storage.test/file", "headers": {}}')
    self.put.side_effect = requests.Timeout("upload timed out")
    with self.assertRaises(requests.Timeout):
      self.uploader.do_upload("boot/test", "unused")
    self.assertTrue(self.stream.closed)

  def test_success_and_explicit_ignore_mark_file_uploaded(self):
    with mock.patch.object(uploader_module.os.path, "getsize", return_value=9), \
         mock.patch.object(uploader_module, "setxattr") as tag, \
         mock.patch.object(uploader_module.time, "monotonic", side_effect=range(6)), \
         mock.patch.object(self.uploader, "do_upload") as upload:
      for status in (200, 201, 412):
        with self.subTest(status=status):
          upload.return_value.status_code = status
          upload.return_value.request.headers = {"Content-Length": "9"}
          self.assertTrue(self.uploader.upload("qlog", "route/qlog", "unused", 1, False))
      self.assertEqual(tag.call_count, 3)

  def test_failures_do_not_mark_file_uploaded(self):
    with mock.patch.object(uploader_module.os.path, "getsize", return_value=9), \
         mock.patch.object(uploader_module, "setxattr") as tag, \
         mock.patch.object(uploader_module.cloudlog, "event") as event:
      self.response(404, "<html>not found</html>")
      self.assertFalse(self.uploader.upload("qlog", "route/qlog", "unused", 1, False))
      for status in (401, 403, 500):
        self.response(200, '{"url": "https://storage.test/file", "headers": {}}')
        self.put.return_value.status_code = status
        self.assertFalse(self.uploader.upload("qlog", "route/qlog", "unused", 1, False))
      tag.assert_not_called()
      self.assertEqual(self.uploader.last_filename, "")
      self.assertEqual(sum(call.args[0] == "upload_failed" for call in event.call_args_list), 4)


class FakeLogHandler(logging.Handler):
  def __init__(self):
    logging.Handler.__init__(self)
    self.condition = threading.Condition()
    self.reset()

  def reset(self):
    with self.condition:
      self.upload_order = []
      self.upload_ignored = []

  def emit(self, record):
    try:
      j = json.loads(record.getMessage())
      with self.condition:
        if j["event"] == "upload_success":
          self.upload_order.append(j["key"])
        if j["event"] == "upload_ignored":
          self.upload_ignored.append(j["key"])
        self.condition.notify_all()
    except Exception:
      pass

  def wait_for_uploads(self, count: int, ignored: bool = False):
    uploads = self.upload_ignored if ignored else self.upload_order
    with self.condition:
      assert self.condition.wait_for(lambda: len(uploads) >= count, timeout=1), "Uploader did not process all files"

log_handler = FakeLogHandler()
cloudlog.addHandler(log_handler)


class TestUploader(UploaderTestCase):
  def setup_method(self):
    super().openpilot_setup_method()
    log_handler.reset()

  def start_thread(self):
    self.end_event = threading.Event()
    self.up_thread = threading.Thread(target=main, args=[self.end_event])
    self.up_thread.daemon = True
    self.up_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.up_thread.join()

  def gen_files(self, lock=False, xattr: bytes | None = None, boot=True) -> list[Path]:
    f_paths = []
    for t in ["qlog", "rlog", "dcamera.hevc", "fcamera.hevc"]:
      f_paths.append(self.make_file_with_data(self.seg_dir, t, 1, lock=lock, upload_xattr=xattr))

    if boot:
      f_paths.append(self.make_file_with_data("boot", f"{self.seg_dir}", 1, lock=lock, upload_xattr=xattr))
    return f_paths

  def gen_order(self, seg1: list[int], seg2: list[int], boot=True) -> list[str]:
    keys = []
    if boot:
      keys += [f"boot/{self.seg_format.format(i)}.zst" for i in seg1]
      keys += [f"boot/{self.seg_format2.format(i)}.zst" for i in seg2]
    keys += [f"{self.seg_format.format(i)}/qlog.zst" for i in seg1]
    keys += [f"{self.seg_format2.format(i)}/qlog.zst" for i in seg2]
    return keys

  def test_upload(self):
    self.gen_files(lock=False)
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_upload_with_wrong_xattr(self):
    self.gen_files(lock=False, xattr=b'0')
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_upload_ignored(self):
    self.set_ignore()
    self.gen_files(lock=False)
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order), ignored=True)
    self.join_thread()

    assert len(log_handler.upload_order) == 0, "Some files were not ignored"
    assert not len(log_handler.upload_ignored) < len(exp_order), "Some files failed to ignore"
    assert not len(log_handler.upload_ignored) > len(exp_order), "Some files were ignored twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not ignored"

    assert log_handler.upload_ignored == exp_order, "Files ignored in wrong order"

  def test_upload_files_in_create_order(self):
    seg1_nums = [0, 1, 2, 10, 20]
    for i in seg1_nums:
      self.seg_dir = self.seg_format.format(i)
      self.gen_files(boot=False)
    seg2_nums = [5, 50, 51]
    for i in seg2_nums:
      self.seg_dir = self.seg_format2.format(i)
      self.gen_files(boot=False)

    exp_order = self.gen_order(seg1_nums, seg2_nums, boot=False)

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_no_upload_with_lock_file(self):
    f_paths = self.gen_files(lock=True, boot=False)
    uploader = Uploader("0000000000000000", Paths.log_root())

    for f_path in f_paths:
      fn = f_path.with_suffix(f_path.suffix.replace(".zst", ""))
      assert all(candidate[2] != str(fn) for candidate in uploader.list_upload_files(metered=False)), "Locked file selected for upload"

  def test_no_upload_with_xattr(self):
    f_paths = self.gen_files(lock=False, xattr=UPLOAD_ATTR_VALUE)
    uploader = Uploader("0000000000000000", Paths.log_root())
    upload_candidates = {candidate[2] for candidate in uploader.list_upload_files(metered=False)}
    assert upload_candidates.isdisjoint(map(str, f_paths)), "Uploaded file selected again"

  def test_clear_locks_on_startup(self, mocker):
    f_paths = self.gen_files(lock=True, boot=False)
    locks_cleared = threading.Event()

    def clear_locks_and_signal(root):
      clear_locks(root)
      locks_cleared.set()

    mocker.patch("openpilot.system.loggerd.uploader.clear_locks", side_effect=clear_locks_and_signal)
    self.start_thread()
    assert locks_cleared.wait(timeout=1), "Uploader did not clear locks on startup"
    self.join_thread()

    for f_path in f_paths:
      lock_path = f_path.with_suffix(f_path.suffix + ".lock")
      assert not lock_path.is_file(), "File lock not cleared on startup"
