import os
import unittest
from unittest import mock

from openpilot.common.test import MonkeyPatch


class TestMonkeyPatchEnvironment(unittest.TestCase):
  def setUp(self):
    self.patch = mock.patch.dict(os.environ, {}, clear=True)
    self.patch.start()
    self.addCleanup(self.patch.stop)
    self.owner = unittest.TestCase()
    self.addCleanup(self.owner.doCleanups)
    self.monkeypatch = MonkeyPatch(self.owner.addCleanup)

  def test_setenv_restores_existing_and_missing(self):
    os.environ["EXISTING"] = "original"
    self.monkeypatch.setenv("EXISTING", "changed")
    self.monkeypatch.setenv("NEW", "value")
    self.assertEqual(os.environ["EXISTING"], "changed")
    self.assertEqual(os.environ["NEW"], "value")
    self.owner.doCleanups()
    self.assertEqual(os.environ["EXISTING"], "original")
    self.assertNotIn("NEW", os.environ)

  def test_delenv_restores_value_after_multiple_changes(self):
    os.environ["EXISTING"] = ""
    self.monkeypatch.delenv("EXISTING")
    self.assertNotIn("EXISTING", os.environ)
    self.monkeypatch.setenv("EXISTING", "new")
    self.monkeypatch.delenv("EXISTING")
    self.owner.doCleanups()
    self.assertEqual(os.environ["EXISTING"], "")

  def test_delenv_missing(self):
    with self.assertRaises(KeyError):
      self.monkeypatch.delenv("MISSING")
    self.monkeypatch.delenv("MISSING", raising=False)
    os.environ["MISSING"] = "later"
    self.owner.doCleanups()
    self.assertEqual(os.environ["MISSING"], "later")

  def test_prepend_and_unrelated_environment(self):
    os.environ["EXISTING"] = "original"
    self.monkeypatch.setenv("EXISTING", "first", prepend=":")
    self.monkeypatch.setenv("NEW", "value", prepend=":")
    os.environ["UNRELATED"] = "keep"
    self.assertEqual(os.environ["EXISTING"], "first:original")
    self.assertEqual(os.environ["NEW"], "value")
    self.owner.doCleanups()
    self.assertEqual(os.environ["UNRELATED"], "keep")
