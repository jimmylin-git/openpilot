import ast
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]


class TestChestnutBuildFailure(unittest.TestCase):
  def test_action_reports_not_ready_and_preserves_compiler_exit_status(self):
    path = ROOT / "openpilot" / "selfdrive" / "modeld" / "SConscript"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chestnut_action")
    for states, compiler_status, expected, checks in (
      ([False] * 10, 0, 1, 10),
      ([True], 0, 0, 1),
      ([False, False, True], 0, 0, 3),
      ([True], 7, 7, 1),
    ):
      with self.subTest(states=states, compiler_status=compiler_status):
        flash = ModuleType("openpilot.system.hardware.chestnut.flash")
        flash.link_up = Mock(side_effect=states)
        time = Mock()
        namespace = {"time": time, "Action": lambda callback, label: callback}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        env = Mock()
        env.Execute.return_value = compiler_status
        with patch.dict(sys.modules, {"openpilot.system.hardware.chestnut.flash": flash}):
          result = namespace["chestnut_action"]("compile command")([], [], env)
        self.assertEqual(result, expected)
        self.assertEqual(flash.link_up.call_count, checks)
        if True in states:
          env.Execute.assert_called_once_with("compile command")
          self.assertEqual(time.sleep.call_count, checks - 1)
        else:
          env.Execute.assert_not_called()
          self.assertEqual(time.sleep.call_count, 10)


if __name__ == "__main__":
  unittest.main()
