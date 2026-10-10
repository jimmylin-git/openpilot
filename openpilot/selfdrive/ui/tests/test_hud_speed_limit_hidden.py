import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TestHudSpeedLimitHidden(unittest.TestCase):
  def test_hud_does_not_construct_update_or_render_speed_limit_indicator(self):
    source = Path(__file__).resolve().parents[1] / "sunnypilot/onroad/hud_renderer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    self.assertNotIn("SpeedLimitRenderer", names)
    self.assertNotIn("speed_limit_renderer", attrs)

  def test_other_hud_overlays_still_render(self):
    source = Path(__file__).resolve().parents[1] / "sunnypilot/onroad/hud_renderer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_render")
    namespace = {"rl": SimpleNamespace(Rectangle=object), "ui_state": SimpleNamespace(torque_bar=False)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    parent = Mock()
    namespace["_render"].__globals__["super"] = Mock(return_value=parent)
    renderers = ("developer_ui", "road_name_renderer", "smart_cruise_control_renderer",
                 "turn_signal_controller", "circular_alerts_renderer", "rocket_fuel")
    obj = SimpleNamespace(**{name: Mock() for name in renderers})
    namespace["ui_state"].sm = object()
    rect = object()
    namespace["_render"](obj, rect)
    parent._render.assert_called_once_with(rect)
    for name in renderers[:-1]:
      getattr(obj, name).render.assert_called_once_with(rect)
    obj.rocket_fuel.render.assert_called_once_with(rect, namespace["ui_state"].sm)


if __name__ == "__main__":
  unittest.main()
