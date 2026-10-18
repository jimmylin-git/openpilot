from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from types import SimpleNamespace


CLUSTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_config import normalize_cluster_brightness_percent


def brightness_namespace():
    tree = ast.parse((CLUSTER_DIR / "main.py").read_text(encoding="utf-8"))
    constants = {
        "MIN_USB_BRIGHTNESS",
        "MAX_USB_BRIGHTNESS",
        "DEFAULT_USB_BRIGHTNESS",
        "OFFROAD_USB_BRIGHTNESS",
        "OFFROAD_BLACKOUT_SECONDS",
    }
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("resolved_usb_brightness", "offroad_blackout_active"):
            nodes.append(node)
    namespace = {"normalize_cluster_brightness_percent": normalize_cluster_brightness_percent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(CLUSTER_DIR / "main.py"), "exec"), namespace)
    return namespace


class BrightnessTests(unittest.TestCase):
    def test_offroad_brightness_failure_remains_pending_until_success(self):
        tree = ast.parse((CLUSTER_DIR / "main.py").read_text(encoding="utf-8"))
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                     and any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                             and ast.unparse(child.value.func) == "usb_display.set_brightness" for child in n.body))
        display = Mock()
        display.set_brightness.side_effect = [RuntimeError("USB timeout"), True]
        namespace = {"usb_display": display, "next_usb_brightness": 2,
                     "brightness_refresh_pending": True, "print": lambda *a, **kw: None}
        code = compile(ast.Module(body=[block], type_ignores=[]), "<brightness retry>", "exec")
        exec(code, namespace)
        self.assertTrue(namespace["brightness_refresh_pending"])
        exec(code, namespace)
        self.assertFalse(namespace["brightness_refresh_pending"])
        self.assertEqual(display.set_brightness.call_count, 2)

    def setUp(self):
        self.ns = brightness_namespace()
        self.resolve = self.ns["resolved_usb_brightness"]

    def test_manual_limits(self):
        for setting, expected in ((0, 5), (1, 5), (3, 5), (5, 5), (10, 10), (13, 13), (30, 30), (70, 70), (100, 70)):
            with self.subTest(setting=setting):
                self.assertEqual(self.resolve(setting, None, auto_enabled=False), expected)

    def test_auto_limits_and_mapping(self):
        source = Mock()
        for ambient, expected in ((0, 5), (30, 5), (40, 14), (50, 24), (65, 38), (80, 51), (100, 70), (120, 70)):
            with self.subTest(ambient=ambient):
                source.ambient_brightness_percent.return_value = ambient
                self.assertEqual(self.resolve(0, source, auto_enabled=True), expected)

    def test_auto_full_ambient_range_is_bounded_and_monotonic(self):
        source = Mock()
        previous = 5
        for ambient in range(30, 101):
            with self.subTest(ambient=ambient):
                source.ambient_brightness_percent.return_value = ambient
                brightness = self.resolve(0, source, auto_enabled=True)
                self.assertGreaterEqual(brightness, previous)
                self.assertLessEqual(brightness, 70)
                self.assertEqual(brightness, round(5 + (ambient - 30) / 70 * 65))
                previous = brightness
        self.assertEqual(previous, 70)

    def test_missing_ambient_uses_thirteen(self):
        self.assertEqual(self.resolve(0, None, auto_enabled=True), 13)
        source = Mock()
        source.ambient_brightness_percent.return_value = None
        self.assertEqual(self.resolve(0, source, auto_enabled=True), 13)

    def test_manual_setting_overrides_auto(self):
        source = Mock()
        self.assertEqual(self.resolve(100, source, auto_enabled=True), 70)
        source.ambient_brightness_percent.assert_not_called()

    def test_offroad_dim_is_one_percent(self):
        self.assertEqual(self.ns["OFFROAD_USB_BRIGHTNESS"], 1)

    def test_offroad_blackout_boundary_and_reentry(self):
        blackout = self.ns["offroad_blackout_active"]
        self.assertFalse(blackout(None, 1000))
        self.assertFalse(blackout(100, 109.999))
        self.assertTrue(blackout(100, 110))
        self.assertTrue(blackout(100, 200))
        tree = ast.parse((CLUSTER_DIR / "main.py").read_text(encoding="utf-8"))
        transition = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "offroad_since" for t in n.targets))
        namespace = {"time": SimpleNamespace(perf_counter=lambda: 300)}
        for offroad, expected in ((True, 300), (False, None), (True, 300)):
            namespace["is_offroad"] = offroad
            exec(compile(ast.Module(body=[transition], type_ignores=[]), "<offroad timer>", "exec"), namespace)
            self.assertEqual(namespace["offroad_since"], expected)
        self.assertFalse(blackout(namespace["offroad_since"], 309.999))
        self.assertTrue(blackout(namespace["offroad_since"], 310))

    def test_blackout_clears_black_without_drawing_hud_for_both_panels(self):
        tree = ast.parse((CLUSTER_DIR / "cluster_renderer.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ClusterUiRenderer")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("render", "set_blackout")]
        raylib = Mock()
        namespace = {"rl": raylib, "rl_color": lambda color: color}
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<blackout render>", "exec"), namespace)
        for width, height in ((1920, 462), (1920, 720)):
            with self.subTest(height=height):
                renderer = SimpleNamespace(width=width, height=height)
                namespace["set_blackout"](renderer, True)
                namespace["render"](renderer, None)
                raylib.clear_background.assert_called_once_with((0, 0, 0))
                raylib.reset_mock()
                namespace["set_blackout"](renderer, False)
                self.assertFalse(renderer._blackout)


if __name__ == "__main__":
    unittest.main()
