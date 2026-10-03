from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


CLUSTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_departure import DepartureReminder


def load_method(file_name, method_name, namespace):
    tree = ast.parse((CLUSTER_DIR / file_name).read_text(encoding="utf-8"))
    method = next(
        node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(CLUSTER_DIR / file_name), "exec"), namespace)
    return namespace[method_name]


class DepartureTests(unittest.TestCase):
    def setUp(self):
        self.reminder = DepartureReminder()

    def update(self, t, **overrides):
        args = {"allowed": True, "valid": True, "model_time": t, "position_x": (0.0,) * 32 + (31.0,)}
        args.update(overrides)
        return self.reminder.update(t, **args)

    def trigger(self, start=10.0):
        self.assertFalse(self.update(start))
        self.assertFalse(self.update(start + 0.2))
        self.assertTrue(self.update(start + 0.4))

    def test_confirmation_and_exact_distance_threshold(self):
        self.assertFalse(self.update(10.0, position_x=(30.0,) * 33))
        self.assertFalse(self.update(10.2))
        self.assertFalse(self.update(10.45))
        self.assertTrue(self.update(10.6))

    def test_repeated_sample_cannot_confirm(self):
        self.assertFalse(self.update(10.0))
        self.assertFalse(self.update(10.4, model_time=10.0))
        self.assertFalse(self.update(10.6, model_time=10.0))

    def test_sample_gap_and_backward_time_restart_confirmation(self):
        self.assertFalse(self.update(10.0))
        self.assertFalse(self.update(11.0))
        self.assertFalse(self.update(10.8))
        self.assertFalse(self.update(11.0))
        self.assertTrue(self.update(11.2))

    def test_invalid_inputs_never_trigger(self):
        cases = [
            {"valid": False},
            {"model_time": 9.0},
            {"model_time": 11.0},
            {"position_x": ()},
            {"position_x": (31.0,) * 32},
            {"position_x": (float("nan"),) + (31.0,) * 32},
            {"position_x": (float("inf"),) * 33},
        ]
        for args in cases:
            with self.subTest(args=args):
                self.reminder = DepartureReminder()
                self.assertFalse(self.update(10.0, **args))
                self.assertFalse(self.update(10.4, **args))

    def test_single_trigger_per_stop_and_display_expiry(self):
        self.trigger()
        for i in range(1, 16):
            self.update(10.4 + i * 0.2)
        self.assertFalse(self.update(13.6))
        self.assertFalse(self.update(13.8))
        self.assertFalse(self.update(14.0, allowed=False))
        self.trigger(14.2)

    def test_blocked_prediction_restarts_confirmation(self):
        self.assertFalse(self.update(10.0))
        self.assertFalse(self.update(10.2, position_x=(10.0,) * 33))
        self.assertFalse(self.update(10.4))
        self.assertFalse(self.update(10.6))
        self.assertTrue(self.update(10.8))

    def test_invalid_data_hides_without_repeated_alert(self):
        self.trigger()
        self.assertFalse(self.update(10.6, valid=False))
        self.assertFalse(self.update(10.8))
        self.assertFalse(self.update(11.2))


class LiveDepartureTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.method = load_method(
            "cluster_live.py",
            "_update_departure_reminder",
            {
                "time": SimpleNamespace(monotonic=lambda: self.now),
                "safe_optional_float": lambda obj, name: getattr(obj, name, None),
                "DEPARTURE_MAX_SAMPLE_GAP_SECONDS": 0.5,
            },
        )
        self.sm = {
            "carState": SimpleNamespace(vEgo=0.0, gasPressed=False, cruiseState=SimpleNamespace(enabled=False)),
            "selfdriveState": SimpleNamespace(enabled=False),
            "modelV2": SimpleNamespace(position=SimpleNamespace(x=(31.0,) * 33)),
        }

        class SubMaster(dict):
            pass

        self.sm = SubMaster(self.sm)
        self.sm.valid = dict.fromkeys(self.sm, True)
        self.sm.logMonoTime = dict.fromkeys(self.sm, int(self.now * 1e9))
        self.source = SimpleNamespace(
            sm=self.sm,
            _service_alive=Mock(return_value=True),
            _service_valid=Mock(return_value=True),
            _departure_reminder=DepartureReminder(),
            vehicle_started=Mock(return_value=True),
        )
        self.state = SimpleNamespace(gear_text="D")

    def sample(self, t):
        self.now = t
        self.sm.logMonoTime = dict.fromkeys(self.sm, int(t * 1e9))
        return self.method(self.source, self.state)

    def test_live_trigger_without_driver_monitoring_or_radar(self):
        self.assertFalse(self.sample(10.0))
        self.assertFalse(self.sample(10.2))
        self.assertTrue(self.sample(10.4))

    def test_disallowed_vehicle_states(self):
        for attribute, value in (("gear_text", "R"), ("gear_text", "P"), ("gear_text", "N")):
            setattr(self.state, attribute, value)
            self.assertFalse(self.sample(10.0))
            self.assertFalse(self.sample(10.4))
        self.state.gear_text = "D"
        car = self.sm["carState"]
        for speed in (0.1, -0.1, 1.0, float("nan")):
            car.vEgo = speed
            self.assertFalse(self.sample(10.0))
            self.assertFalse(self.sample(10.4))
        car.vEgo = 0.0
        for obj, name in ((car, "gasPressed"), (car.cruiseState, "enabled"), (self.sm["selfdriveState"], "enabled")):
            setattr(obj, name, True)
            self.assertFalse(self.sample(10.0))
            self.assertFalse(self.sample(10.4))
            setattr(obj, name, False)
        self.source.vehicle_started.return_value = False
        self.assertFalse(self.sample(10.0))
        self.assertFalse(self.sample(10.4))

    def test_stale_invalid_and_missing_timestamps_clear_banner(self):
        self.sample(10.0)
        self.sample(10.2)
        self.assertTrue(self.sample(10.4))
        self.sm.valid["modelV2"] = False
        self.assertFalse(self.method(self.source, self.state))
        self.sm.valid["modelV2"] = True
        self.sm.logMonoTime.pop("carState")
        self.assertFalse(self.method(self.source, self.state))

    def test_offroad_resets_even_with_stale_model(self):
        self.sample(10.0)
        self.sample(10.2)
        self.assertTrue(self.sample(10.4))
        self.now = 12.0
        self.source.vehicle_started.return_value = False
        self.assertFalse(self.method(self.source, self.state))
        self.source.vehicle_started.return_value = True
        self.assertFalse(self.sample(12.2))
        self.assertFalse(self.sample(12.4))
        self.assertTrue(self.sample(12.6))

    def test_each_required_service_must_be_fresh_alive_and_valid(self):
        for service in self.sm:
            for failure in ("stale", "invalid", "dead"):
                with self.subTest(service=service, failure=failure):
                    self.setUp()
                    self.sample(10.0)
                    self.now = 10.4
                    self.sm.logMonoTime = dict.fromkeys(self.sm, int(self.now * 1e9))
                    if failure == "stale":
                        self.sm.logMonoTime[service] = int(9.0 * 1e9)
                    elif failure == "invalid":
                        self.sm.valid[service] = False
                    else:
                        self.source._service_alive.side_effect = lambda name, service=service: name != service
                    self.assertFalse(self.method(self.source, self.state))


class DepartureRenderingTests(unittest.TestCase):
    def test_banner_and_critical_alert_precedence(self):
        draw = load_method(
            "cluster_renderer.py",
            "_draw_departure_reminder",
            {
                "DESIGN_WIDTH": 1920,
                "DESIGN_HEIGHT": 720,
                "GREEN": (0, 255, 0),
                "WHITE": (255, 255, 255),
            },
        )
        hud = SimpleNamespace(_rounded_rect=Mock(), _draw_text=Mock())
        for show, critical in ((False, False), (True, True)):
            draw(hud, SimpleNamespace(departure_reminder=show, wheel_critical=critical))
            hud._draw_text.assert_not_called()
        draw(hud, SimpleNamespace(departure_reminder=True, wheel_critical=False))
        self.assertEqual(hud._draw_text.call_args.args[0], "READY TO GO")


if __name__ == "__main__":
    unittest.main()
