import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "card.py"


class FakeSubMaster:
  def __init__(self):
    self.frame = 0
    self.alive = {"carControl": True}
    self.valid = {"carControl": True}
    self.control = SimpleNamespace(cruiseControl=SimpleNamespace(cancel=False))

  def __getitem__(self, service):
    return self.control


class TestToyotaCruiseDiagnostics(unittest.TestCase):
  def setUp(self):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Car")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_log_toyota_cruise_state")
    self.logger = Mock()
    namespace = {"cloudlog": self.logger}
    module = ast.Module(body=[
      ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    self.log = namespace[method.name]
    values = {"PCM_CRUISE": {"CRUISE_ACTIVE": 0, "CRUISE_STATE": 0},
              "PCM_CRUISE_2": {"MAIN_ON": 0, "ACC_FAULTED": 0, "LOW_SPEED_LOCKOUT": 0}}
    timestamps = {message: dict.fromkeys(signals, 123456789) for message, signals in values.items()}
    self.parser = SimpleNamespace(bus=0, vl=values, ts_nanos=timestamps)
    self.obj = SimpleNamespace(CP=SimpleNamespace(brand="toyota"), CI=SimpleNamespace(can_parsers={"pt": self.parser}),
                               sm=FakeSubMaster(), CC_prev=SimpleNamespace(cruiseControl=SimpleNamespace(cancel=False)),
                               _cruise_log_state=None, _cruise_log_frame=-500, _last_actuation_nanos=None)
    self.cs = SimpleNamespace(cruiseState=SimpleNamespace(available=False, enabled=False),
                              accFaulted=False, carFaultedNonCritical=False, canValid=True, canTimeout=False)

  def test_received_zero_is_distinguished_from_unreceived_default(self):
    self.log(self.obj, self.cs)
    event = self.logger.event.call_args.kwargs
    self.assertEqual(event["pcm"][1]["signals"]["MAIN_ON"], 0)
    self.assertEqual(event["pcm"][1]["timestamps_ns"]["MAIN_ON"], 123456789)
    self.assertIsNone(event["last_applied_cancel"])
    self.logger.reset_mock()
    self.obj.sm.frame = 500
    self.parser.ts_nanos["PCM_CRUISE_2"]["MAIN_ON"] = 0
    self.log(self.obj, self.cs)
    self.assertEqual(self.logger.event.call_args.kwargs["pcm"][1]["timestamps_ns"]["MAIN_ON"], 0)
    self.assertFalse(self.cs.cruiseState.available)
    self.assertFalse(self.cs.cruiseState.enabled)

  def test_transition_and_five_second_heartbeat_are_logged_without_frame_spam(self):
    self.log(self.obj, self.cs)
    for frame in range(1, 500):
      self.obj.sm.frame = frame
      self.log(self.obj, self.cs)
    self.assertEqual(self.logger.event.call_count, 1)
    self.obj.sm.frame = 500
    self.log(self.obj, self.cs)
    self.assertEqual(self.logger.event.call_count, 2)
    self.cs.cruiseState.available = True
    self.parser.vl["PCM_CRUISE_2"]["MAIN_ON"] = 1
    self.obj.sm.frame = 501
    self.log(self.obj, self.cs)
    self.assertEqual(self.logger.event.call_count, 3)
    self.assertEqual(self.logger.event.call_args.kwargs["pcm"][1]["signals"]["MAIN_ON"], 1)

  def test_requested_and_previously_applied_cancel_are_separate(self):
    self.obj.sm.control.cruiseControl.cancel = True
    self.obj._last_actuation_nanos = 1000
    self.log(self.obj, self.cs)
    event = self.logger.event.call_args.kwargs
    self.assertTrue(event["requested_cancel"])
    self.assertFalse(event["last_applied_cancel"])
    self.assertEqual(event["last_actuation_time_ns"], 1000)
    self.obj.sm.control.cruiseControl.cancel = False
    self.obj.CC_prev.cruiseControl.cancel = True
    self.log(self.obj, self.cs)
    event = self.logger.event.call_args.kwargs
    self.assertFalse(event["requested_cancel"])
    self.assertTrue(event["last_applied_cancel"])

  def test_fault_and_control_connection_changes_are_logged(self):
    self.log(self.obj, self.cs)
    self.cs.accFaulted = True
    self.log(self.obj, self.cs)
    self.assertTrue(self.logger.event.call_args.kwargs["acc_faulted"])
    self.obj.sm.alive["carControl"] = False
    self.log(self.obj, self.cs)
    self.assertFalse(self.logger.event.call_args.kwargs["car_control_alive"])
    self.assertEqual(self.logger.event.call_count, 3)

  def test_unsupported_dsu_uses_its_actual_parser_bus(self):
    self.parser.vl = {"DSU_CRUISE": {"MAIN_ON": 1}}
    self.parser.ts_nanos = {"DSU_CRUISE": {"MAIN_ON": 42}}
    self.parser.bus = 2
    self.log(self.obj, self.cs)
    self.assertEqual(self.logger.event.call_args.kwargs["pcm"], [
      {"message": "DSU_CRUISE", "bus": 2, "signals": {"MAIN_ON": 1}, "timestamps_ns": {"MAIN_ON": 42}},
    ])

  def test_other_brands_are_untouched(self):
    self.obj.CP.brand = "honda"
    self.obj.CI = None
    self.log(self.obj, self.cs)
    self.logger.event.assert_not_called()
    self.assertIsNone(self.obj._cruise_log_state)


if __name__ == "__main__":
  unittest.main()
