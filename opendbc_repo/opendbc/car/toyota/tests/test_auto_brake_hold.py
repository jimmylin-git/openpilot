import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def make_hold(speed_mode=False):
  path = Path(__file__).resolve().parents[1] / "auto_brake_hold.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.Assign))
           and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GearShifter" for t in n.targets))]
  flag = SimpleNamespace(AUTO_BRAKE_HOLD=SimpleNamespace(value=1), AUTO_BRAKE_HOLD_SPEED=SimpleNamespace(value=2))
  namespace = {"ToyotaFlags": flag, "GearShifter": SimpleNamespace(drive="D")}
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
  hold = namespace["AutoBrakeHold"](SimpleNamespace(flags=3 if speed_mode else 1))
  cs = SimpleNamespace(out=SimpleNamespace(gearShifter="D", standstill=True, gasPressed=False,
                                          brakePressed=False, vEgo=0., cruiseState=SimpleNamespace(enabled=False)))
  assert namespace["BRAKE_HOLD_TIMER"] == 50
  return hold, cs


def assert_one_second(hold, cs):
  for _ in range(49):
    hold._update_state(cs)
    assert not hold.active
  hold._update_state(cs)
  assert hold.active


def test_one_second_threshold_and_brake_restarts_full_timer():
  hold, cs = make_hold()
  hold._update_state(cs)
  assert_one_second(hold, cs)
  cs.out.brakePressed = True
  hold._update_state(cs)
  assert not hold.active and hold.counter == 0
  cs.out.brakePressed = False
  assert_one_second(hold, cs)


@pytest.mark.parametrize("condition", ["gas", "moving", "gear", "cruise"])
def test_existing_release_conditions_reset_timer(condition):
  hold, cs = make_hold()
  hold._update_state(cs)
  assert_one_second(hold, cs)
  if condition == "gas":
    cs.out.gasPressed = True
  elif condition == "moving":
    cs.out.standstill = False
  elif condition == "gear":
    cs.out.gearShifter = "P"
  else:
    cs.out.cruiseState.enabled = True
  hold._update_state(cs)
  assert not hold.active and hold.counter == 0


def test_speed_mode_still_requires_exceeding_speed_threshold():
  hold, cs = make_hold(speed_mode=True)
  for _ in range(100):
    hold._update_state(cs)
  assert not hold.active and hold.counter == 0
  cs.out.vEgo = 5.1
  cs.out.standstill = False
  hold._update_state(cs)
  cs.out.vEgo = 0.
  cs.out.standstill = True
  hold._update_state(cs)
  assert_one_second(hold, cs)
  cs.out.gearShifter = "P"
  hold._update_state(cs)
  assert not hold._speed_gear_lock and not hold.active
