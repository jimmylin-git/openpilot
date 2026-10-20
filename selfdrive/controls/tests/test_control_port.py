"""Hardware-free regression coverage for the dpegpu01 control port."""
import ast
from collections import deque
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CV = SimpleNamespace(KPH_TO_MS=1 / 3.6, MS_TO_KPH=3.6, DEG_TO_RAD=math.pi / 180)
DESIRE = SimpleNamespace(none=0, turnLeft=1, turnRight=2, laneChangeLeft=3, laneChangeRight=4)
SOURCE = SimpleNamespace(cruise='cruise', lead0='lead0', lead1='lead1', e2e='e2e', trafficStop='trafficStop', dtsc='dtsc')
LOG = SimpleNamespace(Desire=DESIRE, LongitudinalPlan=SimpleNamespace(LongitudinalPlanSource=SOURCE))


class FakeParams:
  def get_bool(self, key):
    return key in ('dp_lon_traffic_stop', 'dp_lane_turn_desire')

  def get(self, key):
    return 20 if key == 'dp_lane_turn_value' else 0


def load(relative, names=None, **namespace):
  tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
  nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef))
           and (names is None or getattr(n, 'name', None) in names
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
  ns = dict(np=np, math=math, CV=CV, DT_MDL=0.05, Params=FakeParams,
            log=LOG, cloudlog=Mock(), deque=deque, time=SimpleNamespace(monotonic=lambda: 100.0))
  ns.update(namespace)
  future = ast.parse('from __future__ import annotations').body
  exec(compile(ast.Module(body=[*future, *nodes], type_ignores=[]), relative, 'exec'), ns)
  return SimpleNamespace(**ns)


@pytest.mark.parametrize('offset_cm', [-10, 0, 10])
@pytest.mark.parametrize('speed,smoothing', [(0, 0), (0.3, 0), (10, 0.5)])
def test_model_offset_does_not_accumulate(offset_cm, speed, smoothing):
  names = ['get_action_from_model']
  ns = load('selfdrive/modeld/modeld.py', names,
            get_accel_from_plan=lambda *a, **kw: (0.0, False),
            get_curvature_from_plan=lambda *a: 0.01,
            smooth_value=lambda value, prev, smooth: (value + prev) / 2 if smooth else value,
            ModelConstants=SimpleNamespace(T_IDXS=[]),
            Plan=SimpleNamespace(VELOCITY=0, ACCELERATION=1, T_FROM_CURRENT_EULER=2, ORIENTATION_RATE=3),
            LAT_SMOOTH_SECONDS=smoothing, LONG_SMOOTH_SECONDS=0.3, MIN_LAT_CONTROL_SPEED=0.3,
            log=SimpleNamespace(ModelDataV2=SimpleNamespace(Action=SimpleNamespace)))
  offset = 2 * (offset_cm / 100) / 900
  previous = SimpleNamespace(desiredCurvature=0.01 + offset, desiredAcceleration=0)
  for _ in range(1200):
    previous = ns.get_action_from_model({'plan': np.zeros((1, 3, 4, 3))}, previous, 0, 0, speed, offset_cm)
  assert previous.desiredCurvature == pytest.approx(0.01 + offset)


def test_ltd_uses_lane_weakness_and_latches_until_blinker_or_speed_changes():
  ns = load('dragonpilot/selfdrive/controls/lib/lane_turn_desire.py')
  ctrl = ns.LaneTurnController()
  def step(probs, **kw):
    args = dict(blindspot_left=False, blindspot_right=False, left_blinker=True, right_blinker=False,
                v_ego=10 / 3.6, lane_line_probs=probs)
    ctrl.update_lane_turn(**(args | kw))
    return ctrl.get_turn_desire()
  assert step([0.9, 0.9]) == DESIRE.none
  assert step([0.5, 0.9]) == DESIRE.turnLeft
  assert step([0.9, 0.9]) == DESIRE.turnLeft
  assert step([0.9, 0.9], blindspot_left=True) == DESIRE.none
  assert step([0.9, 0.9]) == DESIRE.none
  assert step([0.1, 0.9]) == DESIRE.turnLeft
  assert step([0.9, 0.9], v_ego=20 / 3.6) == DESIRE.none
  assert step([0.1, 0.9], right_blinker=True) == DESIRE.none
  assert step([0.9, 0.9], left_blinker=False, right_blinker=True) == DESIRE.none
  assert step([0.9, 0.1], left_blinker=False, right_blinker=True) == DESIRE.turnRight
  ctrl.enabled = False
  assert step([0.9, 0.9], left_blinker=False, right_blinker=True) == DESIRE.none
  ctrl.enabled = True
  assert step([0.9, 0.9], left_blinker=False, right_blinker=True) == DESIRE.none
  assert step(None) == DESIRE.turnLeft


def stop_setup():
  ns = load('dragonpilot/selfdrive/controls/lib/traffic_stop.py', V_CRUISE_MAX=145.0)
  ctrl = ns.TrafficStopController()
  ctrl._poll_frame = -10000
  return ns, ctrl


def stop_step(ctrl, distance=60, speed=30 / 3.6, rise=0, off=False, **car):
  xs = np.linspace(0, distance, 33).tolist()
  vs = [speed] + ([3.0] * 32 if off else [max(speed * (1 - i / 32), 0.3) for i in range(1, 33)])
  model = SimpleNamespace(position=SimpleNamespace(x=xs, y=[0.0] * 33, z=np.linspace(0, rise, 33).tolist()),
                          velocity=SimpleNamespace(x=vs))
  cs = SimpleNamespace(**(dict(steeringAngleDeg=0, gasPressed=False, leftBlinker=False) | car))
  ctrl.update(model, cs, SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=0)),
              speed, 0, 30)


def test_traffic_obstacle_gate_latch_release_and_distance_sign():
  ns, ctrl = stop_setup()
  for _ in range(20):
    stop_step(ctrl)
  assert ctrl.state == ns.STOPPING
  assert ctrl.stop_dist_m is None and ctrl.near_stop_dist_m is not None
  assert ctrl.output_v_target == ns.V_TARGET_SENTINEL
  for _ in range(20):
    stop_step(ctrl, distance=30, speed=50 / 3.6)
  assert ctrl.stop_dist_m is not None and ctrl._obstacle_latched
  for _ in range(5):
    stop_step(ctrl, distance=25, speed=10 / 3.6)
  assert ctrl.stop_dist_m is not None
  before = ctrl.stop_dist_m
  for _ in range(4):
    stop_step(ctrl, distance=200, speed=30 / 3.6, off=True)
  assert ctrl.stop_dist_m == pytest.approx(before - 4 * (30 / 3.6) * ns.DT_MDL)
  stop_step(ctrl, gasPressed=True)
  assert ctrl.state == ns.CRUISE
  assert ctrl.stop_dist_m is None and ctrl.near_stop_dist_m is None and not ctrl._obstacle_latched
  distances = {}
  for adjustment in (-2, 0, 2):
    _, ctrl = stop_setup()
    ctrl.distance_adjust_m = adjustment
    for _ in range(20):
      stop_step(ctrl)
    distances[adjustment] = ctrl.near_stop_dist_m
  assert distances[2] == pytest.approx(distances[0] - 2)
  assert distances[-2] == pytest.approx(distances[0] + 2)


def test_crest_guard_holds_fast_entry_but_confirms_slow_entry():
  ns, ctrl = stop_setup()
  for _ in range(40):
    stop_step(ctrl, 100, 55 / 3.6, rise=3)
  assert ctrl.state == ns.CRUISE
  for _ in range(ns.CREST_HOLD_FRAMES - 5):
    stop_step(ctrl, 60, 50 / 3.6)
  assert ctrl.state == ns.CRUISE
  _, ctrl = stop_setup()
  for _ in range(ns.UPHILL_CONFIRM_FRAMES - 1):
    stop_step(ctrl, 40, 20 / 3.6, rise=1.5)
  assert ctrl.state == ns.CRUISE
  stop_step(ctrl, 40, 20 / 3.6, rise=1.5)
  assert ctrl.state == ns.STOPPING
  _, ctrl = stop_setup()
  stop_step(ctrl, 60, 40 / 3.6)
  assert ctrl.state == ns.STOPPING


def ocm_setup():
  ns = load('dragonpilot/selfdrive/controls/lib/ocm.py')
  ctrl = ns.OCM()
  ctrl.enabled = True
  return ns, ctrl


def ocm_step(ctrl, **kwargs):
  args = dict(cc=SimpleNamespace(orientationNED=[0, 0, 0]),
              rs=SimpleNamespace(leadOne=SimpleNamespace(status=False)),
              user_ctrl_lon=False, v_ego=120 / 3.6, v_cruise=100 / 3.6)
  ctrl.update_states(**(args | kwargs))


@pytest.mark.parametrize('source', ['lead0', 'lead1', 'e2e', 'trafficStop'])
def test_ocm_yields_to_non_cruise_source_and_cools_down(source):
  ns, ctrl = ocm_setup()
  ocm_step(ctrl)
  assert ctrl.active
  ocm_step(ctrl, plan_source=source)
  assert not ctrl.active and ctrl.just_disabled
  ocm_step(ctrl)
  assert not ctrl.active
  ns.time.monotonic = lambda: 100.5
  ocm_step(ctrl)
  assert ctrl.active


@pytest.mark.parametrize('kwargs', [
  {'a_target': -1.51}, {'a_plan_min': -1.51}, {'dtsc_active': True}, {'user_ctrl_lon': True},
  {'v_ego': 100 / 3.6}, {'cc': SimpleNamespace(orientationNED=[])},
  {'rs': SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=10, vLead=0))},
])
def test_ocm_aborts_without_delaying_safety_paths(kwargs):
  _, ctrl = ocm_setup()
  ocm_step(ctrl)
  assert ctrl.active
  ocm_step(ctrl, **kwargs)
  assert not ctrl.active and ctrl.just_disabled


def test_ocm_overspeed_hysteresis_and_coast_target():
  ns, ctrl = ocm_setup()
  ocm_step(ctrl, v_ego=131 / 3.6)
  assert not ctrl.active
  ocm_step(ctrl, v_ego=130 / 3.6)
  assert ctrl.active
  ocm_step(ctrl, v_ego=135 / 3.6)
  assert ctrl.active
  ocm_step(ctrl, v_ego=135.1 / 3.6)
  assert not ctrl.active
  assert ns.OCM.get_coast_target(-0.24) == pytest.approx(-0.24)
  assert ns.OCM.get_coast_target(0.2) == 0


def test_corolla_coast_calibration_and_legacy_call():
  ns = load('selfdrive/controls/lib/longitudinal_planner.py',
            ['DP_COAST_BP_KPH', 'DP_COAST_BASE', 'get_coast_accel'])
  expected = [(0, -0.10), (20, -0.11), (30, -0.11), (35, -0.14), (45, -0.22),
              (60, -0.24), (80, -0.27), (100, -0.30), (120, -0.33)]
  for speed, coast in expected:
    assert ns.get_coast_accel(0, speed / 3.6) == pytest.approx(coast)
    assert ns.get_coast_accel(0.02, speed / 3.6) == pytest.approx(-np.sin(0.02) * 5.65 + coast)
  assert ns.get_coast_accel(0.02) == pytest.approx(-np.sin(0.02) * 5.65 - 0.3)


def planner_setup(monkeypatch):
  t = np.linspace(0, 10, 33)
  ocm_ns, _ = ocm_setup()
  class Filter:
    def __init__(self, x, *args):
      self.x = x

    def update(self, value):
      self.x = value
      return value

  class DP:
    def __init__(self, *args):
      self.traffic_stop = SimpleNamespace(stop_dist_m=None, near_stop_dist_m=25)
      self.source = SOURCE.cruise
      self.target_source = SOURCE.cruise
      self.comfort_min = -0.1
      self.personality_clip = [-3.5, 1.5]
      self.dtsc = Mock()
      self.dtsc.active = False

    def update(self, sm):
      pass

    def get_accel_clip(self, *args):
      return list(self.personality_clip)

    def get_cruise_min_accel(self, *args):
      return self.comfort_min

    def update_targets(self, sm, v_ego, a_ego, v_cruise):
      self.source = self.target_source
      return v_cruise, a_ego

  mpc = Mock()
  mpc.crash_cnt = 0
  mpc.v_solution = np.full(33, 120 / 3.6)
  mpc.a_solution = np.full(33, -0.5)
  mpc.j_solution = np.zeros(32)
  mpc.source = SOURCE.cruise
  mpc.get_output = -0.5
  fake_cereal = types.ModuleType('cereal')
  fake_cereal.log = LOG
  monkeypatch.setitem(sys.modules, 'cereal', fake_cereal)
  ns = load('selfdrive/controls/lib/longitudinal_planner.py',
            LongitudinalPlannerDP=DP, DPPlanSource=SOURCE, LongitudinalMpc=lambda **kw: mpc,
            LongCtrlState=SimpleNamespace(off=0), FirstOrderFilter=Filter,
            ModelConstants=SimpleNamespace(T_IDXS=t, IDX_N=33), CONTROL_N=17, T_IDXS_MPC=t,
            ACCEL_MIN=-3.5, ACCEL_MAX=2.0, CRUISE_MIN_ACCEL=-1.2, V_CRUISE_MAX=145, V_CRUISE_UNSET=255,
            OCM=ocm_ns.OCM, AEM=Mock, APM=Mock,
            get_accel_from_plan=lambda *args, **kw: (mpc.get_output, False))
  cp = SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2, vEgoStopping=0.5,
                       steerRatio=15, wheelbase=2.7)
  planner = ns.LongitudinalPlanner(cp, init_v=120 / 3.6)
  planner.aem.get_mode.side_effect = lambda mode: mode
  lead = SimpleNamespace(status=False)
  sm = {
    'carState': SimpleNamespace(vEgo=120 / 3.6, vCruise=100, aEgo=0, steeringAngleDeg=0, standstill=False),
    'carControl': SimpleNamespace(orientationNED=[0, 0, 0]),
    'controlsState': SimpleNamespace(longControlState=1, forceDecel=False),
    'selfdriveState': SimpleNamespace(experimentalMode=False, personality=1, enabled=True),
    'radarState': SimpleNamespace(leadOne=lead, leadTwo=lead),
    'liveParameters': SimpleNamespace(angleOffsetDeg=0),
    'modelV2': SimpleNamespace(position=SimpleNamespace(x=t), velocity=SimpleNamespace(x=t),
                              acceleration=SimpleNamespace(x=t),
                              action=SimpleNamespace(desiredAcceleration=0.1, shouldStop=False),
                              meta=SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[1, 1]))),
  }
  return ns, planner, sm


def test_real_planner_overrides_output_not_trajectories_and_honors_source(monkeypatch):
  ns, planner, sm = planner_setup(monkeypatch)
  planner.update(sm, ns.DPFlags.OCM)
  assert planner.ocm.enabled and planner.ocm.active
  assert planner.output_a_target == pytest.approx(-0.33)
  assert np.all(planner.a_desired_trajectory == -0.5)
  assert np.all(planner.v_desired_trajectory == 120 / 3.6)
  planner.mpc.source = SOURCE.trafficStop
  planner.mpc.get_output = -1.0
  planner.update(sm, ns.DPFlags.OCM)
  assert not planner.ocm.active
  assert planner.output_a_target == pytest.approx(-1.0)
  planner.update(sm, 0)
  assert not planner.ocm.enabled and not planner.ocm.active


@pytest.mark.parametrize('condition', ['forced_decel', 'unset_cruise', 'control_off'])
def test_real_planner_does_not_coast_over_forced_slowdown_or_reset(monkeypatch, condition):
  ns, planner, sm = planner_setup(monkeypatch)
  planner.update(sm, ns.DPFlags.OCM)
  assert planner.ocm.active
  planner.mpc.get_output = -1.0
  if condition == 'forced_decel':
    sm['controlsState'].forceDecel = True
  elif condition == 'unset_cruise':
    sm['carState'].vCruise = 255
  else:
    sm['controlsState'].longControlState = 0
  planner.update(sm, ns.DPFlags.OCM)
  assert not planner.ocm.active
  assert planner.output_a_target == pytest.approx(-1.0)


@pytest.mark.parametrize('source,forced,expected', [
  (SOURCE.trafficStop, False, -1.2), (SOURCE.cruise, True, -1.2),
  (SOURCE.cruise, False, -0.1), (SOURCE.dtsc, False, -0.1),
])
def test_real_planner_keeps_braking_authority_and_ungated_aem_distance(monkeypatch, source, forced, expected):
  ns, planner, sm = planner_setup(monkeypatch)
  planner.target_source = source
  sm['controlsState'].forceDecel = forced
  planner.update(sm, ns.DPFlags.AEM)
  assert planner.mpc.update.call_args.kwargs['a_cruise_min_override'] == expected
  assert planner.aem.update_states.call_args.kwargs['stop_dist_m'] == 25
  assert planner.mpc.update.call_args.kwargs['traffic_stop_obstacle_m'] is None


def test_real_planner_turn_limits_apply_with_personality_and_mpc_weights_once(monkeypatch):
  _, planner, sm = planner_setup(monkeypatch)
  sm['carState'].steeringAngleDeg = 100
  planner.update(sm)
  assert np.all(planner.mpc.update.call_args.kwargs['a_max_arr'] == 0)
  planner.mpc.set_weights.assert_called_once_with(True, personality=1)


@pytest.mark.parametrize('distance,source', [(None, SOURCE.cruise), (12.3, SOURCE.trafficStop)])
def test_actual_mpc_update_preserves_weights_and_wires_obstacle_arrays(distance, source):
  t = np.linspace(0, 10, 13)
  ns = load('selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py',
            ['LongitudinalMpc', 'build_traffic_stop_obstacle', 'get_T_FOLLOW',
             'get_stopped_equivalence_factor', 'get_safe_obstacle_distance'],
            log=SimpleNamespace(LongitudinalPersonality=SimpleNamespace(standard=1, relaxed=0, aggressive=2)),
            N=12, T_IDXS=t, T_DIFFS=np.diff(t, prepend=0), FCW_IDXS=t < 5,
            CRUISE_MIN_ACCEL=-1.2, CRUISE_MAX_ACCEL=1.6, COMFORT_BRAKE=2.5, STOP_DISTANCE=6,
            COST_E_DIM=5, LEAD_DANGER_FACTOR=0.75, CRASH_DISTANCE=0.25,
            TRAFFIC_STOP_OBSTACLE_DISABLED_M=1000, MPC_SOURCES=[SOURCE.lead0, SOURCE.lead1, SOURCE.cruise, SOURCE.trafficStop])
  mpc = Mock()
  mpc.x0 = np.array([0, 10, 0])
  mpc.a_prev = np.full(13, -0.4)
  mpc.x_sol = np.zeros((13, 3))
  mpc.yref = np.zeros((13, 6))
  mpc.params = np.zeros((13, 6))
  mpc.process_lead.return_value = np.column_stack([np.full(13, 100), np.full(13, 10)])
  radar = SimpleNamespace(leadOne=SimpleNamespace(status=False, modelProb=0), leadTwo=SimpleNamespace(status=False))
  ns.LongitudinalMpc.update(mpc, radar, 10, a_min_arr=np.full(13, -2.0),
                           a_max_arr=np.full(13, -0.3), traffic_stop_obstacle_m=distance)
  mpc.set_weights.assert_not_called()
  mpc.run.assert_called_once()
  assert mpc.source == source
  assert np.all(mpc.params[:, 0] == -2.0)
  assert np.all(mpc.params[:, 1] == -0.3)
  assert np.all(mpc.params[:, 3] == -0.4)
  if distance is not None:
    assert np.all(mpc.params[:, 2] == distance)
  else:
    assert mpc.params[0, 2] == pytest.approx(10 ** 2 / 5 + 1.55 * 10 + 6)


def test_real_planner_caps_next_mpc_frame_only_while_ocm_enabled(monkeypatch):
  ns, planner, sm = planner_setup(monkeypatch)
  planner.update(sm, ns.DPFlags.OCM)
  assert np.all(planner.mpc.update.call_args.kwargs['a_max_arr'] == 1.5)
  planner.update(sm, ns.DPFlags.OCM)
  assert np.all(planner.mpc.update.call_args.kwargs['a_max_arr'] == pytest.approx(-0.33))
  planner.update(sm, 0)
  assert np.all(planner.mpc.update.call_args.kwargs['a_max_arr'] == 1.5)


@pytest.mark.parametrize('condition', ['lead0', 'lead1', 'e2e', 'dtsc'])
def test_real_planner_yields_to_other_braking_sources_in_same_frame(monkeypatch, condition):
  ns, planner, sm = planner_setup(monkeypatch)
  planner.update(sm, ns.DPFlags.OCM)
  planner.mpc.get_output = -1.0
  flags = ns.DPFlags.OCM
  if condition == 'e2e':
    sm['selfdriveState'].experimentalMode = True
    sm['modelV2'].action.desiredAcceleration = -1.2
  elif condition == 'dtsc':
    planner.dtsc.active = True
    planner.dtsc.get_mpc_constraints.return_value = (np.full(33, -3.5), np.full(33, 1.5))
    flags |= ns.DPFlags.DTSC
  else:
    planner.mpc.source = getattr(SOURCE, condition)
  planner.update(sm, flags)
  assert not planner.ocm.active
  assert planner.output_a_target == pytest.approx(-1.2 if condition == 'e2e' else -1.0)


def test_actual_dp_planner_uses_measured_speed_for_stop_controller():
  ns = load('dragonpilot/selfdrive/controls/lib/longitudinal_planner.py',
            custom=SimpleNamespace(LongitudinalPlanDP=SimpleNamespace(LongitudinalPlanSource=SOURCE)))
  stop = Mock()
  stop.output_v_target, stop.output_a_target = 1000, 0
  obj = SimpleNamespace(traffic_stop=stop, dtsc=SimpleNamespace(output_v_target=1000, output_a_target=0))
  sm = {'carState': SimpleNamespace(vEgo=10), 'carControl': object(), 'modelV2': object(), 'radarState': object()}
  ns.LongitudinalPlannerDP.update_targets(obj, sm, 15, -0.5, 20)
  assert stop.update.call_args.args[3] == 10


def test_ltd_wiring_from_modeld_through_desire_helper():
  model = ast.parse((ROOT / 'selfdrive/modeld/modeld.py').read_text(encoding='utf-8'))
  call = next(n for n in ast.walk(model) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'DH.update')
  assert ast.unparse(call.args[5]) == 'list(modelv2_send.modelV2.laneLineProbs)'
  helper = ast.parse((ROOT / 'selfdrive/controls/lib/desire_helper.py').read_text(encoding='utf-8'))
  call = next(n for n in ast.walk(helper) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith('.update_lane_turn'))
  expression = next(kw.value for kw in call.keywords if kw.arg == 'lane_line_probs')
  code = compile(ast.Expression(expression), '<lane probabilities>', 'eval')
  for probs, expected in [([0.1, 0.8, 0.9, 0.2], [0.8, 0.9]), (None, None), ([], None)]:
    assert eval(code, {'lane_line_probs': probs}) == expected
