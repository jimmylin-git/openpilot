"""Exercise the production radar/AEM source without messaging or vehicle hardware."""
import ast
from collections import deque
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(relative, **namespace):
  tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
  nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
  ns = dict(np=np, math=math, DT_MDL=0.05, deque=deque, cloudlog=logging.getLogger('radar-port-test'))
  ns.update(namespace)
  future = ast.parse('from __future__ import annotations').body
  exec(compile(ast.Module(body=[*future, *nodes], type_ignores=[]), relative, 'exec'), ns)
  return SimpleNamespace(**ns)


@pytest.fixture
def radar():
  filt = load('common/filter_simple.py')
  kalman = load('common/simple_kalman.py')
  base = load('selfdrive/controls/radard.py', FirstOrderFilter=filt.FirstOrderFilter, KF1D=kalman.KF1D)
  return load('dragonpilot/selfdrive/controls/radard_ext.py', Track=base.Track, RadarD=base.RadarD,
              KalmanParams=base.KalmanParams, RADAR_TO_CAMERA=base.RADAR_TO_CAMERA,
              match_vision_to_track=base.match_vision_to_track,
              get_RadarState_from_vision=base.get_RadarState_from_vision, radard=SimpleNamespace())


def track(radar, identifier=1, d=50.0, y=0.0, v_abs=0.0, speed=50 / 3.6):
  t = radar.TrackDP(identifier, v_abs, radar.KalmanParams(0.05))
  t.update(d, y, v_abs - speed, v_abs, True)
  return t


def lead(radar, d=50.0, v=0.0, prob=0.3):
  return SimpleNamespace(x=[d + radar.RADAR_TO_CAMERA], y=[0.0], v=[v], a=[0.0, 0.0],
                         xStd=[2.0], yStd=[1.0], vStd=[2.0], prob=prob)


def select(radar, tracks, message=None, speed=50 / 3.6, **kwargs):
  message = lead(radar) if message is None else message
  args = dict(raw_lead_prob=message.prob, path_x=[0.0, 150.0], path_y=[0.0, 0.0])
  args.update(kwargs)
  return radar.get_lead_ext(speed, True, tracks, message, speed, message.prob, **args)


def test_static_rescue_uses_actual_lead_selection_after_confirmation(radar):
  t = track(radar)
  for _ in range(19):
    assert not select(radar, {1: t})['status']
  result = select(radar, {1: t})
  assert result['status'] and result['radar'] and result['radarTrackId'] == 1
  assert radar._LEAD_STATE_CACHE[0]['rescue']
  assert t.radar_static_frames == 20


def test_rescue_releases_when_vision_evidence_disappears(radar):
  t = track(radar)
  for _ in range(20):
    select(radar, {1: t})
  assert not select(radar, {1: t}, lead(radar, prob=0.0))['status']
  assert t.radar_static_frames == 0
  assert radar._LEAD_STATE_CACHE[0]['track'] is None


def test_lead_two_cannot_promote_stationary_rescue(radar):
  t = track(radar)
  for _ in range(30):
    assert not select(radar, {1: t}, low_speed_override=False)['status']
  assert t.radar_static_frames == 0


def test_rescue_continues_on_duplicate_id_replacement(radar):
  t = track(radar)
  for _ in range(20):
    select(radar, {1: t})
  replacement = track(radar, identifier=2, d=50.5)
  result = select(radar, {2: replacement})
  assert result['radarTrackId'] == 2
  assert replacement.radar_static_frames >= 20


def test_moving_twin_does_not_lend_vision_to_gantry(radar):
  gantry = track(radar)
  moving = track(radar, identifier=2, d=52.0, v_abs=3.5)
  for _ in range(30):
    select(radar, {1: gantry, 2: moving}, lead(radar, v=3.5))
  assert gantry.radar_static_frames == 0


@pytest.mark.parametrize('bad', [
  {'path_x': [0.0, 0.0]}, {'path_y': [0.0]}, {'path_x': None},
  {'steering_angle_deg': 25.0},
])
def test_invalid_path_or_steering_prevents_static_rescue(radar, bad):
  t = track(radar)
  for _ in range(30):
    assert not select(radar, {1: t}, **bad)['status']
  assert t.radar_static_frames == 0


def lanes():
  return [([0.0, 150.0], [y, y]) for y in (-5.25, -1.75, 1.75, 5.25)]


@pytest.mark.parametrize('y', [-2.0, 2.0])
def test_reliable_lane_line_blocks_outside_point_even_if_path_drifts(radar, y):
  t = track(radar, y=y)
  for _ in range(30):
    select(radar, {1: t}, path_y=[-y, -y], lane_lines=lanes(), lane_line_probs=[1.0] * 4)
  assert t.radar_static_frames == 0


def test_large_vehicle_chain_excludes_stationary_objects(radar):
  objects = {i: track(radar, identifier=i, d=20 + 3 * i, y=2.0, v_abs=10) for i in range(3)}
  radar._update_large_vehicle_flags(objects, 50 / 3.6)
  assert all(t.is_large for t in objects.values())
  static = {i: track(radar, identifier=i, d=20 + 3 * i) for i in range(3)}
  radar._update_large_vehicle_flags(static, 50 / 3.6)
  assert not any(t.is_large for t in static.values())


@pytest.mark.parametrize('gated', [False, True])
def test_cutin_requires_lane_crossing_and_is_blocked_while_turning(radar, gated):
  t = track(radar, d=20, y=2.3, v_abs=10)
  lane_ctx = radar._build_lane_ctx(lanes(), [1.0] * 4)
  for frame in range(24):
    t.yRel = 2.3 - frame * 0.05
    t.update_cutin(50 / 3.6, True, 0.0, gated, lane_ctx=lane_ctx)
  assert (t.cutin_frames >= radar.CUTIN_CONFIRM_FRAMES) is (not gated)


@pytest.mark.parametrize('brand,extended', [('toyota', True), ('hyundai', False), ('honda', False), ('gm', False)])
def test_brand_gate_does_not_import_extension_for_other_brands(brand, extended):
  tree = ast.parse((ROOT / 'selfdrive/controls/radard.py').read_text(encoding='utf-8'))
  main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
  block = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'CP.brand in DP_RADARD_EXT_BRANDS')
  assert isinstance(block.body[0], ast.ImportFrom)
  assert block.body[0].module == 'dragonpilot.selfdrive.controls.radard_ext'
  assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.Module(body=block.orelse, type_ignores=[])))
  assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'DP_RADARD_EXT_BRANDS' for t in n.targets))
  assert (brand in ast.literal_eval(assignment.value)) is extended


@pytest.mark.parametrize('left,right', [(False, False), (True, False), (False, True), (True, True)])
def test_planner_supplies_blinker_and_offset_corrected_angle(left, right):
  tree = ast.parse((ROOT / 'selfdrive/controls/lib/longitudinal_planner.py').read_text(encoding='utf-8'))
  call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and ast.unparse(n.func) == 'self.aem.update_states')
  ns = dict(sm={'carState': SimpleNamespace(leftBlinker=left, rightBlinker=right, steeringAngleDeg=23.0),
                'liveParameters': SimpleNamespace(angleOffsetDeg=4.0)})
  values = {kw.arg: eval(compile(ast.Expression(kw.value), '<AEM wiring>', 'eval'), ns)
            for kw in call.keywords if kw.arg in ('blinker', 'steer_angle_deg')}
  assert values == {'blinker': left or right, 'steer_angle_deg': 19.0}


@pytest.mark.parametrize('speed,blinker,steer,stop,expected', [
  (15, False, 0, None, 'blended'), (15, True, 0, None, 'acc'),
  (50, True, 0, 30, 'blended'), (50, True, 21, 30, 'acc'),
  (50, False, 0, None, 'acc'),
])
def test_production_aem_mode_priority(speed, blinker, steer, stop, expected):
  aem = load('dragonpilot/selfdrive/controls/lib/aem.py').AEM()
  for _ in range(100):
    aem.update_states(None, None, speed / 3.6, stop, blinker, steer)
  assert aem.get_mode('acc') == expected


def test_radard_update_wires_lane_data_and_both_lead_slots(radar):
  namespace = radar.RadarD.update.__globals__
  namespace['Track'] = radar.TrackDP
  namespace['get_lead'] = radar.get_lead_ext
  namespace['log'] = SimpleNamespace(RadarState=SimpleNamespace(new_message=SimpleNamespace),
                                     LaneChangeState=SimpleNamespace(off=0))

  class Messages(dict):
    seen = {'modelV2': True}
    recv_frame = {'carState': 0}
    logMonoTime = {'modelV2': 0, 'carState': 0}

    def all_checks(self):
      return True

  model = SimpleNamespace(velocity=SimpleNamespace(x=[50 / 3.6]),
                          leadsV3=[lead(radar), lead(radar)],
                          meta=SimpleNamespace(laneChangeState=0),
                          position=SimpleNamespace(x=[0.0, 150.0], y=[0.0, 0.0]),
                          laneLines=[SimpleNamespace(x=x, y=y) for x, y in lanes()],
                          laneLineProbs=[1.0] * 4)
  messages = Messages(modelV2=model, carState=SimpleNamespace(vEgo=50 / 3.6, steeringAngleDeg=0.0, steeringRateDeg=0.0))
  rr = SimpleNamespace(points=[SimpleNamespace(trackId=1, dRel=50.0, yRel=0.0, vRel=-50 / 3.6, measured=True)], errors=[])
  rd = radar.RadarD(0.0)
  for frame in range(20):
    messages.recv_frame['carState'] = frame
    rd.update(messages, rr)
  assert rd.radar_state.leadOne['radarTrackId'] == 1
  assert not rd.radar_state.leadTwo['status']
  assert rd.radar_state_valid
  rr.points = []
  model.leadsV3 = [lead(radar, prob=0.0), lead(radar, prob=0.0)]
  rd.update(messages, rr)
  assert not rd.tracks
  assert not rd.radar_state.leadOne['status']


@pytest.mark.parametrize('speed', [30.0, 130.0])
def test_static_confirmation_strict_ego_speed_limits(radar, speed):
  t = track(radar, d=20, speed=speed / 3.6)
  for _ in range(30):
    select(radar, {1: t}, lead(radar, d=20), speed=speed / 3.6)
  assert t.radar_static_frames == 0


@pytest.mark.parametrize('center', [-3.0, 3.0])
def test_lane_side_selection_follows_curved_lane_center(radar, center):
  shifted = [([0.0, 150.0], [y - center] * 2) for y in (-5.25, -1.75, 1.75, 5.25)]
  ctx = radar._build_lane_ctx(shifted, [1.0] * 4)
  t = track(radar, y=center - 2.0)
  intr, mode = radar._lane_intrusion(t, 0.0, ctx)
  assert intr == pytest.approx(-0.25)
  assert mode == 'line'
