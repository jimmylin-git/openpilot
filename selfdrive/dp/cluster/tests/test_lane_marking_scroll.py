import ast
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


CLUSTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_DIR))

import cluster_scene as scene
from cluster_models import SimulatorInput
from cluster_simulator import ClusterSimulator
from cluster_utils import clamp


SPECS = ((5, (80, 120, 200, 255), 0.02),)
CYCLE_M = scene.LANE_DASH_LENGTH_M + scene.LANE_DASH_GAP_M


def groups(style="dashed"):
    return scene.cached_lane_offset_strip_groups(0.5, 0.0, 3.6, -12.0, 80.0 + CYCLE_M, SPECS, style)


def intervals(strip_groups, scroll_m=0.0):
    strip = strip_groups[0][0]
    result = []
    start = end = None
    for left, right in zip(strip.left, strip.right, strict=True):
        if left == right:
            if start is not None:
                result.append((start, end))
                start = end = None
        else:
            if start is None:
                start = left.y
            end = left.y
    if start is not None:
        result.append((start, end))
    phase = scroll_m % CYCLE_M
    return [(start - phase, end - phase) for start, end in result]


def test_dashes_move_toward_car_with_gaps_and_constant_length():
    before = intervals(groups())
    after = intervals(groups(), 1.2)
    start = before[2][0]
    end = start + scene.LANE_DASH_LENGTH_M
    assert any(a == pytest.approx(start - 1.2) and b == pytest.approx(end - 1.2) for a, b in after)
    assert all(b - a <= scene.LANE_DASH_LENGTH_M + 1e-6 for a, b in after)
    assert any(after[i + 1][0] - after[i][1] == pytest.approx(scene.LANE_DASH_GAP_M) for i in range(len(after) - 1))


def test_phase_wrap_is_continuous_and_geometry_covers_visible_road():
    assert intervals(groups(), 0.0) == intervals(groups(), CYCLE_M)
    for phase in (0.0, 1.2, CYCLE_M - 0.001, CYCLE_M + 0.001):
        painted = intervals(groups(), phase)
        assert painted
        assert painted[0][0] <= scene.DRIVE_VIEW_REAR_VISIBLE_M
        assert painted[-1][1] >= 80.0 - scene.LANE_DASH_GAP_M
    before = intervals(groups(), CYCLE_M - 0.001)
    after = intervals(groups(), CYCLE_M + 0.001)
    interior_start = before[3][0]
    assert any(a == pytest.approx(interior_start - 0.002) for a, _ in after)


def test_animation_reuses_vertex_tuples_without_growing_geometry_caches():
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    first = scene.build_cluster_scene(state, ground_scroll_m=0.0)
    caches = (scene._LANE_OFFSET_STRIP_CACHE, scene._STYLE_MESH_STRIP_GROUP_CACHE, scene._MERGED_MESH_STRIP_CACHE)
    sizes = tuple(len(cache) for cache in caches)
    for i in range(500):
        frame = scene.build_cluster_scene(state, ground_scroll_m=i * 0.06)
        assert frame.lane_marking_scroll_m == pytest.approx((i * 0.06 * 15.0) % CYCLE_M)
        for before, after in zip(first.lane_markings, frame.lane_markings, strict=True):
            assert before.left is after.left
            assert before.right is after.right
    assert tuple(len(cache) for cache in caches) == sizes


def test_native_vertex_buffers_are_allocated_only_during_warmup():
    tree = ast.parse((CLUSTER_DIR / "cluster_renderer.py").read_text(encoding="utf-8"))
    renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ClusterUiRenderer")
    method = next(n for n in renderer.body if isinstance(n, ast.FunctionDef) and n.name == "_triangle_strip_points_for")
    allocations = []

    def allocate(kind, count):
        assert kind == "struct Vector3[]"
        points = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(count)]
        allocations.append(points)
        return points

    namespace = {"rl": SimpleNamespace(ffi=SimpleNamespace(new=allocate)), "MeshStrip": scene.MeshStrip,
                 "TRIANGLE_STRIP_POINT_CACHE_LIMIT": 256}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<native vertex cache>", "exec"), namespace)
    instance = SimpleNamespace(_triangle_strip_point_cache=OrderedDict())
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    buffers = None
    for i in range(120):
        frame = scene.build_cluster_scene(state, ground_scroll_m=i * 0.06)
        current = [namespace["_triangle_strip_points_for"](instance, strip, len(strip.left))[0]
                   for strip in frame.lane_markings]
        if buffers is None:
            buffers = current
        else:
            assert all(a is b for a, b in zip(buffers, current, strict=True))
    assert len(allocations) == len(buffers)


def test_scene_uses_same_dashed_phase_for_all_four_lane_boundaries(monkeypatch):
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    original = scene.lane_offset_strip_groups
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(scene, "lane_offset_strip_groups", record)
    first = scene.build_cluster_scene(state, ground_scroll_m=0.0)
    calls.clear()
    second = scene.build_cluster_scene(state, ground_scroll_m=1.2)
    assert len(calls) == 4
    assert all(args[6] == "dashed" and not kwargs for args, kwargs in calls)
    assert tuple(args[0] for args, _ in calls) == scene.FIXED_THREE_LANE_MARKING_OFFSETS
    assert first.lane_markings == second.lane_markings
    assert second.lane_marking_scroll_m == pytest.approx(18.0 % CYCLE_M)
    assert first.road_edges == second.road_edges
    assert first.vehicles == second.vehicles


@pytest.mark.parametrize("fail", [False, True])
def test_renderer_translates_only_lane_draws_and_restores_matrix(fail):
    tree = ast.parse((CLUSTER_DIR / "cluster_renderer.py").read_text(encoding="utf-8"))
    renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ClusterUiRenderer")
    method = next(n for n in renderer.body if isinstance(n, ast.FunctionDef) and n.name == "_draw_lane_markings")
    events = []
    rl = SimpleNamespace(rl_push_matrix=lambda: events.append("push"), rl_pop_matrix=lambda: events.append("pop"),
                         rl_translatef=lambda *args: events.append(args))
    namespace = {"rl": rl, "ClusterScene": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<lane drawing>", "exec"), namespace)

    def draw(strip):
        events.append(strip)
        if fail:
            raise RuntimeError("draw failed")

    frame = SimpleNamespace(lane_marking_scroll_m=1.2, lane_markings=("backing", "paint"))
    instance = SimpleNamespace(_draw_strip=draw)
    if fail:
        with pytest.raises(RuntimeError, match="draw failed"):
            namespace["_draw_lane_markings"](instance, frame)
    else:
        namespace["_draw_lane_markings"](instance, frame)
    assert events == ["push", (0.0, -1.2, 0.0), "backing", *(["paint"] if not fail else []), "pop"]


def make_scroll_updater():
    tree = ast.parse((CLUSTER_DIR / "cluster_renderer.py").read_text(encoding="utf-8"))
    renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ClusterUiRenderer")
    method = next(n for n in renderer.body if isinstance(n, ast.FunctionDef) and n.name == "_update_ground_scroll_m")
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in {
                     "GROUND_GRID_SCROLL_SPEED_MPS", "GROUND_GRID_MOVING_THRESHOLD_KPH"} for t in n.targets)]
    clock = SimpleNamespace(now=0.0)
    namespace = {"time": SimpleNamespace(perf_counter=lambda: clock.now), "clamp": clamp, "ClusterUiState": object}
    exec("from __future__ import annotations\n", namespace)
    exec(compile(ast.Module(body=[*constants, method], type_ignores=[]), "<ground scroll>", "exec"), namespace)
    instance = SimpleNamespace(_ground_scroll_m=0.0, _ground_scroll_last_t=None)
    return lambda state: namespace["_update_ground_scroll_m"](instance, state), clock


def test_fixed_scroll_rate_is_independent_of_driving_speed_and_stops_when_parked():
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    results = []
    for speed in (10.0, 50.0, 120.0):
        update, clock = make_scroll_updater()
        moving = replace(state, speed_kph=speed)
        assert update(moving) == 0.0
        clock.now = 0.2
        results.append(update(moving))
        clock.now = 0.4
        assert update(replace(state, speed_kph=0.0)) == results[-1]
    assert results == pytest.approx([0.24, 0.24, 0.24])
