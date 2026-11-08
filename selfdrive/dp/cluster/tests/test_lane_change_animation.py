from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

CLUSTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_config import MODEL_DIRECT_LANE_RECENTER_SECONDS
from cluster_route_replay import RouteLogParser, route_lane_animation_values


def advance(parser, time_s, state, direction="left"):
    parser.model_lane_change_seen = True
    parser.lane_change_state = state
    parser.lane_change_direction = direction
    result = parser._lane_change_values(time_s, direction == "left", direction == "right", 0.0)
    return SimpleNamespace(
        lane_change=result[0], lane_change_phase=result[1], lane_change_progress=result[2],
        lane_change_recenter_start_progress=result[3], lane_change_continuation=result[4],
    )


@pytest.mark.parametrize("direction,sign", [("left", -1.0), ("right", 1.0)])
def test_road_slides_during_starting_without_waiting_for_model_desire(direction, sign):
    parser = RouteLogParser()
    # A fixed desire probability must not stall or instantly complete the animation.
    parser.lane_change_desire_left_prob = parser.lane_change_desire_right_prob = 1.0
    first = advance(parser, 10.0, "laneChangeStarting", direction)
    assert first.lane_change_phase == "changing"
    assert first.lane_change_progress == 0.0
    assert route_lane_animation_values(first, 0.0)[:3] == (0.0, 0.0, 0.0)
    middle = advance(parser, 10.0 + MODEL_DIRECT_LANE_RECENTER_SECONDS / 2, "laneChangeStarting", direction)
    ego, view, grid, highlight, animated = route_lane_animation_values(middle, 0.0)
    assert animated
    assert ego == 0.0
    assert view == pytest.approx(-sign * 0.5)
    assert grid == view
    assert highlight == pytest.approx(sign * 0.5)
    end = advance(parser, 10.0 + MODEL_DIRECT_LANE_RECENTER_SECONDS + 0.01, "laneChangeStarting", direction)
    assert end.lane_change_progress == 1.0
    assert route_lane_animation_values(end, 0.0)[:4] == (0.0, -sign, -sign, 0.0)


def test_preparing_and_blinker_alone_do_not_start_road_animation():
    parser = RouteLogParser()
    first = advance(parser, 0.0, "preLaneChange")
    later = advance(parser, 10.0, "preLaneChange")
    assert first.lane_change_progress == later.lane_change_progress == 0.0
    assert later.lane_change_phase == "preparing"
    assert route_lane_animation_values(later, 0.0)[:3] == (0.0, 0.0, 0.0)
    off = advance(parser, 11.0, "off")
    assert off.lane_change_phase == "idle"


def test_transition_from_preparing_starts_new_clock():
    parser = RouteLogParser()
    advance(parser, 1.0, "preLaneChange")
    started = advance(parser, 5.0, "laneChangeStarting")
    assert started.lane_change_progress == 0.0
    assert advance(parser, 5.2, "laneChangeStarting").lane_change_progress > 0.0


def test_finishing_does_not_restart_and_completed_off_does_not_animate_twice():
    parser = RouteLogParser()
    advance(parser, 0.0, "laneChangeStarting")
    assert advance(parser, 0.4, "laneChangeFinishing").lane_change_progress == pytest.approx(0.4 / 0.85)
    assert advance(parser, 1.0, "laneChangeFinishing").lane_change_progress == 1.0
    off = advance(parser, 1.1, "off")
    assert off.lane_change_phase == "idle"
    assert parser.lane_change_started_t is None
    assert advance(parser, 2.0, "off").lane_change_phase == "idle"


@pytest.mark.parametrize("direction", ["left", "right"])
def test_early_cancellation_smoothly_restores_original_lane(direction):
    parser = RouteLogParser()
    advance(parser, 0.0, "laneChangeStarting", direction)
    current = advance(parser, 0.3, "laneChangeStarting", direction)
    before = route_lane_animation_values(current, 0.0)
    cancelled = advance(parser, 0.3, "off", direction)
    assert cancelled.lane_change_phase == "recentering"
    assert route_lane_animation_values(cancelled, 0.0)[:3] == pytest.approx(before[:3])
    midway = advance(parser, 0.3 + 0.85 / 2, "off", direction)
    assert abs(route_lane_animation_values(midway, 0.0)[2]) == pytest.approx(abs(before[2]) / 2)
    end = advance(parser, 1.2, "off", direction)
    assert end.lane_change_phase == "idle"
    assert route_lane_animation_values(end, 0.0)[:3] == (0.0, 0.0, 0.0)


def test_same_direction_second_change_gets_new_animation():
    parser = RouteLogParser()
    advance(parser, 0.0, "laneChangeStarting")
    advance(parser, 1.0, "laneChangeFinishing")
    second = advance(parser, 1.1, "laneChangeStarting")
    assert second.lane_change_progress == 0.0
    assert advance(parser, 1.4, "laneChangeStarting").lane_change_progress > 0.0


def test_wall_clock_progress_is_independent_of_number_of_frames():
    frequent, skipped = RouteLogParser(), RouteLogParser()
    advance(frequent, 0.0, "laneChangeStarting")
    advance(skipped, 0.0, "laneChangeStarting")
    for frame in range(1, 9):
        advance(frequent, frame * 0.05, "laneChangeStarting")
    a = advance(frequent, 0.425, "laneChangeStarting")
    b = advance(skipped, 0.425, "laneChangeStarting")
    assert a.lane_change_progress == b.lane_change_progress == 0.5


def test_no_model_state_keeps_idle():
    parser = RouteLogParser()
    assert parser._lane_change_values(1.0, True, False, 0.0)[1] == "idle"
