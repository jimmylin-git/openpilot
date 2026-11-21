from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

CLUSTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_config import TURN_SIGNAL_BLINK_PERIOD_SECONDS
from cluster_route_replay import RouteLogParser, route_lane_animation_values


def advance(parser, time_s, state, direction="left", signal=None):
    parser.model_lane_change_seen = True
    parser.lane_change_state = state
    parser.lane_change_direction = direction
    signal = direction if signal is None else signal
    result = parser._lane_change_values(time_s, signal in ("left", "both"), signal in ("right", "both"), 0.0)
    return SimpleNamespace(
        lane_change=result[0], lane_change_phase=result[1], lane_change_progress=result[2],
        lane_change_recenter_start_progress=result[3], lane_change_continuation=result[4],
    )


@pytest.mark.parametrize("direction,sign", [("left", -1.0), ("right", 1.0)])
def test_road_waits_two_blinks_then_slides_until_seventh(direction, sign):
    parser = RouteLogParser()
    period = TURN_SIGNAL_BLINK_PERIOD_SECONDS
    advance(parser, 10.0, "preLaneChange", direction)
    for t in (10.1, 10.0 + period, 10.0 + 2 * period):
        state = advance(parser, t, "laneChangeStarting", direction)
        assert state.lane_change_progress == 0.0
        if t < 10.0 + 2 * period:
            assert state.lane_change_phase == "preparing"
        assert route_lane_animation_values(state, 0.0)[:3] == (0.0, 0.0, 0.0)
    middle = advance(parser, 10.0 + 4 * period, "laneChangeStarting", direction)
    ego, view, grid, highlight, animated = route_lane_animation_values(middle, 0.0)
    assert animated and ego == 0.0
    assert view == pytest.approx(-sign * 0.5)
    assert grid == pytest.approx(view)
    assert highlight == pytest.approx(sign * 0.5)
    end = advance(parser, 10.0 + 6 * period + 0.001, "laneChangeStarting", direction)
    assert end.lane_change_progress == 1.0
    assert route_lane_animation_values(end, 0.0)[:4] == (0.0, -sign, -sign, 0.0)


def test_blinker_alone_and_preparing_do_not_start_animation():
    parser = RouteLogParser()
    assert advance(parser, 0.0, "off").lane_change_phase == "idle"
    assert advance(parser, 5.0, "off").lane_change_phase == "idle"
    state = advance(parser, 6.0, "preLaneChange")
    assert state.lane_change_phase == "preparing"
    assert route_lane_animation_values(state, 0.0)[:3] == (0.0, 0.0, 0.0)


def test_late_model_confirmation_keeps_full_animation_duration():
    parser = RouteLogParser()
    advance(parser, 0.0, "preLaneChange")
    started = advance(parser, 3.0, "laneChangeStarting")
    assert started.lane_change_progress == 0.0
    assert advance(parser, 4.4, "laneChangeStarting").lane_change_progress == pytest.approx(0.5)
    assert advance(parser, 5.81, "laneChangeStarting").lane_change_progress == 1.0


@pytest.mark.parametrize("signal", ["none", "right", "both"])
def test_model_start_requires_matching_single_signal(signal):
    parser = RouteLogParser()
    state = advance(parser, 0.0, "laneChangeStarting", "left", signal)
    assert state.lane_change_phase == "idle"
    assert advance(parser, 5.0, "laneChangeStarting", "left", signal).lane_change_progress == 0.0
    assert advance(parser, 6.0, "laneChangeStarting", "left", "left").lane_change_progress == 0.0
    assert advance(parser, 7.5, "laneChangeStarting", "left", "left").lane_change_progress > 0.0


@pytest.mark.parametrize("end_time", [0.5, 2.0])
@pytest.mark.parametrize("end_state,signal", [
    ("off", "left"), ("preLaneChange", "left"),
    ("laneChangeStarting", "none"), ("laneChangeFinishing", "right"),
    ("laneChangeStarting", "both"),
])
def test_cancel_stops_animation_without_waiting_for_other_signals(end_time, end_state, signal):
    parser = RouteLogParser()
    advance(parser, 0.0, "laneChangeStarting")
    advance(parser, end_time, "laneChangeStarting")
    after = advance(parser, end_time, end_state, signal=signal)
    assert after.lane_change_phase == "idle"
    assert after.lane_change_progress == 0.0
    assert route_lane_animation_values(after, 0.0)[:3] == (0.0, 0.0, 0.0)
    assert route_lane_animation_values(after, 0.2) == (0.2, 0.0, 0.0, None, False)
    assert parser.lane_change_started_t is None


def test_cancellation_does_not_resume_a_stale_animation():
    parser = RouteLogParser()
    advance(parser, 0.0, "laneChangeStarting")
    advance(parser, 2.0, "laneChangeStarting")
    advance(parser, 2.1, "laneChangeStarting", signal="none")
    assert advance(parser, 2.2, "laneChangeStarting").lane_change_progress == 0.0
    assert advance(parser, 3.5, "laneChangeStarting").lane_change_progress == 0.0
    assert advance(parser, 3.7, "laneChangeStarting").lane_change_progress > 0.0


def test_signal_reset_restarts_wait_and_second_change_gets_its_own_clock():
    parser = RouteLogParser()
    advance(parser, 0.0, "preLaneChange")
    advance(parser, 1.0, "off", signal="none")
    advance(parser, 2.0, "preLaneChange")
    first = advance(parser, 2.1, "laneChangeStarting")
    assert first.lane_change_progress == 0.0
    assert advance(parser, 3.3, "laneChangeStarting").lane_change_progress == 0.0
    advance(parser, 3.5, "laneChangeFinishing")
    second = advance(parser, 4.0, "laneChangeStarting")
    assert second.lane_change_progress == 0.0
    assert advance(parser, 5.4, "laneChangeStarting").lane_change_progress == pytest.approx(0.5)


def test_wall_clock_progress_is_independent_of_number_of_frames():
    frequent, skipped = RouteLogParser(), RouteLogParser()
    advance(frequent, 0.0, "laneChangeStarting")
    advance(skipped, 0.0, "laneChangeStarting")
    for frame in range(1, 40):
        advance(frequent, frame * 0.05, "laneChangeStarting")
    a = advance(frequent, 2.8, "laneChangeStarting")
    b = advance(skipped, 2.8, "laneChangeStarting")
    assert a.lane_change_progress == pytest.approx(0.5)
    assert b.lane_change_progress == pytest.approx(a.lane_change_progress)


def test_no_model_state_keeps_idle():
    parser = RouteLogParser()
    assert parser._lane_change_values(1.0, True, False, 0.0)[1] == "idle"
