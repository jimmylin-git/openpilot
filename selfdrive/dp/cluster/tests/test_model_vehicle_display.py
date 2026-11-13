from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cluster_scene as scene
import cluster_live
from cluster_live import OpenpilotLiveSource
from cluster_models import DetectedVehicle, RadarPoint, SimulatorInput
from cluster_simulator import ClusterSimulator


def make_state(vehicles=(), radar_points=(), radar_display_mode=0):
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    return replace(state, detected_vehicles=vehicles, radar_points=radar_points, radar_display_mode=radar_display_mode)


def screen_vehicles(frame, compact=False):
    rl = pytest.importorskip("pyray")
    import cluster_renderer as renderer

    fovy = frame.camera.fovy_deg
    if compact:
        fovy = math.degrees(2 * math.atan(math.tan(math.radians(fovy) * 0.5) / renderer.COMPACT_SCENE_ZOOM))
    camera = rl.Camera3D(renderer.vec3(frame.camera.position), renderer.vec3(frame.camera.target),
                         rl.Vector3(0, 0, 1), fovy, rl.CameraProjection.CAMERA_PERSPECTIVE)
    return renderer.vehicles_without_screen_overlap(frame.vehicles, camera, 1920, 462 if compact else 720,
                                                    compact, False, frame.scene_shift_x_m)


@pytest.mark.parametrize("radar_display_mode", [0, 1])
def test_hidden_model_does_not_hide_radar_vehicle(radar_display_mode):
    radar = DetectedVehicle("TARGET", 25.0, 0.0, source="radarState", probability=1.0, primary=True)
    point = RadarPoint("R1", 25.0, 0.0, source="liveTracks", probability=1.0, relative_speed_mps=-2.0)
    low = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.49, primary=True)
    frame = scene.build_cluster_scene(make_state((radar, low), (point,), radar_display_mode))
    assert len(frame.vehicles) == 2
    assert frame.vehicles[1].source != "modelV2.leadsV3"

    high = replace(low, probability=0.9)
    frame = scene.build_cluster_scene(make_state((radar, high), (point,), radar_display_mode))
    displayed = screen_vehicles(frame)
    assert len(displayed) == 2
    assert displayed[1].label == "M1"
    assert displayed[1].raw_probability == pytest.approx(0.9)


@pytest.mark.parametrize("radar_display_mode", [0, 1])
def test_radar_only_vehicles_are_restored(radar_display_mode):
    radar = DetectedVehicle("TARGET", 25.0, 0.0, source="radarState", probability=1.0, primary=True)
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=1.0, relative_speed_mps=-1.0)
    frame = scene.build_cluster_scene(make_state((radar,), (point,), radar_display_mode))
    assert len(frame.vehicles) == 3
    assert any(vehicle.label == "TARGET" for vehicle in frame.vehicles)
    assert any(vehicle.longitudinal_m == 40.0 for vehicle in frame.vehicles)


@pytest.mark.parametrize("radar_display_mode", [0, 1])
def test_neighbor_survives_when_model_lead_is_below_threshold(radar_display_mode):
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.49, primary=True)
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=1.0, relative_speed_mps=-1.0)
    frame = scene.build_cluster_scene(make_state((lead,), (point,), radar_display_mode))
    assert len(frame.vehicles) == 2
    assert frame.vehicles[1].longitudinal_m == 40.0
    assert frame.vehicles[1].label != "M1"


@pytest.mark.parametrize(("probability", "visible"), [(0.4999, False), (0.5, True), (0.5001, True), (0.6, True)])
def test_model_probability_threshold(probability, visible):
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=probability, primary=True)
    frame = scene.build_cluster_scene(make_state((lead,)))
    assert (len(frame.vehicles) == 2) is visible


def test_model_probability_still_controls_fade_above_threshold():
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6, primary=True)
    frame = scene.build_cluster_scene(make_state((lead,)))
    assert frame.vehicles[1].confidence == pytest.approx(0.5)
    assert frame.vehicles[1].raw_probability == pytest.approx(0.6)


def test_model_probability_grace_is_bounded_and_does_not_promote_new_leads():
    source = OpenpilotLiveSource.__new__(OpenpilotLiveSource)
    source._model_probability_low_since = {}
    high = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6)
    low = replace(high, probability=0.49)
    assert not source._retain_model_probability(low, None, 1.0)
    assert source._retain_model_probability(low, high, 1.0)
    retained = replace(low, model_probability_retained=True)
    assert len(scene.build_cluster_scene(make_state((retained,))).vehicles) == 2
    assert scene.build_cluster_scene(make_state((retained,))).vehicles[1].raw_probability == 0.49
    assert source._retain_model_probability(low, retained, 1.19)
    assert not source._retain_model_probability(low, retained, 1.21)
    assert not source._retain_model_probability(low, low, 1.22)
    assert source._retain_model_probability(low, high, 2.0)
    assert not source._retain_model_probability(replace(low, probability=0.39), retained, 2.01)


def test_probability_recovery_resets_grace_period():
    source = OpenpilotLiveSource.__new__(OpenpilotLiveSource)
    source._model_probability_low_since = {}
    high = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6)
    low = replace(high, probability=0.49)
    assert source._retain_model_probability(low, high, 1.0)
    assert not source._retain_model_probability(high, low, 1.1)
    assert source._retain_model_probability(low, high, 1.2)
    assert source._retain_model_probability(low, replace(low, model_probability_retained=True), 1.39)


def test_model_slot_reused_for_distant_target_does_not_inherit_grace():
    source = OpenpilotLiveSource.__new__(OpenpilotLiveSource)
    source._model_probability_low_since = {}
    high = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6)
    low = replace(high, longitudinal_m=40.0, probability=0.49)
    assert not source._retain_model_probability(low, high, 1.0)


def test_probability_history_is_pruned_when_vehicles_disappear():
    source = OpenpilotLiveSource.__new__(OpenpilotLiveSource)
    source._model_probability_low_since = {}
    source._lane_range_visible = {}
    state = make_state()
    high = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6)
    source._smoothed_state = make_state((high,))
    low = replace(high, probability=0.49)
    vehicles, _ = source._apply_lane_range_hysteresis((low,), (), state, 1.0)
    assert vehicles[0].model_probability_retained
    assert source._model_probability_low_since
    source._smoothed_state = make_state(vehicles)
    source._apply_lane_range_hysteresis((), (), state, 1.1)
    assert not source._model_probability_low_since
    assert not source._lane_range_visible


def make_live_source():
    source = OpenpilotLiveSource.__new__(OpenpilotLiveSource)
    source.parser = SimpleNamespace(radar_detection_t=0.0, live_track_radar_t=0.0, model_detection_t=0.0)
    source._smoothed_state = None
    source._smoothed_state_t = None
    for name in (
        "_vehicle_missing_since", "_vehicle_seen_since", "_vehicle_filter_last_payload",
        "_vehicle_filter_sample_count", "_model_vehicle_seen_since", "_model_vehicle_filter_last_payload",
        "_model_vehicle_filter_sample_count", "_radar_seen_since", "_radar_filter_last_payload",
        "_radar_filter_sample_count", "_radar_missing_since", "_lane_range_visible", "_model_probability_low_since",
    ):
        setattr(source, name, {})
    source._stable_vehicle_keys = set()
    source._stable_model_vehicle_keys = {("M1", "modelV2.leadsV3")}
    source._stable_radar_keys = set()
    source._last_target2_filter_update_t = -999.0
    source._last_model_filter_update_t = -999.0
    source._last_radar_filter_update_t = -999.0
    source._filter_log_disabled = True
    return source


def test_live_probability_grace_and_missing_fade(monkeypatch):
    source = make_live_source()
    clock = [1.0]
    monkeypatch.setattr(cluster_live.time, "monotonic", lambda: clock[0])

    def update(vehicle, now):
        clock[0] = now
        source.parser.model_detection_t = now
        state = make_state((vehicle,) if vehicle else ())
        return source._smooth_scene_state(state)

    high = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.6)
    low = replace(high, probability=0.49)
    assert len(scene.build_cluster_scene(update(high, 1.0)).vehicles) == 2
    retained = update(low, 1.1)
    assert retained.detected_vehicles[0].model_probability_retained
    assert len(scene.build_cluster_scene(retained).vehicles) == 2
    assert len(scene.build_cluster_scene(update(low, 1.31)).vehicles) == 1

    update(high, 2.0)
    update(None, 2.1)
    faded = update(None, 2.5).detected_vehicles[0]
    assert faded.render_phase == "fading"
    assert 0.5 <= faded.probability <= 0.6
    assert not update(None, 2.71).detected_vehicles


def test_live_radar_requires_distinct_updates_and_stable_duration(monkeypatch):
    source = make_live_source()
    clock = [1.0]
    monkeypatch.setattr(cluster_live.time, "monotonic", lambda: clock[0])
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=0.72, relative_speed_mps=-1.0)

    def update(now, sensor_time, points=(point,)):
        clock[0] = now
        source.parser.live_track_radar_t = sensor_time
        return source._smooth_scene_state(make_state(radar_points=points))

    assert not update(1.0, 1.0).radar_points
    assert not update(1.3, 1.0).radar_points
    assert not update(1.31, 1.1).radar_points
    assert update(1.32, 1.21).radar_points


def test_short_lived_radar_candidate_is_not_displayed(monkeypatch):
    source = make_live_source()
    clock = [1.0]
    monkeypatch.setattr(cluster_live.time, "monotonic", lambda: clock[0])
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=0.72, relative_speed_mps=-1.0)
    for now, points in ((1.0, (point,)), (1.1, (point,)), (1.15, ()), (1.2, (point,)), (1.3, (point,))):
        clock[0] = now
        source.parser.live_track_radar_t = now
        assert not source._smooth_scene_state(make_state(radar_points=points)).radar_points


def test_all_model_predictions_can_be_displayed():
    leads = (
        DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9, primary=True),
        DetectedVehicle("M2", 45.0, 3.6, source="modelV2.leadsV3", probability=0.85),
        DetectedVehicle("M3", 65.0, -3.6, source="modelV2.leadsV3", probability=0.95),
    )
    frame = scene.build_cluster_scene(make_state(leads))
    assert {vehicle.label for vehicle in frame.vehicles[1:]} == {"M1", "M2", "M3"}


def test_model_does_not_take_missing_radar_speed():
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9, primary=True)
    point = RadarPoint("R1", 25.0, 0.0, source="liveTracks", probability=1.0, relative_speed_mps=-2.0)
    frame = scene.build_cluster_scene(make_state((lead,), (point,)))
    displayed = screen_vehicles(frame)
    assert displayed[1].raw_probability == pytest.approx(0.9)
    assert displayed[1].relative_speed_mps is None
    assert displayed[1].absolute_speed_kph is None
    assert displayed[1].source == "modelV2.leadsV3"


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("radar_display_mode", [0, 1])
def test_screen_overlap_prefers_pure_model_even_when_match_gate_rejects(compact, radar_display_mode):
    lead = DetectedVehicle("M1", 40.0, 0.0, source="modelV2.leadsV3", probability=0.9,
                           x_std_m=1.0, y_std_m=0.2)
    radar = DetectedVehicle("TARGET", 44.0, 0.0, source="radarState", probability=1.0,
                            relative_speed_mps=-2.0, primary=True)
    assert not scene.detected_vehicles_same_front_vehicle(lead, radar)
    frame = scene.build_cluster_scene(make_state((radar, lead), radar_display_mode=radar_display_mode))
    assert {vehicle.label for vehicle in frame.vehicles[1:]} == {"M1", "TARGET"}
    displayed = screen_vehicles(frame, compact)
    assert [vehicle.label for vehicle in displayed[1:]] == ["M1"]
    assert displayed[1].relative_speed_mps is None


@pytest.mark.parametrize("compact", [False, True])
def test_screen_dedup_keeps_separate_neighbor(compact):
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9)
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=1.0, relative_speed_mps=-1.0)
    frame = scene.build_cluster_scene(make_state((lead,), (point,)))
    assert {vehicle.source for vehicle in screen_vehicles(frame, compact)[1:]} == {"modelV2.leadsV3", "radarPoint"}


def test_screen_overlap_deduplicates_same_source_and_keeps_ego(monkeypatch):
    pytest.importorskip("pyray")
    import cluster_renderer as renderer

    frame = scene.build_cluster_scene(make_state((
        DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9),
        DetectedVehicle("M2", 30.0, 0.0, source="modelV2.leadsV3", probability=0.9),
    )))
    monkeypatch.setattr(renderer, "vehicle_screen_bounds", lambda *args: (0.0, 0.0, 10.0, 10.0))
    displayed = renderer.vehicles_without_screen_overlap(frame.vehicles, None, 1920, 720, False, False, 0.0)
    assert [vehicle.label for vehicle in displayed] == [frame.vehicles[0].label, "M1"]


@pytest.mark.parametrize(("radar_x", "visible"), [(10.0, True), (9.99, False)])
def test_screen_bounds_require_positive_overlap(monkeypatch, radar_x, visible):
    pytest.importorskip("pyray")
    import cluster_renderer as renderer

    frame = scene.build_cluster_scene(make_state((
        DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9),
        DetectedVehicle("TARGET", 40.0, 0.0, source="radarState", probability=1.0, primary=True),
    )))
    monkeypatch.setattr(renderer, "vehicle_screen_bounds",
                        lambda vehicle, *args: (0.0 if vehicle.label == "M1" else radar_x, 0.0, 10.0, 10.0))
    displayed = renderer.vehicles_without_screen_overlap(frame.vehicles, None, 1920, 720, False, False, 0.0)
    assert any(vehicle.label == "TARGET" for vehicle in displayed) is visible


def test_screen_bounds_clip_at_camera_near_plane():
    rl = pytest.importorskip("pyray")
    import cluster_renderer as renderer

    frame = scene.build_cluster_scene(make_state((
        DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9),
    )))
    vehicle = replace(frame.vehicles[1], center=scene.Vec3(0.0, 0.5, 0.0))
    camera = rl.Camera3D(rl.Vector3(0, 0, 1), rl.Vector3(0, 1, 1), rl.Vector3(0, 0, 1),
                         44.0, rl.CameraProjection.CAMERA_PERSPECTIVE)
    assert renderer.vehicle_screen_bounds(vehicle, camera, 1920, 720, False, False, 0.0) is not None


def test_model_uncertainty_prevents_inconsistent_radar_pairing():
    lead = DetectedVehicle("M1", 40.0, 0.0, source="modelV2.leadsV3", probability=0.9,
                           x_std_m=1.0, y_std_m=0.2)
    point = RadarPoint("R1", 40.0, 1.2, source="liveTracks", relative_speed_mps=-1.0)
    radar_lead = DetectedVehicle("TARGET", 40.0, 1.2, source="radarState", probability=1.0, primary=True)
    state = make_state((lead, radar_lead), (point,))
    assert not scene.radar_point_close_to_detected_vehicle(point, lead)
    assert not scene.detected_vehicles_same_front_vehicle(lead, radar_lead)
    assert not OpenpilotLiveSource._model_vehicle_supported_by_radar(lead, (point,))
    assert scene.radar_merge_point_for_vehicle(lead, (point,), state) is None


@pytest.mark.parametrize(
    ("x_std", "y_std", "x_offset", "y_offset", "matches"),
    [(1.0, 0.2, 3.0, 0.9, True), (1.0, 0.2, 3.01, 0.0, False),
     (1.0, 0.2, 0.0, 0.91, False), (2.0, 0.6, 4.0, 1.2, True),
     (2.0, 0.6, 4.01, 0.0, False), (2.0, 0.6, 0.0, 1.21, False)],
)
def test_model_uncertainty_match_boundaries(x_std, y_std, x_offset, y_offset, matches):
    lead = DetectedVehicle("M1", 40.0, 0.0, source="modelV2.leadsV3", x_std_m=x_std, y_std_m=y_std)
    assert scene.model_position_matches(lead, lead.longitudinal_m + x_offset, y_offset) is matches


def test_missing_uncertainty_does_not_change_radar_association():
    lead = DetectedVehicle("M1", 40.0, 0.0, source="modelV2.leadsV3", probability=0.9)
    point = RadarPoint("R1", 40.0, 1.2, source="liveTracks", relative_speed_mps=-1.0)
    radar_lead = DetectedVehicle("TARGET", 40.0, 1.2, source="radarState", probability=1.0, primary=True)
    assert scene.radar_point_close_to_detected_vehicle(point, lead)
    assert scene.detected_vehicles_same_front_vehicle(lead, radar_lead)
    assert OpenpilotLiveSource._model_vehicle_supported_by_radar(lead, (point,))
