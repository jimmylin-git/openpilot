from dataclasses import replace
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cluster_scene as scene
from cluster_models import DetectedVehicle, RadarPoint, SimulatorInput
from cluster_simulator import ClusterSimulator


def make_state(vehicles=(), radar_points=(), radar_display_mode=0):
    state = ClusterSimulator().update(SimulatorInput(), 0.05)
    return replace(state, detected_vehicles=vehicles, radar_points=radar_points, radar_display_mode=radar_display_mode)


@pytest.mark.parametrize("radar_display_mode", [0, 1])
def test_model_probability_controls_display_even_with_high_confidence_radar(radar_display_mode):
    radar = DetectedVehicle("TARGET", 25.0, 0.0, source="radarState", probability=1.0, primary=True)
    point = RadarPoint("R1", 25.0, 0.0, source="liveTracks", probability=1.0, relative_speed_mps=-2.0)
    low = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.79, primary=True)
    frame = scene.build_cluster_scene(make_state((radar, low), (point,), radar_display_mode))
    assert len(frame.vehicles) == 1

    high = replace(low, probability=0.9)
    frame = scene.build_cluster_scene(make_state((radar, high), (point,), radar_display_mode))
    assert len(frame.vehicles) == 2
    assert frame.vehicles[1].label == "M1"
    assert frame.vehicles[1].raw_probability == pytest.approx(0.9)


def test_radar_only_vehicle_is_hidden():
    radar = DetectedVehicle("TARGET", 25.0, 0.0, source="radarState", probability=1.0, primary=True)
    point = RadarPoint("R1", 40.0, 3.6, source="liveTracks", probability=1.0, relative_speed_mps=-1.0)
    frame = scene.build_cluster_scene(make_state((radar,), (point,)))
    assert len(frame.vehicles) == 1


@pytest.mark.parametrize(("probability", "visible"), [(0.7999, False), (0.8, True), (0.8001, True)])
def test_existing_probability_threshold(probability, visible):
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=probability, primary=True)
    frame = scene.build_cluster_scene(make_state((lead,)))
    assert (len(frame.vehicles) == 2) is visible


def test_all_model_predictions_can_be_displayed():
    leads = (
        DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9, primary=True),
        DetectedVehicle("M2", 45.0, 3.6, source="modelV2.leadsV3", probability=0.85),
        DetectedVehicle("M3", 65.0, -3.6, source="modelV2.leadsV3", probability=0.95),
    )
    frame = scene.build_cluster_scene(make_state(leads))
    assert {vehicle.label for vehicle in frame.vehicles[1:]} == {"M1", "M2", "M3"}


def test_radar_speed_can_supplement_model_without_changing_probability():
    lead = DetectedVehicle("M1", 25.0, 0.0, source="modelV2.leadsV3", probability=0.9, primary=True)
    point = RadarPoint("R1", 25.0, 0.0, source="liveTracks", probability=1.0, relative_speed_mps=-2.0)
    frame = scene.build_cluster_scene(make_state((lead,), (point,)))
    assert frame.vehicles[1].raw_probability == pytest.approx(0.9)
    assert frame.vehicles[1].relative_speed_mps == pytest.approx(-2.0)
