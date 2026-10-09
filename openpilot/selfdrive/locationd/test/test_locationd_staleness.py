import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "locationd.py"


def load_estimator():
  tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocationEstimator")
  names = {"handle_log", "camera_odometry_fresh", "reset"}
  methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
  result = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HandleLogResult")
  constants = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id in {"ROTATION_SANITY_CHECK", "TRANS_SANITY_CHECK", "MIN_STD_SANITY_CHECK",
                                      "MAX_FILTER_REWIND_TIME", "YAWRATE_CROSS_ERR_CHECK_FACTOR",
                                      "POSENET_STD_HIST_HALF", "CAM_ODO_ROT_STD_MULT", "CAM_ODO_TRANS_STD_MULT", "CAM_ODO_POSE_DELAY"}]
  logger = Mock()
  namespace = {
    "np": np, "cloudlog": logger,
    "PoseKalman": SimpleNamespace(initial_x=np.zeros(18), initial_P=np.eye(18)),
    "States": SimpleNamespace(GYRO_BIAS=slice(9, 12)),
    "ObservationKind": SimpleNamespace(PHONE_GYRO=1, CAMERA_ODO_ROTATION=2, CAMERA_ODO_TRANSLATION=3),
    "rotate_std": lambda rotation, std: np.sqrt((rotation ** 2) @ (std ** 2)),
  }
  module = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    ast.ImportFrom(module="enum", names=[ast.alias(name="Enum")], level=0),
    *constants, result, ast.ClassDef(name="Estimator", bases=[], keywords=[], body=methods, decorator_list=[]),
  ], type_ignores=[])
  exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
  return namespace["Estimator"], namespace["HandleLogResult"], logger


class TestCameraOdometryStaleness(unittest.TestCase):
  def setUp(self):
    estimator, self.result, self.logger = load_estimator()
    self.est = estimator()
    self.est.kf = SimpleNamespace(x=np.zeros(18), predict_and_observe=Mock(return_value=None), init_state=Mock())
    self.est.camodo_yawrate_time = 100.
    self.est.camodo_yawrate_distribution = np.array([0., 0.001])
    self.est.camodo_yawrate_stale = False
    self.est.device_from_calib = np.eye(3)
    self.est.posenet_stds = np.ones(40)
    self.est._validate_sensor_time = lambda sensor_time, log_time: True
    self.est._validate_timestamp = lambda timestamp: True
    self.est._validate_sensor_source = lambda source: True

  def gyro(self, t, yaw):
    msg = SimpleNamespace(timestamp=int(t * 1e9), source=1, gyroUncalibrated=SimpleNamespace(v=[-yaw, 0., 0.]),
                          which=lambda: "gyroUncalibrated")
    return self.est.handle_log(t, "gyroscope", msg)

  def camera(self, t):
    msg = SimpleNamespace(timestampEof=int((t + 0.1) * 1e9), rot=[0., 0., 0.], trans=[0., 0., 0.],
                          rotStd=[0.001] * 3, transStd=[0.1] * 3)
    return self.est.handle_log(t, "cameraOdometry", msg)

  def test_freshness_boundary_and_missing_data(self):
    self.assertTrue(self.est.camera_odometry_fresh(100.8))
    self.assertFalse(self.est.camera_odometry_fresh(100.800001))
    self.assertFalse(self.est.camera_odometry_fresh(99.199999))
    self.assertFalse(self.est.camera_odometry_fresh(float("nan")))
    self.est.camodo_yawrate_time = None
    self.assertFalse(self.est.camera_odometry_fresh(100.))

  def test_stale_camera_does_not_blame_or_fuse_gyro(self):
    for t in (101., 104., 109.):
      self.assertEqual(self.gyro(t, 0.37), self.result.CAMERA_ODO_STALE)
    self.est.kf.predict_and_observe.assert_not_called()
    self.logger.warning.assert_called_once()
    self.assertFalse(self.est.camera_odometry_fresh(109.))

  def test_fresh_camera_keeps_existing_consistency_threshold(self):
    self.assertEqual(self.gyro(100.1, 0.37), self.result.INPUT_INVALID)
    self.assertEqual(self.gyro(100.1, 0.01), self.result.SUCCESS)
    self.est.kf.predict_and_observe.assert_called_once()

  def test_extreme_gyro_is_rejected_even_with_stale_camera(self):
    self.assertEqual(self.gyro(105., 10.), self.result.INPUT_INVALID)
    self.est.kf.predict_and_observe.assert_not_called()

  def test_nonfinite_gyro_is_rejected_on_every_axis_with_fresh_or_stale_camera(self):
    for t in (100.1, 105.):
      for axis in range(3):
        for value in (np.nan, np.inf, -np.inf):
          values = [0., 0., 0.]
          values[axis] = value
          msg = SimpleNamespace(timestamp=int(t * 1e9), source=1, gyroUncalibrated=SimpleNamespace(v=values),
                                which=lambda: "gyroUncalibrated")
          self.assertEqual(self.est.handle_log(t, "gyroscope", msg), self.result.INPUT_INVALID)
    self.est.kf.predict_and_observe.assert_not_called()

  def test_valid_camera_restores_comparison_without_resetting_filter(self):
    self.assertEqual(self.gyro(105., 0.37), self.result.CAMERA_ODO_STALE)
    self.assertEqual(self.camera(110.), self.result.SUCCESS)
    self.assertTrue(self.est.camera_odometry_fresh(110.1))
    self.assertFalse(self.est.camodo_yawrate_stale)
    self.logger.info.assert_called_once()
    self.assertEqual(self.gyro(110.1, 0.01), self.result.SUCCESS)
    self.assertEqual(self.gyro(110.1, 0.37), self.result.INPUT_INVALID)
    self.est.kf.init_state.assert_not_called()

  def test_invalid_camera_does_not_refresh_timestamp(self):
    msg = SimpleNamespace(timestampEof=int(110.1 * 1e9), rot=[11., 0., 0.], trans=[0., 0., 0.])
    self.assertEqual(self.est.handle_log(110., "cameraOdometry", msg), self.result.INPUT_INVALID)
    self.assertEqual(self.est.camodo_yawrate_time, 100.)

  def test_filter_reset_invalidates_old_camera(self):
    self.est.reset(110.)
    self.assertIsNone(self.est.camodo_yawrate_time)
    np.testing.assert_array_equal(self.est.camodo_yawrate_distribution, [0., 10.])
    self.assertFalse(self.est.camera_odometry_fresh(110.))

  def test_publication_remains_invalid_when_camera_is_stale(self):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in {"odometry_time", "inputs_valid"}]
    condition = compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), "exec")
    for all_valid, critical_valid, simulation, log_time, now, expected in (
      (True, True, False, 100.1, 100.1, True), (True, True, False, 100.1, 105., False),
      (False, True, False, 100.1, 100.1, False), (True, False, False, 100.1, 100.1, False),
      (True, True, True, 100.1, 105., True), (True, True, True, 105., 100.1, False),
      (False, True, True, 100.1, 105., False), (True, False, True, 100.1, 105., False),
    ):
      monotonic = Mock(return_value=now)
      namespace = {"sm": SimpleNamespace(all_valid=Mock(return_value=all_valid), logMonoTime={"cameraOdometry": int(log_time * 1e9)}),
                   "critical_service_inputs_valid": critical_valid, "SIMULATION": simulation,
                   "estimator": self.est, "time": SimpleNamespace(monotonic=monotonic)}
      exec(condition, namespace)
      self.assertEqual(namespace["inputs_valid"], expected)
      self.assertEqual(monotonic.call_count, int(not simulation))


if __name__ == "__main__":
  unittest.main()
