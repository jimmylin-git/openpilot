import random

import numpy as np

from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import Calibrator, INPUTS_NEEDED, INPUTS_WANTED, BLOCK_SIZE, MIN_SPEED_FILTER, \
                                                         MAX_YAW_RATE_FILTER, SMOOTH_CYCLES, HEIGHT_INIT, MAX_ALLOWED_PITCH_SPREAD, MAX_ALLOWED_YAW_SPREAD


def process_messages(c, cam_odo_calib, cycles,
                     cam_odo_speed=MIN_SPEED_FILTER + 1,
                     carstate_speed=MIN_SPEED_FILTER + 1,
                     cam_odo_yr=0.0,
                     cam_odo_speed_std=1e-3,
                     cam_odo_height_std=1e-3):
  old_rpy_weight_prev = 0.0
  for _ in range(cycles):
    assert (old_rpy_weight_prev - c.old_rpy_weight < 1/SMOOTH_CYCLES + 1e-3)
    old_rpy_weight_prev = c.old_rpy_weight
    c.handle_v_ego(carstate_speed)
    c.handle_cam_odom([cam_odo_speed,
                       np.sin(cam_odo_calib[2]) * cam_odo_speed,
                       -np.sin(cam_odo_calib[1]) * cam_odo_speed],
                        [0.0, 0.0, cam_odo_yr],
                        [0.0, 0.0, 0.0],
                        [cam_odo_speed_std, cam_odo_speed_std, cam_odo_speed_std],
                        [0.0, 0.0, HEIGHT_INIT.item()],
                        [cam_odo_height_std, cam_odo_height_std, cam_odo_height_std])

class TestCalibrationd(OpenpilotTestCase):

  def test_invalid_odometry_preserves_calibration(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    vectors = [[MIN_SPEED_FILTER + 1, 0., 0.], [0., 0., 0.], [0., 0., 0.],
               [1e-3] * 3, [0., 0., HEIGHT_INIT.item()], [1e-3] * 3]
    for index in range(len(vectors)):
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(vector=index, value=value):
          bad = [list(v) for v in vectors]
          bad[index][1] = value
          before = (c.idx, c.block_idx, c.valid_blocks, c.old_rpy_weight)
          arrays = [v.copy() for v in (c.rpys, c.heights, c.wide_from_device_eulers)]
          self.assertIsNone(c.handle_cam_odom(*bad))
          self.assertEqual((c.idx, c.block_idx, c.valid_blocks, c.old_rpy_weight), before)
          for actual, expected in zip((c.rpys, c.heights, c.wide_from_device_eulers), arrays, strict=True):
            np.testing.assert_array_equal(actual, expected)
    self.assertIsNotNone(c.handle_cam_odom(*vectors))
    self.assertFalse(c._invalid_cam_odom)

  def test_malformed_and_negative_uncertainty_rejected(self):
    c = Calibrator(param_put=False)
    c.handle_v_ego(MIN_SPEED_FILTER + 1)
    vectors = [[MIN_SPEED_FILTER + 1, 0., 0.], [0.] * 3, [0.] * 3, [1e-3] * 3,
               [0., 0., HEIGHT_INIT.item()], [1e-3] * 3]
    for index in range(len(vectors)):
      bad = [list(v) for v in vectors]
      bad[index] = [0., 0.]
      self.assertIsNone(c.handle_cam_odom(*bad))
    for index in (3, 5):
      bad = [list(v) for v in vectors]
      bad[index][1] = -1.
      self.assertIsNone(c.handle_cam_odom(*bad))
    c.handle_v_ego(np.nan)
    self.assertIsNone(c.handle_cam_odom(*vectors))
    self.assertEqual(c.idx, 0)

  def test_read_saved_params(self):
    msg = messaging.new_message('extrinsicsCalibration')
    msg.extrinsicsCalibration.validBlocks = random.randint(1, 10)
    msg.extrinsicsCalibration.rpyCalib = [random.random() for _ in range(3)]
    msg.extrinsicsCalibration.height = [random.random() for _ in range(1)]
    Params().put("CalibrationParams", msg.to_bytes(), block=True)
    c = Calibrator(param_put=True)

    np.testing.assert_allclose(msg.extrinsicsCalibration.rpyCalib, c.rpy)
    np.testing.assert_allclose(msg.extrinsicsCalibration.height, c.height)
    assert msg.extrinsicsCalibration.validBlocks == c.valid_blocks


  def test_calibration_basics(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED)
    assert c.valid_blocks == INPUTS_WANTED
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)
    c.reset()


  def test_calibration_low_speed_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed=MIN_SPEED_FILTER - 1)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, carstate_speed=MIN_SPEED_FILTER - 1)
    assert c.valid_blocks == 0
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)


  def test_calibration_yaw_rate_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_yr=MAX_YAW_RATE_FILTER)
    assert c.valid_blocks == 0
    np.testing.assert_allclose(c.rpy, np.zeros(3))
    np.testing.assert_allclose(c.height, HEIGHT_INIT)


  def test_calibration_speed_std_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed_std=1e3)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, np.zeros(3))


  def test_calibration_speed_std_height_reject(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_height_std=1e3)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, np.zeros(3))


  def test_calibration_auto_reset(self):
    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0], atol=1e-3)
    process_messages(c, [0.0, MAX_ALLOWED_PITCH_SPREAD*0.9, MAX_ALLOWED_YAW_SPREAD*0.9], BLOCK_SIZE + 10)
    assert c.valid_blocks == INPUTS_NEEDED + 1
    assert c.cal_status == log.ExtrinsicsCalibration.Status.calibrated

    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0])
    process_messages(c, [0.0, MAX_ALLOWED_PITCH_SPREAD*1.1, 0.0], BLOCK_SIZE + 10)
    assert c.valid_blocks == 1
    assert c.cal_status == log.ExtrinsicsCalibration.Status.recalibrating
    np.testing.assert_allclose(c.rpy, [0.0, MAX_ALLOWED_PITCH_SPREAD*1.1, 0.0], atol=1e-2)

    c = Calibrator(param_put=False)
    process_messages(c, [0.0, 0.0, 0.0], BLOCK_SIZE * INPUTS_NEEDED)
    assert c.valid_blocks == INPUTS_NEEDED
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, 0.0])
    process_messages(c, [0.0, 0.0, MAX_ALLOWED_YAW_SPREAD*1.1], BLOCK_SIZE + 10)
    assert c.valid_blocks == 1
    assert c.cal_status == log.ExtrinsicsCalibration.Status.recalibrating
    np.testing.assert_allclose(c.rpy, [0.0, 0.0, MAX_ALLOWED_YAW_SPREAD*1.1], atol=1e-2)
