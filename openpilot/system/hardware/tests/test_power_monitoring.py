from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.system.hardware.power_monitoring import PowerMonitoring, CAR_BATTERY_CAPACITY_uWh, \
  CAR_CHARGING_RATE_W, VBATT_PAUSE_CHARGING, DELAY_SHUTDOWN_TIME_S, MAX_TIME_OFFROAD_S

# Create fake time
ssb = 0.
def mock_time_monotonic():
  global ssb
  ssb += 1.
  return ssb

def set_mock_time(value):
  global ssb
  ssb = value

TEST_DURATION_S = 50
GOOD_VOLTAGE = 12 * 1e3
VOLTAGE_BELOW_PAUSE_CHARGING = (VBATT_PAUSE_CHARGING - 1) * 1e3

def pm_patch(mocker, name, value, constant=False):
  if constant:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", value)
  else:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", return_value=value)


class TestPowerMonitoring(OpenpilotTestCase):
  def setup_method(self):
    self._fixture("mocker").patch("time.monotonic", mock_time_monotonic)
    self.params = Params()

  # Test to see that it doesn't do anything when pandaState is None
  def test_panda_state_present(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(None, False)
    assert pm.get_power_used() == 0
    assert pm.get_car_battery_capacity() == (CAR_BATTERY_CAPACITY_uWh / 10)

  # Test to see that it doesn't integrate offroad when ignition is True
  def test_offroad_ignition(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(GOOD_VOLTAGE, True)
    assert pm.get_power_used() == 0

  # Test to see that it integrates with discharging battery
  def test_offroad_integration_discharging(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_power_usage = ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_power_used() - expected_power_usage) < 10

  # Test to check positive integration of car_battery_capacity
  def test_car_battery_integration_onroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    expected_capacity = ((TEST_DURATION_S/3600) * CAR_CHARGING_RATE_W * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check positive integration upper limit
  def test_car_battery_integration_upper_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh - 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    estimated_capacity = CAR_BATTERY_CAPACITY_uWh + (CAR_CHARGING_RATE_W / 3600 * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Test to check negative integration of car_battery_capacity
  def test_car_battery_integration_offroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_capacity = CAR_BATTERY_CAPACITY_uWh - ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check negative integration lower limit
  def test_car_battery_integration_lower_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    estimated_capacity = 0 - ((1/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Test to check policy of stopping charging after MAX_TIME_OFFROAD_S
  def test_max_time_offroad(self, mocker):
    MOCKED_MAX_OFFROAD_TIME = 3600
    pm_patch(mocker, "MAX_TIME_OFFROAD_S", MOCKED_MAX_OFFROAD_TIME, constant=True)
    pm = PowerMonitoring()
    start_time = ssb
    ignition = False
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME - 1)
    assert not pm.should_shutdown(ignition, True, start_time, False)
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME)
    assert pm.should_shutdown(ignition, True, start_time, False)

  # Voltage and battery estimates are not trusted on this port, only the offroad timer
  def test_low_voltage_does_not_shutdown(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 0)
    self.params.put("MaxTimeOffroad", 0, block=True)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    ignition = False
    start_time = ssb
    for _ in range(DELAY_SHUTDOWN_TIME_S * 2):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
    assert not pm.should_shutdown(ignition, True, start_time, True)

  # Test to check policy of not stopping charging when DisablePowerDown is set
  def test_disable_power_down(self):
    self.params.put_bool("DisablePowerDown", True, block=True)
    self.params.put("MaxTimeOffroad", 5, block=True)
    pm = PowerMonitoring()
    start_time = ssb
    set_mock_time(start_time + 3600)
    assert not pm.should_shutdown(False, True, start_time, True)

  # Test to check policy of not stopping charging when ignition
  def test_ignition(self):
    self.params.put("MaxTimeOffroad", 5, block=True)
    pm = PowerMonitoring()
    start_time = ssb
    set_mock_time(start_time + 3600)
    assert not pm.should_shutdown(True, True, start_time, True)

  def test_onroad_never_shuts_down(self):
    self.params.put("MaxTimeOffroad", 5, block=True)
    pm = PowerMonitoring()
    set_mock_time(ssb + 3600)
    assert not pm.should_shutdown(False, True, None, True)

  # Harness status and started_seen do not block the offroad timer
  def test_timer_without_harness_or_started(self):
    self.params.put("MaxTimeOffroad", 5, block=True)
    pm = PowerMonitoring()
    start_time = ssb
    set_mock_time(start_time + DELAY_SHUTDOWN_TIME_S)
    assert pm.should_shutdown(False, False, start_time, False)

  def test_delay_shutdown_time(self):
    self.params.put("MaxTimeOffroad", 1, block=True)
    pm = PowerMonitoring()
    offroad_timestamp = ssb
    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S - 1)
    assert not pm.should_shutdown(False, True, offroad_timestamp, True), \
                     f"Should not shutdown before {DELAY_SHUTDOWN_TIME_S} seconds offroad time"
    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S)
    assert pm.should_shutdown(False, True, offroad_timestamp, True), \
                    f"Should shutdown after {DELAY_SHUTDOWN_TIME_S} seconds offroad time"

  def test_interaction_restarts_timer(self):
    self.params.put("MaxTimeOffroad", 10, block=True)
    pm = PowerMonitoring()
    offroad_timestamp = ssb
    interaction = offroad_timestamp + 500
    self.params.put("LastInteractionMonotonic", interaction, block=True)
    set_mock_time(offroad_timestamp + 10 * 60 + 1)
    assert not pm.should_shutdown(False, True, offroad_timestamp, True)
    set_mock_time(interaction + 10 * 60 - 1)
    assert pm.should_shutdown(False, True, offroad_timestamp, True)

  def test_interaction_before_offroad_ignored(self):
    self.params.put("MaxTimeOffroad", 10, block=True)
    pm = PowerMonitoring()
    offroad_timestamp = ssb + 1000
    self.params.put("LastInteractionMonotonic", offroad_timestamp - 100, block=True)
    set_mock_time(offroad_timestamp + 10 * 60 - 1)
    assert pm.should_shutdown(False, True, offroad_timestamp, True)

  @parameterized.expand(
    [
      # No max time set – fallback to default (30 hours)
      (None, 0, False),
      (None, MAX_TIME_OFFROAD_S + 1, True),  # exceeds 30h (1800+ mins)

      # Valid max time values (in minutes)
      (60, 59, False),  # under limit
      (60, 120, True),  # over limit
      (10, 8, False),  # under limit
      (10, 11, True),  # over limit

      # Edge case: max time is zero → no limit enforced
      (0, 0, False),
      (0, 400, False),

      # Invalid max time formats or negative values → fallback to 30 hours
      (-100, 100, False),  # should fallback to 30h
      (-1, MAX_TIME_OFFROAD_S + 1, True),  # should fallback to 30h, and exceed it
    ]
  )
  def test_max_time_offroad_exceeded(self, max_time_offroad, offroad_time_min, expected_result):
    # Set the parameter if provided
    if max_time_offroad is not None:
      self.params.put("MaxTimeOffroad", max_time_offroad, block=True)

    # Convert offroad time from minutes to seconds
    offroad_time_s = offroad_time_min * 60

    pm = PowerMonitoring()
    result = pm.max_time_offroad_exceeded(offroad_time_s)

    assert result == expected_result
