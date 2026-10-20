from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Longitudinal",
    "key": "dp_lon_traffic_stop",
    "type": "toggle_item",
    "title": lambda: tr("Traffic Light / Stop Sign Stopping Beta"),
    "description": lambda: tr(
      "Uses the driving model's own predicted path to detect an upcoming red "
      "light or stop sign and brings the car to a smooth stop, without needing "
      "a dedicated traffic-light recognition model.<br>Ported from the "
      "openpilot carrot (cp) branch.<br>The MPC stop obstacle is delayed until "
      "braking is needed, then retained for that stop. Uphill detection suppresses "
      "new stops above 30 km/h and requires 0.5 seconds of confirmation at lower "
      "speeds; this can also suppress a real uphill stop."),
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
  },
  {
    "section": "Longitudinal",
    "key": "dp_lon_traffic_stop_distance_adjust_m",
    "type": "spin_button_item",
    "title": lambda: tr("Traffic Stop Distance Adjust"),
    "description": lambda: tr(
      "Fine-tune where the car stops relative to the detected stop line. "
      "Positive values stop further back, negative values stop closer.<br>"
      "This is on top of a fixed camera-to-bumper correction - start small "
      "and adjust based on your own installation."),
    "default": "0",
    "min_val": -5,
    "max_val": 5,
    "step": 1,
    "suffix": lambda: tr("m"),
    "depends_on": "dp_lon_traffic_stop == 1",
    "flags": "PERSISTENT",
    "param_type": "INT",
  },
]
