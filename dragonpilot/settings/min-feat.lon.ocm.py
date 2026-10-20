from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Longitudinal",
    "key": "dp_lon_ocm",
    "type": "toggle_item",
    "title": lambda: tr("Overtaking Coasting Mode (OCM)"),
    "description": lambda: tr("After you accelerate above your set speed with the gas pedal (2-30 km/h over, at 50 km/h or faster), coast back down toward the set speed. Cancels when a lead car, traffic light, curve or downhill takes over. Uses a Corolla TSS2 coasting calibration; zero throttle or braking is not guaranteed on other vehicles."),
   #"condition": "openpilotLongitudinalControl",
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
  },
]
