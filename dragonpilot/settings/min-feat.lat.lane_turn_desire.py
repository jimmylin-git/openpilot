from dragonpilot.settings import tr

ITEMS = [
  {
    "section": "Lateral",
    "key": "dp_lane_turn_desire",
    "type": "toggle_item",
    "title": lambda: tr("Use Lane Turn Desires (LTD)"),
    "description": lambda: tr("Below the set speed, a single blinker can request a turn when either ego-lane line becomes weak (probability 0.5 or lower). The turn stays latched while that blinker remains on, speed stays below the limit, and no blindspot or road edge blocks that side. Missing lane probabilities retain the original behavior. From SunnyPilot."),
    "flags": "PERSISTENT",
    "param_type": "BOOL",
    "default": "0",
    "on_change": [{
      "target": "dp_lane_turn_value",
      "action": "set_visible",
      "condition": "value == '1'"  # 修改此處
    }]
  },
  {
    "section": "Lateral",
    "key": "dp_lane_turn_value",
    "type": "spin_button_item",
    "title": lambda: tr("Adjust Lane Turn Speed"),
    "description": lambda: tr("Set the maximum speed for lane turn desires (Default is 20 km/h)."),
    "flags": "PERSISTENT",
    "param_type": "INT",
    "default": "20",
    "min_val": 10,
    "max_val": 30,
    "step": 5,
    "suffix": lambda: tr(" km/h"),
    "initially_visible_by": {
      "param": "dp_lane_turn_desire",
      "condition": "value == '1'", # 修改此處
      "default": 0
    }
  },
]
