"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from pathlib import Path

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


_RADAR_ENABLE_PATH = Path("/data/params/d/HondaBoschARadarOn")


class HondaSettings(BrandSettings):
  def __init__(self):
    super().__init__()

    self.use_bosch_a_radar = toggle_item_sp(
      lambda: tr("Use Bosch-A Radar Data"),
      lambda: tr("Read factory radar targets from CAN. Read-only; factory AEB and longitudinal control are unaffected. Requires a vehicle restart."),
      initial_state=_RADAR_ENABLE_PATH.exists(),
      callback=self._on_bosch_a_radar_toggled,
      enabled=lambda: ui_state.is_offroad(),
    )

    self.items = [self.use_bosch_a_radar]

  def _on_bosch_a_radar_toggled(self, state: bool):
    if state:
      _RADAR_ENABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
      _RADAR_ENABLE_PATH.touch()
    else:
      _RADAR_ENABLE_PATH.unlink(missing_ok=True)

  def update_settings(self):
    pass
