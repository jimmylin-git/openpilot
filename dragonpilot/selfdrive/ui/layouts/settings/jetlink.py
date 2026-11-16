"""
dp - jetlink settings panel (tici).

Ported from the jetlink parts of zoompilot's sunnypilot models panel
(selfdrive/ui/sunnypilot/layouts/settings/models.py): the Jetlink setting
(Off / USB / iOS, offroad only, turns ADB off) and the failover note.

dp has no sunnypilot model manager. Its big-model menu is built from the same
sunnypilot chestnut catalog the Jetlink phone/Mac apps list, and from what the
far end reports in its hello (the model it has loaded and the ones it has
built). The default is to follow the far end: whatever is picked on the phone
drives. The live connection rows are dp's: zoompilot folds them into the
descriptions.
"""
import threading
import time

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.list_view import multiple_button_item, text_item, button_item
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.widgets.scroller_tici import Scroller
from dragonpilot import jetlink_adapter
from dragonpilot.selfdrive.ui import jetlink_ui
from dragonpilot.selfdrive.ui.jetlink_ui import LINK_MODES, LINK_MODE_TITLES, LINK_PARAM, NONE_TEXT

DESCRIPTION = ("Run big models on an external device connected to the comma USB-C port (USB for Jetson / Linux PC / Mac, iOS for iPhone). "
               + "Enabling this disables ADB; changes are offroad-only. If the connection drops or lags, comma immediately falls back to its small model.")

# dp: the iPhone charging toggle, sealed. The param stays declared so an older value can be cleared
CHARGE_PARAM = "JetlinkChargePhone"
CHARGE_SEALED_NOTE = ("Phone charging from comma is permanently disabled. USB-C phone charging has caused motherboard damage on a comma 3 (C3), "
                      + "and camera failures and frozen displays on clone devices. "
                      + "Charge your iPhone using a car charger, MagSafe, or a USB-C hub with PD input.")

PROGRESS_STAGES = {"connect": "Connecting", "fetch": "Downloading", "upload": "Uploading", "build": "Building", "failed": "Failed"}


def _bool_text(v: bool) -> str:
  return "Yes" if v else "No"


class JetlinkLayout(Widget):
  def __init__(self):
    super().__init__()
    self._last_description = None
    # what the far end reported, read at most once a second (files and params, not per frame)
    self._far = {'loaded': NONE_TEXT, 'built': "", 'model': NONE_TEXT, 'following': True, 'at': 0.0}

    self._link_item = multiple_button_item(
      "Jetlink", DESCRIPTION,
      buttons=[LINK_MODE_TITLES[m] for m in LINK_MODES],
      selected_index=LINK_MODES.index(jetlink_ui.link_mode()),
      button_width=250, callback=self._on_link_mode)

    self._state_item = text_item("Connection Status", lambda: jetlink_ui.STATE_TEXT[ui_state.jetlink_state],
                                 description=lambda: jetlink_ui.status_note(self._far['model']))
    self._transport_item = text_item("Transport", self._transport)
    self._present_item = text_item("External Device Online", lambda: _bool_text(bool(ui_state.jetlink and ui_state.jetlink.present)))
    self._port_item = text_item("USB Port", self._port)
    self._model_item = text_item("Model for Driving", self._model)
    self._ready_item = text_item("External Engine Built", lambda: _bool_text(bool(ui_state.jetlink and ui_state.jetlink.ready)))
    self._progress_item = text_item("Progress", self._progress)
    self._reason_item = text_item("Unavailable Reason", lambda: (ui_state.jetlink.reason if ui_state.jetlink and ui_state.jetlink.reason else NONE_TEXT))
    self._adb_item = text_item("ADB", lambda: "Disabled by Jetlink" if ui_state.adb_blocked else ("On" if ui_state.params.get_bool("AdbEnabled") else "Off"))

    # dp: phone charging is sealed (jetlink.comma.port.PHONE_CHARGING_SEALED); the toggle is gone and the
    # panel says why. A value an earlier build stored is cleared, so nothing reads as on
    self._charge_item = text_item("Charge iPhone from comma", "Disabled",
                                  description=CHARGE_SEALED_NOTE)
    try:
      if ui_state.params.get_bool(CHARGE_PARAM):
        ui_state.params.put_bool(CHARGE_PARAM, False)
    except Exception:
      pass

    # dp: the big-model menu (follow the far end, or a catalog model) and the far end's own models
    self._menu: list[dict] = []
    self._menu_labels: dict[str, str | None] = {}
    self._refreshing = False
    self._refresh_result = ""
    self._source_item = button_item("Big Model Source", "Select", description=self._source_description, callback=self._open_source_dialog)
    self._far_end_item = text_item("External Device Model", self._far_end_loaded, description=self._far_end_built)
    self._refresh_item = button_item("Refresh Model List", lambda: "Refreshing..." if self._refreshing else "Refresh",
                                     description=lambda: self._refresh_result or "Fetch the shared sunnypilot / Jetlink catalog to name models on your phone.",
                                     callback=self._refresh_catalog, enabled=lambda: not self._refreshing)

    self._items = [self._link_item, self._state_item, self._transport_item, self._present_item, self._port_item,
                   self._source_item, self._far_end_item, self._model_item, self._ready_item, self._progress_item,
                   self._refresh_item, self._reason_item, self._charge_item, self._adb_item]
    self._scroller = Scroller(self._items, line_separator=True, spacing=0)

  # -- values -------------------------------------------------------------------

  @staticmethod
  def _transport() -> str:
    j = ui_state.jetlink
    return j.transport if j is not None else NONE_TEXT

  @staticmethod
  def _port() -> str:
    j = ui_state.jetlink
    if j is None or j.port is None:
      return NONE_TEXT
    return "No device" if j.port == "empty" else "Device present"

  def _model(self) -> str:
    """The model that will drive: the far end's loaded one when following, else the comma's pick."""
    return self._far['model']

  @staticmethod
  def _progress() -> str:
    p = jetlink_ui.progress()
    if p is None:
      return NONE_TEXT
    stage, frac, msg = p
    label = PROGRESS_STAGES.get(stage, stage)
    pct = f" {frac * 100:.0f}%" if 0.0 < frac < 1.0 else ""
    return f"{label}{pct} {msg}".strip()

  # -- the model menu -------------------------------------------------------------

  @staticmethod
  def _following_now() -> bool:
    return jetlink_adapter.follows_far_end(ui_state.params.get(jetlink_adapter.KEYS.big_model))

  def _following(self) -> bool:
    return self._far['following']

  def _refresh_far_end(self) -> None:
    now = time.monotonic()
    if now - self._far['at'] < 1.0:
      return
    loaded, built = jetlink_adapter.far_end_models()
    menu = jetlink_adapter.model_menu(ui_state.params)
    by_sha = {r['sha256']: r['name'] for r in menu if r['sha256']}
    name = (lambda s: by_sha.get(s, s[:16]) if s else None)
    following = self._following_now()
    j = ui_state.jetlink
    if following and loaded:
      model = name(loaded)
    elif j is not None:
      model = j.active_model or j.model or j.default_model or NONE_TEXT
    else:
      model = NONE_TEXT
    self._far = {'loaded': name(loaded) or "Not reported",
                 'built': ("Built on external device: " + ", ".join(name(b) for b in built))
                          if built else "No built models reported yet (shown after connecting).",
                 'model': model, 'following': following, 'at': now}

  def _source_description(self) -> str:
    if self._far['following']:
      return "Follow external device: use the big model selected in the phone (or Jetson / Mac) app. No separate download on comma."
    slot = ui_state.params.get(jetlink_adapter.KEYS.big_model) or {}
    return (f"Selected by comma: {slot.get('displayName') or slot.get('ref', '')[:10]}. "
            + "If missing on the external device, comma downloads and uploads it for building while parked.")

  def _far_end_loaded(self) -> str:
    return self._far['loaded']

  def _far_end_built(self) -> str:
    return self._far['built']

  def _open_source_dialog(self):
    self._menu = jetlink_adapter.model_menu(ui_state.params)
    loaded, _ = jetlink_adapter.far_end_models()
    follow_label = f"Follow External Device (current: {jetlink_adapter.model_name(loaded, ui_state.params) or 'Unknown'})"
    self._menu_labels = {follow_label: None}
    for row in self._menu:
      notes = [n for n, on in (("loaded", row['loaded']), ("built externally", row['built']), ("downloaded on comma", row['downloaded'])) if on]
      label = row['name'] + (f" ({', '.join(notes)})" if notes else "")
      self._menu_labels[label] = row['ref']
    slot = ui_state.params.get(jetlink_adapter.KEYS.big_model)
    current = follow_label if self._following_now() else next((k for k, v in self._menu_labels.items() if v == (slot or {}).get('ref')), "")
    dialog = MultiOptionDialog("Select Big Model", list(self._menu_labels.keys()), current,
                               callback=lambda result: self._on_source_selected(result, dialog))
    gui_app.push_widget(dialog)

  def _on_source_selected(self, result, dialog):
    if result != DialogResult.CONFIRM or dialog.selection not in self._menu_labels:
      return
    if not ui_state.is_offroad():
      return  # the far end switches engines only parked, as the link setting does
    jetlink_adapter.pick_model(self._menu_labels[dialog.selection], ui_state.params)
    self._far['at'] = 0.0

  def _refresh_catalog(self):
    if self._refreshing:
      return
    self._refreshing = True
    self._refresh_result = "Refreshing..."

    def work():
      try:
        n = jetlink_adapter.refresh_catalog(ui_state.params)
        self._refresh_result = f"Updated: {n} big models."
      except Exception as e:
        self._refresh_result = f"Refresh failed: {e}"
      finally:
        self._refreshing = False

    threading.Thread(target=work, daemon=True).start()

  # -- events -------------------------------------------------------------------

  def _on_link_mode(self, index: int):
    if not ui_state.is_offroad():
      # the gadget changes only once the car is parked; put the buttons back
      self._link_item.action_item.set_selected_button(LINK_MODES.index(jetlink_ui.link_mode()))
      return
    ui_state.params.put(LINK_PARAM, int(index))
    if LINK_MODES[index] != "off" and ui_state.params.get_bool("AdbEnabled"):
      ui_state.params.put_bool("AdbEnabled", False)

  def _update_state(self):
    self._refresh_far_end()
    j = ui_state.jetlink
    mode = jetlink_ui.link_mode()
    on = mode != "off"
    self._link_item.action_item.set_selected_button(LINK_MODES.index(mode))
    self._link_item.action_item.set_enabled(ui_state.is_offroad())
    for item in (self._state_item, self._transport_item, self._present_item, self._port_item,
                 self._model_item, self._ready_item, self._progress_item, self._source_item, self._far_end_item, self._refresh_item):
      item.set_visible(on or (j is not None and j.present))
    self._source_item.action_item.set_enabled(ui_state.is_offroad())
    self._reason_item.set_visible(bool(j is not None and j.reason))
    self._charge_item.set_visible(mode == "ios")

    status = jetlink_ui.link_status()
    description = f"{DESCRIPTION} {status}".strip()
    if j is None and not on:
      description += " (Jetlink package not found on this device)"
    if description != self._last_description:
      self._last_description = description
      self._link_item.set_description(description)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    ui_state.update_params()

  def _render(self, rect):
    self._scroller.render(rect)
