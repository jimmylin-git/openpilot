"""
dp - egpu: what the settings panel and the onroad / sidebar icons show of the eGPU big model.

Ported from sunnypilot's ChestnutState icon logic (selfdrive/ui/ui_state.py, layouts/sidebar.py,
mici/onroad/hud_renderer.py) and model_info.py. Reads ui_state.egpu_* (taken on the 5 Hz params pass).

The icons are comma's chestnut icons, the same files the jetlink port ships
(dragonpilot/selfdrive/assets/icons/jetlink*.png): sunnypilot draws the chestnut with them too.
Status strings use English, matching the device UI.
"""
from openpilot.selfdrive.ui.ui_state import ui_state, EgpuState, JetlinkState
from dragonpilot.selfdrive.ui import jetlink_ui
from dragonpilot.egpu.bundles import pointer_for, onnx_present, model_compiled
from dragonpilot.egpu.constants import (ST_IDLE, ST_RESOLVING, ST_DOWNLOADING, ST_COMPILING_WARP, ST_COMPILING, ST_DONE,
                                        ST_FAILED, ST_CANCELLED, ST_WAITING_EGPU, BUSY_STATES)

NONE_TEXT = jetlink_ui.NONE_TEXT

STATE_TEXT = {
  EgpuState.DISCONNECTED: "Offline",
  EgpuState.NO_MODEL: "Not in use",
  EgpuState.READY: "Ready",
  EgpuState.LOADING: "Loading",
  EgpuState.ACTIVE: "Big model",
  EgpuState.FAILED: "Failed",
}

STATE_COLOR = {
  EgpuState.DISCONNECTED: jetlink_ui.GREY,
  EgpuState.NO_MODEL: jetlink_ui.ORANGE,
  EgpuState.READY: jetlink_ui.GREEN,
  EgpuState.LOADING: jetlink_ui.YELLOW,
  EgpuState.ACTIVE: jetlink_ui.GREEN,
  EgpuState.FAILED: jetlink_ui.ORANGE,
}

# same drawing rules as sunnypilot's chestnut icon, through jetlink's JetlinkIcons
_AS_JETLINK = {
  EgpuState.DISCONNECTED: JetlinkState.DISCONNECTED,
  EgpuState.NO_MODEL: JetlinkState.UNCOMPILED,   # orange: plugged in but not used
  EgpuState.READY: JetlinkState.READY,
  EgpuState.LOADING: JetlinkState.LOADING,       # pulses
  EgpuState.ACTIVE: JetlinkState.ACTIVE,
  EgpuState.FAILED: JetlinkState.FAILED,
}


def icon_for(icons: "jetlink_ui.JetlinkIcons", state: EgpuState):
  return icons.for_state(_AS_JETLINK[state])


def badge_text(state: EgpuState) -> str:
  """the onroad badge's word: the model's short name while it drives"""
  if state == EgpuState.ACTIVE and ui_state.egpu_bundle:
    return str(ui_state.egpu_bundle.get("short_name", STATE_TEXT[state]))
  return STATE_TEXT[state]


def model_name() -> str:
  b = ui_state.egpu_bundle
  return b["display_name"] if b else "None selected (using built-in small model)"


def model_state(ref: str) -> str:
  """'compiled' | 'downloaded' | 'none' for one catalog model, from the files on disk"""
  p = pointer_for(ui_state.params, ref)
  if p and model_compiled(p["oid"]):
    return "compiled"
  if onnx_present(p):
    return "downloaded"
  return "none"


def bundle_label(b: dict, state: str) -> str:
  active = ui_state.egpu_bundle
  if active and active.get("ref") == b["ref"]:
    tag = " (active)"
  else:
    tag = {"compiled": " (compiled)", "downloaded": " (downloaded)"}.get(state, "")
  return f"{b['display_name']}{tag}"


def status_note() -> str:
  """sunnypilot's Model Status failover note, for the eGPU"""
  state = ui_state.egpu_state
  name = (ui_state.egpu_bundle or {}).get("short_name", "Big model")
  if state == EgpuState.DISCONNECTED:
    return "No eGPU detected. You can select and download a big model now; compilation will finish automatically when the eGPU is connected."
  if not ui_state.egpu_enabled:
    return "eGPU big models are disabled. Using the built-in small model."
  if state == EgpuState.NO_MODEL:
    if ui_state.egpu_bundle is None:
      return "eGPU detected, but no compiled big model is available. Using the built-in small model."
    return "Using the built-in small model for this drive (eGPU connected after ignition). The big model will be used at the next ignition."
  if state == EgpuState.FAILED:
    return f"{name} is unavailable. Using the built-in small model for this drive; retrying at the next ignition."
  if state == EgpuState.LOADING:
    return f"{name} is loading. Model output will not be published until loading completes."
  if state == EgpuState.ACTIVE:
    return f"Using {name} on the eGPU. On error, the built-in small model will be used for the rest of this drive."
  return f"{name} will be used at the next ignition, with fallback to the built-in small model on loading or runtime errors."


def _mmss(seconds) -> str:
  s = int(seconds or 0)
  return f"{s // 60}:{s % 60:02d}"


def download_text() -> str:
  st = ui_state.egpu_status or {}
  state = st.get("state", ST_IDLE)
  name = st.get("name", "")
  if state == ST_RESOLVING:
    return f"{name}: resolving model"
  if state == ST_DOWNLOADING:
    return f"{name}: downloading ONNX {st.get('progress', 0.0):.0f}%"
  if state == ST_COMPILING_WARP:
    return f"Compiling warp {_mmss(st.get('elapsed'))}"
  if state == ST_COMPILING:
    return f"{name}: compiling {_mmss(st.get('elapsed'))}"
  if state == ST_WAITING_EGPU:
    if st.get("error", "").startswith("PCIe"):
      return f"{name}: waiting for eGPU connection ({st['error']})"
    return f"{name}: connect eGPU to compile"
  if state == ST_DONE:
    return f"{name}: done"
  if state == ST_FAILED:
    return f"{name}: failed"
  if state == ST_CANCELLED:
    return f"{name}: cancelled"
  return NONE_TEXT


def busy() -> bool:
  return (ui_state.egpu_status or {}).get("state") in (*BUSY_STATES, ST_WAITING_EGPU)
