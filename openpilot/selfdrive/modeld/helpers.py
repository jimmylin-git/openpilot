import os
from pathlib import Path

from openpilot.common.file_chunker import get_chunk_name, get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id
from openpilot.sunnypilot.modeld_v2.helpers import dump_oob, load_oob  # noqa: F401

MODELS_DIR = Path(__file__).resolve().parent / 'models'
CHESTNUT_POWERED_VOLTAGE = 5000
CHESTNUT_PCIE_READY = 0x78


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def chestnut_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return True
    except Exception:
      pass
  return False

def model_file_exists(path: str | Path) -> bool:
  path = Path(path)
  if path.is_file():
    return True

  manifest_path = Path(get_manifest_path(str(path)))
  if not manifest_path.is_file():
    return False
  try:
    num_chunks = int(manifest_path.read_text().strip())
  except (OSError, ValueError):
    return False
  return num_chunks > 0 and all(Path(get_chunk_name(str(path), i, num_chunks)).is_file() for i in range(num_chunks))


def chestnut_compiled() -> bool:
  bundled_model = modeld_pkl_path(chestnut=True)
  if model_file_exists(bundled_model):
    return True

  if (override := os.environ.get("COMBINED_MODEL_PKL")) and model_file_exists(override):
    return True

  from openpilot.common.hardware.hw import Paths
  from openpilot.sunnypilot.models.helpers import get_selected_bundle
  bundle = get_selected_bundle(source="chestnut")
  if bundle is None or not bundle.models:
    return False
  return model_file_exists(Path(Paths.model_root()) / bundle.models[0].artifact.fileName)


def chestnut_ready(state) -> bool:
  return state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE and not state.supplyFault and state.pcieLtssm == CHESTNUT_PCIE_READY
