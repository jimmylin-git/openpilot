from pathlib import Path

from openpilot.common.file_chunker import get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id
from openpilot.common.model_pickle import load_oob as _load_oob

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_oob(f):
  return _load_oob(f)

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

def chestnut_compiled() -> bool:
  path = modeld_pkl_path(chestnut=True)
  return (path.is_file() or Path(get_manifest_path(path)).is_file()) and all(
    (MODELS_DIR / f'big_driving_warp_{size}_tinygrad.pkl').is_file() for size in ('1344x760', '1928x1208'))
