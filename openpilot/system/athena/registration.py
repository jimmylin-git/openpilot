#!/usr/bin/env python3
import time
import json
import jwt
from typing import cast
from pathlib import Path

from datetime import datetime, timedelta, UTC
from openpilot.common.api import api_get, get_key_pair
from openpilot.common.params import Params
from openpilot.common.spinner import Spinner
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.common.hardware import HARDWARE, PC
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

def is_registered_device() -> bool:
  dongle = Params().get("DongleId")
  return dongle not in (None, UNREGISTERED_DONGLE_ID)


def register(show_spinner=False) -> str | None:
  """
  All devices built since March 2024 come with all
  info stored in /persist/. This is kept around
  only for devices built before then.

  With a backend update to take serial number instead
  of dongle ID to some endpoints, this can be removed
  entirely.
  """
  params = Params()

  dongle_id: str | None = params.get("DongleId")
  if dongle_id is None and Path(Paths.persist_root()+"/comma/dongle_id").is_file():
    # not all devices will have this; added early in comma 3X production (2/28/24)
    with open(Paths.persist_root()+"/comma/dongle_id") as f:
      dongle_id = f.read().strip()

  # Create registration token, in the future, this key will make JWTs directly
  jwt_algo, private_key, public_key = get_key_pair()

  if not public_key:
    dongle_id = UNREGISTERED_DONGLE_ID
    cloudlog.warning("missing public key")
    if show_spinner:
      # MR.ONE: a device without a key pair is not authorized - never enter the OS
      spinner = Spinner()
      try:
        while True:
          spinner.update("registering device - no key, contact MR.ONE")
          time.sleep(5)
      finally:
        spinner.close()
  elif dongle_id is None:
    if show_spinner:
      spinner = Spinner()
      spinner.update("registering device")

    # Block until we get the imei
    serial = HARDWARE.get_serial()
    start_time = time.monotonic()
    imei='865420071781912'
    while imei is None:
      try:
        imei = HARDWARE.get_imei()
      except Exception:
        cloudlog.exception("Error getting imei, trying again...")
        time.sleep(1)

      if time.monotonic() - start_time > 60 and show_spinner:
        spinner.update(f"registering device - serial: {serial}, IMEI: {imei}")

    backoff = 0
    start_time = time.monotonic()
    while True:
      try:
        register_token = jwt.encode({'register': True, 'exp': datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)},
                                    cast(str, private_key), algorithm=jwt_algo)
        cloudlog.info("getting pilotauth")
        resp = api_get("v2/pilotauth/", method='POST', timeout=15,
                       imei=imei, imei2="", serial=serial)

        # ========== MR.ONE: only an authorized dongle id lets the device through ==========
        # Not authorized / pending approval / any other error (402, 403, bad JSON, ...):
        # keep waiting on the registration screen - never skip into the OS.
        if resp is not None and resp.status_code == 200:
          try:
            dongleauth = json.loads(resp.text)
            dongle_id = dongleauth.get("dongle_id")
          except Exception:
            dongle_id = None
          if dongle_id and dongle_id != UNREGISTERED_DONGLE_ID:
            break  # authorized
        if show_spinner:
          spinner.update(f"registering device - serial: {serial}, contact MR.ONE")
        cloudlog.info("Device not authorized yet, retrying...")
        time.sleep(2)  # avoid hammering the backend
        continue  # keep waiting for backend authorization
        # =====================================
      except NotImplementedError:
        # dependency issues with PyJWT will hang the registration test in backoff loop otherwise
        raise
      except Exception:
        cloudlog.exception("failed to authenticate")
        backoff = min(backoff + 1, 15)
        time.sleep(backoff)

    if show_spinner:
      spinner.close()

  if dongle_id:
    params.put("DongleId", dongle_id, block=True)
    set_offroad_alert("Offroad_UnregisteredHardware", (dongle_id == UNREGISTERED_DONGLE_ID) and not PC)
  return dongle_id


if __name__ == "__main__":
  print(register())
