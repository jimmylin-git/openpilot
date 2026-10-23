"""Dashy eGPU management using the existing egpu_manager parameter protocol."""
import asyncio
import logging

from aiohttp import web

from dragonpilot import jetlink_adapter
from dragonpilot.egpu import constants as c
from dragonpilot.egpu.bundles import cached_bundles, pointer_for, model_compiled, onnx_present, active_ready

logger = logging.getLogger("dashy")


def is_offroad(params):
  return params.get_bool("IsOffroad") and not params.get_bool("IsOnroad")


def snapshot(params):
  active = params.get(c.P_ACTIVE_BUNDLE)
  bundles = cached_bundles(params)
  if isinstance(active, dict) and active.get("ref") and all(b["ref"] != active["ref"] for b in bundles):
    bundles.append(active)
  models = []
  for bundle in bundles:
    pointer = pointer_for(params, bundle["ref"])
    models.append({
      "ref": bundle["ref"], "name": bundle.get("display_name") or bundle["ref"][:10],
      "compiled": bool(pointer and model_compiled(pointer["oid"])), "downloaded": onnx_present(pointer),
    })
  progress = params.get(c.P_STATUS) or {}
  pending = params.get(c.P_DOWNLOAD_REF)
  maintenance = bool(params.get(c.P_DELETE_REF)) or params.get_bool(c.P_CLEAR_CACHE)
  return {
    "offroad": is_offroad(params), "enabled": params.get_bool(c.P_ENABLED),
    "present": params.get_bool(c.P_PRESENT), "loading": params.get_bool(c.P_LOADING),
    "running": params.get_bool(c.P_RUNNING), "failed": params.get_bool(c.P_FAILED),
    "ready": active_ready(active), "active": active or {}, "models": models,
    "pending": pending, "progress": progress,
    "busy": bool(pending) or progress.get("state") in c.BUSY_STATES or maintenance,
    "jetlink_enabled": int(params.get(jetlink_adapter.KEYS.link) or 0) != 0,
  }


def mutate(params, data):
  if not is_offroad(params):
    raise web.HTTPConflict(text="Changes require offroad mode")
  if not isinstance(data, dict):
    raise web.HTTPBadRequest(text="Expected a JSON object")
  action, value = data.get("action"), data.get("value")
  if action == "enabled":
    if type(value) is not bool:
      raise web.HTTPBadRequest(text="Enabled must be a boolean")
    if value and int(params.get(jetlink_adapter.KEYS.link) or 0) != 0:
      raise web.HTTPConflict(text="Disable Jetlink before enabling eGPU")
    params.put_bool(c.P_ENABLED, value)
  elif action == "refresh":
    params.put(c.P_MANIFEST_SYNC, 0)
  elif action == "cancel":
    params.put_bool(c.P_AUTO_DEFAULT, False)
    params.remove(c.P_DOWNLOAD_REF)
  elif action in ("select", "delete", "clear"):
    progress = params.get(c.P_STATUS) or {}
    if (params.get(c.P_DOWNLOAD_REF) or progress.get("state") in c.BUSY_STATES
        or params.get(c.P_DELETE_REF) or params.get_bool(c.P_CLEAR_CACHE)):
      raise web.HTTPConflict(text="Wait for the current operation to finish or cancel it")
    bundles = cached_bundles(params)
    refs = {b["ref"] for b in bundles}
    active = params.get(c.P_ACTIVE_BUNDLE)
    if isinstance(active, dict) and active.get("ref"):
      refs.add(active["ref"])
    if action != "clear":
      if not isinstance(value, str) or (value not in refs and not (action == "select" and value == "")):
        raise web.HTTPBadRequest(text="Model is not in the catalog")
      if action == "select" and value and value not in {b["ref"] for b in bundles}:
        raise web.HTTPConflict(text="Refresh the catalog before selecting this model")
    params.put_bool(c.P_AUTO_DEFAULT, False)
    if action == "clear":
      params.put_bool(c.P_CLEAR_CACHE, True)
    elif action == "delete":
      params.put(c.P_DELETE_REF, value)
    elif value:
      params.put(c.P_DOWNLOAD_REF, value)
    else:
      params.remove(c.P_ACTIVE_BUNDLE)
      params.remove(c.P_DOWNLOAD_REF)
  else:
    raise web.HTTPBadRequest(text="Unknown eGPU action")
  return {"ok": True}


async def get_status(request):
  try:
    return web.json_response(await asyncio.to_thread(snapshot, request.app["cache"].params))
  except Exception:
    logger.exception("Could not read eGPU settings")
    return web.json_response({"error": "Could not read eGPU settings; see Dashy logs"}, status=503)


async def post_action(request):
  origin = request.headers.get("Origin")
  if origin and origin != f"{request.scheme}://{request.host}":
    raise web.HTTPForbidden(text="Cross-origin changes are not allowed")
  if request.content_type != "application/json":
    raise web.HTTPUnsupportedMediaType()
  try:
    data = await request.json()
  except ValueError:
    raise web.HTTPBadRequest(text="Invalid JSON")
  result = mutate(request.app["cache"].params, data)
  request.app["cache"].invalidate()
  return web.json_response(result)
