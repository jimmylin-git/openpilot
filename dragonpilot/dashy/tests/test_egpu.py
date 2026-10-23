"""Hardware-free Dashy API and feature-removal regressions."""
import ast
import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "dragonpilot/dashy/web/dist"
REF = "a" * 40


class Params:
  def __init__(self, **values):
    self.values = {"IsOffroad": True, "IsOnroad": False, **values}

  def get(self, key):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value):
    self.values[key] = value

  def put_bool(self, key, value):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


def load_api():
  spec = importlib.util.spec_from_file_location("egpu_constants", ROOT / "dragonpilot/egpu/constants.py")
  assert spec is not None and spec.loader is not None
  constants = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(constants)
  namespace = {
    "asyncio": asyncio, "logging": logging, "web": web, "c": constants,
    "jetlink_adapter": SimpleNamespace(KEYS=SimpleNamespace(link="Jetlink")),
    "cached_bundles": lambda p: [{"ref": REF, "display_name": "<Model>"}],
    "pointer_for": lambda p, ref: {"oid": ref, "size": 10},
    "model_compiled": lambda oid: True, "onnx_present": lambda pointer: True,
    "active_ready": lambda active: bool(active and active.get("sha256")),
  }
  source = ROOT / "dragonpilot/dashy/egpu.py"
  tree = ast.parse(source.read_text(encoding="utf-8"))
  tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
  exec(compile(tree, str(source), "exec"), namespace)
  return SimpleNamespace(**namespace)


def make_app(params=None):
  api = load_api()
  app = web.Application()
  cache = SimpleNamespace(params=params or Params(), invalidate=Mock())
  app["cache"] = cache
  app.router.add_get("/api/egpu", api.get_status)
  app.router.add_post("/api/egpu", api.post_action)
  app.router.add_get("/", lambda request: web.FileResponse(WEB / "index.html"))
  app.router.add_static("/", WEB)
  return app, api, cache


def test_catalog_and_live_status_contract():
  api = load_api()
  p = Params()
  p.put(api.c.P_ACTIVE_BUNDLE, {"ref": REF, "display_name": "Selected", "sha256": REF})
  p.put(api.c.P_STATUS, {"state": "downloading", "progress": 42})
  result = api.snapshot(p)
  assert result["offroad"] and result["ready"] and result["busy"]
  assert result["models"] == [{"ref": REF, "name": "<Model>", "compiled": True, "downloaded": True}]
  assert result["progress"]["progress"] == 42
  assert not result["running"]


@pytest.mark.parametrize("state", [
  {"IsOffroad": False}, {"IsOnroad": True}, {"IsOffroad": False, "IsOnroad": True},
  {"IsOffroad": None, "IsOnroad": None},
])
@pytest.mark.parametrize("action", ["enabled", "select", "delete", "cancel", "clear", "refresh"])
def test_all_mutations_require_explicit_offroad(state, action):
  api = load_api()
  p = Params(**state)
  before = dict(p.values)
  with pytest.raises(web.HTTPConflict):
    api.mutate(p, {"action": action, "value": REF})
  assert p.values == before


@pytest.mark.parametrize("data", [
  [], None, {"action": "unknown"}, {"action": "enabled", "value": "true"},
  {"action": "select", "value": "../model"}, {"action": "delete", "value": ""},
  {"action": "select", "value": []}, {"action": "delete", "value": {}},
])
def test_bad_input_never_writes_params(data):
  api = load_api()
  p = Params()
  before = dict(p.values)
  with pytest.raises(web.HTTPBadRequest):
    api.mutate(p, data)
  assert p.values == before


def test_actions_use_existing_manager_protocol_and_cancel_does_not_requeue():
  api = load_api()
  p, c = Params(), api.c
  api.mutate(p, {"action": "enabled", "value": True})
  assert p.get_bool(c.P_ENABLED)
  api.mutate(p, {"action": "select", "value": REF})
  assert p.get(c.P_DOWNLOAD_REF) == REF and not p.get_bool(c.P_AUTO_DEFAULT)
  api.mutate(p, {"action": "cancel"})
  assert p.get(c.P_DOWNLOAD_REF) is None and not p.get_bool(c.P_AUTO_DEFAULT)
  api.mutate(p, {"action": "delete", "value": REF})
  assert p.get(c.P_DELETE_REF) == REF
  p.remove(c.P_DELETE_REF)
  api.mutate(p, {"action": "clear"})
  assert p.get_bool(c.P_CLEAR_CACHE)
  p.remove(c.P_CLEAR_CACHE)
  p.put(c.P_ACTIVE_BUNDLE, {"ref": REF})
  api.mutate(p, {"action": "select", "value": ""})
  assert p.get(c.P_ACTIVE_BUNDLE) is None
  api.mutate(p, {"action": "refresh"})
  assert p.get(c.P_MANIFEST_SYNC) == 0


@pytest.mark.parametrize("pending", ["download", "compile", "delete", "clear"])
@pytest.mark.parametrize("action", ["select", "delete", "clear"])
def test_conflicting_manager_operations_are_rejected(pending, action):
  api = load_api()
  p, c = Params(), api.c
  if pending == "download":
    p.put(c.P_DOWNLOAD_REF, REF)
  elif pending == "compile":
    p.put(c.P_STATUS, {"state": c.ST_COMPILING})
  elif pending == "delete":
    p.put(c.P_DELETE_REF, REF)
  else:
    p.put_bool(c.P_CLEAR_CACHE, True)
  before = dict(p.values)
  with pytest.raises(web.HTTPConflict):
    api.mutate(p, {"action": action, "value": REF})
  assert p.values == before


def test_jetlink_is_not_changed_and_conflicts_block_enabling():
  api = load_api()
  p = Params(Jetlink=1)
  with pytest.raises(web.HTTPConflict):
    api.mutate(p, {"action": "enabled", "value": True})
  api.mutate(p, {"action": "enabled", "value": False})
  assert p.get("Jetlink") == 1
  assert not p.get_bool(api.c.P_ENABLED)


def test_active_model_removed_from_catalog_remains_deletable_not_selectable():
  api = load_api()
  other = "b" * 40
  p = Params()
  p.put(api.c.P_ACTIVE_BUNDLE, {"ref": other, "sha256": other})
  assert other in {row["ref"] for row in api.snapshot(p)["models"]}
  with pytest.raises(web.HTTPConflict):
    api.mutate(p, {"action": "select", "value": other})
  api.mutate(p, {"action": "delete", "value": other})
  assert p.get(api.c.P_DELETE_REF) == other


def test_http_endpoints_static_entry_and_failure_responses():
  async def run():
    app, api, cache = make_app()
    async with TestClient(TestServer(app)) as client:
      response = await client.get("/api/egpu")
      assert response.status == 200 and (await response.json())["offroad"]
      for path, content in [("/", "/pages/egpu.html"), ("/pages/egpu.html", "/js/egpu.js"),
                            ("/js/egpu.js", "/api/egpu")]:
        response = await client.get(path)
        assert response.status == 200 and content in await response.text()
      response = await client.post("/api/egpu", json={"action": "select", "value": REF},
                                   headers={"Origin": str(client.make_url("/")).rstrip("/")})
      assert response.status == 200 and (await response.json())["ok"]
      assert cache.params.get(api.c.P_DOWNLOAD_REF) == REF
      cache.invalidate.assert_called_once()
      response = await client.post("/api/egpu", json={"action": "cancel"}, headers={"Origin": "https://other.example"})
      assert response.status == 403
      response = await client.post("/api/egpu", data="action=cancel")
      assert response.status == 415
      response = await client.post("/api/egpu", data="{bad", headers={"Content-Type": "application/json"})
      assert response.status == 400
      cache.params.put_bool("IsOnroad", True)
      response = await client.post("/api/egpu", json={"action": "cancel"})
      assert response.status == 409
      response = await client.get("/api/egpu")
      assert response.status == 200 and not (await response.json())["offroad"]
      cache.params.get = Mock(side_effect=OSError("test read failure"))
      response = await client.get("/api/egpu")
      assert response.status == 503 and "error" in await response.json()
  asyncio.run(run())


def test_production_server_wires_both_endpoints():
  source = (ROOT / "dragonpilot/dashy/serverd.py").read_text(encoding="utf-8")
  tree = ast.parse(source)
  routes = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
  assert "app.router.add_get('/api/egpu', egpu.get_status)" in routes
  assert "app.router.add_post('/api/egpu', egpu.post_action)" in routes


def test_tdx_removed_but_traffic_stop_and_gps_remain():
  for path in ["selfdrive/ui/onroad/hud_renderer.py", "selfdrive/ui/mici/onroad/hud_renderer.py"]:
    source = (ROOT / path).read_text(encoding="utf-8")
    assert "tdx" not in source.lower()
    assert "def _draw_traffic_stop" in source
  source = (ROOT / "system/manager/process_config.py").read_text(encoding="utf-8")
  assert "tdxd" not in source
  assert 'PythonProcess("gpsd"' in source and 'PythonProcess("cluster_d"' in source
  assert '"tdx"' not in (ROOT / "cereal/services.py").read_text(encoding="utf-8")
  assert '"tdx"' not in (ROOT / "selfdrive/ui/ui_state.py").read_text(encoding="utf-8")
  assert not (ROOT / "dragonpilot/selfdrive/controls/lib/tdx/tdxd.py").exists()
  assert not (ROOT / "dragonpilot/selfdrive/controls/lib/tdx/freeway_shapes.json").exists()


@pytest.mark.parametrize("mici", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_actual_hud_updates_without_tdx_and_retains_stop_banner(mici, fresh):
  path = ROOT / ("selfdrive/ui/mici/onroad/hud_renderer.py" if mici else "selfdrive/ui/onroad/hud_renderer.py")
  tree = ast.parse(path.read_text(encoding="utf-8"))
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HudRenderer")
  methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_update_state", "_update_traffic_stop")]
  car = SimpleNamespace(vCruiseCluster=80, vEgoCluster=10, vEgo=10, steeringAngleDeg=5,
                        leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False)
  class SM(dict):
    recv_frame = {"carState": 2 if fresh else 0, "longitudinalPlanDP": 2}
    alive = {"longitudinalPlanDP": True}
  sm = SM(carState=car, controlsState=SimpleNamespace(), radarState=SimpleNamespace(leadOne=SimpleNamespace(status=False)),
          longitudinalPlanDP=SimpleNamespace(targets=[SimpleNamespace(braking=True)]),
          carControl=SimpleNamespace(longActive=True), selfdriveState=SimpleNamespace(enabled=True))
  namespace = {
    "ui_state": SimpleNamespace(sm=sm, started_frame=1, is_metric=True),
    "rl": SimpleNamespace(get_time=lambda: 1), "SET_SPEED_NA": 255, "TRAFFIC_STOP_IDX": 0,
    "TS_HOLD_TIME": 0.5, "CV": SimpleNamespace(MS_TO_KPH=3.6), "DEBUG_TEST_UI": False,
  }
  exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
  obj = SimpleNamespace(v_ego_cluster_seen=False, _ts_last_braking_t=-1e9, set_speed=0, _engaged=False,
                        _dp_indicator_show_left=False, _dp_indicator_show_right=False,
                        _dp_indicator_count_left=0, _dp_indicator_count_right=0)
  obj._update_dp_indicator_side_state = lambda *args: (False, 0, None)
  if not mici:
    obj._update_traffic_stop = lambda message: namespace["_update_traffic_stop"](obj, message)
  namespace["_update_state"](obj)
  assert obj.ts_braking == fresh
  assert obj.speed == (36 if fresh else 0)
