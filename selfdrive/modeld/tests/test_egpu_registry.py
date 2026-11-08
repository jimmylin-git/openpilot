import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import urllib.error

import pytest


ROOT = Path(__file__).resolve().parents[3]
REF = "a" * 40
PIN = {"oid": "b" * 64, "size": 123}
OLD = {"oid": "c" * 64, "size": 456}


@pytest.fixture
def registry(monkeypatch):
  monkeypatch.syspath_prepend(str(ROOT / "jetlink_repo"))
  return importlib.import_module("jetlink.registry.catalog"), importlib.import_module("jetlink.registry.lfs")


def load_function(path, name, namespace):
  tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
  node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
  exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
  return namespace[name]


def opener_for(routes):
  calls = []

  def opener(request, timeout):
    url = request if isinstance(request, str) else request.full_url
    calls.append(url)
    value = routes[url]
    if value is None:
      raise urllib.error.HTTPError(url, 404, "missing", {}, None)
    return io.BytesIO(value.encode() if isinstance(value, str) else json.dumps(value).encode())

  return opener, calls


def bundle():
  return {"ref": REF, "is_big": True, "minimum_selector_version": "19", "index": 20}


@pytest.mark.parametrize("extra", [None, {"bundles": [], "pointers": {REF: PIN}}])
def test_catalog_fetch_preserves_optional_pins(registry, extra):
  catalog, _ = registry
  opener, _ = opener_for({
    catalog.CATALOG_URL: {"bundles": [bundle()]},
    catalog.CATALOG_URL_TEMPLATE.format(version=27): None,
    catalog.EXTRA_CATALOG_URL: extra,
  })
  result = catalog.fetch_catalogs(opener=opener)
  assert result["bundles"] == [bundle()]
  assert result.get("pointers", {}) == ({REF: PIN} if extra else {})


@pytest.mark.parametrize("ref,pin", [
  ("bad", PIN), (REF, None), (REF, {"oid": "bad", "size": 123}),
  (REF, {"oid": PIN["oid"], "size": True}), (REF, {"oid": PIN["oid"], "size": "123"}),
  (REF, {"oid": PIN["oid"], "size": 0}), (REF, {"oid": PIN["oid"], "size": -1}),
])
def test_catalog_rejects_malformed_pins(registry, ref, pin):
  catalog, _ = registry
  assert catalog.pinned_pointers({"pointers": {ref: pin}}) == {}


def export_routes(lfs, subject, folders):
  return {
    lfs.POINTER_URL.format(ref=REF): None,
    lfs.COMMIT_PATCH_URL.format(ref=REF): f"Subject: [PATCH] {subject}\n\n",
    lfs.DRIVING_MODELS_TREE_URL: [{"type": "directory", "path": f"{f}-uuid"} for f in folders],
    **{f"{lfs.DRIVING_MODELS_TREE_URL}/{f}-uuid?recursive=true": [
      {"type": "file", "path": f"{f}-uuid/12864/{lfs.BIG_ONNX}", "lfs": p},
    ] for f, p in folders.items()},
  }


def test_latest_export_in_pull_request_wins(registry):
  _, lfs = registry
  routes = export_routes(lfs, "New model (#123)", {"11111111": OLD, "22222222": PIN})
  routes[lfs.PULL_PATCH_URL.format(number="123")] = (
    "Subject: [PATCH 1/2] Use 11111111-uuid/12864\n\n" +
    "Subject: [PATCH 2/2] Use 22222222-uuid/12864\n\n"
  )
  opener, calls = opener_for(routes)
  assert lfs.fetch_pointer(REF, opener=opener) == lfs.Pointer(**PIN)
  assert not any("/11111111-uuid?" in url for url in calls)


@pytest.mark.parametrize("squashed", [False, True])
def test_missing_newest_named_export_does_not_use_old_model(registry, squashed):
  _, lfs = registry
  subject = "New model (#123)" if squashed else "Use 11111111-uuid then 22222222/12864"
  routes = export_routes(lfs, subject, {"11111111": OLD})
  if squashed:
    routes[lfs.PULL_PATCH_URL.format(number="123")] = (
      "Subject: [PATCH 1/2] Use 11111111-uuid/12864\n\n" +
      "Subject: [PATCH 2/2] Use 22222222/12864\n\n"
    )
  opener, calls = opener_for(routes)
  with pytest.raises(lfs.RegistryError, match="22222222.*not in.*yet"):
    lfs.fetch_pointer(REF, opener=opener)
  assert not any("/11111111-uuid?" in url for url in calls)


def test_bare_hex_word_does_not_block_export_lookup(registry):
  _, lfs = registry
  opener, _ = opener_for(export_routes(lfs, "Use 11111111-uuid, debug deadbeef", {"11111111": PIN}))
  assert lfs.fetch_pointer(REF, opener=opener) == lfs.Pointer(**PIN)


def test_missing_commit_export_does_not_fall_back_to_pull_export(registry):
  _, lfs = registry
  routes = export_routes(lfs, "Use 22222222/12864 (#123)", {"11111111": OLD})
  routes[lfs.PULL_PATCH_URL.format(number="123")] = "Subject: [PATCH] Use 11111111-uuid/12864\n\n"
  opener, calls = opener_for(routes)
  with pytest.raises(lfs.RegistryError, match="22222222.*not in.*yet"):
    lfs.fetch_pointer(REF, opener=opener)
  assert not any("/11111111-uuid?" in url for url in calls)


def test_direct_pointer_and_deleted_pointer_fallback(registry):
  _, lfs = registry
  pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{PIN['oid']}\nsize {PIN['size']}\n"
  opener, _ = opener_for({lfs.POINTER_URL.format(ref=REF): pointer})
  assert lfs.fetch_pointer(REF, opener=opener) == lfs.Pointer(**PIN)
  patch = f"Subject: [PATCH] compiled\n\ndiff --git a/selfdrive/modeld/models/{lfs.BIG_ONNX} b/selfdrive/modeld/models/{lfs.BIG_ONNX}\n"
  patch += "\n".join(f"-{line}" for line in pointer.splitlines())
  opener, _ = opener_for({lfs.POINTER_URL.format(ref=REF): None, lfs.COMMIT_PATCH_URL.format(ref=REF): patch})
  assert lfs.fetch_pointer(REF, opener=opener) == lfs.Pointer(**PIN)


@pytest.mark.parametrize("resolve", [False, True])
def test_refresh_pins_override_shared_cache_without_network_lookup(registry, monkeypatch, resolve):
  catalog, lfs = registry
  adapter = importlib.import_module("dragonpilot.jetlink_adapter")
  monkeypatch.setitem(sys.modules, "openpilot.common.params", SimpleNamespace(Params=object))
  monkeypatch.setattr(catalog, "fetch_catalogs", lambda: {"bundles": [bundle()], "pointers": {REF: PIN}})
  monkeypatch.setattr(lfs, "fetch_pointer", lambda *a, **k: pytest.fail("pinned model must not be looked up"))
  data = {adapter.KEYS.pointers: {REF: OLD, "d" * 40: OLD}}
  params = SimpleNamespace(get=data.get, put=lambda key, value, **kw: data.update({key: value}))
  assert adapter.refresh_catalog(params, resolve=resolve) == 1
  assert data[adapter.KEYS.pointers][REF] == PIN
  assert data[adapter.KEYS.pointers]["d" * 40] == OLD
  assert data[adapter.KEYS.catalog]["pointers"][REF] == PIN

  constants = importlib.import_module("dragonpilot.egpu.constants")
  namespace = {**vars(constants), "Params": object}
  pointer_for = load_function("dragonpilot/egpu/bundles.py", "pointer_for", namespace)
  data[constants.P_POINTERS] = {REF: OLD}
  assert pointer_for(params, REF) == PIN
  resolve_model = load_function("dragonpilot/egpu/manager.py", "resolve", {"pointer_for": pointer_for})
  remembered = []
  manager = SimpleNamespace(params=params, _remember_pointer=lambda ref, p: remembered.append((ref, p)))
  assert resolve_model(manager, bundle()) == PIN
  assert remembered == [(REF, PIN)]
  data[adapter.KEYS.catalog] = {"bundles": [bundle()]}
  assert pointer_for(params, REF) == OLD


def test_jetlink_runtime_pins_update_even_without_new_catalog_entries(registry, monkeypatch):
  catalog, _ = registry
  models_cls = importlib.import_module("jetlink.openpilot.models").Models
  old_catalog = {"bundles": [bundle()]}
  data = {"pointers": {REF: OLD}, "catalog": old_catalog}
  op = SimpleNamespace(
    keys=SimpleNamespace(pointers="pointers", catalog="catalog"), catalog_selector=19,
    get=data.get, put=lambda key, value, **kw: data.update({key: value}),
    log=SimpleNamespace(warning=lambda *args: None, exception=lambda *args: pytest.fail("unexpected failure")),
  )
  models = models_cls(op)
  assert models.model_index()[0]["oid"] == OLD["oid"]
  monkeypatch.setattr(catalog, "fetch_catalogs", lambda: {**old_catalog, "pointers": {REF: PIN}})
  assert models.big_catalog(old_catalog) == old_catalog
  assert models.resolve_pointer(REF) == (PIN["oid"], PIN["size"])
  assert models.model_index()[0]["oid"] == PIN["oid"]


def test_refresh_unpinned_models_resolves_and_logs_individual_failures(registry, monkeypatch):
  catalog, lfs = registry
  adapter = importlib.import_module("dragonpilot.jetlink_adapter")
  monkeypatch.setitem(sys.modules, "openpilot.common.params", SimpleNamespace(Params=object))
  other = {**bundle(), "ref": "d" * 40}
  monkeypatch.setattr(catalog, "fetch_catalogs", lambda: {"bundles": [bundle(), other]})

  def fetch(ref, timeout):
    if ref == other["ref"]:
      raise catalog.NetworkError("test network failure")
    return lfs.Pointer(**PIN)

  monkeypatch.setattr(lfs, "fetch_pointer", fetch)
  errors = []
  monkeypatch.setattr(adapter, "_log_failure", lambda message, exc: errors.append(message))
  data = {}
  params = SimpleNamespace(get=data.get, put=lambda key, value, **kw: data.update({key: value}))
  assert adapter.refresh_catalog(params) == 2
  assert data[adapter.KEYS.pointers] == {REF: PIN}
  assert "test network failure" in errors[0]
  assert len(data[adapter.KEYS.catalog]["bundles"]) == 2


def test_refresh_network_failure_keeps_previous_catalog_and_pointers(registry, monkeypatch):
  catalog, _ = registry
  adapter = importlib.import_module("dragonpilot.jetlink_adapter")
  monkeypatch.setitem(sys.modules, "openpilot.common.params", SimpleNamespace(Params=object))

  def fail():
    raise catalog.NetworkError("test network failure")

  monkeypatch.setattr(catalog, "fetch_catalogs", fail)
  data = {adapter.KEYS.catalog: {"bundles": [bundle()]}, adapter.KEYS.pointers: {REF: OLD}}
  params = SimpleNamespace(get=data.get, put=lambda *a, **kw: pytest.fail("failed refresh must not write"))
  with pytest.raises(catalog.NetworkError, match="test network failure"):
    adapter.refresh_catalog(params)
  assert data[adapter.KEYS.pointers][REF] == OLD


@pytest.mark.parametrize("resume", [False, True])
def test_download_still_verifies_and_resumes_onnx(registry, tmp_path, resume):
  _, lfs = registry
  payload = b"test ONNX bytes"
  pointer = lfs.Pointer(hashlib.sha256(payload).hexdigest(), len(payload))
  dest = tmp_path / "model.onnx"
  if resume:
    dest.with_suffix(".onnx.part").write_bytes(payload[:4])

  def opener(request, timeout):
    response = io.BytesIO(payload[4:] if resume else payload)
    response.status = 206 if resume else 200
    if resume:
      assert request.get_header("Range") == "bytes=4-"
    return response

  assert lfs.lfs_download("https://example.invalid/model", pointer, dest, opener=opener) == dest
  assert dest.read_bytes() == payload


def test_download_hash_failure_does_not_publish_model(registry, tmp_path):
  _, lfs = registry
  dest = tmp_path / "model.onnx"
  with pytest.raises(lfs.VerifyError, match="hash"):
    lfs.lfs_download("https://example.invalid/model", lfs.Pointer("b" * 64, 3), dest,
                     opener=lambda *a, **kw: io.BytesIO(b"bad"))
  assert not dest.exists()
  assert not dest.with_suffix(".onnx.part").exists()
