"""Offroad preparation and transactional selection of model-bound host-warp artifacts."""

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog

PREPARATION_VERSION = 1
RETRY_SECONDS = 3600


def cache_root() -> Path:
  return Path(Paths.model_root()) / '.host-warp'


def atomic_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
    temporary = Path(file.name)
    json.dump(value, file, sort_keys=True)
    file.flush()
    os.fsync(file.fileno())
  try:
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
  try:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
      raise ValueError(f"Expected an object in {path}")
    return value
  except FileNotFoundError:
    return {}


def bundle_identity(bundle) -> dict:
  if bundle is None:
    return {}
  value = bundle.to_dict() if hasattr(bundle, 'to_dict') else bundle
  return {key: value.get(key) for key in ('ref', 'generation', 'runner', 'is20hz', 'minimumSelectorVersion', 'overrides')} | {
    'artifacts': [{key: model.get('artifact', {}).get(key) for key in ('fileName', 'downloadUri', 'chunks')}
                  for model in value.get('models', [])],
  }


def source_for_bundle(params, bundle) -> dict | None:
  """Only consume source provenance explicitly published for this exact catalog artifact."""
  catalog = params.get('ModelManager_ModelsCache_Chestnut') or {}
  identity = bundle_identity(bundle)
  for entry in catalog.get('bundles', []):
    if entry.get('ref') != identity.get('ref'):
      continue
    source = entry.get('host_warp')
    if source is None:
      return None
    if not isinstance(source, dict) or not isinstance(source.get('onnx'), dict):
      raise ValueError("Invalid host-warp source descriptor")
    if identity.get('runner') != 'tinygrad':
      raise ValueError("Automatic host warp requires the modeld_v2 tinygrad runner")
    artifact_hashes = [artifact.get('downloadUri', {}).get('sha256') for artifact in identity['artifacts']]
    if len(artifact_hashes) != 1 or source.get('baseline_sha256') != artifact_hashes[0]:
      raise ValueError("Host-warp source does not match the selected baseline artifact")
    filename = identity['artifacts'][0].get('fileName', '')
    if not filename or '/' in filename or '\\' in filename or filename in ('.', '..'):
      raise ValueError("Invalid baseline artifact filename")
    if source.get('model_type') != 'supercombo':
      raise ValueError("Automatic host warp supports only supercombo")
    uri = source.get('onnx', {}).get('url', '')
    digest = source.get('onnx', {}).get('sha256', '')
    if urlparse(uri).scheme != 'https' or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
      raise ValueError("Host-warp ONNX requires HTTPS and a lowercase SHA-256")
    for name in ('model_size', 'camera_size'):
      size = source.get(name)
      if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 or v % 2 for v in size):
        raise ValueError(f"Invalid host-warp {name}")
    return source
  return None


@lru_cache(maxsize=1)
def runtime_identity() -> str:
  from openpilot.sunnypilot.modeld_v2.host_warp import warp_source_hash
  digest = hashlib.sha256(warp_source_hash().encode())
  for relative in ('openpilot/sunnypilot/modeld_v2/compile_modeld.py', 'openpilot/sunnypilot/modeld_v2/host_warp.py',
                   'openpilot/sunnypilot/modeld_v2/modeld.py', 'openpilot/sunnypilot/models/host_warp_worker.py',
                   'openpilot/sunnypilot/models/host_warp.py', 'openpilot/selfdrive/modeld/compile_modeld.py',
                   'openpilot/selfdrive/modeld/get_model_metadata.py', 'openpilot/system/camerad/cameras/nv12_info.py'):
    digest.update((Path(BASEDIR) / relative).read_bytes())
  return digest.hexdigest()


def request_for_bundle(params, bundle, device_type: str) -> dict | None:
  source = source_for_bundle(params, bundle)
  if source is None:
    return None
  return {'version': PREPARATION_VERSION, 'bundle': bundle_identity(bundle), 'source': source,
          'runtime': runtime_identity(), 'device': device_type}


def request_key(request: dict) -> str:
  return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def artifact_files(path: Path) -> list[Path]:
  from openpilot.common.file_chunker import get_chunk_name, get_manifest_path
  manifest = Path(get_manifest_path(str(path)))
  if manifest.exists():
    count = int(manifest.read_text().strip())
    if count < 1:
      raise ValueError("Invalid host-warp chunk manifest")
    return [manifest, *[Path(get_chunk_name(str(path), i, count)) for i in range(count)]]
  return [path]


def file_records(path: Path, *, hashes: bool) -> list[dict]:
  records = []
  for file in artifact_files(path):
    stat = file.stat()
    record = {'name': file.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if hashes:
      with file.open('rb') as stream:
        record['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    records.append(record)
  return records


def ready_artifact(request: dict, *, verify_hashes: bool = False) -> Path | None:
  directory = cache_root() / request_key(request)
  receipt = read_json(directory / 'ready.json')
  if not receipt:
    return None
  if receipt.get('request') != request:
    raise ValueError("Host-warp readiness receipt does not match the model/runtime")
  path = directory / 'model.pkl'
  records = file_records(path, hashes=verify_hashes)
  expected = receipt.get('files', [])
  if not verify_hashes:
    expected = [{key: record[key] for key in ('name', 'size', 'mtime_ns')} for record in expected]
  if records != expected:
    raise ValueError("Host-warp artifact changed after offroad validation")
  return path


def select_artifact(bundle, cam_w: int, cam_h: int) -> Path | None:
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.params import Params
  if os.environ.get('HOST_WARP_PREPARING'):
    return None
  device_type = HARDWARE.get_device_type()
  if device_type not in ('tici', 'tizi') or bundle is None:
    return None
  try:
    request = request_for_bundle(Params(), bundle, device_type)
    if request is None or request['source']['camera_size'] != [cam_w, cam_h]:
      return None
    return ready_artifact(request)
  except (OSError, ValueError, KeyError) as error:
    cloudlog.exception(f"Automatic host-warp selection rejected; retaining original model: {error}")
    return None


def status_for_bundle(bundle) -> str:
  try:
    state = read_json(cache_root() / 'status.json')
    if state.get('bundle') != bundle_identity(bundle):
      return 'Host warp: waiting for offroad preparation'
    return f"Host warp: {state.get('message', 'waiting')}"
  except (OSError, ValueError) as error:
    cloudlog.error(f"Cannot read host-warp preparation status: {error}")
    return 'Host warp: status unavailable; original model retained'


def reject_artifact(path: Path, error: Exception) -> None:
  (path.parent / 'ready.json').unlink(missing_ok=True)
  state = read_json(cache_root() / 'status.json')
  atomic_json(cache_root() / 'status.json', {**state, 'state': 'failed', 'message': f"validation failed; original model retained: {error}",
                                          **attempt_time()})


def attempt_time() -> dict:
  return {'attempted_at': time.monotonic(), 'boot_id': boot_id()}


@lru_cache(maxsize=1)
def boot_id() -> str:
  return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


class HostWarpPreparation:
  def __init__(self, params, device_type: str | None):
    self.params, self.device_type = params, device_type
    self.child: subprocess.Popen | None = None
    self.request: dict | None = None
    self.started_at = 0.
    self.log = None
    self.work: Path | None = None
    self.last_status: dict | None = None
    self.verified_key: str | None = None

  def status(self, bundle, state: str, message: str, **extra) -> None:
    value = {'bundle': bundle_identity(bundle), 'state': state, 'message': message, **extra}
    if value != self.last_status:
      atomic_json(cache_root() / 'status.json', value)
      cloudlog.info(f"Host warp: {message}")
      self.last_status = value

  def stop(self) -> None:
    if self.child is not None:
      try:
        os.killpg(self.child.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
      try:
        self.child.wait(timeout=1)
      except subprocess.TimeoutExpired:
        try:
          os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
        self.child.wait()
      self.child = None
    if self.log is not None:
      self.log.close()
      self.log = None
    # Remove only this attempt's files; never touch completed caches or model bundles.
    if self.work is not None:
      for file in self.work.iterdir():
        if file.is_file():
          file.unlink()
      self.work.rmdir()
      self.work = None

  def tick(self, bundle) -> None:
    if not self.params.get_bool('IsOffroad') or self.params.get_bool('IsDriverViewEnabled') or self.params.get_bool('IsLiveStreaming'):
      self.stop()
      self.status(bundle, 'paused', 'paused while cameras or driving are active; original model retained')
      return
    if self.device_type not in ('tici', 'tizi'):
      self.status(bundle, 'unsupported', 'device uses the original AMD path')
      return
    if bundle is None:
      self.stop()
      self.status(bundle, 'waiting', 'waiting for a selected Chestnut model')
      return
    request = request_for_bundle(self.params, bundle, self.device_type)
    if request is None:
      self.stop()
      self.status(bundle, 'waiting_source', 'waiting for verified ONNX source; original model retained')
      return
    key = request_key(request)
    if self.child is not None:
      if request != self.request or time.monotonic() - self.started_at > 3600:
        self.stop()
        self.status(bundle, 'paused', 'preparation cancelled; original model retained')
        return
      result = self.child.poll()
      if result is None:
        return
      if result != 0:
        self.stop()
        self.status(bundle, 'failed', 'preparation failed; original model retained (see prepare.log)',
                    key=key, **attempt_time())
        return
      assert self.work is not None
      receipt = read_json(self.work / 'ready.json')
      if receipt.get('request') != request:
        raise ValueError("Worker did not validate the requested model")
      if file_records(self.work / 'model.pkl', hashes=True) != receipt.get('files'):
        raise ValueError("Worker output changed after validation")
      destination = cache_root() / key
      if destination.exists():
        os.replace(destination, cache_root() / f'{key}-rejected-{time.monotonic_ns()}')
      os.replace(self.work, destination)
      self.work = None
      self.stop()
      self.status(bundle, 'ready', 'ready; automatically selected next model start', key=key)
      return
    try:
      cached = ready_artifact(request, verify_hashes=self.verified_key != key)
    except (OSError, ValueError, KeyError) as error:
      reject_artifact(cache_root() / key / 'model.pkl', error)
      self.status(bundle, 'failed', f"cached artifact rejected; original model retained: {error}", key=key, **attempt_time())
      return
    if cached is not None:
      self.verified_key = key
      self.status(bundle, 'ready', 'ready; automatically selected next model start', key=key)
      return
    previous = read_json(cache_root() / 'status.json')
    if (previous.get('key') == key and previous.get('boot_id') == boot_id()
        and 0 <= time.monotonic() - previous.get('attempted_at', 0) < RETRY_SECONDS):
      return
    from openpilot.selfdrive.modeld.helpers import chestnut_present
    if not chestnut_present():
      self.status(bundle, 'waiting_hardware', 'waiting for Chestnut power; original model retained')
      return
    cache_root().mkdir(parents=True, exist_ok=True)
    self.work = Path(tempfile.mkdtemp(prefix='prepare-', dir=cache_root()))
    atomic_json(self.work / 'request.json', request)
    self.log = (cache_root() / 'prepare.log').open('w')
    self.request = request
    self.started_at = time.monotonic()
    self.child = subprocess.Popen([sys.executable, '-m', 'openpilot.sunnypilot.models.host_warp_worker', str(self.work)],
                                  cwd=BASEDIR, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
    self.status(bundle, 'preparing', 'preparing and validating while offroad; original model retained', key=key)


def main() -> None:
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.params import Params
  from openpilot.sunnypilot.models.helpers import get_selected_bundle
  params = Params()
  preparation = HostWarpPreparation(params, HARDWARE.get_device_type())
  try:
    while True:
      bundle = get_selected_bundle(params, 'chestnut')
      try:
        preparation.tick(bundle)
      except (OSError, ValueError, KeyError) as error:
        preparation.stop()
        cloudlog.exception(f"Host-warp preparation failed: {error}")
        preparation.status(bundle, 'failed', f"preparation failed; original model retained: {error}",
                           **attempt_time())
        time.sleep(30)
      time.sleep(1)
  finally:
    preparation.stop()
