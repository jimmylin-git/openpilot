"""Explicit delivery of carrot's generic model; independent of sunnypilot selection."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

SOURCE_COMMIT = 'c4702e4238f5e6aeb2e14911a5b635234f821efd'
DEFAULT_MANIFEST_URL = 'https://upload.shind0.synology.me/models/comma4-big-cinque-v3/manifest.json'


@dataclass(frozen=True)
class BigModelManifest:
  model_id: str
  filename: str
  size: int
  sha256: str
  url: str

  @classmethod
  def from_dict(cls, value, manifest_url=DEFAULT_MANIFEST_URL):
    if not isinstance(value, dict):
      raise ValueError('Model manifest must be an object')
    for name in ('model_id', 'filename'):
      if not isinstance(value.get(name), str) or not re.fullmatch(r'[A-Za-z0-9._-]+', value[name]):
        raise ValueError(f'Invalid {name}')
    if not value['filename'].endswith('.pkl'):
      raise ValueError('This port accepts precompiled generic PKLs only')
    if type(value.get('size')) is not int or not 0 < value['size'] <= 4 * 1024**3:
      raise ValueError('Invalid model size')
    if not isinstance(value.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', value['sha256']):
      raise ValueError('Invalid model SHA256')
    if not isinstance(value.get('url'), str) or not value['url']:
      raise ValueError('Invalid model URL')
    url = urljoin(manifest_url, value['url'])
    if urlparse(url).scheme != 'https' or urlparse(url).netloc != urlparse(manifest_url).netloc:
      raise ValueError('Model must use the manifest HTTPS origin')
    return cls(value['model_id'], value['filename'], value['size'], value['sha256'], url)

  @property
  def cache_filename(self):
    return f'{Path(self.filename).stem}-{self.sha256[:16]}.pkl'


def model_cache_dir() -> Path:
  return Path(os.environ.get('CARROT_BIG_MODEL_DIR', '/data/media/0/carrot/models'))


def model_path(model, cache_dir=None) -> Path:
  return (cache_dir or model_cache_dir()) / model.cache_filename


def active_manifest():
  # This experiment never changes the existing selected model or startup defaults.
  return None


def fetch_manifest(url=DEFAULT_MANIFEST_URL):
  if urlparse(url).scheme != 'https':
    raise ValueError('Manifest must use HTTPS')
  with urlopen(Request(url, headers={'User-Agent': 'carrot-port/1'}), timeout=15) as response:
    data = response.read(65537)
  if len(data) > 65536:
    raise ValueError('Manifest too large')
  return BigModelManifest.from_dict(json.loads(data), url)


def require_parked():
  if os.name != 'posix' or not Path('/TICI').is_file():
    raise RuntimeError('Carrot runtime validation requires a C3 vehicle')
  from openpilot.common.params import Params
  if not Params().get_bool('IsOffroad'):
    raise RuntimeError('Carrot port is unqualified: parked testing only')


def main():
  import argparse
  from openpilot.selfdrive.modeld.precompiled_model import ensure_precompiled

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', default=DEFAULT_MANIFEST_URL)
  parser.add_argument('--cache', type=Path, default=model_cache_dir())
  args = parser.parse_args()
  require_parked()
  model = fetch_manifest(args.manifest)
  args.cache.mkdir(parents=True, exist_ok=True)
  path = ensure_precompiled(model, args.cache)
  if path is None:
    raise RuntimeError('Artifact is rejected; inspect last_failure.json before retrying')
  if json.loads((path.parent / 'installed.json').read_text())['format'] != 'comma-generic-onnx':
    raise ValueError('This port requires a generic model artifact')
  print(json.dumps({'model': str(path), 'sha256': model.sha256, 'source_commit': SOURCE_COMMIT, 'qualified': False}, indent=2))


if __name__ == '__main__':
  main()
