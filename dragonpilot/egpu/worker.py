#!/usr/bin/env python3
"""Fresh eGPU interpreter, private to and terminated with its owning modeld."""
import ctypes
import os
import sys


def main():
  parent_pid = int(sys.argv[3])
  if ctypes.CDLL(None).prctl(1, 9, 0, 0, 0) != 0:
    raise RuntimeError('cannot install eGPU parent-death signal')
  if os.getppid() != parent_pid:
    sys.exit(1)
  os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))

  import tg_env  # Must precede every tinygrad import.
  os.environ['DEV'] = 'QCOM'
  os.environ.setdefault('AM_POWER_LIMIT', '100')
  os.environ.setdefault('HCQDEV_WAIT_TIMEOUT_MS', '3000')
  tg_env.check()

  import mmap
  from multiprocessing.connection import Connection
  import numpy as np
  from tinygrad.runtime.ops_qcom import dcache_flush
  from dragonpilot.egpu.constants import PCIE_WAIT, model_root, model_pkl_name, warp_pkl_name
  from dragonpilot.egpu.usb import wait_pcie_link
  from dragonpilot.egpu.runner import EgpuModelState
  from openpilot.common.swaglog import cloudlog

  conn = Connection(int(sys.argv[1]))
  try:
    width, height, frame_size, bundle = conn.recv()
    memory = mmap.mmap(int(sys.argv[2]), 2 * frame_size)
    frames = {key: np.ndarray((frame_size,), dtype=np.uint8, buffer=memory, offset=i * frame_size)
              for i, key in enumerate(('img', 'big_img'))}
    tg_env.patch_fetch_fw()
    up, status = wait_pcie_link(PCIE_WAIT)
    if not up:
      raise RuntimeError(f'eGPU PCIe not ready: {status}')
    model = EgpuModelState(width, height, bundle,
                          os.path.join(model_root(), model_pkl_name(bundle['sha256'])),
                          os.path.join(model_root(), warp_pkl_name(width, height)))
    model.warmup()
    flush = dcache_flush().fxn
    frame_start = frames['img'].ctypes.data
    cache_start = frame_start & ~63
    cache_lines = (frame_start + 2 * frame_size - cache_start + 63) // 64
    conn.send('ready')
    while True:
      transforms, inputs = conn.recv()
      # memfd is CPU write-back memory, unlike camerad's GPU buffers. Clean both
      # shared frames on every request, even when from_blob reuses its mapping.
      flush(ctypes.c_uint64(cache_start), cache_lines)
      conn.send(model.run(frames, transforms, inputs))
  except EOFError:
    pass  # The owning modeld closed its private connection.
  except Exception as exc:
    cloudlog.exception('eGPU worker failed')
    try:
      conn.send({'error': repr(exc)})
    except OSError:
      cloudlog.exception('eGPU worker could not report failure to modeld')
  finally:
    conn.close()


if __name__ == '__main__':
  main()
