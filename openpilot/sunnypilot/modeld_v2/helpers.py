"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import struct
import pickle
import tempfile
import shutil
import enum
import inspect
import logging
import os
import threading


logger = logging.getLogger(__name__)


def _patch_system_flock_acquire():
  from tinygrad.runtime.support.system import System
  original = System.flock_acquire
  if getattr(original, '_model_pickle_compat', False):
    return
  locks: dict[str, int] = {}
  mutex = threading.Lock()
  pid = os.getpid()

  def flock_acquire(name: str) -> int:
    nonlocal pid
    with mutex:
      if pid != os.getpid():
        for fd in locks.values():
          os.close(fd)
        locks.clear()
        pid = os.getpid()
      if name not in locks:
        locks[name] = original(name)
      # Each caller owns its descriptor; closing one must not invalidate another.
      return os.dup(locks[name])

  def after_fork():
    nonlocal mutex
    mutex = threading.Lock()

  if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=after_fork)
  flock_acquire._model_pickle_compat = True
  System.flock_acquire = flock_acquire


def _pad_args(func, args, kwargs):
  signature = inspect.signature(func)
  parameters = list(signature.parameters.values())
  positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
  new_args = list(args)
  if not any(p.kind == p.VAR_POSITIONAL for p in parameters):
    new_args = new_args[:len(positional)]
  for param in positional[len(new_args):]:
    if param.name in kwargs:
      break
    new_args.append(param.default if param.default is not param.empty else None)
  new_kwargs = dict(kwargs)
  for param in parameters:
    if param.kind == param.KEYWORD_ONLY and param.name not in new_kwargs and param.default is param.empty:
      new_kwargs[param.name] = None
  signature.bind(*new_args, **new_kwargs)
  return new_args, new_kwargs


def _dynamic_factory(real_class):
  if not isinstance(real_class, type) or issubclass(real_class, enum.Enum):
    return real_class

  def factory(*args, **kwargs):
    try:
      return real_class(*args, **kwargs)
    except TypeError:
      signature = inspect.signature(real_class)
      try:
        signature.bind(*args, **kwargs)
      except TypeError:
        new_args, new_kwargs = _pad_args(real_class, args, kwargs)
        logger.warning("Model pickle compatibility: adapting %s.%s constructor (%d -> %d positional arguments)",
                       real_class.__module__, real_class.__name__, len(args), len(new_args))
        return real_class(*new_args, **new_kwargs)
      raise

  class DynamicMeta(type(real_class)):
    def __call__(cls, *args, **kwargs):
      return factory(*args, **kwargs)

  class DynamicProxy(real_class, metaclass=DynamicMeta):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
      return factory(*args, **kwargs)

  DynamicProxy.__name__ = real_class.__name__
  DynamicProxy.__module__ = real_class.__module__
  return DynamicProxy


class DynamicTinygradUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    real_class = super().find_class(module, name)
    if module.startswith('tinygrad.'):
      _patch_system_flock_acquire()
      return _dynamic_factory(real_class)
    return real_class


def load_pickle(f):
  return DynamicTinygradUnpickler(f).load()


def load_oob(f):
  def read_size():
    header = f.read(8)
    if len(header) != 8:
      raise EOFError("Truncated model pickle size header")
    size = struct.unpack('<q', header)[0]
    if size < 0:
      raise ValueError(f"Invalid model pickle size: {size}")
    return size

  opcode_size = read_size()
  opcodes = f.read(opcode_size)
  if len(opcodes) != opcode_size:
    raise EOFError("Truncated model pickle opcodes")

  def buffers():
    while True:
      size = read_size()
      pb = pickle.PickleBuffer(bytearray(size))
      view = pb.raw()
      offset = 0
      while offset < size:
        count = f.readinto(view[offset:])
        if not count:
          raise EOFError("Truncated model pickle buffer")
        offset += count
      yield pb
  return DynamicTinygradUnpickler(io.BytesIO(opcodes), buffers=buffers()).load()


def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_cb(buffer):
      data = buffer.raw()
      tmp.write(struct.pack('<q', len(data)))
      tmp.write(data)
      buffer.release()
      return False

    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_cb).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)
