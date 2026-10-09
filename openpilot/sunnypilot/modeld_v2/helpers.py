"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import struct
import pickle
import inspect
import importlib
import tempfile
import shutil
import enum

from openpilot.common.model_pickle import load_oob as _load_oob


def _pad_args(func, args, kwargs):
  sig = inspect.signature(func)
  params = list(sig.parameters.values())
  positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
  new_args = list(args)
  new_kwargs = dict(kwargs)
  has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
  if not has_varargs:
    new_args = new_args[:len(positional)]

  for param in positional[len(new_args):]:
    if param.name in new_kwargs or param.default is inspect.Parameter.empty:
      break
    new_args.append(param.default)
  for param in params:
    if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
      if param.name not in new_kwargs and param not in positional[:len(new_args)]:
        if param.default is not inspect.Parameter.empty:
          new_kwargs[param.name] = param.default
  return new_args, new_kwargs


def _dynamic_factory(real_class):
  if isinstance(real_class, type) and issubclass(real_class, enum.Enum):
    return real_class

  def factory(*args, **kwargs):
    try:
      sig = inspect.signature(real_class)
    except (TypeError, ValueError):
      return real_class(*args, **kwargs)
    try:
      sig.bind(*args, **kwargs)
    except TypeError:
      args, kwargs = _pad_args(real_class, args, kwargs)
      sig.bind(*args, **kwargs)
    return real_class(*args, **kwargs)

  class DynamicMeta(type(real_class)):
    def __call__(cls, *args, **kwargs):
      return factory(*args, **kwargs)

  class DynamicProxy(real_class, metaclass=DynamicMeta):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
      # Pickle NEWOBJ restores attributes separately and must not invoke __init__.
      if real_class.__new__ is object.__new__:
        return object.__new__(real_class)
      return real_class.__new__(real_class, *args, **kwargs)

  DynamicProxy.__name__ = real_class.__name__
  DynamicProxy.__module__ = real_class.__module__
  return DynamicProxy


class DynamicTinygradUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    real_class = getattr(importlib.import_module(module), name)
    if module.startswith("tinygrad"):
      return _dynamic_factory(real_class)
    return real_class


def load_oob(f):
  return _load_oob(f, DynamicTinygradUnpickler)


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
