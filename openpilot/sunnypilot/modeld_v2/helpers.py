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
  return pickle.Unpickler(io.BytesIO(opcodes), buffers=buffers()).load()


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
