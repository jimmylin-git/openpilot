import io
import pickle
import struct


def _read_exact(f: io.BufferedIOBase | io.RawIOBase, size: int, description: str) -> bytearray:
  if size < 0:
    raise ValueError(f"negative {description} length: {size}")
  data = bytearray(size)
  with memoryview(data) as view:
    offset = 0
    while offset < size:
      count = f.readinto(view[offset:])
      if not count:
        raise EOFError(f"incomplete {description}: expected {size} bytes, got {offset}")
      offset += count
  return data


def load_oob(f: io.BufferedIOBase | io.RawIOBase, unpickler: type[pickle.Unpickler] = pickle.Unpickler):
  size = struct.unpack('<q', _read_exact(f, 8, "model opcode header"))[0]
  opcodes = _read_exact(f, size, "model opcodes")

  def buffers():
    while first := f.read(1):
      header = first + _read_exact(f, 7, "model buffer header")
      size = struct.unpack('<q', header)[0]
      yield pickle.PickleBuffer(_read_exact(f, size, "model buffer"))

  return unpickler(io.BytesIO(opcodes), buffers=buffers()).load()
