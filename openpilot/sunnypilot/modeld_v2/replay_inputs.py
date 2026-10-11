"""Recorded camera inputs for the isolated CTMV2 experiment, not driving replay."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EncodedFrame:
  frame_id: int
  ordinal: int
  sof: int
  eof: int


def pair_frames(road: list[EncodedFrame], wide: list[EncodedFrame], start: int, count: int):
  if count < 1:
    raise ValueError("Replay count must be positive")
  selected = sorted((frame for frame in road if start <= frame.frame_id < start + count), key=lambda frame: frame.frame_id)
  if [frame.frame_id for frame in selected] != list(range(start, start + count)):
    raise ValueError("Requested narrow-camera frame IDs are missing or duplicated")
  pairs = []
  for frame in selected:
    if not wide:
      raise ValueError("Missing wide-camera encoder metadata")
    extra = min(wide, key=lambda item: abs(item.sof - frame.sof))
    if abs(extra.sof - frame.sof) > 20_000_000:
      raise ValueError(f"Camera skew exceeds 20 ms at frame {frame.frame_id}")
    pairs.append((frame, extra))
  if any(b[0].ordinal <= a[0].ordinal or b[1].ordinal <= a[1].ordinal for a, b in zip(pairs, pairs[1:], strict=False)):
    raise ValueError("Replay would reuse or reverse a camera frame")
  return pairs


def pad_nv12(decoded: np.ndarray, camera_size: tuple[int, int], frame_info: tuple) -> np.ndarray:
  width, height = camera_size
  stride, y_height, uv_height = frame_info[:3]
  if decoded.dtype != np.uint8 or decoded.shape != (width * height * 3 // 2,):
    raise ValueError("Decoded NV12 shape/dtype differs from camera layout")
  if stride < width or y_height < height or uv_height < height // 2:
    raise ValueError("Invalid padded NV12 layout")
  padded = np.zeros(stride * (y_height + uv_height), dtype=np.uint8)
  padded[:stride * y_height].reshape(y_height, stride)[:height, :width] = decoded[:width * height].reshape(height, width)
  padded[stride * y_height:].reshape(uv_height, stride)[:height // 2, :width] = decoded[width * height:].reshape(height // 2, width)
  return padded


class RecordedInputs:
  def __init__(self, segment: Path, start: int, count: int, camera_size: tuple[int, int], frame_info: tuple):
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.common.transformations.model import get_warp_matrix
    from openpilot.tools.lib.logreader import LogReader

    self.segment, self.camera_size, self.frame_info = segment, camera_size, frame_info
    encoders: dict[str, list[EncodedFrame]] = {'narrowRoadEncodeIdx': [], 'wideRoadEncodeIdx': []}
    calibrations = []
    sensors = set()
    device_types = set()
    offsets = []
    for message in LogReader(str(segment / 'rlog.zst')):
      kind = message.which()
      if kind in encoders:
        event = getattr(message, kind)
        if str(event.type) != 'fullHEVC':
          raise ValueError("Replay requires full HEVC encoder metadata")
        encoders[kind].append(EncodedFrame(event.frameId, event.segmentId, event.timestampSof, event.timestampEof))
      elif kind == 'extrinsicsCalibration':
        event = message.extrinsicsCalibration
        if str(event.calStatus) == 'calibrated':
          calibrations.append((message.logMonoTime, np.asarray(event.rpyCalib, dtype=np.float32)))
      elif kind == 'narrowRoadCameraState':
        sensors.add(str(message.narrowRoadCameraState.sensor))
      elif kind == 'deviceState':
        device_types.add(str(message.deviceState.deviceType))
      elif kind == 'initData':
        offsets.extend(float(entry.value) for entry in message.initData.params.entries if entry.key == 'CameraOffset')
    if offsets != [0.0]:
      raise ValueError("Replay requires a recorded zero CameraOffset; nonzero offset smoothing cannot be reconstructed")
    if len(sensors) != 1 or len(device_types) != 1:
      raise ValueError("Replay requires unambiguous camera sensor/device metadata")
    camera = DEVICE_CAMERAS[(next(iter(device_types)), next(iter(sensors)))]
    self.pairs = pair_frames(encoders['narrowRoadEncodeIdx'], encoders['wideRoadEncodeIdx'], start, count)
    self.encoder_counts = []
    for records in encoders.values():
      if sorted(frame.ordinal for frame in records) != list(range(len(records))):
        raise ValueError("HEVC ordinals are missing or duplicated")
      self.encoder_counts.append(len(records))
    self.transforms = []
    for road, _ in self.pairs:
      prior = [item for item in calibrations if item[0] <= road.sof]
      if not prior:
        raise ValueError(f"No preceding calibrated transform for frame {road.frame_id}")
      rpy = max(prior, key=lambda item: item[0])[1]
      if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("Invalid recorded calibration")
      self.transforms.append(np.stack((get_warp_matrix(rpy, camera.narrow_road.intrinsics, False),
                                       get_warp_matrix(rpy, camera.wide_road.intrinsics, True))).astype(np.float32))

  def _decode(self, name: str, wanted: list[int], expected_count: int):
    from openpilot.tools.camerastream.ffmpeg_decoder import Decoder
    from openpilot.tools.lib.vidindex import hevc_index

    path = self.segment / name
    entries, size, prefix = hevc_index(str(path))
    if len(entries) != expected_count or any(kind == 0 for kind, _ in entries):
      raise ValueError(f"{name}: encoder count differs or unsupported B-frame reordering")
    starts = [i for i, (kind, _) in enumerate(entries[:wanted[0] + 1]) if kind == 2]
    if not starts:
      raise ValueError(f"{name}: no preceding intra frame")
    start = starts[-1]
    targets = set(wanted)
    with path.open('rb') as stream, Decoder('hevc') as decoder:
      for i in range(start, wanted[-1] + 1):
        offset = entries[i][1]
        end = entries[i + 1][1] if i + 1 < len(entries) else size
        stream.seek(offset)
        packet = stream.read(end - offset)
        decoded = decoder.decode(prefix + packet if i == start else packet)
        if decoded is None:
          raise RuntimeError(f"{name}: decoder delayed frame {i}; cannot prove ordinal alignment")
        if (decoder.width, decoder.height) != self.camera_size:
          raise ValueError(f"{name}: dimensions differ from --camera")
        if i in targets:
          yield pad_nv12(decoded, self.camera_size, self.frame_info)

  def __iter__(self):
    road = self._decode('fcamera.hevc', [pair[0].ordinal for pair in self.pairs], self.encoder_counts[0])
    wide = self._decode('ecamera.hevc', [pair[1].ordinal for pair in self.pairs], self.encoder_counts[1])
    try:
      yield from zip(self.transforms, zip(road, wide, strict=True), strict=True)
    finally:
      road.close()
      wide.close()
