import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openpilot.sunnypilot.modeld_v2.replay_inputs import EncodedFrame, RecordedInputs, pad_nv12, pair_frames


class TestReplayInputs(unittest.TestCase):
  def test_pairs_by_sof_not_same_frame_id_or_camera_state_offset(self):
    road = [EncodedFrame(1607, 405, 100_000_000, 115_000_000), EncodedFrame(1608, 406, 150_000_000, 165_000_000)]
    wide = [EncodedFrame(1606, 404, 99_800_000, 114_800_000), EncodedFrame(1607, 405, 149_800_000, 164_800_000)]
    pairs = pair_frames(road, wide, 1607, 2)
    self.assertEqual([(r.ordinal, w.ordinal) for r, w in pairs], [(405, 404), (406, 405)])

  def test_skew_boundary_and_invalid_pairing(self):
    road = [EncodedFrame(1, 0, 100_000_000, 115_000_000)]
    pair_frames(road, [EncodedFrame(2, 0, 120_000_000, 135_000_000)], 1, 1)
    for wide in ([], [EncodedFrame(2, 0, 120_000_001, 135_000_001)]):
      with self.subTest(wide=wide), self.assertRaises(ValueError):
        pair_frames(road, wide, 1, 1)
    for count, frames in ((0, road), (2, road), (1, road + road)):
      with self.subTest(count=count), self.assertRaises(ValueError):
        pair_frames(frames, road, 1, count)
    with self.assertRaisesRegex(ValueError, 'reuse'):
      pair_frames(road + [EncodedFrame(2, 1, 101_000_000, 116_000_000)], road, 1, 2)

  def test_nv12_planes_use_padded_uv_offset_and_zero_padding(self):
    decoded = np.arange(12, dtype=np.uint8)
    actual = pad_nv12(decoded, (4, 2), (8, 4, 2, 48))
    self.assertEqual(actual.shape, (48,))
    np.testing.assert_array_equal(actual[:32].reshape(4, 8)[:2, :4], decoded[:8].reshape(2, 4))
    np.testing.assert_array_equal(actual[32:36], decoded[8:])
    self.assertTrue((actual[16:32] == 0).all())
    self.assertTrue((actual[36:] == 0).all())
    for invalid in (decoded.astype(np.float32), decoded[:-1], decoded.reshape(3, 4)):
      with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
        pad_nv12(invalid, (4, 2), (8, 4, 2, 48))
    with self.assertRaises(ValueError):
      pad_nv12(decoded, (4, 2), (2, 4, 2, 16))

  def test_recording_metadata_uses_encoder_ordinals_and_rejects_unknown_transforms(self):
    def event(kind, **fields):
      return types.SimpleNamespace(which=lambda: kind, logMonoTime=0, **{kind: types.SimpleNamespace(**fields)})

    messages = [
      event('initData', params=types.SimpleNamespace(entries=[types.SimpleNamespace(key='CameraOffset', value=b'0.0')])),
      event('deviceState', deviceType='tici'),
      event('narrowRoadCameraState', sensor='test-sensor'),
      event('extrinsicsCalibration', calStatus='calibrated', rpyCalib=[0, 0, 0]),
      event('narrowRoadEncodeIdx', frameId=1607, segmentId=0, timestampSof=100, timestampEof=200, type='fullHEVC'),
      event('wideRoadEncodeIdx', frameId=1606, segmentId=0, timestampSof=99, timestampEof=199, type='fullHEVC'),
    ]
    camera = types.SimpleNamespace(narrow_road=types.SimpleNamespace(intrinsics=np.eye(3)),
                                   wide_road=types.SimpleNamespace(intrinsics=np.eye(3)))
    modules = {
      'openpilot.tools.lib.logreader': types.SimpleNamespace(LogReader=lambda _: messages),
      'openpilot.common.transformations.camera': types.SimpleNamespace(DEVICE_CAMERAS={('tici', 'test-sensor'): camera}),
      'openpilot.common.transformations.model': types.SimpleNamespace(get_warp_matrix=lambda *_: np.eye(3)),
    }
    with mock.patch.dict('sys.modules', modules):
      replay = RecordedInputs(Path('unused'), 1607, 1, (4, 2), (8, 4, 2, 48))
      self.assertEqual([(r.frame_id, w.frame_id) for r, w in replay.pairs], [(1607, 1606)])
      self.assertEqual(replay.encoder_counts, [1, 1])
      np.testing.assert_array_equal(replay.transforms[0], np.stack((np.eye(3), np.eye(3))))
      offset = messages[0].initData.params.entries[0]
      for value in (b'0.1', b'nan', b'inf'):
        offset.value = value
        with self.subTest(offset=value), self.assertRaisesRegex(ValueError, 'CameraOffset'):
          RecordedInputs(Path('unused'), 1607, 1, (4, 2), (8, 4, 2, 48))
      offset.value = b'0.0'
      messages[3].logMonoTime = 201
      with self.assertRaisesRegex(ValueError, 'preceding'):
        RecordedInputs(Path('unused'), 1607, 1, (4, 2), (8, 4, 2, 48))
      messages[3].logMonoTime = 0
      messages[4].narrowRoadEncodeIdx.segmentId = 1
      with self.assertRaisesRegex(ValueError, 'ordinals'):
        RecordedInputs(Path('unused'), 1607, 1, (4, 2), (8, 4, 2, 48))

  def test_decoder_rejects_count_reordering_delay_and_dimensions(self):
    replay = object.__new__(RecordedInputs)
    replay.camera_size, replay.frame_info = (4, 2), (8, 4, 2, 48)
    decoder = mock.MagicMock()
    decoder.__enter__.return_value = decoder
    decoder.width, decoder.height = 4, 2
    decoder.decode.return_value = np.arange(12, dtype=np.uint8)
    index = mock.Mock(return_value=([(2, 0), (1, 4)], 8, b'prefix'))
    modules = {
      'openpilot.tools.camerastream.ffmpeg_decoder': types.SimpleNamespace(Decoder=lambda _: decoder),
      'openpilot.tools.lib.vidindex': types.SimpleNamespace(hevc_index=index),
    }
    with tempfile.TemporaryDirectory() as directory, mock.patch.dict('sys.modules', modules):
      replay.segment = Path(directory)
      (replay.segment / 'fcamera.hevc').write_bytes(b'abcdefgh')
      output = list(replay._decode('fcamera.hevc', [1], 2))
      self.assertEqual(len(output), 1)
      self.assertEqual(decoder.decode.call_args_list[0].args[0], b'prefixabcd')
      for entries, count, decoded, width in (
        ([(2, 0), (1, 4)], 3, np.arange(12, dtype=np.uint8), 4),
        ([(2, 0), (0, 4)], 2, np.arange(12, dtype=np.uint8), 4),
        ([(1, 0), (1, 4)], 2, np.arange(12, dtype=np.uint8), 4),
        ([(2, 0), (1, 4)], 2, None, 4),
        ([(2, 0), (1, 4)], 2, np.arange(12, dtype=np.uint8), 6),
      ):
        index.return_value = (entries, 8, b'prefix')
        decoder.decode.return_value, decoder.width = decoded, width
        with self.subTest(entries=entries, count=count, width=width), self.assertRaises((ValueError, RuntimeError)):
          list(replay._decode('fcamera.hevc', [1], count))


if __name__ == '__main__':
  unittest.main()
