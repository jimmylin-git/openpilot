"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import unittest

from openpilot.sunnypilot.modeld_v2.compile_modeld import validate_chestnut_host_warp
from openpilot.sunnypilot.modeld_v2.modeld import _validate_camera_input_abi


CAM_W, CAM_H = 1928, 1208


def host_warp_artifact():
  return {
    'metadata': {
      'camera_input_abi': 'warped_yuv_v1',
      'frame_skip': 4,
      'warp_dev': 'QCOM',
      'model': {'input_shapes': {'features_buffer': (1, 24, 512)}},
    },
    'run_policy': object(),
    (CAM_W, CAM_H): object(),
  }


class TestChestnutHostWarp(unittest.TestCase):
  def test_compiler_requires_chestnut_supercombo_on_amd(self):
    validate_chestnut_host_warp('supercombo', True, True, 'AMD:LLVM')

    with self.assertRaisesRegex(ValueError, 'supercombo and CHESTNUT'):
      validate_chestnut_host_warp('vision_policy', True, True, 'AMD:LLVM')
    with self.assertRaisesRegex(ValueError, 'supercombo and CHESTNUT'):
      validate_chestnut_host_warp('supercombo', True, False, 'AMD:LLVM')
    with self.assertRaisesRegex(ValueError, 'AMD inference'):
      validate_chestnut_host_warp('supercombo', True, True, 'QCOM')
    validate_chestnut_host_warp('supercombo', False, False, None)
    with self.assertRaisesRegex(ValueError, 'AMD inference'):
      validate_chestnut_host_warp('supercombo', True, True, None)

  def test_runtime_accepts_only_matching_chestnut_host_warp_abi(self):
    artifact = host_warp_artifact()
    self.assertTrue(_validate_camera_input_abi(artifact['metadata'], artifact, True, CAM_W, CAM_H))
    self.assertFalse(_validate_camera_input_abi({}, {}, False, CAM_W, CAM_H))

  def test_runtime_rejects_incompatible_host_warp_artifacts(self):
    cases = [
      (lambda artifact: artifact['metadata'].update(camera_input_abi='future_abi'), True, 'Unsupported camera input ABI'),
      (lambda artifact: artifact['metadata'].pop('model'), True, 'Chestnut supercombo'),
      (lambda artifact: artifact['metadata'].update(warp_dev='AMD'), True, 'warp_dev=QCOM'),
      (lambda artifact: artifact.pop((CAM_W, CAM_H)), True, 'no warp for camera size'),
      (lambda artifact: artifact['metadata'].update(frame_skip=1), True, 'invalid frame_skip'),
      (lambda artifact: artifact.update(run_model=object()), True, 'Chestnut supercombo'),
      (lambda artifact: artifact.update(), False, 'Chestnut supercombo'),
    ]
    for change, chestnut, message in cases:
      with self.subTest(message=message):
        artifact = host_warp_artifact()
        change(artifact)
        with self.assertRaisesRegex(RuntimeError, message):
          _validate_camera_input_abi(artifact['metadata'], artifact, chestnut, CAM_W, CAM_H)
