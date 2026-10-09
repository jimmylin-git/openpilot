import ctypes
import importlib.util
import unittest
from unittest.mock import patch

from tinygrad.runtime.autogen import comgr, comgr_3


class TestComgrCompatibility(unittest.TestCase):
  def test_version_detection_selects_matching_binding(self):
    spec = importlib.util.find_spec("tinygrad.runtime.support.compiler_amd")
    for major, binding, hip in ((2, comgr, 4), (3, comgr_3, 3), (4, comgr_3, 3)):
      with self.subTest(major=major):
        def get_version(major_ptr, minor_ptr, version=major):
          ctypes.cast(major_ptr, ctypes.POINTER(ctypes.c_uint64))[0] = version
          ctypes.cast(minor_ptr, ctypes.POINTER(ctypes.c_uint64))[0] = 0

        compiler = importlib.util.module_from_spec(spec)
        with patch.object(comgr, "amd_comgr_get_version", side_effect=get_version):
          spec.loader.exec_module(compiler)
        self.assertIs(compiler.comgr, binding)
        self.assertEqual(compiler.comgr.AMD_COMGR_LANGUAGE_HIP, hip)

  def test_missing_rocm_preserves_legacy_import_behavior(self):
    spec = importlib.util.find_spec("tinygrad.runtime.support.compiler_amd")
    compiler = importlib.util.module_from_spec(spec)
    with patch.object(comgr, "amd_comgr_get_version", side_effect=AttributeError("ROCm not installed")):
      spec.loader.exec_module(compiler)
    self.assertIs(compiler.comgr, comgr)

  def test_comgr3_types_match_legacy_handle_layouts(self):
    for name in ("amd_comgr_data_t", "amd_comgr_data_set_t", "amd_comgr_action_info_t",
                 "amd_comgr_metadata_node_t", "amd_comgr_symbol_t", "amd_comgr_disassembly_info_t",
                 "amd_comgr_symbolizer_info_t", "amd_comgr_code_object_info_t"):
      with self.subTest(name=name):
        old, new = getattr(comgr, name), getattr(comgr_3, name)
        self.assertEqual(ctypes.sizeof(old), ctypes.sizeof(new))
        self.assertEqual(old._real_fields_, new._real_fields_)
    handle = comgr_3.amd_comgr_data_t(handle=42)
    self.assertEqual(handle.handle, 42)
    self.assertEqual(comgr_3.AMD_COMGR_INTERFACE_VERSION_MAJOR, 3)

  def test_comgr3_language_binding_passes_correct_enum_and_handle(self):
    observed = []
    signature = ctypes.CFUNCTYPE(comgr_3.amd_comgr_status_t, comgr_3.amd_comgr_action_info_t, comgr_3.amd_comgr_language_t)

    @signature
    def set_language(action, language):
      observed.append((action.handle, language))
      return 0

    # Use a fresh module so the lazy binding cannot retain the test callback.
    spec = importlib.util.find_spec("tinygrad.runtime.autogen.comgr_3")
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    with patch.object(binding.dll, "amd_comgr_action_info_set_language", set_language, create=True):
      self.assertEqual(binding.amd_comgr_action_info_set_language(
        binding.amd_comgr_action_info_t(handle=42), binding.AMD_COMGR_LANGUAGE_HIP), 0)
    self.assertEqual(observed, [(42, 3)])


if __name__ == "__main__":
  unittest.main()
