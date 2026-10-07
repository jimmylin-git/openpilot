## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

SCons builds stock models and warps with [compile_native.py](../compile_native.py).
Native driving artifacts contain input/output specs and recurrent state outputs.
Existing stock ONNX files with image/history inputs retain the unified `run_model` format.

Driver monitoring builds `dmonitoring_model_native.pkl` and `dm_warp_*_native.pkl`
as a matched model/warp pair. If the native model is absent, the loader logs its
use of the existing tracked legacy artifacts and their metadata/device sidecars.
Invalid native artifacts are reported rather than replaced with legacy results.
Rebuild on the target device; host CPU tests do not certify QCOM or Chestnut execution.
