"""
dp - eGPU (chestnut / USB AMD GPU) big-model support, ported from sunnypilot, models compiled on the device.

Layout:
  constants.py    URLs, USB ids, cache keys, tinygrad flags, param keys, paths (no heavy imports)
  usb.py          eGPU presence detection (sysfs)
  bundles.py      catalog merge, local state (ONNX / compiled / active)
  manager.py      daemon: presence param, catalog fetch, ONNX download (jetlink registry), compile, delete, clear
  tg_env.py       puts the vendored sunnypilot tinygrad (tinygrad_spm/) first on sys.path, in a fresh interpreter
  compile.py      on-device compile: QCOM warp JIT per camera + eGPU model JIT per ONNX (stateful or queued)
  runner.py       EgpuModelState: QCOM warp -> crop over USB -> eGPU model
  action.py       shared output-to-action conversion, without importing a GPU runtime
  joining.py      small-model-first routing, background retry, control-gated switching
  ipc.py          private memfd frame buffers and bounded per-frame IPC waits
  worker.py       fresh eGPU interpreter owned by modeld, killed on parent death
  modeld_egpu.py  legacy standalone diagnostic entry point (not launched by manager)

Production modeld always loads the local small model. On eGPU installations it keeps
publishing small-model output while the worker loads/warms up, then reports
dp_egpu_ready until control is released and at least three small frames have run.
Only then may big-model output replace small-model output. The worker uses normal
scheduling outside core 7. Its load timeout is 150 seconds; the frame thread waits
at most 80 ms for an inference reply (not including copying frames into shared memory).
An error, unplug, disable, watchdog failure or dropped-frame ratio above 0.0075
returns to the retained small model after resetting its history. After a five-second
cooldown the next worker may load in the background, even during control; switching
back to big still requires no control. A new worker cannot overlap an unreaped one.

This replaces production's built-in USBGPU / standalone modeld_egpu selection with
the selected, locally compiled eGPU bundle. Existing artifacts are reused; it does
not download or compile onroad. No bundle means the local small model continues.
Jetlink's existing adapter/library version and the cluster/UI layout are retained.

The router and IPC design follow herizon1054/openpilot dpegpu00 at a45a1959e2.
Hardware-free regression tests cover routing, faults, timing, transport and UI
state. They do not certify real inference latency, memory pressure, output
continuity, driver behavior or vehicle safety; validate those separately before
using the changed runtime on a vehicle. New dp_egpu_ready registration requires
rebuilding libparams; do not deploy Python files alone over a stale prebuilt.
"""
