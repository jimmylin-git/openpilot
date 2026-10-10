# Model Selector Version Compatibility

This document explains the version compatibility mechanism used by the Model Selector system, and the rationale behind certain version constraints and JSON file management strategies.

## Overview

The Model Selector is responsible for selecting and validating model bundles based on their metadata and version constraints. Each model bundle is distributed via a JSON file and includes a `minimumSelectorVersion` field indicating the minimum selector version required to load it.

To ensure robust compatibility and prevent mismatches between model expectations and selector capabilities, the selector enforces two version boundaries:

* **`REQUIRED_MIN_SELECTOR_VERSION`**: the oldest selector version we support.
* **`CURRENT_SELECTOR_VERSION`**: the current version of the selector logic.

## Version Compatibility Check

A model bundle is considered compatible if:

```python
REQUIRED_MIN_SELECTOR_VERSION <= bundle["minimumSelectorVersion"] <= CURRENT_SELECTOR_VERSION
```

This ensures:

* **Old bundles are rejected** if they rely on deprecated selector behavior.
* **Future bundles are ignored** if they expect logic that the current selector doesn't yet implement.

## Handling Breaking Changes

When a deep change in selector behavior requires *all* models to be recompiled (e.g., due to a major architectural update), we:

1. **Create a new JSON file** (e.g., from `models_v4.json` to `models_v5.json`).
2. **Assign updated `minimumSelectorVersion` values** in the new bundles.

This allows older selector versions to continue using the previous JSON file, while newer versions point to the new one, preventing cross-contamination.

## Why `REQUIRED_MIN_SELECTOR_VERSION` Still Matters

Despite using new JSON files to isolate breaking changes, `REQUIRED_MIN_SELECTOR_VERSION` plays a critical role:

### 1. **Cached Bundle Validation**

Model bundles are cached locally (e.g., in-memory or on disk). A user might have previously loaded a now-invalid bundle from an older JSON file.

`REQUIRED_MIN_SELECTOR_VERSION` prevents the selector from reloading or trusting that stale cached bundle, even if the original JSON is gone.

### 2. **Explicit Deprecation Boundary**

By raising `REQUIRED_MIN_SELECTOR_VERSION`, we declare older bundles officially unsupported, even if they technically still exist in a legacy JSON file.

### 3. **Avoiding Race Conditions**

Some clients may have intermittent access to updated JSONs. The runtime check ensures version compatibility is enforced independently of external file state.

## Summary

| Component                       | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `minimumSelectorVersion`        | Declares the minimum selector version required to load a model bundle |
| `REQUIRED_MIN_SELECTOR_VERSION` | Prevents loading bundles that are too old (e.g., from stale cache)    |
| `CURRENT_SELECTOR_VERSION`      | Prevents loading bundles that are too new or forward-incompatible     |
| JSON file renaming              | Isolates bundles by selector generation to handle full recompiles     |

This layered strategy ensures safe evolution of the model selection system while maintaining backward compatibility and runtime protection against stale or incompatible bundles.

## Experimental Chestnut QCOM host warp

This implements the guarded pre-upload workflow from carrot-wip
[`local_gpu_warp.py`](https://github.com/ajouatom/openpilot/blob/c60cde06cb10908da5e559732a6a0b2b1ff1d0d3/openpilot/selfdrive/modeld/local_gpu_warp.py)
and its [validation notes](https://github.com/ajouatom/openpilot/blob/c60cde06cb10908da5e559732a6a0b2b1ff1d0d3/docs/c3_preupload_warp.md),
adapted to modeld_v2's supercombo format. The upstream code is MIT licensed;
see the repository's [license](../../../LICENSE).

The mechanisms match, but the artifact formats are not interchangeable.
Carrot wraps its generic precompiled runtime; modeld_v2 retains runtime support
for the earlier opt-in `warped_yuv_v2` experiment. Old `warped_yuv_v1` artifacts
are rejected. The experimental ONNX-to-host-warp compiler and automatic offroad
preparation/receipt selection have been removed: the current Chestnut catalog
provides precompiled PKLs without verified corresponding ONNX sources.

Downloaded PKL bundles remain the normal big-model path. No ONNX download or
compilation is required to use them. The existing general ONNX compiler and
small-model tooling remain available. Carrot generic artifacts are not yet
supported by this loader; renaming a generic PKL or changing its metadata does
not convert its ABI. A future adapter must use the artifact's matching runtime
and validate its inputs, state and outputs before selection.

### Runtime contract

* Only C3/C3X (`tici`/`tizi`) attempts QCOM preparation. C4 (`mici`) and other
  devices stay on the artifact's AMD path.
* Raw NV12 frames and transforms are copied into a persistent local QCOM buffer
  **on every frame**. A cached `from_blob` mapping is not used as a substitute
  for CPU/GPU cache coherence.
* The local warp uses the same implementation as the compiled AMD reference.
  Its source and the full tinygrad Python source tree are hashed at build
  time. A mismatch disables QCOM and logs a same-model AMD fallback. The local
  JIT cache is separate, source/version/layout keyed, and atomically written.
* Before accepting QCOM, initialization compares random NV12 inputs under
  identity, projective and border-clamped transforms against the actual AMD
  warp. Shape, dtype, repeat instability or unexplained pixels reject QCOM.
  Only nearest-neighbour differences within **0.00025 source pixels** of a
  half-pixel boundary are accepted, and every differing value must be one of
  the correct camera/plane's adjacent source pixels. No image-error percentage
  or arbitrary intensity tolerance is used.
* Probes run no model inference and clear their host inputs afterward.
  QCOM initialization/validation/frame-preparation failures retain the same
  model, recurrent queues and AMD warp. Policy inference errors are not retried
  after potentially advancing state; they use the existing small-model failure
  handling. Warmup clears the experiment's inputs and recurrent queues.
* QCOM results are read into a compact host buffer, then explicitly uploaded
  once to AMD before policy dispatch. At 512x256, image data is 393,216 bytes;
  the total is this plus the actual float32 controls/previous-feature payload,
  padded to 512 bytes. **393,728 is not assumed for every supercombo**: models
  with larger hidden features need larger control payloads.
* Logs include `warp_backend`, calculated `usb_input_bytes`, `local_prepare_ms`,
  `input_upload_ms`, `model_call_ms`, `output_read_ms`, total model execution
  and frame-drop percentage. These are host-call timings, not pure GPU kernel
  or USB-wire measurements. AMD fallback upload timing includes reference warp
  dispatch; its control upload remains inside policy dispatch.

### Bounded camera pairing

Stock modeld and modeld_v2 share carrot's `camera_sync.py` implementation from
commit `c60cde06cb10908da5e559732a6a0b2b1ff1d0d3`, independently of host warp.
Each call receives fresh frames, compares their current SOF timestamps, and
accepts skew up to and including 20 ms in either direction. Only the older
camera advances during resynchronization, with the original ten-iteration
bound. Timeouts or an unsuccessful resync skip model dispatch; stale buffers
are not reused. Single-camera operation uses each fresh frame for both inputs.

This replaces the previous-exposure-plus-25-ms rule, preserving valid
23/77-ms alternating exposure intervals. Unlike carrot's old strict 10 ms
limit, our previous final skew check only logged and proceeded, so the EV9
frame-drop improvement cannot be assumed here. Real frame gaps still reach the
existing dropped-frame and cameraOdometry validity checks. No camera driver,
pose-validity, downstream fault check or Cluster renderer is changed.
Both Chestnut and small models use this pairing rule without rebuilding their
artifacts. Host tests reproduce carrot's EV9 timing cases, boundary rejection,
missing frames and resync behavior; on-device timing and driving validation
remain required.

### Building and testing

This path is experimental and must not be selected for driving based only on
reduced transfer size. A prior isolated stock-model trial on the V23 branch
averaged 91.2 ms over 100 warmed runs (P95 136.0 ms; 99/100 over the 50 ms
frame budget). That was not a CTMV2 comparison or a camera-drop road test, but
it is a strong reason to keep this ABI opt-in. Host/interpreter tests cover
payload layout, state advancement/equivalence, JIT replay/pickle,
validation, cache round trips and failure paths. They do not establish QCOM/AMD
hardware equivalence, real-model output equivalence or sustained 20 Hz.

Existing experimental artifacts can still be tested in an isolated parked
process using `CHESTNUT_COMBINED_MODEL_PKL`. This override accepts chunked PKLs
and applies only to Chestnut, preserving the selected bundle's generation,
constants and overrides. It is not an automatic readiness check or support for
carrot generic artifacts. A missing/incomplete override raises an initialization
error. Do not use the global `COMBINED_MODEL_PKL` for a Chestnut-only test:
it also applies to the small model.

Previously created `.host-warp` preparation caches are no longer read or selected.
No installed model or cache is deleted by this removal. The host-warp status UI
and offroad preparation process are no longer registered.

### CTMV2 captured-policy extraction research

`openpilot.sunnypilot.modeld_v2.extract_policy.extract_ctmv2` constructs an
**unqualified, in-memory candidate** from the inspected CTMV2 checkpoint. It is
not imported by modeld, registered as a process, published as a ready artifact,
or selected automatically. Existing TT/IDM and other model paths are unchanged.
It does not modify the source artifact or replace installed PKLs.

The candidate removes the six camera-plane kernels and their concatenation,
retains the original model kernels/weights and scratch arena, and uploads a
compact input once. An AMD-local copy fills the original warped-image view
before the policy/history kernels execute. Original control offsets are
preserved, including the transform header and CTMV2's 16,384-float previous
feature. At 1928x1208 the input shrinks from 7,536,760 to 459,264 bytes; these
are buffer sizes, not measured USB-wire traffic or a performance guarantee.

Extraction rejects other checkpoints/layouts, unexpected graph wrappers,
recurrent dependencies in the warp prefix, and policy programs whose packed
input accesses cannot be statically bounded to the control region. It depends
on this tinygrad captured-graph representation, not a universal PKL conversion.
Static checks do not prove warp semantics, output equivalence or state safety.

Before any runtime integration, require parked hardware checks of original
versus extracted outputs and history across changing inputs, QCOM versus the
original AMD warp, serialization/reload, same-model fallback, sustained latency
and Cluster contention. No QCOM frontend or ready artifact has been installed
for this candidate. Never use it for driving based only on static inspection.

#### Compact upload and AMD-resident feature feedback

The explicit research option `extract_ctmv2(..., resident_state=True)` removes
the previous feature from the host payload. It retains the original history
kernels and their state queues; it does not substitute another model or change
their sampling semantics. Each call uploads a 120-byte transform/control header
and 393,216 image bytes in a 393,728-byte aligned buffer. AMD-local copies fill
the original control layout and warp-output view. A dedicated 65,536-byte AMD
buffer feeds the prior hidden feature into the next call and receives output
floats `[2066:18450]` after the model finishes. Initialize it to zero before the
first call. The returned public output is `(1, 2066)` float32 (8,264 bytes),
excluding hidden state and trailing padding; this return contract is for the
isolated experiment, not the production parser.

This follows carrot's data-movement/state-residency approach without assuming
its generic artifact or state ABI is compatible with CTMV2. The default
extractor still returns the original output and accepts host previous features.
Both candidates remain unqualified and unselected by modeld.

`openpilot.sunnypilot.modeld_v2.verify_extracted_policy` connects this candidate
to the existing coherent-copy QCOM warp backend in a parked-only process:

```sh
python -m openpilot.sunnypilot.modeld_v2.verify_extracted_policy \
  --model /absolute/path/to/driving_cinque_terre_model_v2_september_08_2026_tinygrad.pkl \
  --camera 1928x1208 --warmup 10 --runs 100 --roundtrip
```

Use the vehicle's existing Python environment. The path can be the base path
of a complete chunked model. The tool pins the inspected v25 source SHA, rejects
onroad operation, concurrent modeld and absent Chestnut, and changes no vehicle
parameters or installed artifacts. Do not run against an untrusted pickle.
It builds QCOM warp in a temporary cache, demands exact QCOM/AMD pixels, compares
public outputs, resident features and all four history queues across changing
inputs, and reports mean/P95/P99 plus counts exceeding the 50 ms budget.
Unlike the older host-warp path, sampling-boundary differences fail this
qualification experiment rather than being accepted. `--roundtrip` reloads the
candidate before a further compared frame and needs additional RAM/VRAM/disk.
Errors stop the experiment; it does not silently switch models/backends.

Host tests execute the copy schedule on the tinygrad interpreter and cover
payload size, feature offsets, changing frames, guards and timing thresholds.
They are not evidence of real CTMV2/QCOM/AMD equivalence. Even successful parked
results report `qualified: false`: live frame IDs/Chestnut dropped frames,
representative camera/calibration inputs, sustained performance, same-model
fallback and production selection still need validation. TT/IDM, original PKLs
and Cluster remain on their existing paths.
