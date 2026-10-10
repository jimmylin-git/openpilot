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

## Chestnut runtime recovery

The tinygrad model launcher runs a separate supervisor. Startup has a 120-second
deadline; after startup, model publication must complete within 5 seconds.
Camera/inference activity alone does not renew that deadline. The supervisor also
detects native USB calls that never return, including calls embedded in downloaded
compiled model bundles.

On a stall or nonzero exit, the supervisor terminates and reaps the old model
process before launching a fresh small-model-only process. Chestnut remains
disabled until the supervisor is started again (normally the next drive). A failed
small-model replacement is not restarted repeatedly. Manager shutdown stops both
processes without triggering recovery.

Recovery does not make an unhealthy USB bridge reliable or guarantee 20 Hz model
performance. It does not bypass driver monitoring, calibration, or engagement
checks. Missing/invalid model messages remain subject to existing safety checks;
no stale predictions are published during recovery.

## Experimental Chestnut host warp

Legacy supercombo models can be rebuilt with `--chestnut-host-warp`. This is a new
camera input ABI, not a runtime switch for an existing compiled bundle. The
compiler records `camera_input_abi=warped_yuv_v1` and embeds a QCOM warp for each
requested camera resolution. Unmarked bundles retain their existing behavior.

In this path, raw NV12 camera buffers and calibration matrices are processed on
QCOM. The resulting uint8 model-sized YUV tensors are staged on CPU and copied
over USB to AMD; temporal image queues and inference remain on AMD. Packed policy
inputs remain live host arrays. A pair of 512x256 YUV420 images occupies 393,216
bytes, versus 9,609,216 bytes for two raw 1928x1208 buffers with the current
allocation layout. This reduces image transport bytes, not necessarily total
execution time.

Compile only in a parked maintenance session with the normal model process
stopped and reaped. Do not run the compiler concurrently with vehicle inference:

```sh
CHESTNUT=1 DEV=USB+AMD:LLVM GMMU=0 FLOAT16=1 JIT_BATCH_SIZE=0 \
  python -m openpilot.sunnypilot.modeld_v2.compile_modeld \
  --model-type supercombo --supercombo-onnx /path/to/source.onnx \
  --model-size 512x256 --camera-resolutions 1928x1208 \
  --chestnut-host-warp --benchmark-runs 3 --output /data/host-warp-test.pkl
```

Use the same source ONNX, camera frames, and calibration matrices for both
baselines. Validate warped pixels, recurrent inputs, outputs, serialized replay,
and sustained end-to-end latency before selecting the new bundle. CPU unit tests
check geometry/packing/history equivalence; QCOM-versus-AMD numerical equivalence
and 20 Hz performance still require hardware measurements. A stock-model test is
not proof of CTMV2 equivalence. Do not overwrite an existing downloaded bundle.
