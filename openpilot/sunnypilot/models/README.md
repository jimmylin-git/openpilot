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

## Chestnut PKL inference

Downloaded PKL bundles remain the normal big-model path, including TT/IDM and
CTMV2's original PKL. No ONNX download or compilation is required to use them.
The existing general ONNX compiler and small-model tooling remain available.

The CTMV2 captured-policy extractor, recorded/synthetic verifier, compact upload,
AMD-resident feature experiment and pre-upload QCOM warp runtime have been
removed. Experimental camera-input ABI artifacts are rejected explicitly;
select an original PKL instead. The Chestnut-only experimental model override,
preparation process and status UI are no longer supported. Installed models,
recordings and caches are not deleted or selected by this cleanup. Model bundle
selection, generation settings, original recurrent state and the existing
small-model failure fallback remain unchanged.

## Bounded camera pairing

Stock modeld and modeld_v2 retain the shared `camera_sync.py` implementation
adapted from carrot's MIT-licensed implementation at commit
`c60cde06cb10908da5e559732a6a0b2b1ff1d0d3`, independently of the removed warp
experiment. Each call receives fresh frames, compares their current SOF
timestamps, and accepts skew up to and including 20 ms in either direction.
Only the older camera advances during resynchronization, with the original
ten-iteration bound. Timeouts or unsuccessful resync skip model dispatch;
stale buffers are not reused. Single-camera operation uses each fresh frame
for both inputs.

Real frame gaps still reach the existing dropped-frame and cameraOdometry
validity checks. No camera driver, downstream fault check or Cluster renderer
is changed. Both Chestnut and small models use this pairing rule without
rebuilding their artifacts. Host tests cover exposure timing, skew boundaries,
missing frames and resync behavior; this does not establish driving performance
or eliminate Chestnut execution-time overruns.
