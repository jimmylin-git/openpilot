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

## Carrot generic Chestnut pipeline (experimental branch)

### Automatic offline operation

This branch registers `carrot_offline` as an offroad-only vehicle process.
No laptop, SSH command or network is needed to run its parked smoke test
after the model/runtime have been prepared. On the first installation, keep
the vehicle device powered **offroad with Wi-Fi** until the Models page's
**Carrot Offline Test** row says the local files are ready. A Git source
installation cannot include the external 776 MB weights; initial preparation
therefore still requires a network connection on the device itself.
The process downloads/resumes/verifies the default carrot artifact and runtime
automatically, independently of the existing selected models.

On later offroad sessions it verifies and uses the saved files without
requesting a remote manifest. When a supported eGPU USB ID (ADD1:0001 or
3801:0001) is detected, regardless of its product name or firmware string,
it automatically runs changing synthetic frames on both camera layouts,
checks finite parsed outputs, resets state, and saves timing results.
Unrelated USB devices and bootloader IDs do not trigger testing. USB absence
means power status is unknown; USB presence permits a test attempt but does
not guarantee that the GPU is ready. Initialization failures are reported.
It never forces Chestnut power, changes ignition/vehicle parameters or runs
alongside modeld. If Chestnut is not available offroad, the screen explicitly
shows that it is waiting; **starting ignition does not bypass this condition**.
The manager stops this process when switching onroad, and the original
selected model continues driving. This is not an automatic carrot road test.

The Models page displays preparation, download, waiting, test progress,
completion or failure. Completion means only that the synthetic parked smoke
test ran, never that driving equivalence was proven. Results are saved under
the independent carrot cache as `offline-status.json`, `offline-result.json`
and, on failure, `offline-error.json`; no uploads are required.
Latest results persist across reboot until replaced by the next completed
test. GPU failures are not repeatedly retried in the same offroad session.
USB transport, PSP bootloader and SMU initialization timeouts are device
failures, not proof of model corruption. Old rejections for these startup
failures can be recovered after reboot using hash-verified local files, without
fetching a manifest or discarding the failure record. Model/format errors remain
rejected, and recovery never grants driving qualification.
Download failures are recorded and preparation retries after 60 seconds.

Automatic and explicit diagnostic tests share a nonblocking parked-session
file lock across both camera layouts. A busy automatic test waits without
loading the GPU; an explicit diagnostic reports failure without retrying.
The runtime's own GPU lock remains authoritative for other users such as
modeld. Startup contention on that USB GPU lock is not model corruption.
Older misclassified lock failures can recover after reboot with verified
local artifacts; the original failure record is retained.

The explicit QCOM diagnostic saves 100 per-frame timing samples, stage
mean/P95/P99/max, slow/fast means and the ten slowest frames. Its response
contains timing JSON; normal inference and reset retain the original pipe
response. QCOM input copy, warp submission and warp readback are measured
separately without introducing an extra synchronization. Warp readback and
model output readback still include waiting for outstanding GPU work; these
are not pure kernel or USB bulk-transfer measurements.
Return the device to a network later to retrieve these files for diagnosis.
Hardware testing of this automatic workflow is still required.

The `carrot-chestnut-pipeline` branch ports the MIT-licensed generic model
runtime, coherent QCOM warp, compact upload, AMD state feedback, isolated
worker/client, verified artifact delivery and diagnostics from
`ajouatom/openpilot` carrot-wip commit
`c4702e4238f5e6aeb2e14911a5b635234f821efd`. It does not restore the removed
CTMV2 extractor or relabel an existing PKL as a generic artifact.
The new modules retain carrot's source layout under `selfdrive/modeld`.

This is a **parked-only, unqualified integration**, not a default driving
backend. Without `CARROT_CHESTNUT_PKL`, existing TT/IDM/model selection and
fallback behavior are unchanged. Small models ignore this variable. Vehicle
controls, message publication, Cluster and the bounded camera pairing below
are retained. The offroad preparation/test process is automatic; there are no
model-selector changes, onroad/build hooks or vehicle parameter changes.

### Downloading carrot's model and pinned runtime

On a parked C3 vehicle, using its existing Python environment:

```sh
cd /data/openpilot
export PYTHONPATH=/data/openpilot/pydeps:/data/openpilot
/usr/local/venv/bin/python -m openpilot.selfdrive.modeld.carrot_model
```

The default manifest selects Cinque V3 generic PKL, SHA256
`e758b96df27858ea97122d18554930d04f9f8bda417417074edfb3a72b008d0b`
(776,634,338 bytes), with checkpoint ending
`f78ed37d-afad-4dbc-8050-40ea885eedde/12864`. Its companion catalog currently
pins tinygrad `d5e17c935daf11f6318e45aade9528f71b8fbdcc` and a 2,731,184-byte
runtime archive, SHA256
`a25b81e90d5259c27bc49b3a93ab499c1909a33e133cba974cf023cd94727eeb`.
These identify the checked server version, not future manifest contents.
`--manifest` accepts another trusted HTTPS generic-model manifest; `--cache`
changes the independent download cache. Existing sunnypilot PKLs are not
overwritten. Downloading does not select or qualify a model.
Execution currently requires the checked two-camera uint8 image contract,
8-float desire input, two-float traffic/action inputs and `(1,18452)` float32
output contract. Other generic models need their own compatibility tests
before these guards are changed. Output-slice metadata is decoded with a
slice-only unpickler; arbitrary metadata globals are rejected.

The downloader verifies sizes, hashes, HTTPS origin, GPU architecture
`gfx1200`, camera resolutions, frame skip and serialization. It extracts the
verified runtime into a hash-named directory. The worker verifies model and
runtime archive hashes, checkpoint and runtime location before loading the
artifact. It selects the isolated tinygrad before importing model code; the
repository tinygrad is not replaced. Only use manifests/PKLs from a trusted
source: pickle executes code, and hashes establish identity, not trust.

### Parked execution and opt-in adapter

With Chestnut connected/PCIe ready and no other process owning its GPU:

```sh
/usr/local/venv/bin/python -m openpilot.selfdrive.modeld.precompiled_runner \
  /data/media/0/carrot/models/precompiled/e758b96df27858ea97122d18554930d04f9f8bda417417074edfb3a72b008d0b/model.pkl \
  --camera 1928x1208
```

This smoke test executes the downloaded model, not a synthetic stand-in.
For isolated integration testing, set `CARROT_CHESTNUT_PKL` to that absolute
installed path before constructing sunnypilot's Chestnut `ModelState`.
The adapter checks Offroad before construction and every run; do not add this
variable to persistent launch settings or bypass the parked guard.
Initialization/inference errors remain explicit and the existing modeld
failure handling is retained. This is not an automatic switch back to TT/IDM
within a generic model's recurrent sequence.

The QCOM path copies mutable host input coherently each frame, prepares two
`(6,128,256)` uint8 images and uploads controls/alignment plus images in one
393,728-byte buffer for the checked Cinque V3 specs. State inputs and
`next_state_*` outputs alias AMD buffers; no CPU previous-feature refill is
used. Warmup resets those buffers through the worker protocol. On QCOM
initialization/validation failure, carrot's explicit logged same-model AMD
warp fallback remains; check the reported backend before attributing any
performance improvement to compact upload.

For a one-shot **parked diagnostic only**, after the automatic test has completed
and its GPU worker has exited:

```sh
python -m openpilot.selfdrive.modeld.carrot_offline --sampling-boundary-diagnostic
```

This explicit mode permits only repeatable, source-pixel-verified sampling-boundary
differences in the existing warp probes. All other differences still fail, and
AMD fallback cannot count as a successful QCOM diagnostic. Both client and worker
check Offroad before execution. Results, upload size and probe differences are
saved separately in `qcom-diagnostic-{status,result,error}.json`; the original
automatic test records are preserved. Each layout uses 10 warmup frames and 100
timing samples plus reset. `qualified` remains false: this is neither continuous
output/state equivalence nor driving qualification. Default exact validation
and model selection are unchanged. Do not run this alongside another GPU worker
or repeat a failed GPU initialization within the same offroad session.

Necessary deviations from carrot:

* Keep our parser, generation-12/20-Hz constants and existing action/control
  helpers; generic metadata fixtures test their output compatibility.
* No carrot automatic model-selection/status parameters or build pipeline:
  delivery and selection are explicit and separate from the original catalog.
* QCOM pixel validation requires exact agreement. Sampling-boundary
  differences are diagnostic only and trigger the logged AMD fallback.
* The optional ChestnutState callback executes after the isolated worker
  returns, not at the exact GPU enqueue boundary.

Host tests cover delivery/resume/rejection, parser/spec contracts, worker
protocol/failure cleanup, coherent copy, compact packing and state feedback.
They do not prove hardware equivalence or sustained 20 Hz. Before enabling
driving, require actual QCOM/AMD comparisons on representative recordings,
continuous output/state equivalence, reload/reset validation, end-to-end
P95/P99 and live frame-ID/Chestnut dropped-frame measurements, plus same-model
fallback and Cluster contention tests. No qualification receipt or driving
enablement is supplied by this branch.

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
