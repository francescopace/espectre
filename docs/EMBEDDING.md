# Embedding Guide

This guide is for firmware teams that want to integrate the ESPectre sensing engine into their own ESP32 firmware instead of shipping one of the published frontends. It complements [ARCHITECTURE.md](ARCHITECTURE.md), which describes the internal layering in detail.

It assumes C++17, an ESP-IDF application or equivalent host build, and familiarity with callbacks and task ownership. A **snapshot** is one immutable view of runtime state, a **listener** receives runtime events, and a **capability** reports whether the selected backend supports an optional control. If you only need an existing ESPectre firmware image, use [SETUP.md](SETUP.md) instead.

## Five-minute integration

The shortest supported integration uses `espectre_sdk.h` and `RuntimeFrontendController`:

```cpp
#include "espectre_sdk.h"

class ProductFrontend : public espectre::IRuntimeListener {
 public:
  bool setup() {
    espectre::RuntimeConfig config;  // documented defaults, ready to use
    runtime_.set_config(config);
    return runtime_.setup(this);
  }

  void loop() { runtime_.loop(); }

  void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
    if (!snapshot.ready_to_publish) {
      return;
    }
    publish_motion(snapshot.motion_state == espectre::MotionState::MOTION);
  }

 private:
  espectre::RuntimeFrontendController runtime_;
};
```

On ESP-IDF, replace the bare `RuntimeConfig` with `espectre::make_runtime_sensing_config_from_kconfig()` to drive the sensing settings from menuconfig.

Before adding product-specific behavior, enforce these runtime constraints:

- Gate everything user-visible on `snapshot.ready_to_publish`. The runtime emits snapshots while it calibrates, and motion state is not meaningful before that flag is true.
- Read `runtime_.snapshot()` for on-demand state. The controller refreshes it before forwarding each listener callback; frontends do not maintain a second cache.
- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Ask `capabilities()` before exposing a control, rather than assuming the active runtime supports it.

## What you embed

| Layer | Contents | Dependencies |
|-------|----------|--------------|
| `src/cpp/espectre_sdk.h` | Stable full-runtime SDK facade | Header only |
| `src/cpp/espectre_core_sdk.h` | Optional core-only detector facade | C++17 standard library only |
| `src/cpp/core/` | Lightweight and High-Accuracy detectors, feature extraction, filters, CSI format | C++17 standard library only |
| `src/cpp/runtime/` | Runtime contracts, snapshots, events, ESPectre Protocol model, traffic generation | Portable, host-testable |
| `src/cpp/runtime/esp_idf/` | CSI capture, Wi-Fi lifecycle, sensing pipeline, traffic generation, NVS persistence | ESP-IDF `>= 5.5` |
| `src/cpp/frontend/` | ESPHome, Native Direct/MQTT, Matter, and Streamer reference integrations | Frontend-specific stacks |

The layering is strict: `core` has no upward or SDK dependencies, and `runtime` contracts stay platform-agnostic, so the sensing logic can be compiled, tested, and simulated on a host machine without ESP-IDF.

### Stability tiers

| Tier | What it covers | Change policy |
|------|----------------|---------------|
| Stable runtime | Everything reachable from `espectre_sdk.h` | Follows the SDK version contract below |
| Core-only extension | Detector classes and documented public methods exposed by `espectre_core_sdk.h` | Follows source compatibility; algorithm internals and exact numeric output may evolve as documented below |
| Internal | Headers and declarations not identified as either facade's public API | May change in any release; they ship because the runtime and core detector definitions need them to compile |

The frontend layer is a set of reference integrations, not a supported API. Read it for patterns; do not link against it.

## Supported hardware

ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6, using standard single-antenna Wi-Fi CSI with AGC active and HT20 bandwidth. No extra sensors or radio hardware are required. See [SETUP.md](SETUP.md) for the current per-frontend target matrix.

Set `RuntimeConfig::wifi_band_policy` to choose `BAND_2G`, `BAND_5G`, or `AUTO`. `BAND_2G` is the default and is supported by every target; `BAND_5G` and `AUTO` require dual-band silicon, currently ESP32-C5 among the published targets. The runtime applies that choice and pins an 802.11n protocol ceiling plus HT20 on the selected band or bands. Unsupported policies fail setup instead of falling back silently, and packets outside the HT20 contract are dropped and counted.

## Choosing A Detection Profile

Choose Lightweight Detection when sensing must leave more CPU time and working memory for the rest of the product. It runs fewer feature trackers and less per-packet computation, but gives up accuracy and cross-environment robustness relative to High-Accuracy Detection. Choose High Accuracy when detection quality is the priority and the product can afford its additional feature state and neural inference.

Lightweight adapts its threshold from about 10 seconds of clean, ready quiet-room coverage after temporal warmup; missing or burst-concentrated slots extend wall-clock calibration instead of counting as evidence. After that, a long quiet stretch can still lower the live threshold if the opening was noisier than the rest of the session. The runtime reports those drops through `IRuntimeListener::on_threshold_changed()`, the same hook used for control writes and calibration finish; live telemetry still carries the per-sample comparison value. High Accuracy uses a trained threshold and skips that calibration, although it still needs CSI readiness and one feature window of warmup. A runtime-switching build may contain both detector implementations and ML weights in flash even while Lightweight is active; budget flash separately from active detector CPU and working memory.

## Integration paths

### Full runtime (recommended)

Your firmware owns boot, provisioning, networking policy, OTA, and the product surface; the ESPectre runtime owns CSI capture, calibration, detection, and eventing behind two contracts:

- `IEspectreRuntime` (`runtime/runtime_interface.h`): `setup()`, `loop()`, runtime threshold/detector control, recalibration, and snapshot access.
- `IRuntimeListener` (`runtime/runtime_events.h`): callbacks for motion-state changes, periodic updates, threshold/detector changes (including Lightweight settled-level recovery), calibration lifecycle, live telemetry, and runtime faults. If you publish a writable threshold control, override `on_threshold_changed()` rather than inferring the live value from telemetry.

`RuntimeFrontendController` wires configuration, detector persistence, and the runtime backend together. The Native and Matter frontends are compact reference integrations for this path.

### Core-only

If your firmware already owns Wi-Fi and CSI capture, include `espectre_core_sdk.h` and consume the detectors directly. The `core` detectors accept normalized CSI payloads and expose motion state, movement metric, and threshold control. Apply the same temporal admission as the shipped pipeline before `process_packet()`: retain the candidate nearest each `csi_target_pps` slot center, enforce the target-derived half-slot minimum spacing, and leave missing slots invalid.

After each `update_state()`, re-read `get_threshold()`: Lightweight can lower it without a setter call, and the core-only path has no `on_threshold_changed()` hook. `core/temporal_csi_sampler.h` is the production sampler; it is internal to the bundle rather than part of the supported `espectre_sdk.h` facade. Use `runtime/esp_idf/csi_pipeline.cpp` as the reference for normalization, temporal admission, evaluation cadence, and hit filtering before committing to custom wiring.

## Header map

| Header | Use it for |
|--------|------------|
| `espectre_sdk.h` | Stable full-runtime facade and recommended integration entry point |
| `espectre_core_sdk.h` | Opt-in core-only facade for integrations that already own normalized CSI capture |
| `runtime/espectre_sdk_version.h` | Compile-time SDK version and the `ESPECTRE_SDK_VERSION_AT_LEAST()` guard |
| `runtime/runtime_interface.h` | `RuntimeConfig` and the backend contract |
| `runtime/runtime_events.h` | `IRuntimeListener` and the threading contract |
| `runtime/runtime_snapshot.h` | `RuntimeSnapshot`: what every callback delivers |
| `runtime/runtime_capabilities.h` | Which controls the active runtime honors |
| `runtime/runtime_sensing_schema.h` | Defaults and valid ranges for every tunable |
| `runtime/runtime_config_utils.h` | Validators and name/enum conversion |
| `runtime/runtime_diagnostics.h` | Capture and link counters, plus the sampler that turns them into rates |
| `runtime/csi_traffic_types.h` | Runtime traffic-source and generator mode enums used by `RuntimeConfig` |
| `runtime/esp_idf/runtime_frontend_controller.h` | The recommended entry point |
| `runtime/esp_idf/runtime_sensing_kconfig.h` | Build a config from menuconfig |
| `runtime/espectre_protocol.h` | Wire types, payload builders, command parsers |
| `runtime/mqtt_transport.h` | Implement to reach your own MQTT client |
| `runtime/direct_websocket_protocol.h` | Versioned Direct request envelopes, parsing, and response/event builders |
| `runtime/direct_websocket_service.h` | Implement to expose the local Direct WebSocket boundary |
| `runtime/ota_service.h` | Implement to reach your own update channel |
| `runtime/firmware_version.h` | The application version reported on the wire |
| `core/detector_types.h`, `core/csi_types.h`, `core/filter_config.h`, `core/detector_limits.h` | Stable value types, dimensions, defaults, and ranges shared by both facades |
| **Core-only extension** | **Headers below are reached only through `espectre_core_sdk.h`** |
| `core/lightweight_detector.h`, `core/high_accuracy_detector.h`, `core/filtered_turbulence_ring.h` | The core-only detector path and its shared filtered-sample storage |
| `core/base_detector.h` | The shared detector lifecycle both detectors inherit |
| `core/csi_format.h` | CSI layout and the subcarrier band the detectors measure on |
| `core/detector_limits.h`, `core/filters.h`, `core/utils.h` | Detector limits, filter state, and numeric helpers used by the public detector definitions |
| `core/csi_features.h`, `core/ml_feature_trackers.h`, `core/l1_delta_tracker.h` | Feature extraction and tracker types embedded in the public detector definitions |
| `core/threshold.h` | Detector threshold validation and startup calibrator used by the core-only implementation |

## Runtime contract

### Threading

The control surface is single-owner. Internal bounded mailboxes protect callback-to-loop handoff, but they do not make control calls thread-safe.

- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Every `IRuntimeListener` callback is delivered on the caller's task: from `loop()` for sensing events, or inline on the task that invoked a control method. Work raised in the Wi-Fi CSI callback is deferred through an internal mailbox first, so no listener callback runs in interrupt or Wi-Fi driver context.
- Keep callbacks bounded and non-blocking. A slow callback delays the next `loop()` iteration; sufficiently long work can fill the bounded CSI mailbox and drop incoming frames. Queue network publication, NVS writes, and other potentially blocking work for a separate task.
- Call `set_*_runtime()` only from the owner task. The shipped MQTT, Direct WebSocket, and OTA adapters queue stack events and deliver application callbacks from the frontend loop, so Native follows this rule without external locks.

### Lifecycle

`set_config()` -> `setup(listener)` -> `loop()` repeatedly -> `shutdown()`. The controller is reusable after `shutdown()`: the configuration survives and `set_config()` becomes effective again. `setup()` is idempotent, and a failed `setup()` leaves the controller un-setup so you can fix the config and retry.

### Errors

The control surface reports failure through `bool` returns and never throws. A `false` means the call was rejected or could not be applied, and the runtime is unchanged. There are three reasons a control call returns false:

1. The value is outside the range published in `runtime_sensing_schema.h`.
2. The active runtime does not advertise the matching capability.
3. The backend refused the change.

Asynchronous failures arrive instead through `IRuntimeListener::on_runtime_fault()`. Calibration outcome is reported by `on_calibration_finished(snapshot, success)`; a `false` there is not fatal, the runtime keeps sensing with the configured threshold.

### Capabilities

`RuntimeCapabilities` defaults every flag to false, so a runtime declares what it offers rather than inheriting a permissive default. Read `controller.capabilities()` after `setup()` and expose only what it advertises. The controller already refuses capability-gated calls; this check keeps unsupported controls out of the product interface.

### Diagnostics

The runtime exposes cumulative capture and link counters separately from the sensing snapshot. `RuntimeFrontendController::diagnostics()` reads the totals, and `RuntimeDiagnosticsSampler` turns two reads into rates without requiring a separate timer:

```cpp
// once, at frontend startup
sampler_.reset(runtime_.diagnostics(), now_ms);

// whenever the existing periodic sensing callback runs
latest_ = sampler_.sample(runtime_.diagnostics(), now_ms);
```

`RuntimeDiagnosticsSample::csi_admitted_pps` is the detector input rate after temporal admission. `csi_accepted_pps` is the identity-accepted supply. Compare admitted PPS with `RuntimeConfig::csi_target_pps` together with `csi_occupancy_ratio`, same-slot excess, missing-slot, stale, and out-of-order rates when a deployment underperforms. Occupancy is diagnostic telemetry and does not change the device send rate. MQTT `stats` publishes the same occupancy as `csi_occupancy`; the SDK field name remains `csi_occupancy_ratio`.

The shipped ESP-IDF runtime always collects these counters. Native and ESPHome refresh their cache from the same sensing update that feeds the periodic status log, then expose the cache only on an explicit `stats` request or a `Refresh Diagnostics` button press. `CONFIG_ESPECTRE_DEBUG_TELEMETRY` controls additional timing and load logs, not availability of these counters.

### Versioning

`ESPECTRE_SDK_VERSION_STRING` identifies the SDK sources you compiled against. Use `ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch)` to guard code that needs a given release.

ESPectre uses Semantic Versioning for the published C++ source API:

- Patch releases preserve source compatibility and documented lifecycle, validation, ownership, threading, capability, and error semantics. Detector coefficients and generated model weights may change when validation gates demonstrate a compatible quality fix; exact floating-point telemetry is not a compatibility guarantee.
- Minor releases may append fields, add callbacks with default implementations, and add types, functions, or overloads. Existing calls keep their meaning, closed enums do not gain values, and removals require a prior deprecation in a released minor version.
- Major releases may remove deprecated APIs or otherwise break source compatibility, with migration notes in `CHANGELOG.md`.
- Prerelease and rolling `preview` or `develop` bundles may change before the corresponding final release. The compatibility promise begins at the final numeric release.

The SDK is distributed and consumed as source. It does not promise a stable binary ABI: rebuild the SDK and integration together with the same C++ standard library and ESP-IDF toolchain. Construct public configuration and snapshot structs with their defaults, then assign named fields as shown in this guide; positional aggregate initialization is outside the compatibility contract so new fields can be appended safely.

Everything reachable from `espectre_sdk.h` belongs to the stable runtime surface. `espectre_core_sdk.h` is a separate, explicit opt-in for custom capture pipelines: its detector classes and documented public methods follow the same source-compatibility rules, while feature trackers, generated weights, and other headers reached only as implementation dependencies are not independent extension points.

First-party firmware, host tests, and CMake configuration resolve the string from `git describe` on numeric tags. The result is either the tag or a moving identity such as `<tag>-<commit-count>-g<hash>`. A checkout without usable Git history must pass `-DESPECTRE_GIT_VERSION=...` or set `ESPECTRE_GIT_VERSION`; ESPHome GitHub clones use this override.

Published SDK bundles stamp the same identity into `espectre_sdk_version.h` and `idf_component.yml`, so an unpacked archive compiles without `.git`. There is no in-tree numeric fallback. Rolling GitHub tags remain `snapshot` for `preview` and `snapshot-dev` for `develop`. SDK identity is separate from `espectre_firmware_version()`, which reports the application version, and `ESPECTRE_PROTOCOL_VERSION`, which versions the wire format.

## Build integration

Both surfaces build the same sources; they differ only in how you select the optional capability groups.

- **CMake / ESP-IDF**: include `src/cpp/espectre_sources.cmake` and consume the source lists (`ESPECTRE_CORE_SOURCES`, `ESPECTRE_RUNTIME_ESP_IDF_SOURCES`, and the per-capability lists for Direct WebSocket, MQTT, provisioning, and OTA) plus `ESPECTRE_SHARED_INCLUDE_DIRS`. The frontend `CMakeLists.txt` files show the working combinations.
- **Vendored ESP-IDF component**: drop `src/cpp/` into your project's `components/` directory and add `espectre` to your own component's `REQUIRES`. The sensing runtime is always built; the optional groups are opt-in under the "ESPectre SDK" menuconfig menu.
- **Toolchain**: C++17, ESP-IDF `>= 5.5` for the `runtime/esp_idf` services. Repository builds use ESP-IDF `5.5.5`.

`ESPECTRE_SHARED_INCLUDE_DIRS` puts the SDK root on the include path, so both the flat form (`#include "runtime_interface.h"`) and the layer-prefixed form (`#include "runtime/runtime_interface.h"`) work. Prefer the prefixed form: the shared tree contains generic basenames such as `utils.h` and `filters.h`, and the prefix keeps them from colliding with headers of your own.

### Optional capability groups

| Menuconfig option | `espectre_sources.cmake` variable | Adds |
|-------------------|-----------------------------------|------|
| `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` | `ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES` | Shared bootstrap, control, sysinfo, and MQTT payload helpers |
| `ESPECTRE_SDK_ENABLE_MQTT` | `ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES` | `EspIdfMqttTransport` over `esp-mqtt` |
| `ESPECTRE_SDK_ENABLE_PROVISIONING` | `ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES` | Device config store and Wi-Fi provisioning |
| `ESPECTRE_SDK_ENABLE_OTA` | `ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES` | `HttpsOtaService` |
| `ESPECTRE_SDK_ENABLE_STREAM_RUNTIME` | `ESPECTRE_RUNTIME_STREAMER_FRONTEND_SUPPORT_SOURCES` | The `RuntimeProfile::STREAM` backend |

Each group is off by default, so a minimal integration does not pay for transports it never calls. Implementing `IMqttTransport`, `IDirectWebSocketService`, or `IOtaService` yourself needs no group at all: the interfaces are header-only. `DirectWebSocketServiceConfig` keeps its generic Origin allowlist empty; `for_first_party_portals()` explicitly selects the official production and validation portals. The Native reference app adds `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES` explicitly because Direct WebSocket and mDNS are frontend-owned deployment choices rather than a general SDK default.

## Published SDK channels

ESPectre publishes source-first SDK bundles alongside the firmware release channels:

| Channel | Source | Intended use |
|---------|--------|--------------|
| `release` | semver GitHub Release and `https://espectre.dev/artifacts/sdk/release/` | Production integrations and reproducible open-source or commercial builds |
| `preview` | rolling `snapshot` GitHub prerelease and `https://espectre.dev/artifacts/sdk/preview/` | Validate `main` before the next release |
| `develop` | rolling `snapshot-dev` GitHub prerelease and `https://espectre.dev/artifacts/sdk/develop/` | Pre-main validation from `develop` |

Each SDK bundle includes:

- `src/cpp/espectre_sdk.h`
- `src/cpp/espectre_core_sdk.h`
- `src/cpp/core/`
- `src/cpp/runtime/`
- `src/cpp/runtime/esp_idf/espectre_config/`
- `src/cpp/espectre_sources.cmake`
- `src/cpp/espectre_git_version.cmake`
- `src/cpp/CMakeLists.txt`
- `src/cpp/idf_component.yml`
- `src/cpp/Kconfig.projbuild`
- `src/cpp/Doxyfile`
- generated `src/cpp/core/ml_weights.h`

The published bundle is not a chip-specific binary library. It is a versioned source package with stamped packaging metadata, suitable for vendoring or unpacking into your own firmware tree. Its `.tar.gz` and `.zip` archives are generated deterministically from the source commit timestamp, and the accompanying SDK manifest records a SHA-256 digest for each archive so consumers can verify downloaded bytes.

## Detection profile behavior

- **Lightweight Detection** (`DetectionAlgorithm::LIGHTWEIGHT`) uses `LightweightDetector`, requires no training data, and adapts its probability threshold to the session at startup and again if a later quiet stretch proves the opening was too noisy. Mirror `on_threshold_changed()` if your product publishes that threshold.
- **High-Accuracy Detection** (`DetectionAlgorithm::HIGH_ACCURACY`) uses `HighAccuracyDetector` with a trained model (`core/ml_weights.h`) and a fixed default threshold. The training and export pipeline is documented in [ML_TRAINING.md](ML_TRAINING.md).
- Shared defaults, ranges, and validation live in `runtime/runtime_sensing_schema.h` and are documented in [SETUP.md](SETUP.md).

## Validation assets

- [README.md](performance/README.md) publishes the current benchmark and validation metrics per chip and detector.
- `test/cpp/` builds the full sensing stack on a host machine, including integration suites that replay real CSI recordings through the production pipeline; `test/python/` mirrors the algorithm behavior for parity checks.
- `test/cpp/suites/runtime/test_sdk_surface.cpp` compiles against `espectre_sdk.h` alone, so it fails if the facade stops reaching the documented surface or a published default drifts out of its range.
- `test/python/test_sdk_surface_invariants.py` checks the surface against its own documentation: every facade header appears in the API reference and this guide's header map, and no type reachable from the facade is left as an unresolved forward declaration.
- The dataset collection and quality workflow is documented in [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md).

## Generated API reference

The headers carry Doxygen-compatible documentation. Generate a browsable reference for the supported surface from the repository root with:

```bash
python3 .github/scripts/generate_sdk_api.py
```

The generator stamps Doxygen `PROJECT_NUMBER` from the same `git describe` identity used by SDK bundles, then writes `docs/web/artifacts/sdk/api/`. The output is not committed, so it never drifts from the headers.

An unpacked SDK bundle ships this guide and `src/cpp/Doxyfile` rewritten to write `output/api/` and stamped with that bundle's version, so `doxygen src/cpp/Doxyfile` from the bundle root rebuilds a matching reference without the website tree. The published reference for the current site commit is at `https://espectre.dev/artifacts/sdk/api/`, rebuilt from source on every deploy.

## Licensing

ESPectre is dual-licensed: GPLv3 for open-source use, with commercial licenses available for embedding into proprietary firmware. See [LICENSING.md](../LICENSING.md).
