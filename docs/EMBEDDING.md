# Embedding Guide

This guide is for firmware teams that want to integrate the ESPectre sensing engine into their own ESP32 firmware instead of shipping one of the published frontends. It complements [ARCHITECTURE.md](ARCHITECTURE.md), which describes the internal layering in detail.

It assumes C++17, an ESP-IDF application or equivalent host build, and familiarity with callbacks and task ownership. A **snapshot** is one immutable view of runtime state, a **listener** receives runtime events, and a **capability** reports whether the selected backend supports an optional control. If you only need an existing ESPectre firmware image, use [SETUP.md](SETUP.md) instead.

## Five-minute integration

Include one header, implement one interface, and drive one object:

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
    runtime_.record_snapshot(snapshot);
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

Three rules cover most integration mistakes:

- Gate everything user-visible on `snapshot.ready_to_publish`. The runtime emits snapshots while it calibrates, and motion state is not meaningful before that flag is true.
- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Ask `capabilities()` before exposing a control, rather than assuming the active runtime supports it.

## What you embed

| Layer | Contents | Dependencies |
|-------|----------|--------------|
| `src/cpp/espectre_sdk.h` | The SDK facade: the supported surface in one include | Header only |
| `src/cpp/core/` | Lightweight and High-Accuracy detectors, feature extraction, filters, CSI format | C++17 standard library only |
| `src/cpp/runtime/` | Runtime contracts, snapshots, events, ESPectre Protocol model, adaptive traffic pacing | Portable, host-testable |
| `src/cpp/runtime/esp_idf/` | CSI capture, Wi-Fi lifecycle, sensing pipeline, traffic generation, NVS persistence | ESP-IDF `>= 5.1` |
| `src/cpp/frontend/` | ESPHome, native BLE/MQTT, Matter, and streamer reference integrations | Frontend-specific stacks |

The layering is strict: `core` has no upward or SDK dependencies, and `runtime` contracts stay platform-agnostic, so the sensing logic can be compiled, tested, and simulated on a host machine without ESP-IDF.

### Stability tiers

| Tier | What it covers | Change policy |
|------|----------------|---------------|
| Supported | Everything reachable from `espectre_sdk.h` | Follows the SDK version contract below |
| Internal | Every other header in the bundle | May change in any release; it ships because the runtime needs it to compile |

The frontend layer is a set of reference integrations, not a supported API. Read it for patterns; do not link against it.

## Supported hardware

ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6, using standard single-antenna Wi-Fi CSI with AGC active and HT20 bandwidth. No extra sensors or radio hardware are required. See [SETUP.md](SETUP.md) for the current per-frontend target matrix.

Set `RuntimeConfig::wifi_band_policy` to choose `BAND_2G`, `BAND_5G`, or `AUTO`. `BAND_2G` is the default and is supported by every target; `BAND_5G` and `AUTO` require dual-band silicon, currently ESP32-C5 among the published targets. The runtime applies that choice and pins an 802.11n protocol ceiling plus HT20 on the selected band or bands. Unsupported policies fail setup instead of falling back silently, and packets outside the HT20 contract are dropped and counted.

## Choosing A Detection Profile

Choose Lightweight Detection when sensing must leave more CPU time and working memory for the rest of the product. It runs fewer feature trackers and less per-packet computation, but gives up accuracy and cross-environment robustness relative to High-Accuracy Detection. Choose High Accuracy when detection quality is the priority and the product can afford its additional feature state and neural inference.

Lightweight adapts its threshold during up to about 10 seconds of quiet startup coverage. High Accuracy uses a trained threshold and skips that calibration, although it still needs CSI readiness and one feature window of warmup. A runtime-switching build may contain both detector implementations and ML weights in flash even while Lightweight is active; budget flash separately from active detector CPU and working memory.

## Integration paths

### Full runtime (recommended)

Your firmware owns boot, provisioning, networking policy, OTA, and the product surface; the ESPectre runtime owns CSI capture, calibration, detection, and eventing behind two contracts:

- `IEspectreRuntime` (`runtime/runtime_interface.h`): `setup()`, `loop()`, runtime threshold/detector control, recalibration, and snapshot access.
- `IRuntimeListener` (`runtime/runtime_events.h`): callbacks for motion-state changes, periodic updates, threshold/detector changes, calibration lifecycle, live telemetry, and runtime faults.

`RuntimeFrontendController` wires configuration, detector persistence, and the runtime backend together; the native and Matter frontends are the reference integrations for this path and stay intentionally small.

### Core-only

If your firmware already owns Wi-Fi and CSI capture, you can consume the detectors directly: `core` detectors accept normalized CSI payloads and expose motion state, movement metric, and threshold control. Use `runtime/esp_idf/csi_pipeline.cpp` as the reference for normalization, evaluation cadence, and hit filtering before committing to a custom wiring.

## Header map

| Header | Use it for |
|--------|------------|
| `espectre_sdk.h` | The facade. Includes everything below and documents the contracts |
| `runtime/espectre_sdk_version.h` | Compile-time SDK version and the `ESPECTRE_SDK_VERSION_AT_LEAST()` guard |
| `runtime/runtime_interface.h` | `RuntimeConfig`, and the backend contract |
| `runtime/runtime_events.h` | `IRuntimeListener`, and the threading contract |
| `runtime/runtime_snapshot.h` | `RuntimeSnapshot`: what every callback delivers |
| `runtime/runtime_capabilities.h` | Which controls the active runtime honors |
| `runtime/runtime_sensing_schema.h` | Defaults and valid ranges for every tunable |
| `runtime/runtime_config_utils.h` | Validators, and name/enum conversion |
| `runtime/runtime_diagnostics.h` | Capture and link counters, and the sampler that turns them into rates |
| `runtime/csi_traffic_types.h` | Runtime traffic-source and generator mode enums used by `RuntimeConfig` |
| `runtime/esp_idf/runtime_frontend_controller.h` | The recommended entry point |
| `runtime/esp_idf/runtime_sensing_kconfig.h` | Build a config from menuconfig |
| `runtime/espectre_protocol.h` | Wire types, payload builders, command parsers |
| `runtime/mqtt_transport.h` | Implement to reach your own MQTT client |
| `runtime/ble_bindings.h` | Implement to reach your own BLE stack |
| `runtime/ota_service.h` | Implement to reach your own update channel |
| `runtime/firmware_version.h` | The application version reported on the wire |
| `core/lightweight_detector.h`, `core/high_accuracy_detector.h` | The core-only detector path |
| `core/base_detector.h` | The shared detector lifecycle both detectors inherit |
| `core/csi_format.h` | CSI layout, and the subcarrier band the detectors measure on |
| `core/detector_limits.h`, `core/filters.h`, `core/utils.h` | Detector limits, filter state, and numeric helpers used by the public detector definitions |
| `core/csi_features.h`, `core/ml_feature_trackers.h`, `core/l1_delta_tracker.h` | Feature extraction and tracker types embedded in the public detector definitions |
| `core/ml_weights.h` | Generated ML model metadata and weights reachable through `HighAccuracyDetector` |
| `core/threshold.h` | Detector threshold validation and algorithm-name helpers reachable through the runtime contract |

## Runtime contract

### Threading

The runtime carries no internal locking.

- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Every `IRuntimeListener` callback is delivered on the caller's task: from `loop()` for sensing events, or inline on the task that invoked a control method. Work raised in the Wi-Fi CSI callback is deferred through an internal mailbox first, so no listener callback runs in interrupt or Wi-Fi driver context.
- Because callbacks run on your own task, blocking in them is allowed. Publishing over MQTT or writing NVS from a callback costs loop latency, not CSI frames.
- The `set_*_runtime()` controls are the one surface reached from elsewhere in practice: the Native frontend applies BLE and MQTT commands straight from their stack callbacks. Prefer queueing such a request and applying it from your loop task.
- The transport seams follow their own stack instead: `IOtaService` callbacks arrive on the OTA worker task, and `IBleBindings` callbacks on the BLE host task.

### Lifecycle

`set_config()` -> `setup(listener)` -> `loop()` repeatedly -> `shutdown()`. The controller is reusable after `shutdown()`: the configuration survives and `set_config()` becomes effective again. `setup()` is idempotent, and a failed `setup()` leaves the controller un-setup so you can fix the config and retry.

### Errors

The control surface reports failure through `bool` returns and never throws. A `false` means the call was rejected or could not be applied, and the runtime is unchanged. There are three reasons a control call returns false:

1. The value is outside the range published in `runtime_sensing_schema.h`.
2. The active runtime does not advertise the matching capability.
3. The backend refused the change.

Asynchronous failures arrive instead through `IRuntimeListener::on_runtime_fault()`. Calibration outcome is reported by `on_calibration_finished(snapshot, success)`; a `false` there is not fatal, the runtime keeps sensing with the configured threshold.

### Capabilities

`RuntimeCapabilities` defaults every flag to false, so a runtime declares what it offers rather than inheriting a permissive default. Read `controller.capabilities()` after `setup()` and expose only what it advertises; the controller already refuses capability-gated calls, so this is about not showing a control that cannot work.

### Diagnostics

The runtime exposes cumulative capture and link counters separately from the sensing snapshot. `RuntimeFrontendController::diagnostics()` reads the totals, and `RuntimeDiagnosticsSampler` turns two reads into rates without requiring a separate timer:

```cpp
// once, at frontend startup
sampler_.reset(runtime_.diagnostics(), now_ms);

// whenever the existing periodic sensing callback runs
latest_ = sampler_.sample(runtime_.diagnostics(), now_ms);
```

`RuntimeDiagnosticsSample::csi_accepted_pps` is the rate the detector actually sees, which is the number to compare against `RuntimeConfig::traffic_generator_rate` when a deployment underperforms.

The shipped ESP-IDF runtime always collects these counters. Native and ESPHome refresh their cache from the same sensing update that feeds the periodic status log, then expose the cache only on an explicit `stats` request or a `Refresh Diagnostics` button press. `CONFIG_ESPECTRE_DEBUG_TELEMETRY` controls additional timing and load logs, not availability of these counters.

### Versioning

`ESPECTRE_SDK_VERSION_STRING` identifies the SDK sources you compiled against, and `ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch)` guards code that needs a given release. This is distinct from `espectre_firmware_version()`, which reports *your* application version, and from `ESPECTRE_PROTOCOL_VERSION`, which versions the wire format. The release tooling keeps the header and `idf_component.yml` in agreement and fails the SDK build if they drift.

## Build integration

Both surfaces build the same sources; they differ only in how you select the optional capability groups.

- **CMake / ESP-IDF**: include `src/cpp/espectre_sources.cmake` and consume the source lists (`ESPECTRE_CORE_SOURCES`, `ESPECTRE_RUNTIME_ESP_IDF_SOURCES`, and the per-capability lists for BLE, MQTT, provisioning, and OTA) plus `ESPECTRE_SHARED_INCLUDE_DIRS`. The frontend `CMakeLists.txt` files show the working combinations.
- **Vendored ESP-IDF component**: drop `src/cpp/` into your project's `components/` directory and add `espectre` to your own component's `REQUIRES`. The sensing runtime is always built; the optional groups are opt-in under the "ESPectre SDK" menuconfig menu.
- **Toolchain**: C++17, ESP-IDF `>= 5.1` for the `runtime/esp_idf` services.

`ESPECTRE_SHARED_INCLUDE_DIRS` puts the SDK root on the include path, so both the flat form (`#include "runtime_interface.h"`) and the layer-prefixed form (`#include "runtime/runtime_interface.h"`) work. Prefer the prefixed form: the shared tree contains generic basenames such as `utils.h` and `filters.h`, and the prefix keeps them from colliding with headers of your own.

### Optional capability groups

| Menuconfig option | `espectre_sources.cmake` variable | Adds |
|-------------------|-----------------------------------|------|
| `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` | `ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES` | Shared bootstrap, control, sysinfo, and MQTT payload helpers |
| `ESPECTRE_SDK_ENABLE_MQTT` | `ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES` | `EspIdfMqttTransport` over `esp-mqtt` |
| `ESPECTRE_SDK_ENABLE_BLE` | `ESPECTRE_RUNTIME_ESP_IDF_BLE_SOURCES` | `NimbleBleBindings` |
| `ESPECTRE_SDK_ENABLE_PROVISIONING` | `ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES` | Device config store and Wi-Fi provisioning |
| `ESPECTRE_SDK_ENABLE_OTA` | `ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES` | `HttpsOtaService` |
| `ESPECTRE_SDK_ENABLE_STREAM_RUNTIME` | `ESPECTRE_RUNTIME_STREAMER_FRONTEND_SUPPORT_SOURCES` | The `RuntimeProfile::STREAM` backend |

Each group is off by default, so a minimal integration does not pay for transports it never calls. Implementing `IMqttTransport`, `IBleBindings`, or `IOtaService` yourself needs no group at all: the interfaces are header-only.

## Published SDK channels

ESPectre publishes source-first SDK bundles alongside the firmware release channels:

| Channel | Source | Intended use |
|---------|--------|--------------|
| `stable` | semver GitHub Release and `https://espectre.dev/artifacts/sdk/stable/` | Production integrations and reproducible open-source or commercial builds |
| `snapshot` | rolling `snapshot` GitHub prerelease and `https://espectre.dev/artifacts/sdk/main/` | Validate `main` before the next stable release |
| `snapshot-dev` | rolling `snapshot-dev` GitHub prerelease only | Pre-main validation from `develop` |

Each SDK bundle includes:

- `src/cpp/espectre_sdk.h`
- `src/cpp/core/`
- `src/cpp/runtime/`
- `src/cpp/runtime/esp_idf/espectre_config/`
- `src/cpp/espectre_sources.cmake`
- `src/cpp/CMakeLists.txt`
- `src/cpp/idf_component.yml`
- `src/cpp/Kconfig.projbuild`
- generated `src/cpp/core/ml_weights.h`

The published bundle is not a chip-specific binary library. It is a versioned source package with stamped packaging metadata, suitable for vendoring or unpacking into your own firmware tree. Its `.tar.gz` and `.zip` archives are generated deterministically from the source commit timestamp, and the accompanying SDK manifest records a SHA-256 digest for each archive so consumers can verify downloaded bytes.

## Detection profile behavior

- **Lightweight Detection** (`DetectionAlgorithm::LIGHTWEIGHT`) uses `LightweightDetector`, requires no training data, and adapts its probability threshold to the session.
- **High-Accuracy Detection** (`DetectionAlgorithm::HIGH_ACCURACY`) uses `HighAccuracyDetector` with a trained model (`core/ml_weights.h`) and a fixed default threshold. The training and export pipeline is documented in [ML_TRAINING.md](ML_TRAINING.md).
- Shared defaults, ranges, and validation live in `runtime/runtime_sensing_schema.h` and are documented in [SETUP.md](SETUP.md).

## Validation assets

- [README.md](performance/README.md) publishes the current benchmark and validation metrics per chip and detector.
- `test/cpp/` builds the full sensing stack on a host machine, including integration suites that replay real CSI recordings through the production pipeline; `test/python/` mirrors the algorithm behavior for parity checks.
- `test/cpp/suites/runtime/test_sdk_surface.cpp` compiles against `espectre_sdk.h` alone, so it fails if the facade stops reaching the documented surface or a published default drifts out of its range.
- `test/python/test_sdk_surface_invariants.py` checks the surface against its own documentation: every facade header appears in the API reference and this guide's header map, and no type reachable from the facade is left as an unresolved forward declaration.
- The dataset collection and quality workflow is documented in [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md).

## Generated API reference

The headers carry Doxygen-compatible documentation. Generate a browsable reference for the supported surface with:

```bash
doxygen docs/Doxyfile
```

The output lands in `output/api/`. It is generated on demand and is not committed, so it never drifts from the headers.

The same command works from an unpacked SDK bundle, which ships this guide and `docs/Doxyfile` alongside the sources. The published reference for the current release is at `https://espectre.dev/artifacts/sdk/api/`, rebuilt from source on every deploy.

## Licensing

ESPectre is dual-licensed: GPLv3 for open-source use, with commercial licenses available for embedding into proprietary firmware. See [LICENSING.md](../LICENSING.md).
