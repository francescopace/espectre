# Embedding Guide

This guide is for firmware teams that want to integrate the ESPectre sensing
engine into their own ESP32 firmware instead of shipping one of the published
frontends. It complements [ARCHITECTURE.md](ARCHITECTURE.md), which describes
the internal layering in detail.

## What you embed

| Layer | Contents | Dependencies |
|-------|----------|--------------|
| `src/cpp/core/` | Classic and ML detectors, feature extraction, filters, CSI format | C++17 standard library only |
| `src/cpp/runtime/` | Runtime contracts, snapshots, events, ESPectre Protocol model, adaptive traffic pacing | Portable, host-testable |
| `src/cpp/runtime/esp_idf/` | CSI capture, Wi-Fi lifecycle, sensing pipeline, traffic generation, NVS persistence | ESP-IDF `>= 5.1` |
| `src/cpp/frontend/` | ESPHome, native BLE/MQTT, Matter, and streamer reference integrations | Frontend-specific stacks |

The layering is strict: `core` has no upward or SDK dependencies, and `runtime`
contracts stay platform-agnostic, so the sensing logic can be compiled, tested,
and simulated on a host machine without ESP-IDF.

## Supported hardware

ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6, using standard
single-antenna Wi-Fi CSI with AGC active and HT20 bandwidth. No extra sensors
or radio hardware are required. See [SETUP.md](SETUP.md) for the current
per-frontend target matrix.

## Integration paths

### Full runtime (recommended)

Your firmware owns boot, provisioning, networking policy, OTA, and the product
surface; the ESPectre runtime owns CSI capture, calibration, detection, and
eventing behind two contracts:

- `IEspectreRuntime` (`runtime/runtime_interface.h`): `setup()`, `loop()`,
  runtime threshold/detector control, recalibration, and snapshot access.
- `IRuntimeListener` (`runtime/runtime_events.h`): callbacks for motion-state
  changes, periodic updates, threshold/detector changes, calibration
  lifecycle, live telemetry, and runtime faults.

`RuntimeFrontendController` wires configuration, detector persistence, and the
runtime backend together; the native and Matter frontends are the reference
integrations for this path and stay intentionally small.

### Core-only

If your firmware already owns Wi-Fi and CSI capture, you can consume the
detectors directly: `core` detectors accept normalized CSI payloads and expose
motion state, movement metric, and threshold control. Use
`runtime/esp_idf/csi_pipeline.cpp` as the reference for normalization,
evaluation cadence, and hit filtering before committing to a custom wiring.

## Build integration

- CMake / ESP-IDF: include `src/cpp/espectre_sources.cmake` and consume the
  source lists (`ESPECTRE_CORE_SOURCES`, `ESPECTRE_RUNTIME_ESP_IDF_SOURCES`,
  and the per-capability lists for BLE, MQTT, provisioning, and OTA) plus
  `ESPECTRE_SHARED_INCLUDE_DIRS`. The frontend `CMakeLists.txt` files show the
  working combinations.
- PlatformIO: `src/cpp/library.json` packages the same layers with an
  equivalent source filter.
- Component-shaped bundle root: `src/cpp/` now also exposes `CMakeLists.txt`,
  `idf_component.yml`, and `Kconfig.projbuild`, so the published SDK bundle can
  be unpacked directly into a vendored ESP-IDF component tree.
- Toolchain: C++17, ESP-IDF `>= 5.1` for the `runtime/esp_idf` services.

## Published SDK channels

ESPectre now publishes source-first SDK bundles alongside the firmware release
channels:

| Channel | Source | Intended use |
|---------|--------|--------------|
| `stable` | semver GitHub Release and `https://espectre.dev/sdk/stable/` | Production integrations and reproducible open-source or commercial builds |
| `snapshot` | rolling `snapshot` GitHub prerelease and `https://espectre.dev/sdk/main/` | Validate `main` before the next stable release |
| `snapshot-dev` | rolling `snapshot-dev` GitHub prerelease only | Pre-main validation from `develop` |

Each SDK bundle includes:

- `src/cpp/core/`
- `src/cpp/runtime/`
- `src/cpp/runtime/esp_idf/espectre_config/`
- `src/cpp/espectre_sources.cmake`
- `src/cpp/library.json`
- `src/cpp/CMakeLists.txt`
- `src/cpp/idf_component.yml`
- `src/cpp/Kconfig.projbuild`
- generated `src/cpp/core/ml_weights.h`

The published bundle is not a chip-specific binary library. It is a versioned
source package with stamped packaging metadata, suitable for vendoring or
unpacking into your own firmware tree.

## Detector behavior

- **Classic** requires no training data: it self-calibrates at startup from
  the ambient channel and adapts its probability threshold to the session.
- **ML** ships with a trained model (`core/ml_weights.h`) and a fixed default
  threshold, with the training and export pipeline documented in
  [ML_TRAINING.md](ML_TRAINING.md).
- Shared defaults, ranges, and validation live in
  `runtime/runtime_sensing_schema.h` and are documented in [SETUP.md](SETUP.md).

## Validation assets

- [README.md](performance/README.md) publishes the current benchmark and
  validation metrics per chip and detector.
- `test/cpp/` builds the full sensing stack on a host machine, including
  integration suites that replay real CSI recordings through the production
  pipeline; `test/python/` mirrors the algorithm behavior for parity checks.
- The dataset collection and quality workflow is documented in
  [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md).

## Licensing

ESPectre is dual-licensed: GPLv3 for open-source use, with commercial
licenses available for embedding into proprietary firmware. See
[LICENSING.md](../LICENSING.md).
