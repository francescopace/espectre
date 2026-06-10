# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0] - Unreleased - Core/runtime/frontend split

### Highlights

- **Internal architecture refactored**: the firmware codebase is now split into `core`, `runtime`, and `frontend` layers to separate reusable detection logic from ESP-IDF orchestration and ESPHome integration.
- **ESPHome kept as the production frontend**: the existing ESPHome component model is preserved while the implementation now lives under the new `src/` layout.
- **Foundation prepared for future frontends and runtimes**: the runtime contract is now explicit, making it possible to add a Matter frontend or alternate runtimes without cloning the motion pipeline.
- **ESPHome packaging no longer depends on repository symlinks**: shared code is now wired from canonical `core` / `runtime` sources, making ESPHome builds more reliable on Windows and in checkout/archive workflows where symlinks may be lost.
- **Matter firmware is now part of the published firmware surface**: CI, stable releases, snapshot releases, and the web flasher can all resolve Matter artifacts for all supported Matter targets except `ESP32-S2`.
- **Custom BLE protocol moved into its own frontend**: the standalone GATT transport now lives in a dedicated BLE firmware instead of being embedded in the ESPHome adapter.

### Added

- **New product-first source layout**:
  - `src/cpp/core/` for detectors, filters, thresholds, math, and shared domain logic
  - `src/cpp/runtime/` for CSI, Wi-Fi, calibration, gain lock, and traffic orchestration
  - `src/cpp/frontend/esphome/espectre/` for the ESPHome adapter and external component root
  - `src/cpp/frontend/ble/espectre/` for the standalone BLE adapter and firmware app
  - `src/cpp/frontend/matter/espectre/` for the Matter adapter and esp-matter firmware app
- **Matter frontend scaffold using managed `espressif/esp_matter`**:
  - `MatterFrontend` adapter over `IEspectreRuntime`
  - occupancy endpoint plus ESPectre vendor cluster for diagnostics and runtime controls
  - ESP-IDF firmware app under `src/cpp/frontend/matter/app/` with `main/idf_component.yml`
  - host-side tests in `test_matter_frontend`
  - verified `esp32c3` build/flash path using ESP-IDF managed toolchains and component registry dependencies
- **Published firmware manifests for the web flasher**:
  - stable manifest attached to official GitHub releases
  - main/snapshot manifest attached to the rolling `snapshot` release
  - browser-side frontend/channel/chip selection on `espectre.dev/flash/`
- **Runtime/core logging decoupled from ESPHome headers** via `espectre_log.h`, so non-ESPHome frontends can reuse the same runtime sources.
- **Frontend-oriented runtime contract** with:
  - `IEspectreRuntime`
  - `RuntimeSnapshot`
  - runtime events/listener callbacks
  - runtime capability reporting
- **`ARCHITECTURE.md`** documenting the new structure, rationale, and standalone core reuse model.

### Changed

- **CSI live streaming unified around the C++ streamer frontend**: the host collector now targets the versioned ESP-IDF streamer protocol, the legacy MicroPython UDP producer has been removed, and existing `.npz` datasets remain readable while newly collected samples gain optional device/stream metadata for realtime fusion workflows.
- **Streamer protocol simplified for ML and realtime fusion**: FTM telemetry was removed, `stimulus_id` is now optional instead of gating packet emission, and the packet header now carries a stable `gain_locked` flag plus device-side timing metadata without changing the compact HT20 CSI payload layout.
- **NBVI removed from the active runtime path**: MVS now uses the same fixed 12-subcarrier set as ML, startup calibration computes only the adaptive threshold in RAM, and no disk-backed calibration buffer is required anymore.
- **Subcarrier selection fully centralized and fixed across the repo**: C++ runtime, Micro-ESPectre, tests, and analysis/training tools now all use the same shared default set `[12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]`; runtime/config override paths were removed and related docs/tooling were simplified accordingly.
- **SPIFFS removed from active firmware layouts**: ESPHome now relies on the default board partition table, while Matter keeps a custom layout without SPIFFS so more flash can be allocated to OTA app slots and the runtime matches the new in-RAM startup flow.
- **Matter ESP32-C5 build flow simplified**: `idf.py set-target esp32c5` now uses the same standard path as the other published Matter targets; the repository and CI no longer need a special `--preview` branch for ESP-IDF 5.5.
- **Matter CI now includes QEMU boot smoke coverage for `ESP32`, `ESP32-S3`, and `ESP32-C3`**: the shared QEMU action can now consume either ESPHome builds or prebuilt Matter merged images, so the emulatable Matter targets get crash/boot regression coverage in addition to plain artifact builds.
- **Matter QEMU smoke uses a CI-only no-BLE overlay**: the QEMU path now disables CHIPoBLE via a dedicated sdkconfig overlay instead of changing the production firmware path, allowing emulated boot smoke without relaxing the normal BLE-based commissioning configuration for real devices.
- **Matter QEMU smoke now requires a pre-WiFi application marker**: the shared QEMU action can enforce an expected runtime log, and the Matter CI path now waits for `ESPectre Matter smoke marker: endpoint ... configured, starting Matter stack` while filtering the known QEMU Wi-Fi PHY assert, instead of treating a bootloader-only path as success.
- **`ESpectreComponent` is now a thin frontend adapter**: setup/orchestration responsibilities were moved behind the runtime facade.
- **ESPHome is now HA-focused again**: the `ble_channel_*` options and the custom game transport were removed from the ESPHome frontend and examples.
- **ESPHome local development path now uses `src/cpp/frontend/esphome`** as the external-components root.
- **Native tests and CI build plumbing were updated** to follow the new `src/` layout.
- **Repository CLI `./me` renamed to `./espectre`**: the repo now exposes explicit `micro`, `esphome`, `ble`, `matter`, and `streamer` namespaces instead of a single monolithic entrypoint.
- **Host-side C++ tests now use a layered `CMake + CTest` suite under `test/`**: PlatformIO-specific scaffolding was removed, suites were regrouped by `core` / `runtime` / `integration` / `frontend`, shared support code was consolidated, and coverage reporting now includes per-layer breakdowns.
- **Matter firmware startup ordering was hardened** so the shared runtime initializes after `esp_matter::start()`, allowing the Matter stack to bring up Wi-Fi before CSI-specific runtime configuration runs.
- **Documentation was aligned** to the new structure, `.venv` activation flow, and the practical ESP32-C3 Matter build/flash workflow.
- **ESPHome build metadata now points directly at shared sources** instead of relying on symlinked files inside the component directory.
- **Project narrative was realigned around the new platform direction**:
  - `README.md` now presents ESPectre as a modular Wi-Fi sensing platform, not only as a single ESPHome motion sensor
  - `ROADMAP.md` now promotes `v3` as the `core` / `runtime` / `frontend` platform phase with `Matter`, `v4` as a local web/orchestration layer, and keeps 3D localization as a stage-gated research track
  - `ARCHITECTURE.md` now explicitly connects the source split to multi-frontend firmware targets and future multi-device orchestration

### Fixed

- **Matter vendor cluster registration**: the custom ESPectre diagnostics/control cluster is now created as a server cluster, so the firmware can boot cleanly on hardware instead of failing during endpoint setup.
- **ESP32-C3 Matter bring-up**: build defaults now include the flash size, BLE, IPv6, and HKDF options required by `esp-matter`, and the runtime Wi-Fi init order now matches the Matter stack lifecycle.

### Notes

- This is an architectural/frontend release, not an algorithm change.
- The small `core` / `runtime` edits included for the Matter frontend are enabling changes only (mainly removing residual ESPHome-only logging dependencies), so the shared motion-detection behavior remains unchanged.
- The `core` layer is now reusable as a clean standalone building block for future SDK-style embedding, even though no separately packaged public SDK exists yet.

---

## [2.8.0] - 2026-05-21 - Detection hardening, ML cross-chip reliability, and runtime motion policy

### Highlights

- **Detection and calibration hardened across stacks**: multi-strategy NBVI band selection, Hampel on by default, stricter NBVI defaults, conservative hint-band fallback, unified 12-subcarrier defaults, and a 100-packet detection window.
- **Runtime motion policy aligned between firmware and Micro-ESPectre**: edge-driven binary motion publish, configurable `motion_on_hits` / `motion_off_hits` (default `3` / `3`), and `evaluation_interval` decoupled from `publish_interval`.
- **ML detector reliability improved on all chips**: datasets recollected, chip-grouped training, updated 9-feature model (`9 -> 32 -> 16 -> 1`), Hampel-aligned weights, raw-std features for ML, and a more gradual temperature-scaled Movement Score for Home Assistant.
- **Traffic generator defaults to `ping`**: ICMP ping replaces DNS as the default CSI traffic source on both ESPHome and Micro-ESPectre (`dns` remains available).

### Detection and calibration

- **Traffic generator default is `ping`**: improves compatibility with routers that ignore or rate-limit root-domain DNS queries.
- **Hampel enabled by default** (`hampel_enabled=true`, threshold `5.0` MAD).
- **NBVI selects the lowest-FP band among four strategies** per window (Entropy Spaced, MAD Clustered, Classic Spaced, Classic Clustered); scoring exposes `nbvi_classic`, `nbvi_entropy`, and `nbvi_mad`.
- **NBVI defaults tightened**: `alpha` `0.75`, `percentile` `5`, `noise_gate_percentile` `15`; calibration FP uses the runtime adaptive threshold (`P95 x 1.1`).
- **Hint-band fallback is more conservative**: keeps the current/default band when both candidates already meet the `<=5%` FP target and the hint is not meaningfully worse.
- **Default detection window is 100 packets** (`DETECTOR_DEFAULT_WINDOW_SIZE` / `SEG_WINDOW_SIZE`).
- **Motion state is edge-driven and filterable**: binary `IDLE <-> MOTION` transitions publish immediately; `motion_on_hits` / `motion_off_hits` apply consecutive-hit filtering to both MVS and ML (default `3` / `3`).
- **`evaluation_interval` decoupled from `publish_interval`**: detector evaluation defaults to every `25` packets; metric/log publishing keeps its own cadence.
- **Micro-ESPectre runtime matches firmware policy**: same evaluation/hit filtering before MQTT publish; `info` reports the new knobs; `./me deploy` ships `runtime_policy.py`.
- **Micro-ESPectre per-packet overhead reduced** for ML and MVS on device.

### ML detector

- **ML uses raw turbulence std, not CV normalization**: fixes ESP32 feature-scale mismatch with gain-locked chips; CV normalization remains MVS-only.
- **Movement Score is more gradual for Home Assistant**: temperature scaling before sigmoid; default threshold `5.0` unchanged for binary detection.
- **Production model refreshed**: 9 inputs (`turb_mean`, `turb_std`, `turb_max`, `turb_min`, `turb_iqr`, `turb_skewness`, `turb_autocorr`, `turb_mad`, `waveform_length`); topology `9 -> 32 -> 16 -> 1`; weights retrained on Hampel-filtered input.
- **Training pipeline hardened**: `StratifiedGroupKFold` by chip, stratified internal validation, hard-positive mining for near-threshold motion.
- **Per-chip datasets recollected** under stricter quality controls (gain-locked, 128 SC HT20, balanced baseline/motion); used for NBVI validation, MVS tests, and ML training.

### Added

- **`examples/espectre-s3-touch-lcd.yaml`**: Waveshare-compatible ESP32-S3 Touch LCD 1.47" profile with on-device motion status.
- **`./me detect`**: live host-side ML inference from the UDP CSI stream (replaces `tools/12_test_motion_stream.py`).
- **CSI research notebooks**: `notebooks/01_csi_data_explorer.ipynb`, `02_feature_extraction_and_ml.ipynb`.

### Changed

- **ESPHome baseline `2026.5.0`**; examples/QEMU now require `min_version: 2026.5.0`.
- **Docs and config templates**: `ping` default traffic generator; `./me detect` documented alongside `stream` and `collect`.
- **Removed `examples/uart/`**; optional `hardware_uart: UART0` documented for USB-UART bridge setups.

### Fixed

- **Cross-stack real-data test pairing**: C++ harness matches Python by using full ISO timestamps with fractional seconds.
- **Micro-ESPectre collector and dataset QA**: monotonic packet drain during interactive collect; consistent pair-ratio rounding in quality reports.
- **Code scanning and collect tooling**: addressed reported alerts; safer temp files, UDP bind handling, and `--bind-ip` on `./me collect`.

### CI and project maintenance

- **DCO commit-signoff enforcement** and **CodeQL** workflow for C++ and Python.
- **QEMU smoke tests stabilized**: PHY emulator limits handled, ESP32 coverage restored, unsupported C6 matrix entries removed.
- **Workflow permissions and Dependabot grouping** tuned to reduce noise.

---

## [2.7.0] - 2026-03-17 - ESPectre configuration over BLE and subcarrier normalization

### Highlights

- **BLE control unlocks standalone integrations**: ESPectre can now be used even without Home Assistant by building custom BLE clients.
- **Runtime threshold is now configurable via BLE**: the current BLE command channel enables live threshold updates and can be extended to additional runtime parameters.
- **Web game moved from Serial to Web Bluetooth**: `docs/game` is now an example BLE client instead of a Web Serial-only path.
- **CSI normalization supports more payload variants**: runtime handling now covers `256->128`, `228->114`, and `114->128` remap paths before HT20 processing.
- **Behavior and validation are aligned across stacks**: ESPHome/C++ and Micro-ESPectre/Python follow the same CSI-length normalization logic, with dedicated tests for `128/256/114/228`.

### Fixed

- **Extended CSI normalization paths (ESPHome/C++ and Micro-ESPectre/Python)**: runtime now consistently handles `256->128` (double HT-LTF), `228->114`, and `114->128` remap paths before HT20 processing, reducing packet drops on short/double CSI payload variants (#93).
- **Cross-stack alignment for CSI length handling**: Micro-ESPectre normalization behavior and runtime logs are now aligned with ESPHome component behavior.

### Added

- **Unit tests for new CSI payload scenarios**: added coverage for `114-byte` and `228-byte` CSI handling in C++ (`test_csi_manager`) and dedicated Python tests for `128/256/114/228` normalization paths (`test/python/test_utils.py`).

### Changed

- **Web game transport migrated to BLE**: `docs/game` now uses Web Bluetooth (desktop Chrome/Edge) with a custom ESPectre BLE service for telemetry (`movement`, `threshold`) and sysinfo notifications, replacing the previous Web Serial path.
- **BLE control channel opened to generic clients**: `docs/game` is now an example client built with the Web Bluetooth API, but ESPectre can be controlled by any standard BLE client implementing the same commands.
- **Legacy USB serial monitor path removed from firmware loop**: USB Serial/JTAG attach detection for game sysinfo emission has been cleaned up as it is no longer needed with BLE transport.
- **Practical BLE benefits documented in this release**: the game/channel flow is now compatible with standard ESP32 boards (no native USB serial/JTAG required), does not require an active wired serial connection, and enables runtime threshold tuning over BLE from any compatible BLE client (not only `docs/game`).

---

## [2.6.0] - 2026-03-08 - ESP32-C5 Support, Context-Aware Calibration, and Stricter Validation Targets

### Highlights

- **More robust runtime on modern chips (ESP32-C5/C6)**: WiFi lifecycle handling is hardened, dual-band protocol/bandwidth APIs are used correctly (with safe fallback), C5 is forced to 2.4 GHz, and C5 CSI `114-byte` payloads are normalized to HT20 `128-byte` internal layout
- **Safer calibration and detector state transitions**: calibration start failures are now handled explicitly, detector buffers are cold-cleared after calibration/channel switches, and NBVI input/band-size validation is hardened
- **Stricter quality bar for motion validation**: Python and C++ performance targets are now unified to `Recall >95%` and `FP <5%` for both MVS and ML, with docs updated accordingly (`PERFORMANCE.md`, `test/cpp/README.md`)

### Reliability and Runtime Fixes

- **Threshold handling unified across stacks**: validation is aligned to `0.0-10.0` across ESPHome/C++ and Micro-ESPectre/Python (HA number, Serial, MQTT, detector setters); MQTT now propagates detector rejection correctly; `factory_reset` restores ML threshold to `5.0`
- **Serial command parsing hardening**: `T:<value>` now uses validated `strtof` parsing (`endptr`, finite/range checks)
- **Startup fail-fast behavior**: setup now marks the component failed when WiFi initialization/handler registration fails
- **Auxiliary task stability**: DNS task always clears `running_` on early exits, avoiding stale "already running" states
- **Safety guards in diagnostics utilities**: progress-bar width/marker bounds are now clamped to prevent fixed-buffer edge cases

### Calibration, ML, and Dataset Pipeline

- **NBVI hot-path optimization**: reduced allocations, enforced memory-bounded chunked validation reads (avoids `std::bad_alloc`/`abort()` on low-heap targets), and replaced O(window) shifts with ring buffer + running statistics
- **Context-aware grid-search metadata workflow**: `tools/11_refresh_gridsearch_metadata.py` introduced and then simplified to a single C++-aligned evaluation path (legacy hardcoded subcarrier override removed)
- **Metadata consistency cleanup**: `gain_locked` is now the single source of truth in `.npz` and `dataset_info.json`; deprecated `use_cv_normalization` and `label_id` metadata removed
- **ML pipeline alignment (training + inference)**: both stacks now use `[12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]`; models were retrained/re-exported with validated seed, and feature extraction was simplified to the selected 12 runtime features

### Tooling and Developer Experience

- **Micro-ESPectre deploy diagnostics improved**: `./me deploy` now performs a REPL health-check and reports explicit remediation for ROM boot-loop (`invalid header`) with `./me flash --erase`
- **C5 support in `me` CLI expanded**: C5 auto/manual selection, `--chip c5`, correct `esp32c5` target mapping, and C5 firmware artifact selection (`ESP32_CSI_C5.bin`)
- **Flash mapping hardening**: per-chip offsets aligned with MicroPython board deploy options (including C5 `0x2000`)
- **Optional BSSID lock in Micro-ESPectre**: `WIFI_BSSID` support added in `src/main.py`
- **ESP-IDF mock alignment**: WiFi mock headers updated to modern protocol bitmasks, band-mode enums, and dual-band API signatures
- **General cleanup**: removed unused `BaseDetector` amplitude getters and refreshed stale comments/documentation (including C5 tested status in setup/examples, S2 still experimental)
- **ESPHome 2026.2.4 validation**: project configuration was re-validated after upgrading ESPHome from `2026.2.0` to `2026.2.4` (`esphome config examples/espectre-c6-dev.yaml`), with full Python and C++ test suites passing

---

## [2.5.1] - 2026-02-23 - HT STBC Multi-Antenna Router Fix

### Fixed

- **ESP32-C5/C6 STBC multi-antenna router fix**: Multi-antenna routers with STBC TX send two HT training fields per frame (HT-LTF1 + HT-LTF2), causing the CSI callback to receive 256-byte packets instead of the expected 128 bytes for HT20. On ESP32-C5/C6, `wifi_csi_acquire_config_t` has no field to disable HT STBC capture (unlike older chips). ESPectre now accepts these packets and takes the first 64 subcarriers (HT-LTF1), which is a valid channel estimate (#76, espressif/esp-csi#238)
- **Micro-ESPectre NBVI calibration on ESP32-C3**: Fixed OOM crashes during calibration caused by large in-memory allocations in the streaming NBVI computation phase; calibration now completes successfully on C3 with ~59 KB free heap
- **ESPHome 2026.2.0+ compatibility**: Ensure SPIFFS inclusion for newer ESPHome versions (#87)
- **CI develop branch**: Use local component configs instead of main branch for CI builds on develop

### Changed

- **Micro-ESPectre NBVI calibration speed**: Packet collection rate improved ~3x on ESP32-C3 (28 → 80 pps) by skipping sqrt on guard band subcarriers (excluded from NBVI selection), caching `math.sqrt` locally, and using integer arithmetic in the magnitude loop

### Added

- **Performance documentation**: RAM, Flash, and detection timing benchmarks for ESP32-C3 and ESP32-C6 in PERFORMANCE.md
- **Performance logging**: Lightweight DEBUG-level logging for heap usage (startup/post-calibration) and detection time (~10s interval)
- **git_ref substitution**: All example YAML files now use a `git_ref` substitution, making it easy to switch between branches, tags, or commits
- **Snapshot builds**: Automated pre-release builds on every push to main, providing pre-compiled firmware for testing fixes before official releases

### Documentation

- Clarify that NBVI calibration applies only to MVS mode
- Add media section in README

---

## [2.5.0] - 2026-02-15 - ML Detector, Training Pipeline & Pre-built Firmware

### Highlights

- **ML Detector (Experimental)**: Neural network-based motion detection with ~3s boot time (no calibration needed)
- **Training Pipeline**: Collect data, train, and export models for both platforms
- **Pre-built Firmware**: Ready-to-flash binaries for all 6 ESP32 variants via GitHub Releases
- **PCA & P95 Removed**: Simplified to MVS + ML detectors, NBVI-only calibration

### ML Detector

First experimental release of a neural network-based motion detector, available in both C++ (ESPHome) and Python (Micro-ESPectre).

- **Configuration**: `detection_algorithm: ml` in YAML
- **Performance**: ~97-100% F1 score depending on chip
- **Boot time**: ~3 seconds (vs ~10s for MVS) — no band calibration needed
- **Zero dependencies**: Manual MLP inference, no TFLite required

```yaml
espectre:
  detection_algorithm: ml
```

The pre-trained model shipped with this release was trained on a limited dataset collected in a single environment. It performs well in initial testing, but **we need your help to make it better**. If you try the ML detector, consider contributing baseline (empty room) and movement recordings from your environment — the more diverse the training data, the more robust the model becomes. See [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) for how to collect and submit data via pull request.

For architecture and feature details, see [ALGORITHMS.md](ALGORITHMS.md#ml-neural-network-detector).

#### Training Pipeline

Collect labeled data, train a model, and export weights for both platforms:

```bash
./me collect --label <name> --duration <sec>   # Collect data
python tools/10_train_ml_model.py               # Train model
```

Exports `ml_weights.py` (Python), `ml_weights.h` (C++), and TFLite checkpoint. See [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) for the full workflow.

### Pre-built Firmware & Easy Install

New `release.yml` workflow builds and publishes firmware for all supported chips (ESP32, S2, S3, C3, C5, C6) on every tagged release.

SETUP.md now offers two installation paths:

- **Option A: Web Flash** — Download from Releases, flash via [ESPConnect](https://thelastoutpostworkshop.github.io/ESPConnect/) in Chrome
- **Option B: ESPHome CLI** — Traditional `esphome run` workflow

Based on PR #77 by [@WLaoDuo](https://github.com/WLaoDuo).

### Removed

**PCA Detection Algorithm**: While PCA itself is a well-known statistical technique, our implementation was based on Espressif's open-source esp-radar library. Since Espressif has transitioned this library to closed source, we have removed our PCA implementation to ensure full compliance with our GPLv3 license.

MVS remains the recommended algorithm with excellent performance. Future development will focus on the ML detector, which shows very promising results in early testing and requires no initial calibration.

**P95 Calibrator**: NBVI (Normalized Band Variance Index) is now the sole calibration algorithm. NBVI consistently outperforms P95 by selecting non-consecutive subcarriers for better spectral diversity and resilience to narrowband interference.

### Improvements

- **Gain lock**: Median-based calibration (replaces mean), signed FFT gain fix, CV normalization when gain lock is skipped
- **Bug fixes**: Double amplitude calculation fix, stack allocation in Hampel filter

### Micro-ESPectre (R&D Platform)

- **Extended hardware support**: `me` CLI now supports ESP32, C3, S3, C6 with auto-detection and SHA256 firmware verification
- **ML detector filter support**: ML detector now accepts low-pass and Hampel filter parameters, matching C++ implementation
- **Import standardization**: All `src/` modules now use `try/except ImportError` pattern for MicroPython/CPython compatibility
- **Bug fixes**: Signed int8 CSI parsing, ESP32 flash offset corrected, LowPass default cutoff aligned to 11.0 Hz

### For Contributors

<details>
<summary>Architecture changes and internal improvements</summary>

#### Architecture

- **Calibrator simplification**: Removed `ICalibrator` interface and `BaseCalibrator` base class, merging all functionality into `NBVICalibrator`
- **Window size centralization**: `segmentation_window_size` (default: 75) is now defined in a single source of truth (`DETECTOR_DEFAULT_WINDOW_SIZE` in C++, `SEG_WINDOW_SIZE` in Python) and passed to both detector and calibrator
- **Calibration buffer size**: Now calculated as `10 × window_size` (default: 750 packets), automatically adapts if window size changes
- **CV normalization**: Both platforms use CV normalization consistently when gain lock is skipped
- **Unified window size in tools**: All analysis tools now use `SEG_WINDOW_SIZE` from `config.py` instead of hardcoded values

</details>

---

## [2.4.0] - 2026-01-24 - Live Recalibration, Adaptive Threshold & PCA

### Highlights

- **Two detection algorithms**: Choose between MVS (default) and PCA (Principal Component Analysis)
- **Recalibrate from Home Assistant**: New switch to trigger recalibration without reflashing
- **Adaptive threshold by default**: No manual tuning needed - works out of the box

### New Features

#### Dual Detection Algorithms

Two motion detection algorithms are now available:

| Algorithm | Configuration |
|-----------|---------------|
| **MVS** (default) | `detection_algorithm: mvs` |
| **PCA** (experimental) | `detection_algorithm: pca` |

#### Calibrate Switch

New Home Assistant switch for triggering recalibration without reflashing:

- `switch.espectre_calibrate`: Turn ON to recalibrate, auto-turns OFF when complete
- Useful after room layout changes or furniture moves

#### Adaptive Threshold

The `segmentation_threshold` parameter is now optional:

- **Default**: Adaptive threshold calculated as P95 × 1.1 during calibration
- **Manual override**: Specify value in YAML to use fixed threshold

### Improvements

#### Dual Band Selection Algorithms

Two automatic subcarrier selection algorithms:

| Algorithm | Method | Configuration |
|-----------|--------|---------------|
| **NBVI** (default) | 12 non-consecutive subcarriers | `segmentation_calibration: nbvi` |
| **P95** | 12 consecutive subcarriers | `segmentation_calibration: p95` |

#### HT20-Only Mode

Simplified to WiFi 4 (802.11n) HT20 mode for stable 64 subcarriers across all ESP32 variants:

- Consistent performance on C3, C6, S3, and original ESP32
- Reduced memory footprint

#### Other Improvements

- **Lower threshold minimum**: 0.1 (was 0.5) for high-sensitivity applications
- **ESP32-C3 dev config**: Added `espectre-c3-dev.yaml`

### Bug Fixes

- **ESP32-C3 boot crash**: Fixed duplicate `register_component` calls
- **USB Serial JTAG**: Correct ESPHome macro for detection
- **CSI data overflow**: Limit to 128 bytes to prevent overflow

### Micro-ESPectre (R&D Platform)

- **SHA256 firmware verification**: New `verify` command detects outdated firmware
- **CSI Stream Protocol v2**: Auto-detected chip type, contributor tracking
- **Refactored calibrators**: Cleaner architecture with `p95_calibrator.py`, `nbvi_calibrator.py`, `threshold.py`

### For Contributors

<details>
<summary>Architecture changes, testing, and CI/CD improvements</summary>

#### Architecture

- New `IDetector` interface for polymorphic detection (`MVSDetector`, `PCADetector`)
- Centralized threshold calculation in `threshold.h` / `threshold.py`
- Legacy `csi_processor` module removed

#### Multi-Chip Test Suite

Tests run on ESP32-C6 and ESP32-S3 with real CSI data:

| Chip | Recall | FP Rate | F1-Score |
|------|--------|---------|----------|
| C6 | 98.8% | 0.0% | 99.4% |
| S3 | 99.1% | 14.3% | 92.9% |

#### CI/CD

- **NPZ data loading**: C++ tests use same NPZ files as Python via [cnpy](https://github.com/rogersce/cnpy)
- **QEMU smoke tests**: Catches crashes on S3, C3, C6 before deploy
- **Stale bot**: Auto-close inactive issues after 30+7 days
- **Coverage threshold**: CI fails below 80%

</details>

### Documentation

- **Roadmap update**: Added 3D Localization in v4.x (30-50cm indoor tracking with phase-coherent antenna array)
- Added Part 2 Medium article link to README
- Updated bug report template with crash debug section
- Added Home Assistant dashboard screenshot to README

---

## [2.3.0] - 2025-12-31 - End of Year Edition

### ESPectre - The Game 

As a thank you to the community, I'm closing the year with something fun: a browser-based reaction game where your physical movement controls the gameplay through WiFi sensing. No controller needed!

Beyond the fun, it's actually useful for threshold tuning, calibration verification, and coverage testing. Uses Web Serial API (Chrome/Edge) with real-time CSI streaming at ~100 Hz.

**Play now:** [espectre.dev/game](https://espectre.dev/game)

### Features

#### Sensor Entity Customization

Full control over exposed sensor entities with standard ESPHome options: `internal`, `icon`, `filters`, `disabled_by_default`.

```yaml
espectre:
  movement_sensor:
    name: "Movement"
    internal: true
    filters:
      - multiply: 100
  motion_sensor:
    icon: "mdi:motion-sensor"
```

See [#51](https://github.com/francescopace/espectre/issues/51).

#### External Traffic Mode

Support for multi-device deployments with reduced network overhead.

- **`traffic_generator_rate: 0`**: Disable internal traffic generator and rely on external WiFi traffic
- **`publish_interval`**: Control sensor update rate independently from traffic source
- **UDP Listener**: Opens port 5555 to receive external UDP packets for CSI generation
- **95% less network overhead**: One broadcast source feeds all ESPectre devices

External traffic source: [`espectre_traffic_generator.py`](../examples/espectre_traffic_generator.py) - standalone script with daemon mode and Home Assistant integration.

See [#50](https://github.com/francescopace/espectre/issues/50).

#### Traffic Generator Ping Mode

New `traffic_generator_mode: ping` option using ICMP echo requests instead of DNS queries. Use when DNS mode has low packet rates (~18 pps instead of ~100 pps) due to routers not responding to root domain queries.

See [#48](https://github.com/francescopace/espectre/issues/48).

#### Gain Lock Mode

New `gain_lock` option to control AGC/FFT gain locking behavior:

- **`auto`** (default): Enable gain lock but skip if signal too strong (AGC < 30)
- **`enabled`**: Always force gain lock (may freeze if too close to AP)
- **`disabled`**: Never lock gain (less stable CSI but works at any distance)

Solves the issue where devices too close to the AP (RSSI > -40 dB) would freeze during calibration. See [TUNING.md](TUNING.md) for AGC threshold details.

### Improvements

#### WiFi Channel Change Detection

Automatic detection and buffer reset when AP switches channel (auto-channel, roaming), preventing false positives from CSI data spikes.

Fixes [#46](https://github.com/francescopace/espectre/issues/46).

#### Multi-Window NBVI Calibration

Optimized subcarrier selection with multi-window validation, gain lock phase exclusion (first 300 packets), and updated noise gate percentile (10% → 25%).

#### Calibration Fallback with Normalization

When NBVI calibration fails, normalization is still calculated and the shared default subcarriers `[12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]` are used, preventing 2000%+ motion values from missing normalization.

### Platform Support

#### ESP32-C3 Super Mini Tested

Added example configuration `espectre-c3.yaml`. Full feature support including gain lock and NBVI calibration.

#### ESP32 (Original/WROOM-32) Tested

Tested on ESP32-WROOM-32D Mini (CH340). Fixed NBVI calibration not starting on platforms without gain lock.

**Known limitations** (ESP32 original only):
- AGC/FFT gain lock not available
- External traffic generator must start **after** ESP32 connects to WiFi
- Broadcast mode not supported; use unicast instead

See [espressif/esp-csi#247](https://github.com/espressif/esp-csi/issues/247).

---

## [2.2.0] - 2025-12-19 - Gain Lock, Low-Pass Filter & ML Data Collection

### Gain Lock (AGC/FFT Stabilization)

Automatic gain control locking for stable CSI measurements, based on [Espressif esp-csi](https://github.com/espressif/esp-csi) recommendations.

- **Two-phase calibration**: Gain Lock (3s, 300 pkt) → NBVI (~7.5s, 10 × window_size pkt)
- Gain lock happens BEFORE NBVI calibration to ensure clean data
- Eliminates amplitude variations caused by automatic gain control
- Supported on ESP32-S3, C3, C5, C6 (not available on ESP32, S2)
- New files: `gain_controller.h`, `gain_controller.cpp`

### Baseline Variance Normalization

Automatic attenuation for consistent thresholds across devices and environments.

- **Always enabled** - no configuration needed
- During calibration, calculates baseline variance using selected subcarriers
- If baseline > 0.25: attenuate with `scale = 0.25 / baseline_variance`
- If baseline ≤ 0.25: no scaling needed (scale = 1.0)
- Prevents over-amplification of weak signals while taming strong ones
- Removed `normalization_enabled` and `normalization_target` parameters

### Low-Pass Filter

New 1st order Butterworth IIR filter to reduce high-frequency RF noise.

- Cutoff frequency: 11 Hz (human movement: 0.5-10 Hz, RF noise: >15 Hz)
- Reduces false positives in noisy environments (51% → 2%)
- Disabled by default: enable with `lowpass_enabled: true`
- Processing pipeline: Normalization → Hampel → **Low-Pass** → Buffer

### NBVI Improvements

Optimized parameters and restricted search range for better subcarrier selection:

| Parameter | Old | New | Effect |
|-----------|-----|-----|--------|
| `alpha` | 0.3 | 0.5 | Balanced weight between signal strength and stability |
| `min_spacing` | 2 | 1 | Allow adjacent subcarriers for better quality selection |
| `window_size` | 100 | 200 | Larger window (2s) for more stable baseline detection |
| `GUARD_BAND_LOW` | 6 | 11 | Exclude noisy edge subcarriers |
| `GUARD_BAND_HIGH` | 58 | 52 | Exclude noisy edge subcarriers |

Dynamic null subcarrier detection replaces hardcoded lists - adapts to local RF conditions.

### Performance

**Lazy Variance Evaluation**: Moving variance calculated only at publish time.
- ~99% CPU savings for variance calculation
- New API: `csi_processor_update_state()` (C++), `seg.update_state()` (Python)

### Automatic sdkconfig

The ESPHome component now auto-configures all required sdkconfig options:
- `CONFIG_ESP_WIFI_CSI_ENABLED`, `CONFIG_PM_ENABLE`, AMPDU settings, buffer sizes, tick rate
- YAML files only need platform-specific options (WiFi 6, CPU frequency, PSRAM)

### ML Data Collection

New infrastructure for building labeled CSI datasets (groundwork for 3.x):
- `me collect` CLI subcommand for recording labeled samples
- `.npz` format for ML-ready datasets
- `csi_utils.py` module with `CSIReceiver`, `CSICollector`, `MVSDetector`

### Configuration Changes

**Removed options** (now automatic):
- `normalization_enabled`, `normalization_target`, manual sdkconfig options

**Default values**: All filters disabled, normalization always active.

**Enhanced logging**: Movement logs now include WiFi channel and RSSI:

```
[I][espectre]: [######--|----] 43% | mvmt:0.43 thr:1.00 | IDLE | 101 pkt/s | ch:3 rssi:-47
```

### Testing & Documentation

- **324 pytest tests** with CI integration (`test-python` job)
- Python coverage uploaded to Codecov
- New `docs/ALGORITHMS.md` with scientific documentation of MVS, NBVI, Hampel filter

---

## [2.1.0] - 2025-12-10 - Made for ESPHome Compliance

### Made for ESPHome Compliance

**All example configurations now meet "Made for ESPHome" requirements**

#### WiFi Provisioning
- **BLE provisioning**: `esp32_improv` for easy setup via ESPHome/HA Companion app
- **USB provisioning**: `improv_serial` for web.esphome.io configuration (not yet supported on ESP32-C5)
- **Captive Portal**: Fallback AP "ESPectre Fallback" for WiFi configuration
- **No hardcoded credentials**: Removed `YOUR_WIFI_SSID` placeholders

#### Dashboard Adoption
- **`dashboard_import`**: One-click adoption from ESPHome Dashboard
- **`project` metadata**: Version tracking for firmware updates

#### Code Cleanup
- Renamed `espectre_component.cpp/.h` → `espectre.cpp/.h`
- Component ID standardized to `espectre_csi`
- Updated `me` CLI: `erase_flash` → `erase-flash` (esptool deprecation fix)

### Performance Optimization

**Unified variance algorithm and optimized Hampel filter across both platforms**

This release focuses on code uniformity between MicroPython and C++ implementations, improving numerical stability and performance.

#### Algorithm Uniformity
- **Two-pass variance**: Both platforms now use the same numerically stable algorithm
  - Formula: `Var(X) = Σ(x - μ)² / n` (more stable than `E[X²] - E[X]²`)
  - Eliminates catastrophic cancellation risk with float32
  - Identical behavior between MicroPython and C++

#### Hampel Filter Optimization
- **C++ (ESPHome)**: Eliminated dynamic memory allocation
  - Pre-allocated static buffers in `hampel_turbulence_state_t`
  - Insertion sort replaces `qsort()` for small arrays (N=3-11)
  - **~20-25μs saved per packet** (no malloc/free overhead)
  
- **MicroPython**: Pre-allocated buffers and circular buffer
  - Eliminated list creation per call
  - Insertion sort for small arrays
  - **~120μs saved per packet**

#### Validation
- New test script `16_test_optimization_equivalence.py` using real CSI data
- Verified with 2000 real CSI packets (baseline + movement)
- Maximum variance difference: 9.41e-14 (effectively zero)

| Change | C++ Impact | MicroPython Impact |
|--------|------------|-------------------|
| Two-pass variance | Unchanged (already used) | +25μs (acceptable) |
| Hampel optimization | -20-25μs | -120μs |
| **Net improvement** | **-20-25μs/pkt** | **-95μs/pkt** |

### Test Suite & Code Coverage

- **140 test cases** (+72 from 2.0.0) with real CSI data
- **Full device testing**: All tests run on both native and ESP32-C6 via `IWiFiCSI` dependency injection
- **Codecov integration**: Coverage badge, PR comments, 80% threshold
- **84% line coverage**, 94% function coverage
- **Refactoring**: Shared utilities in `utils.h`, configurable `CalibrationManager`

---

## [2.0.0] - 2025-12-06 - ESPHome Native Integration

### Major - ESPHome Native Integration

**Complete platform migration from ESP-IDF to ESPHome**

This release represents a major architectural shift from standalone ESP-IDF firmware to a native ESPHome component, enabling seamless Home Assistant integration.

> ⚠️ **Note**: Extensively tested on ESP32-S3 and ESP32-C6, but bugs may still exist. Community contributions, bug reports, and support for additional ESP32 variants are welcome!

### Two-Platform Strategy

**ESPectre now follows a dual-platform development model:**

| Platform | Role | Focus | Target |
|----------|------|-------|--------|
| **ESPectre** (ESPHome - C++) | Production | Motion detection only | Smart home users, Home Assistant |
| **Micro-ESPectre** (Micro Python) | R&D | Features + Filters for research | Researchers, developers, academics |

**ESPectre** focuses on core motion detection for Home Assistant integration.
**Micro-ESPectre** provides features (variance, skewness, kurtosis, entropy, IQR, spatial_*, temporal_*) and advanced filters (Butterworth, Wavelet, Savitzky-Golay) for research/ML applications.

**New Architecture:**
- **Native ESPHome component**: Full C++ implementation as ESPHome external component
- **Home Assistant auto-discovery**: Automatic device and sensor registration via Native API
- **YAML configuration**: All parameters configurable via simple YAML files
- **OTA updates**: Wireless firmware updates via ESPHome

**Implementation:**
- `components/espectre/`: Complete ESPHome component with Python config and C++ implementation
- Modular C++ architecture: `calibration_manager`, `csi_manager`, `sensor_publisher`, etc.
- Binary sensor for motion detection
- Movement score sensor
- Adjustable threshold (number entity) - controllable from Home Assistant

### Micro-ESPectre

**R&D Platform for Wi-Fi CSI Motion Detection - Pure Python implementation for MicroPython**

Micro-ESPectre is the research and development platform of the ESPectre project, designed for rapid prototyping, algorithmic experimentation, and academic/industrial research. It implements motion detection algorithms in pure Python, enabling fast iterations without compilation overhead.

**Key Features:**
- **Instant Deploy**: ~5 seconds to update code (no compilation)
- **MQTT Integration**: Runtime configuration via MQTT commands
- **Auto Calibration Algorithm**: Automatic subcarrier selection (F1=97.6%)
- **Analysis Tools**: Complete suite for CSI analysis and algorithm optimization
- **Feature Extraction**: Statistical features (variance, skewness, kurtosis, entropy, IQR)
- **Confidence Score**: Experimental motion detection confidence estimation
- **NVS Persistence**: Persistent configuration on filesystem

**Advanced Applications (ML/DL ready):**
- People counting
- Activity recognition (walking, falling, sitting, sleeping)
- Localization and tracking
- Gesture recognition

**Dependencies:** 
- [`micropython-esp32-csi`](https://github.com/francescopace/micropython-esp32-csi) - Custom MicroPython fork with native CSI support for ESP32 family 
- MQTT broker (e.g., Mosquitto)

### Test Suite Refactoring

**Migration from Unity (ESP-IDF) to host-side CMake/CTest for ESPHome consistency**

The test suite has been migrated from ESP-IDF's Unity framework to a host-side CMake/CTest workflow, aligning with the ESPHome development flow and enabling native (desktop) test execution without hardware.

**Complete test suite with 68 test cases organized in 5 suites and Memory leak detection:**

| Suite | Tests | Focus |
|-------|-------|-------|
| `test_csi_processor` | 19 | API, initialization, validation, memory management |
| `test_hampel_filter` | 16 | Outlier removal filter behavior |
| `test_calibration` | 21 | NBVI algorithm, variance, percentile calculations |
| `test_calibration_file_storage` | 9 | Calibration persistence and file I/O |
| `test_motion_detection` | 3 | MVS performance with real CSI data (2000 packets) |

```bash
# Run tests locally
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build --output-on-failure
```

### CI/CD Pipeline

**GitHub Actions integration for automated quality assurance**

- **Automated testing**: Runs on push to `main`/`develop` and pull requests
- **ESPHome build verification**: Compiles `espectre.yaml` to validate component
- **Status badge**: Real-time CI status displayed in README
- **Path filtering**: Only triggers on changes to `components/espectre/` or `test/`

---

## [1.5.0] - 2025-12-03 - Automatic Subcarrier Selection

### Automatic Subcarrier Selection
- Zero-configuration subcarrier selection using NBVI (Normalized Baseline Variability Index) algorithm. 
- Auto-calibration at boot, re-calibration after factory_reset.
- Formula: `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`. 
- Achieves F1=97.6% (Recall 95.3%, Precision 100%, FP 0%). 

---

## [1.4.0] - 2025-11-28 - Major Refactoring & Technical Debt Reduction

### Major Refactoring
- **Feature extraction module**: Extracted to `csi_features.c/h`, reduced `csi_processor.c` by 50%
- **Configuration centralization**: All defaults in `espectre.h`, validation in `validation.h/c`
- **Two-pass variance**: Numerically stable calculation
- **Traffic generator**: Max rate 1000 pps (was 50), default 100 pps
- **CLI migration**: Bash → Python (cross-platform)
- **Wi-Fi Theremin**: `tools/web/espectre-theremin.html` for CSI sonification
- **Removed**: Redundant segmentation parameters (min_length, max_length, k_factor)

---

## [1.3.0] - 2025-11-22 - ESP32-C6 Platform Support

### ESP32-C6 Platform Support
- **WiFi 6 (802.11ax)** support with proper CSI configuration
- **Runtime-configurable parameters**: threshold, window_size via MQTT
- **Web Monitor**: `tools/web/espectre-monitor.html` with real-time visualization
- **System monitoring**: CPU/RAM usage in stats command
- **MQTT optimization**: Simplified message format, removed segment tracking

---

## [1.2.1] - 2025-11-17

### Wi-Fi Optimization
ESP-IDF best practices: disabled power save (`WIFI_PS_NONE`), configurable country code, HT20 bandwidth.

---

## [1.2.0] - 2025-11-16 - Simplified Architecture & MVS Segmentation

### Simplified Architecture
- **MVS algorithm**: Moving Variance Segmentation with adaptive threshold
- **Amplitude-based features**: +151% separation improvement for skewness/kurtosis
- **Traffic generator**: ICMP ping-based (was UDP broadcast)
- **64 subcarriers**: All available (was 52 filtered)
- **10 features**: Added temporal_delta_mean, temporal_delta_variance

---

## [1.1.0] - 2025-11-08

### Auto-Calibration System
- **Fisher's criterion**: Automatic feature selection (4-6 from 8)
- **Butterworth filter**: Order 4, cutoff 8Hz
- **Wavelet filter**: Daubechies db4 for high-noise environments
- **NVS persistence**: Configuration survives reboots
- **Modular architecture**: Split into 10 specialized modules

---

## [1.0.0] - 2025-11-01

### Initial Release
CSI-based movement detection for ESP32-S3. Hampel + Savitzky-Golay filters, 15 features, 4-state detection (IDLE/MICRO/DETECTED/INTENSE), MQTT publishing, CLI tool. 10-100 pps, <50ms latency, 3-8m range.

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.
