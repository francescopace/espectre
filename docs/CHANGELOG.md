# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0] - Unreleased - Modular multi-frontend platform

### Highlights

- **Firmware architecture is now modular**: the codebase is split into `core`, `runtime`, and `frontend` layers so ESPHome, native, Matter, and streamer can share the same sensing foundations.
- **Native firmware is now a first-class frontend**: the standalone BLE/MQTT path is no longer embedded in ESPHome and now ships as its own firmware target.
- **Matter is now part of the published firmware surface**: releases, snapshots, CI, and the web flasher now include Matter artifacts for the supported targets.
- **Streamer workflows were promoted and cleaned up**: the C++ streamer path is now the main live-streaming implementation, with collector-driven discovery and broader multi-chip CLI support.
- **ESPectre Protocol is now a shared platform service**: BLE, MQTT, provisioning, telemetry, and command handling now form a reusable baseline across ESP-IDF frontends.
- **Classic startup calibration is now motion-first, with an internal quiet-only fallback**.
- **The production ML feature set is now Core-6**, improving cross-device stability and keeping the runtime compact.
- **The roadmap now frames `v3` as the modular multi-frontend platform phase**.

### Added

- **New source layout under `src/cpp/`** with shared `core`, `runtime`, and `frontend` layers.
- **Shared runtime/frontend infrastructure** with explicit runtime contracts, common frontend orchestration, and reusable ESP-IDF protocol services.
- **Shared ESP-IDF runtime debug telemetry** for periodic heap, configured CPU frequency, runtime-loop load, loop timing, and detector evaluation timing across frontends.
- **Pipelined firmware hardware benchmark** for ESPHome and Native Classic/ML variants, with ML builds overlapping Classic monitoring and generated per-chip performance reports.
- **Matter frontend and release surface**, including published artifacts for releases, snapshots, and the web flasher.
- **Shared HTTPS OTA service for ESP-IDF frontends** under `runtime/esp_idf`.
- **L1-delta as the primary Classic runtime metric** in both the Python and C++ runtimes.
- **Motion-first startup calibration for Classic thresholds** with internal quiet-only fallback.
- **Parallel multi-detector live collect** through `./espectre collect --detector classic,ml`.
- **BLE-assisted Wi-Fi provisioning for the streamer firmware** via `tools/web/espectre-ble.html`
- **Uplink CSI record batching in the streamer transport**: up to 8 records per UDP datagram via `ESPECTRE_STREAM_TX_BATCH_RECORDS` (default 4), cutting uplink packet rate and airtime pressure.
- **ESP32-specific streamer `sdkconfig` profile** with shallower Wi-Fi TX/RX buffer caps and lwIP IPv6 disabled to fit the original ESP32 resource budget.
- **Updated architecture documentation** in `docs/ARCHITECTURE.md`

Historical decision context for the Classic and ML promotions now lives in:

- [`docs/adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`](adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md)
- [`docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [`docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`](adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md)

### Changed

- **ESPHome, native, and Matter now share the same runtime foundations**: frontend setup, diagnostics, status reporting, and standalone Wi-Fi policy were consolidated to reduce duplication and keep behavior aligned.
- **The C++ source tree was normalized around explicit naming and layer placement**: `runtime/esp_idf/protocol/` became `frontend_support/`, `csi_manager` and `standalone_wifi_manager` became `csi_pipeline` and `standalone_wifi_service`, the streamer adapter is now `streamer_frontend`, HTTPS OTA follows the `ota_service_https` variant pattern, CSI layout constants moved from `utils.h` into `csi_format.h`, threshold validation moved into `threshold.h`, and shared `core/` and `runtime/` headers no longer include ESP-IDF-only headers.
- **ESPectre Protocol was extracted from the native frontend into shared runtime code** so multiple ESP-IDF frontends can reuse the same telemetry, command, BLE, and provisioning helpers.
- **Native firmware was simplified into a dedicated standalone frontend**: BLE telemetry, MQTT diagnostics, device identity, and subscription behavior were cleaned up around the shared protocol contract.
- **Streamer workflows were modernized**: multi-chip CLI support was expanded, collection is now collector-driven, the C++ streamer protocol became the primary live-streaming path, and ESP32-C3 transport defaults were tuned for high-rate capture.
- **Dataset and sensing defaults were normalized across the project**: room-state labels were simplified, empty-room validation became part of the standard workflow, the active runtime path now uses one fixed shared subcarrier set, and Classic startup calibration remains adaptive-threshold tuning only, but now uses a shared motion-first bootstrap with a quiet-first fallback inside the same startup budget.
- **Classic startup fallback was hardened at the calibration boundary**: the final packet now participates in motion confirmation, partial movement is capped to the validated quiet-anchor band instead of setting a motion-level threshold, and trusted pre-motion samples remain available to the variance recovery vote.
- **Classic variance recovery can now be disabled explicitly** while retaining the same L1-delta startup threshold calibration, enabling L1-only experiments without source changes.
- **Default runtime subcarriers were moved away from the DC bin**: the shared fixed 12-subcarrier set is now `[14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50]`, improving current Classic real-data validation while keeping one cross-chip default band.
- **Hardware gain lock was removed completely**: ESPectre now keeps AGC active on all chips and uses one shared CV-normalized turbulence path (`std/mean`) across runtime, collection, datasets, and tooling. This avoids the forced-gain instability and Wi-Fi RX/TX problems that may lead to packet loss.
- **Matter build and CI flows were hardened**: published targets use the standard ESP-IDF path, commissioning behavior is stricter, and QEMU smoke tests now validate real application startup markers.
- **Firmware build optimization is consistent across frontends**: native, Matter, and streamer now default to ESP-IDF size optimization, matching ESPHome's release-oriented `-Os` profile for comparable firmware size and detector timing.
- **Shared detector timing is now continuously aggregated**: every runtime state-evaluation tick contributes to thread-safe duration, sample-count, minimum, and maximum statistics, while the firmware benchmark uses an exact sample-weighted average and excludes empty telemetry windows.
- **Repository tooling and docs were aligned with the new platform direction**: `./me` became `./espectre`, host-side tools now live at the top level (`collect`, `ui`, `mqtt`, `monitor`), `micro` is limited to MicroPython device commands, ESP-IDF frontend namespaces focus on build/flash, serial logs use the frontend-agnostic `monitor` command, the MQTT monitor was renamed from `espectre-monitor.html` to `espectre-mqtt.html`, ESPHome packaging no longer relies on symlinks, and the main docs were rewritten around the modular multi-frontend architecture.
- **Published firmware was reduced to one installable image per supported target**: releases and snapshots now publish the base ESPHome, Matter, and native factory images plus one unified firmware manifest, while streamer remains a source-built workflow because its Wi-Fi credentials are supplied at build time.
- **ESP32-S2 support was removed** because the target had no recorded hardware validation; supported firmware targets are now ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6.
- **Streamer transport and host collect were hardened for long live sessions**: the streamer now prefers PSRAM-backed staging where available, exposes more precise retry/duplicate telemetry (`wifi_dup`, `stim_dup`, `retry`), suspends BLE during sustained active streaming to reduce coexistence pressure, and uses chip-specific Wi-Fi defaults for `ESP32` and `ESP32-S3` transport tuning.
- **Streamer wire and dataset metadata now use a clean-break contract**: the old dedicated streamer metadata was removed from the streamer-collector exchange, CSI stream header, host parser, and checked-in capture datasets, with repository `.npz` samples migrated to dataset format `1.2`.
- **Host collection and serial monitoring are more resilient**: `collect` now reports requested and effective `SO_RCVBUF`, rolling status output was simplified around live session state, and `monitor` now uses the project `pyserial` path with auto-reconnect, port reuse, and raw-byte mode.
- **Host-side analysis tooling was modularized internally**: shared helpers were split into `tools/lib/` so end-user entrypoints stay at the top level, with dataset metadata resolution, CSI I/O, plotting helpers, path helpers, and paired variance-baseline sweep logic now living in focused internal modules instead of the old monolithic helper files.
- **The tooling support detector moved from the historical moving-variance baseline to the Classic/L1-delta path**: the legacy `optimal_threshold_gridsearch` metadata was retired, dataset pair validation and the detection-methods comparison now replay the production Classic startup calibration directly on the selected quiet capture, ML sample weighting uses l1_delta-guided modes (`l1_gridsearch`, `l1_hard_negative`), and the remaining variance-baseline research tools keep self-calibrating their thresholds instead of reading metadata.
- **The production ML feature set is now the mixed "Core-6" set** (`turb_mad_over_mean`, `turb_skewness`, `turb_autocorr`, `l1_delta`, `l1_delta_std`, `l1_delta_waveform_length`), replacing the relative-8 turbulence set.
- **ML seed-search candidate gating became much faster and reliable again**: the long-recording gate is now evaluated in-process on cached feature streams (the per-packet replay is paid once per search instead of once per candidate), gate aggregates now cover every curated long recording instead of the last one per chip, promoted artifacts still get a final full pytest verification, and the gate subprocesses disable `pytest-xdist` so the summary table the trainer parses is emitted again.
- **Micro-ESPectre was reorganized under src/python/micro_espectre/**: the runtime/device sources now live in a dedicated subdirectory.
- **ESPHome baseline `2026.6.0`**; examples/QEMU now require `min_version: 2026.6.0`.
- **The Python baseline was raised from `3.12` to `3.14`** across the main workflow and the ML training environment.
- **The ML training stack was migrated from TensorFlow/Keras to PyTorch**: the trainer now runs on the PyTorch MLP path, exports the same runtime weights, and no longer produces the unused TFLite/scaler artifacts.
- **The ML runtime now exposes direct probabilities on a `0.0-1.0` scale**: Python, C++, and training-side reference inference use the raw sigmoid output, so the published movement metric is now a probability and the default binary decision threshold is `0.5`.
- **ML documentation was split by concern**: dataset collection stays in `docs/ML_DATA_COLLECTION.md`, while training, export, and validation guidance now live in `docs/ML_TRAINING.md`.
- **The roadmap was realigned around the platform split**: `v3` now defines the reusable local platform phase, while `v4` is positioned as an optional privacy-first orchestration layer across multiple ESPectre nodes.

---

## [2.8.0] - 2026-05-21 - Detection hardening, ML cross-chip reliability, and runtime motion policy

- Detection and calibration were hardened across stacks: tighter NBVI defaults, Hampel enabled by default, a 100-packet default window, and a clearer edge-driven motion policy.
- ML reliability improved across chips with shared CV-normalized turbulence, a refreshed 9-feature model, and stricter training/data quality controls.
- `ping` became the default CSI traffic source, `./me detect` was added for live ML inference, and notebooks plus CI/test coverage were expanded.

---

## [2.7.0] - 2026-03-17 - ESPectre configuration over BLE and subcarrier normalization

- BLE runtime control became a first-class standalone integration surface, including live threshold updates and a Web Bluetooth example client.
- CSI normalization was extended to `256->128`, `228->114`, and `114->128` payload remaps, with aligned behavior and tests across C++ and Micro-ESPectre.

---

## [2.6.0] - 2026-03-08 - ESP32-C5 Support, Context-Aware Calibration, and Stricter Validation Targets

- ESP32-C5 support was added and runtime handling on newer chips (`C5`/`C6`) was hardened.
- Calibration, thresholds, dataset metadata, and ML feature selection were aligned more strictly across C++ and Micro-ESPectre.
- Validation targets were tightened to `Recall >95%` and `FP <5%`, with related tooling and deploy diagnostics improved.

---

## [2.5.1] - 2026-02-23 - HT STBC Multi-Antenna Router Fix

- Fixed HT STBC CSI handling on ESP32-C5/C6 with multi-antenna routers by accepting 256-byte packets and using the first HT20 estimate.
- Fixed Micro-ESPectre NBVI calibration memory issues on ESP32-C3, improved calibration speed, and refreshed performance/snapshot documentation.

---

## [2.5.0] - 2026-02-15 - ML Detector, Training Pipeline & Pre-built Firmware

- Added the first experimental ML detector in both ESPHome/C++ and Micro-ESPectre/Python, with a training and weight-export pipeline.
- Added pre-built firmware releases for all supported ESP32 variants.
- Removed the PCA detector and the older P95 calibrator, leaving MVS plus NBVI as the main non-ML path at the time.

---

## [2.4.0] - 2026-01-24 - Live Recalibration, Adaptive Threshold & PCA

- Added live recalibration, adaptive thresholds by default, and a choice between MVS and experimental PCA detection.
- Standardized the runtime around HT20 CSI, improved calibration/subcarrier handling, and expanded tooling, tests, and Micro-ESPectre support.

---

## [2.3.0] - 2025-12-31 - End of Year Edition

- Added `ESPectre - The Game`, a browser-based motion-controlled tuning and demo client.
- Added sensor customization, external traffic mode, `ping` traffic generation, and configurable gain-lock behavior.
- Improved channel-change handling, NBVI calibration, and board support, including tested ESP32-C3 and original ESP32 paths.

---

## [2.2.0] - 2025-12-19 - Gain Lock, Low-Pass Filter & ML Data Collection

- Added gain-lock stabilization, low-pass filtering, and baseline variance normalization to make calibration more stable.
- Tightened NBVI behavior, moved variance evaluation to publish time, and auto-configured the required ESP-IDF options in the ESPHome path.
- Added the first labeled ML data-collection infrastructure (`me collect`, `.npz`, and `csi_utils.py`) plus broader testing/documentation.

---

## [2.1.0] - 2025-12-10 - Made for ESPHome Compliance

- All example configs were brought in line with "Made for ESPHome" requirements, including provisioning, dashboard import, and project metadata.
- Variance and Hampel behavior were unified and optimized across C++ and MicroPython.
- The test suite and coverage pipeline were expanded substantially.

---

## [2.0.0] - 2025-12-06 - ESPHome Native Integration

- Major platform migration from standalone ESP-IDF firmware to an ESPHome native integration for Home Assistant.
- Established the dual-platform model: ESPHome/C++ for production, and Micro-ESPectre/MicroPython for R&D and rapid experimentation.
- Migrated tests and CI toward the ESPHome-oriented workflow with host-side CMake/CTest coverage.

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
