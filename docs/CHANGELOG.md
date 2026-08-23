# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0-rc1] - Unreleased - Modular multi-frontend platform

This first release candidate brings production sensing onto the shared v3 architecture. It also introduces the Matter and Streamer frontends and defines the protocol, SDK, licensing, and browser-tool contracts planned for v3.0.0. Matter interoperability remains limited while validation expands to more controllers.

### Highlights

- **Production sensing uses a layered architecture with separate `core`, `runtime`, and `frontend` responsibilities**. ESPHome, Native, Matter, and Streamer reuse the same detector implementations, CSI policy, runtime contracts, and ESP-IDF services without introducing dependencies from lower layers back into a frontend.
- **The Matter occupancy frontend publishes full-flash firmware for every supported chip**. It generates and persists per-device onboarding data and exposes the QR code through serial output, the web flasher, and `./espectre matter qr`. Controller validation remains limited in this release candidate.
- **The Streamer frontend moves collection control to the PC**. The host collector discovers device IP addresses through mDNS, paces one or more devices, receives batched raw CSI over UDP, and preserves PHY metadata in saved captures. Post-capture validation reuses the dataset validator, warns below 85% temporal occupancy, fails below 70%, and caps quality scores by occupancy.
- **Two production detection profiles target different resource budgets**. Lightweight Detection uses a low-cost, two-feature weighted model with startup calibration. High-Accuracy Detection uses an eight-feature phaseless neural model for higher accuracy and stronger generalization without quiet-room threshold calibration.
- **Hardware gain locking was removed**. Every production detector feature is gain- and scale-invariant, so all maintained firmware paths keep AGC enabled.
- **Temporal CSI sampling replaces packet-count windows**. Packet timestamps define a stable slot grid that admits only the candidate nearest each slot. Missing slots remain missing, so a short burst of packets cannot fill a detector window and create artificial movement evidence.
- **The `./espectre` CLI covers the firmware lifecycle across supported frontends**. It builds, flashes, deploys, monitors, diagnoses, and collects from one command surface. The interactive MQTT client discovers devices from retained `info` and `status` topics, selects among multiple matches, and uses each device's published command catalog for help, completion, and command execution.
- **The new [espectre.dev](https://espectre.dev) site brings the device workflow to the browser**. It flashes published firmware, configures Native devices over Bluetooth, monitors live detection over MQTT WebSockets, manages OTA updates, and hosts the motion game and Wi-Fi Theremin.
- **One versioned protocol covers setup over BLE and operation over MQTT**. BLE provisions Wi-Fi, MQTT, and identity settings. MQTT handles discovery, persistent device-label updates, telemetry, diagnostics, OTA, and live sensing commands.
- **Supported settings and detection profiles can be changed at runtime**. Threshold, motion-hit debounce, detector selection, CSI traffic ownership, and traffic source are available where each frontend supports them. Native and ESPHome persist detector, debounce, and traffic selections; thresholds remain session-only, and Micro-ESPectre keeps every runtime write session-only.
- **The embeddable C++ SDK has a documented, source-stable public surface**. `espectre_sdk.h` exposes the recommended runtime API, `espectre_core_sdk.h` explicitly opts custom CSI integrations into the lower-level detectors, and the embedding contract defines Semantic Versioning, validation, capability, ownership, threading, error, and no-stable-ABI guarantees. Generated Doxygen pages stamp the same `git describe` identity as the matching SDK bundle.
- **ESPectre is dual-licensed** under GPLv3 or a separate commercial license for proprietary and closed-source integrations. Contributions remain subject to the CLA and DCO checks.
- **GitHub Releases keep firmware images direct and group build-specific compliance files into one archive**. Each channel or version publishes factory and OTA images for ESPHome, factory and OTA images for Native, factory images for Matter, and `firmware-compliance-<channel-or-version>.zip`; the ESPectre website continues to expose the individual SBOMs, notices, and license archives next to their corresponding factory images.

### Breaking changes and migration

- **Detector identifiers, C++ names, and the metric scale changed without compatibility aliases**. Replace `mvs` with `lightweight`, `ml` with `high_accuracy`, `MVSDetector` with `LightweightDetector`, `MLDetector` with `HighAccuracyDetector`, and the corresponding `DetectionAlgorithm::MVS` / `ML` values with `LIGHTWEIGHT` / `HIGH_ACCURACY`. Update movement and threshold integrations from the former `0–10` assumptions to the shared `0.0–1.0` probability scale.
- **The repository command wrapper is now `./espectre`**. Run MicroPython device commands under `./espectre micro`, and run `collect` and `mqtt` from the repository root. The former `micro-espectre/me` wrapper and its `ui`, `detect`, and `stream` commands are removed, as are the MQTT shell `web` and `webui` utilities.
- **ESPHome timing and traffic configuration now follows the temporal sampler**. Replace `segmentation_window_size`, `evaluation_interval`, `publish_interval`, and `traffic_generator_rate` with `segmentation_window_size_ms`, `evaluation_interval_ms`, `publish_interval_ms`, a positive `csi_target_pps`, and an explicit `csi_traffic_mode`. Drop `segmentation_threshold`, `gain_lock`, and `selected_subcarriers`.
- **The legacy ESPHome BLE channel was removed**. Drop `ble_channel_enabled`, `ble_server_id`, `ble_control_char_id`, `ble_sysinfo_char_id`, `ble_telemetry_char_id`, and `ble_telemetry_interval_ms`; use Native firmware for the shared Bluetooth provisioning protocol.
- **Micro-ESPectre configuration and MQTT topics changed**. Replace `SEG_WINDOW_SIZE`, `EVALUATION_INTERVAL`, `PUBLISH_INTERVAL`, and `TRAFFIC_GENERATOR_RATE` with `SEGMENTATION_WINDOW_SIZE_MS`, `EVALUATION_INTERVAL_MS`, `PUBLISH_INTERVAL_MS`, and `CSI_TARGET_PPS`, respectively; remove `SEG_THRESHOLD`, `SELECTED_SUBCARRIERS`, and the gain-lock settings. Remove `MQTT_CLIENT_ID`, replace `MQTT_TOPIC` with `MQTT_TOPIC_PREFIX`, and update broker ACLs and subscriptions for `{MQTT_TOPIC_PREFIX}/{device_id}/...`.
- **The supported build and hardware baseline changed**. ESPHome requires at least `2026.7.0`, host and ML workflows require Python `3.14`, ESP-IDF integrations require ESP-IDF `>= 5.5`, and PlatformIO-backed builds are no longer supported. ESP32-S2 configurations must move to ESP32, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6.
- **Dataset metadata moved from format `1.0` to `1.2`**. Consumers of 2.8.0 `.npz` captures must migrate to the versioned metadata schema.
- **C++ integrations must move to the supported SDK facade**. Include `espectre_sdk.h`, follow the `core -> runtime -> frontend` dependency direction, and update code that depends on the former `components/espectre/` layout or namespace.
- **ESPHome example configurations moved under the ESPHome frontend**. Replace repository paths beginning with `examples/` with `src/cpp/frontend/esphome/examples/`, including package URLs that reference an example YAML file.
- **ESPHome firmware URLs now use the new filename patterns**: `espectre-esphome-<channel-or-version>-<chip>.bin` for full-flash images and `espectre-esphome-<channel-or-version>-<chip>-ota.bin` for OTA images.

For detector design, feature decisions, integration guidance, and validation results, see [ALGORITHMS.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/ALGORITHMS.md), [FEATURES.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/FEATURES.md), [EMBEDDING.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/EMBEDDING.md), and [performance/README.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/performance/README.md).

---

## [2.8.0] - 2026-05-21 - Detection hardening, ML cross-chip reliability, and runtime motion policy

- Hardened detection and calibration across stacks with tighter NBVI defaults, Hampel enabled by default, a 100-packet default window, and a clearer edge-driven motion policy.
- Improved ML reliability across chips with shared CV-normalized turbulence, a refreshed 9-feature model, and stricter training/data quality controls.
- Made `ping` the default CSI traffic source, added `./me detect` for live ML inference, and expanded notebooks and CI/test coverage.

---

## [2.7.0] - 2026-03-17 - ESPectre configuration over BLE and subcarrier normalization

- Added BLE runtime control as a first-class standalone integration surface, including live threshold updates and a Web Bluetooth example client.
- Extended CSI normalization to `256->128`, `228->114`, and `114->128` payload remaps, with aligned behavior and tests across C++ and Micro-ESPectre.

---

## [2.6.0] - 2026-03-08 - ESP32-C5 support, context-aware calibration, and stricter validation targets

- Added ESP32-C5 support and hardened runtime handling on newer chips (`C5`/`C6`).
- Aligned calibration, thresholds, dataset metadata, and ML feature selection more strictly across C++ and Micro-ESPectre.
- Tightened validation targets to `Recall >95%` and `FP <5%` and improved the related tooling and deploy diagnostics.

---

## [2.5.1] - 2026-02-23 - HT STBC multi-antenna router fix

- Fixed HT STBC CSI handling on ESP32-C5/C6 with multi-antenna routers by accepting 256-byte packets and using the first HT20 estimate.
- Fixed Micro-ESPectre NBVI calibration memory issues on ESP32-C3, improved calibration speed, and refreshed performance/snapshot documentation.

---

## [2.5.0] - 2026-02-15 - ML detector, training pipeline, and pre-built firmware

- Added the first experimental ML detector in both ESPHome/C++ and Micro-ESPectre/Python, with a training and weight-export pipeline.
- Added pre-built firmware releases for all supported ESP32 variants.
- Removed the PCA detector and the older P95 calibrator, leaving MVS plus NBVI as the main non-ML path at the time.

---

## [2.4.0] - 2026-01-24 - Live recalibration, adaptive threshold, and PCA

- Added live recalibration, adaptive thresholds by default, and a choice between MVS and experimental PCA detection.
- Standardized the runtime around HT20 CSI, improved calibration/subcarrier handling, and expanded tooling, tests, and Micro-ESPectre support.

---

## [2.3.0] - 2025-12-31 - End-of-year edition

- Added `ESPectre - The Game`, a browser-based motion-controlled tuning and demo client.
- Added sensor customization, external traffic mode, `ping` traffic generation, and configurable gain-lock behavior.
- Improved channel-change handling, NBVI calibration, and board support, including tested ESP32-C3 and original ESP32 paths.

---

## [2.2.0] - 2025-12-19 - Gain lock, low-pass filter, and ML data collection

- Added gain-lock stabilization, low-pass filtering, and baseline variance normalization to make calibration more stable.
- Tightened NBVI behavior, moved variance evaluation to publish time, and auto-configured the required ESP-IDF options in the ESPHome path.
- Added the first labeled ML data-collection infrastructure (`me collect`, `.npz`, and `csi_utils.py`) plus broader testing/documentation.

---

## [2.1.0] - 2025-12-10 - Made for ESPHome compliance

- Updated all example configs to meet "Made for ESPHome" requirements, including provisioning, dashboard import, and project metadata.
- Unified and optimized variance and Hampel behavior across C++ and MicroPython.
- Expanded the test suite and coverage pipeline substantially.

---

## [2.0.0] - 2025-12-06 - ESPHome native integration

- Migrated the platform from standalone ESP-IDF firmware to an ESPHome native integration for Home Assistant.
- Established the dual-platform model: ESPHome/C++ for production, and Micro-ESPectre/MicroPython for R&D and rapid experimentation.
- Moved tests and CI to the ESPHome-oriented workflow with host-side CMake/CTest coverage.

---

## [1.5.0] - 2025-12-03 - Automatic subcarrier selection

- Added zero-configuration subcarrier selection using the Normalized Baseline Variability Index (NBVI) algorithm.
- Calibrated automatically at boot and after `factory_reset`.
- Defined NBVI as `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`.
- Achieved F1 97.6%, recall 95.3%, precision 100%, and FP 0%.

---

## [1.4.0] - 2025-11-28 - Major refactoring and technical debt reduction

- Extracted feature calculation into `csi_features.c/h`, reducing `csi_processor.c` by 50%.
- Centralized defaults in `espectre.h` and validation in `validation.h/c`.
- Replaced variance calculation with a numerically stable two-pass implementation.
- Increased the traffic generator maximum from 50 to 1000 pps, with a default of 100 pps.
- Migrated the CLI from Bash to Python for cross-platform use.
- Added `tools/web/espectre-theremin.html` for CSI sonification.
- Removed the redundant `min_length`, `max_length`, and `k_factor` segmentation parameters.

---

## [1.3.0] - 2025-11-22 - ESP32-C6 platform support

- Added Wi-Fi 6 (`802.11ax`) support and the corresponding CSI configuration.
- Made `threshold` and `window_size` configurable at runtime through MQTT.
- Added `tools/web/espectre-monitor.html` for real-time visualization.
- Added CPU and RAM usage to the `stats` command.
- Simplified the MQTT message format and removed segment tracking.

---

## [1.2.1] - 2025-11-17 - Wi-Fi optimization

- Applied ESP-IDF Wi-Fi practices by disabling power saving (`WIFI_PS_NONE`), making the country code configurable, and setting HT20 bandwidth.

---

## [1.2.0] - 2025-11-16 - Simplified architecture and MVS segmentation

- Added Moving Variance Segmentation (MVS) with an adaptive threshold.
- Switched to amplitude-based features, improving separation for skewness and kurtosis by 151%.
- Replaced UDP broadcast traffic generation with ICMP ping.
- Used all 64 available subcarriers instead of filtering the set to 52.
- Added `temporal_delta_mean` and `temporal_delta_variance`, bringing the feature count to 10.

---

## [1.1.0] - 2025-11-08 - Automatic calibration system

- Used Fisher's criterion to select four to six features automatically from a set of eight.
- Applied a fourth-order Butterworth filter with an 8 Hz cutoff.
- Added a Daubechies `db4` wavelet filter for high-noise environments.
- Persisted configuration in NVS across reboots.
- Split the implementation into 10 specialized modules.

---

## [1.0.0] - 2025-11-01 - Initial release

- Added CSI-based movement detection for ESP32-S3 with Hampel and Savitzky-Golay filters, 15 features, four detection states (`IDLE`, `MICRO`, `DETECTED`, and `INTENSE`), MQTT publishing, and a CLI tool.
- Supported traffic rates from 10 to 100 pps, latency below 50 ms, and a range of 3 to 8 m.
