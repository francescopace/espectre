# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0-rc1] - Unreleased - Modular multi-frontend platform

This first release candidate moves production sensing onto the shared v3 architecture. It introduces the Native and Matter frontends, a common protocol and SDK, browser-based device tools, and one raw CSI collection path. Matter controller validation remains limited in this release candidate.

### Highlights

- **ESPHome, Native, and Matter share one layered production architecture**. The `core`, `runtime`, and `frontend` layers keep detector implementations, CSI policy, runtime contracts, and ESP-IDF services independent from any one frontend. Native adds a standalone BLE-free firmware with Direct HTTP and optional MQTT. Matter publishes standard occupancy, generates per-device onboarding data, and exposes the persisted QR code through serial output, the web flasher, and `./espectre matter qr`; controller validation remains limited.
- **Lightweight and High Accuracy provide two production detection profiles**. Lightweight uses a low-cost two-feature weighted model with startup calibration. High Accuracy uses an eight-feature phaseless neural model and does not require quiet-room threshold calibration. All production detector features are gain- and scale-invariant, so maintained firmware keeps AGC enabled instead of locking hardware gain.
- **Temporal CSI sampling replaces packet-count windows**. Packet timestamps define a stable slot grid that retains the candidate nearest each slot and leaves missing slots empty. Traffic generators avoid catch-up bursts, ping remains the default source, and detector resets do not rephase the sampling grid.
- **One versioned protocol covers Direct HTTP and optional MQTT**. ESPectre frontends expose Direct HTTP on port `62587`, advertise `_espectre._tcp.local.`, and enforce the methods declared by their capability profiles. Supported detector, motion-hit, and traffic selections persist across reboot on the ESP-IDF frontends, while thresholds remain session-only. ESPHome republishes entity state after Direct writes, and Matter uses Direct for controls that its standard occupancy clusters do not expose.
- **Raw CSI collection uses HTTP v2 across supported ESPectre frontends**. `./espectre collect` discovers raw-capable devices, selects external traffic, drives the shared UDP marker generator, and saves CSI V8 records with PHY and device provenance. HTTP does not pace or decimate records, and bounded firmware queues report drops explicitly. Post-capture validation warns below 85% temporal occupancy and fails below 70%.
- **ML corpus curation and promotion keep selection and holdout distinct**. New captures require explicit environment and dataset-role metadata before validation. `train_ml_model.py --evaluate-selection` runs candidate deployment gates without opening holdout, while `--evaluate-gates` remains the final read-only check for a fixed candidate.
- **The `./espectre` CLI covers the supported firmware and device workflows**. It builds, flashes, deploys, monitors, diagnoses, discovers devices, sends Direct requests, collects CSI, and opens the interactive MQTT client from one command surface. Discovery uses the first-party `_espectre._tcp.local.` record rather than frontend-specific ESPHome or Matter services.
- **The [espectre.dev](https://espectre.dev) site provides browser-based flashing and device tools**. It provisions Native and ESPHome over standard Improv Serial, negotiates each device's Direct capabilities, configures and monitors supported C++ frontends, manages Native MQTT and OTA settings, reads Matter onboarding data, and hosts the motion game and Wi-Fi Theremin. ESPHome images retain their fallback access point but no longer include BLE provisioning.
- **The hosted portal discovers ESPectre devices on the LAN without an extension, cloud relay, or address-range scan**. Each attempt resolves a fresh `.local` hostname containing a 96-bit nonce, so cached positive or negative results from earlier attempts cannot satisfy the lookup. An eligible ESPectre performs the bounded `_espectre._tcp.local.` browse and returns validated endpoints. Discovery requires IPv4, working mDNS, and browser Local Network Access.
- **Micro-ESPectre is a small, read-only sensing frontend**. The deployed application uses Lightweight Detector, the native ICMP generator, five Direct HTTP queries, one telemetry SSE stream, mDNS discovery, and serial logging. The High Accuracy ML model remains available in Python for host research and C++/Python validation but is not copied to the device.
- **The embeddable C++ SDK has a documented public surface**. `espectre_sdk.h` exposes the runtime API, while `espectre_core_sdk.h` exposes lower-level detectors for custom CSI integrations. The embedding contract covers source compatibility, capabilities, ownership, threading, errors, optional components, and the absence of a stable binary ABI. Raw CSI callbacks have an explicit bounded, non-blocking, allocation-free contract, and OTA quiescing stops an active raw session safely.
- **ESPectre is dual-licensed and publishes release compliance artifacts**. The project is available under GPLv3 or a separate commercial license, with contributions subject to the CLA and DCO checks. Releases provide factory and OTA images for ESPHome and Native, factory images for Matter, SDK bundles, and `firmware-compliance-<channel-or-version>.zip`. Native OTA accepts only a strictly newer release, prerelease, or rolling `git describe` identity.

### Breaking changes and migration

- **Detector identifiers, C++ names, and the metric scale changed without compatibility aliases**. Replace `mvs` with `lightweight`, `ml` with `high_accuracy`, `MVSDetector` with `LightweightDetector`, `MLDetector` with `HighAccuracyDetector`, and the corresponding `DetectionAlgorithm::MVS` / `ML` values with `LIGHTWEIGHT` / `HIGH_ACCURACY`. Update movement and threshold integrations from the former `0–10` assumptions to the shared `0.0–1.0` probability scale.
- **The repository command wrapper is now `./espectre`**. Run MicroPython commands under `./espectre micro`, and run `collect` and `mqtt` from the repository root. Replace the former `micro-espectre/me` `ui`, `detect`, `stream`, and collection workflows with the browser tools, the MQTT client, or `./espectre collect` as appropriate. Raw collection now uses HTTP v2, and `--pps` controls the external generator.
- **ESPHome configuration now follows the temporal sampler**. Replace `segmentation_window_size`, `evaluation_interval`, and `traffic_generator_rate` with `segmentation_window_size_ms`, `evaluation_interval_ms`, a positive `csi_target_pps`, and an explicit `csi_traffic_mode`. Drop `segmentation_threshold`, `gain_lock`, `selected_subcarriers`, the configurable publish interval, and the legacy BLE channel settings `ble_channel_enabled`, `ble_server_id`, `ble_control_char_id`, `ble_sysinfo_char_id`, `ble_telemetry_char_id`, and `ble_telemetry_interval_ms`. Use Improv Serial for USB provisioning.
- **Micro-ESPectre is now read-only at runtime**. MQTT transport and commands, device-side High Accuracy deployment, and the UDP CSI streamer are removed. Flash the matching project firmware once, keep Wi-Fi settings in `config_local.py`, use `micro deploy` for `.mpy -O3` application updates, and monitor through Direct HTTP, SSE, or serial output.
- **The supported build and hardware baseline changed**. ESPHome requires at least `2026.7.0`, host and ML workflows require Python `3.14`, ESP-IDF integrations require ESP-IDF `>= 5.5`, and PlatformIO-backed builds are no longer supported. ESP32-S2 is supported by ESPHome, Native, and Micro-ESPectre, but not Matter.
- **Dataset metadata moved from format `1.0` to `1.2`**. Consumers of 2.8.0 `.npz` captures must migrate to the versioned metadata schema.
- **C++ integrations must move to the supported SDK facade**. Include `espectre_sdk.h`, follow the `core -> runtime -> frontend` dependency direction, and update code that depends on the former `components/espectre/` layout or namespace.
- **ESPHome examples and firmware names changed**. Replace repository paths beginning with `examples/` with `src/cpp/frontend/esphome/examples/`, including package URLs that reference an example YAML file. Full-flash images now use `espectre-esphome-<channel-or-version>-<chip>.bin`, and OTA images use `espectre-esphome-<channel-or-version>-<chip>-ota.bin`.

For detector design, feature decisions, integration guidance, and validation results, see [ALGORITHMS.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/ALGORITHMS.md), [FEATURES.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/FEATURES.md), [SDK.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/SDK.md), and [performance/README.md](https://github.com/francescopace/espectre/blob/3.0.0-rc1/docs/performance/README.md).

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
