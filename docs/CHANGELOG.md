# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0-rc1] - Unreleased - Modular multi-frontend platform

This is the first release candidate for the v3 platform. It brings the production sensing paths into one shared architecture, publishes new firmware frontends, and establishes the SDK, protocol, and tooling contracts intended for v3.0.0. Matter interoperability remains limited while controller coverage is expanded.

### Highlights

- **One sensing platform, multiple frontends**: ESPHome, Native, Matter, and Streamer now reuse the same `core`, runtime, CSI policy, detector implementations, and ESP-IDF services.
- **Native and Matter join the published firmware surface**: release and snapshot artifacts now cover ESPHome, Native, and Matter across ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. Streamer remains source-built because its Wi-Fi credentials are supplied at build time.
- **A supported embeddable SDK is available**: `espectre_sdk.h` exposes the documented public C++ surface, with version macros, optional ESP-IDF component capabilities, Doxygen reference generation, and an embedding guide.
- **The production detectors share one runtime contract**: Classic uses vote-free weighted fusion of turbulence autocorrelation and aggregated turbulence IQR, while ML uses the promoted seven-feature `7 -> 24 -> 12 -> 1` phaseless model. Both publish movement on a `0.0–1.0` probability scale.
- **Setup and operation are available from one toolset**: `./espectre` now covers firmware workflows, collection, monitoring, and local browser tools, while `espectre.dev` provides flashing, BLE configuration, MQTT monitoring, the motion game, and the Wi-Fi Theremin.
- **ESPectre is dual-licensed** under GPLv3 or a separate commercial license for proprietary integrations. Contributions remain subject to the CLA and DCO checks.

### Added

- **Native firmware as a first-class frontend**, with BLE provisioning and control, MQTT telemetry and diagnostics, Home Assistant MQTT Discovery, persisted detector selection, and HTTPS OTA support.
- **Matter occupancy firmware and release artifacts**, with per-device onboarding data generated from the device RNG and persisted in a factory partition. The same onboarding QR code is available from the web flasher, serial output, and `./espectre matter qr`.
- **Shared ESPectre Protocol services** for BLE, MQTT, provisioning, telemetry, status, device information, commands, and OTA across ESP-IDF frontends.
- **Home Assistant MQTT Discovery for Native and Micro-ESPectre**. Discovery is enabled by default in published Native firmware and remains opt-in for Micro-ESPectre.
- **ESPHome intensity and on-demand CSI diagnostics**. The new intensity sensor stays meaningful across automatic Classic calibration, while the diagnostics surface reports traffic, CSI callback, accepted and filtered packet rates, Wi-Fi channel, and RSSI only when requested.
- **Native on-demand MQTT diagnostics** for traffic and CSI rates, Wi-Fi channel, and RSSI through the `stats` command.
- **Configurable SDK capability groups** for MQTT, BLE, provisioning, OTA, frontend support, and stream runtime when ESPectre is consumed as an ESP-IDF component.
- **Parallel detector inspection during collection** through `./espectre collect --detector classic,ml`.
- **Streamer discovery and collection improvements**, with collector-driven mDNS discovery, UDP record batching, and per-record PHY mode, LTF type, and channel-width metadata. Streamer credentials remain build-time `sdkconfig` values; the frontend does not expose a BLE control plane.
- **CSI amplitude heatmap generation** through `tools/plot_heatmap.py`.

### Changed

- **ESPHome now builds directly with ESP-IDF** through ESPHome 2026.7's native backend. The external component registers the canonical shared sensing sources as a local ESP-IDF component, examples no longer pin an alternate toolchain, and CI plus release packaging consume native build artifacts.
- **PlatformIO integration was removed** from the repository CLI, firmware benchmarks, SDK bundles, and published install surfaces. C++ consumers now use the supported CMake source lists or vendored ESP-IDF component layout.
- **Native, Matter, and Streamer builds now select their environment automatically**: the repository CLI prefers local ESP-IDF, falls back to the pinned Docker image when local ESP-IDF is absent, asks before the first interactive image download, and exposes explicit backend and pull-policy controls for reproducible or non-interactive builds.
- **The website sitemap now omits ignored change-frequency hints** and receives source-accurate `lastmod` dates from Git history and published SDK metadata during the Pages build.
- **Classic detection now uses its final v3 two-feature model**: gain-invariant turbulence autocorrelation and robust IQR over `W=5` adjacent-bin aggregated turbulence feed a fixed weighted logistic fusion. The packet magnitude frame is shared across both streams; the additional state is one window-sized ring plus filter state, while the former complex frequency-coherence path is host-only. Startup calibration adapts the probability threshold to the current session and can recover once an unrepresentative startup period settles.
- **ML detection now uses the promoted DCT-backed subband seven-feature model** with 505 parameters. Physical-time subband spread replaces the full-band shape-spread history and shares the trajectory tracker already used by coherent innovation and excess path, removing exactly `22,400` bytes of requested dynamic float storage at the default window (`24,720 -> 2,320` bytes). Host training, Python replay, C++ replay, and firmware inference use the same DCT-mode arithmetic and exported feature order; the retired full-band feature remains available only in host research tooling. Current performance and parity evidence is published in [performance/README.md](performance/README.md); on-device CPU and peak-RAM measurements remain pending.
- **Threshold modes were removed**. Classic calibrates automatically at startup, ML uses its trained threshold, and runtime threshold changes apply only to the current session.
- **Motion activation now requires four consecutive evaluation hits by default**, corresponding to approximately one second at the default `250 ms` evaluation cadence.
- **Detector evaluation cadence is configured directly in milliseconds** with a `250 ms` default and advances only from packet timestamps, so confirmation timing no longer depends on the CSI packet rate and sources without usable timestamps no longer fall back to packet counting.
- **Detector windows are configured directly in milliseconds** through `segmentation_window_size_ms`, with a `1000 ms` default resolved from measured CSI cadence across firmware, ESPHome, Micro-ESPectre, replay, validation, and training. The augmented ML path now trains stable lower-rate windows, and live detection stays on hold below the supported `80 pps` floor.
- **Periodic telemetry publishing now uses a monotonic `1000 ms` heartbeat** instead of counting 100 packets. Heartbeats report the packets accepted during the interval, publish zero-rate stalls, and never force detector evaluation.
- **Sensing now enforces one classifier-first HT20, HT-LTF, 64-subcarrier contract** across firmware, Micro-ESPectre, collection, datasets, training, and C++ replay. Format changes reset detector state, and rejected formats are exposed through diagnostics.
- **The Wi-Fi band is explicit while HT20 remains mandatory**. The validated default remains 2.4 GHz; ESP32-C5 integrations may select 5 GHz or automatic band choice, but detection quality on 5 GHz has not yet been characterized.
- **CSI traffic generation is adaptive by default** across ESPHome, Native, Matter, Micro-ESPectre, and host collection. It responds to sustained local backpressure and CSI delivery feedback; fixed host pacing remains available through `--fixed`.
- **Wi-Fi and CSI startup policy is shared across ESP-IDF frontends**: association, HT20 policy, CSI initialization, and traffic startup now follow one lifecycle.
- **Matter defaults to the Classic detector** and is documented as available with limited controller validation. ESPHome remains the most extensively validated Home Assistant integration.
- **Hardware gain locking was removed**. All maintained device paths keep AGC enabled and use scale-invariant detector inputs.
- **Published firmware separates installation and update artifacts**: releases provide 15 full-flash images for ESPHome, Native, and Matter, plus five application-only OTA images for Native.
- **ML training moved from TensorFlow/Keras to PyTorch** and now separates training data, model-selection replays, sealed holdouts, excluded recordings, and low-RSSI stress diagnostics. Training exports by default; use `--no-export` for candidate-only experiments.
- **ML augmentation caching now keeps only the selected mixed rows**, caches lightweight source-admission metadata separately, avoids persisting complete intermediate seed views, and supports explicit age- and size-based cache pruning.
- **Host-side feature caching is now column-granular**: replay coordinates are stored once, each feature owns an independently versioned column, and adding a variant no longer invalidates sibling columns used by seed searches or model comparisons. Cold cache producers use per-key process locks, and augmented host views reuse the same granular artifacts.
- **Dataset validation is detector-independent** and evaluates shared scale-invariant feature evidence. Detector-specific promotion results remain in the performance report.
- **The generated detector performance report now publishes only reserved `selection + holdout` evidence** for both Classic and ML. Training-role recordings remain regression-tested but no longer appear in detector tables, input counts, or replay work.
- **The documentation and website now follow the modular platform structure**, with task-oriented setup, detection, hardware, embedded integration, and use case guides.
- **Website analytics now separates intent, transport connection, first valid data, and verified outcomes**. SPA routes update the Google tag state before manual page views, configuration writes are confirmed through sysinfo before reporting success, OTA and SDK outcomes have dedicated events, MQTT and BLE durations preserve their original entry point, and interrupted game sessions are explicit.

### Fixed

- **Release and snapshot publication now fail closed on stale or unvalidated sources**, reuse one verified Pages build, publish rolling tags atomically, and produce reproducible SDK archives with SHA-256 digests.
- **Wi-Fi channel changes no longer leave sensing or streaming in a stale CSI session**. Frontends now invalidate the session, reset the active detector or Streamer transport, and rearm capture outside the Wi-Fi callback.
- **Native Wi-Fi can reassociate correctly after BLE coexistence or protocol renegotiation**, including after a station stop event.
- **ESP-IDF frontends no longer attempt the unsupported 802.11n-only protocol configuration** before applying the shared Wi-Fi policy.
- **C++ and Python ML inference no longer diverge near the decision threshold** because compiler-dependent fused multiply-add contraction is disabled for the exported inference path.
- **Classic startup calibration can recover from a noisy or otherwise unrepresentative opening period** without requiring a reboot or manual threshold mode.
- **Adaptive traffic pacing no longer over-corrects under temporary CSI surplus or isolated callback deficits**, improving sustained sensing and Streamer collection on the original ESP32.
- **Streamer long-session handling is more resilient**, with PSRAM-backed staging where available, improved retry and duplicate telemetry, BLE suspension during sustained streaming, and chip-specific transport defaults.
- **C++ and Python detector replays now follow the same timing, cadence, calibration, and state-transition behavior**.
- **Classic settled-level recovery was recalibrated for temporal windows**, restoring the weak-link S3 recall floor without increasing the measured normal-link or quiet-room false-positive tails.
- **Cache and generated-artifact publication now fails safely under overlap or interruption**: nested provenance parameters are no longer dropped from feature-index identities, generated reports track implementation and capture revisions, related model outputs publish as a rollback-capable set, and seed-search rollback removes artifacts that were absent before the search.

### Breaking changes and migration

- **The command wrapper is now `./espectre`**. The previous `./me` name was removed, host commands such as `collect`, `ui`, `mqtt`, and `monitor` moved to the top level, and `micro` now contains only MicroPython device operations.
- **`collect` no longer saves without an explicit `--label`**. Omitting the label starts inspection-only mode, and the former `--no-save` option was removed.
- **`monitor` no longer resets the device when attaching**. Pass `--reset` when boot-time output or a clean restart is required.
- **Movement and threshold integrations must use the shared `0.0–1.0` probability scale**. The previous Classic `0–10` amplitude assumptions are no longer supported.
- **Direct `ClassicDetector` integrations must use the aggregated-IQR surface**. The C++ coherence-lag constructor argument and `get_chan_freq_coh_curve_std()` accessor were removed; use the three-argument constructor ending in `autocorr_lag` and `get_turb_iqr_over_mean_aggr()`.
- **ESP32-S2 support was removed** because it had no recorded hardware validation. The supported firmware targets are ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6.
- **The ESPHome baseline is now `2026.7.0`**, and the host and ML workflows now require Python `3.14`. PlatformIO-backed ESPHome builds and the former PlatformIO SDK manifest are no longer supported; use ESPHome's native ESP-IDF backend or the published CMake and ESP-IDF component surfaces.
- **Streamer and checked-in dataset metadata use a clean-break format**: Streamer protocol V7 carries per-record PHY metadata, and repository datasets use dataset format `1.2`. Consumers of the previous wire or dataset metadata must migrate.
- **C++ integrators should include the supported SDK facade** through `espectre_sdk.h` and follow the v3 `core -> runtime -> frontend` dependency direction. Internal v2 source paths and generic header names are not stable compatibility surfaces.
- **Evaluation cadence configuration now uses milliseconds**. Migrate `evaluation_interval`, `EVALUATION_INTERVAL`, and `CONFIG_ESPECTRE_EVALUATION_INTERVAL` to `evaluation_interval_ms`, `EVALUATION_INTERVAL_MS`, and `CONFIG_ESPECTRE_EVALUATION_INTERVAL_MS`, respectively. The legacy `25`-packet setting is replaced by a timestamp-driven `250 ms` interval with no packet-count fallback.
- **Periodic publish configuration now uses milliseconds**. Migrate `publish_interval`, `PUBLISH_INTERVAL`, and `CONFIG_ESPECTRE_PUBLISH_INTERVAL` to `publish_interval_ms`, `PUBLISH_INTERVAL_MS`, and `CONFIG_ESPECTRE_PUBLISH_INTERVAL_MS`, respectively; the default changes from 100 packets to `1000 ms`.
- **Legacy detector tooling was removed**, including threshold modes, the moving-variance baseline cluster, stale notebooks, the ML `--promote` flag, and detector-guided sample-weight options.

Detailed detector design, feature decisions, integration guidance, and validation results are maintained in [ALGORITHMS.md](ALGORITHMS.md), [FEATURES.md](FEATURES.md), [EMBEDDING.md](EMBEDDING.md), and [performance/README.md](performance/README.md).

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
