# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0-rc1] - Unreleased - Modular multi-frontend platform

This is the first release candidate for the v3 platform. It brings the production sensing paths into one shared architecture, publishes new firmware frontends, and establishes the SDK, protocol, and tooling contracts intended for v3.0.0. Matter interoperability remains limited while controller coverage is expanded.

### Highlights

- **One sensing platform, multiple frontends**: ESPHome, Native, Matter, and Streamer now reuse the same `core`, runtime, CSI policy, detector implementations, and ESP-IDF services.
- **Native and Matter join the published firmware surface**: `release`, `preview`, and `develop` artifacts now cover ESPHome, Native, and Matter across ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. Streamer remains source-built because its Wi-Fi credentials are supplied at build time.
- **A supported embeddable SDK is available**: `espectre_sdk.h` exposes the documented public C++ surface, with version macros, optional ESP-IDF component capabilities, Doxygen reference generation, and an embedding guide.
- **The production detection profiles share one runtime contract**: Lightweight uses vote-free weighted fusion of turbulence autocorrelation and aggregated turbulence IQR, while High Accuracy uses the promoted eight-feature `8 -> 24 -> 12 -> 1` phaseless ML model. Both publish movement on a `0.0–1.0` probability scale.
- **Setup and operation are available from one toolset**: `./espectre` now covers firmware workflows, collection, and monitoring, while `espectre.dev` provides flashing, BLE configuration, MQTT live detection, the motion game, and the Wi-Fi Theremin.
- **ESPectre is dual-licensed** under GPLv3 or a separate commercial license for proprietary integrations. Contributions remain subject to the CLA and DCO checks.

### Added

- **Native firmware as a first-class frontend**, with BLE provisioning for Wi-Fi, MQTT, identity, and OTA, plus MQTT telemetry and diagnostics, Home Assistant MQTT Discovery, persisted detector selection, and HTTPS OTA support.
- **Matter occupancy firmware and release artifacts**, with per-device onboarding data generated from the device RNG and persisted in a factory partition. The same onboarding QR code is available from the web flasher, serial output, and `./espectre matter qr`.
- **Shared ESPectre Protocol services** for BLE, MQTT, provisioning, telemetry, status, device information, commands, and OTA across ESP-IDF frontends.
- **Home Assistant MQTT Discovery for Native and Micro-ESPectre**. Discovery is enabled by default in published Native firmware and remains opt-in for Micro-ESPectre. The adapter includes writable Threshold and Motion Hits numbers, a Calibrate switch, CSI traffic-control selects, the ESPHome CSI diagnostic sensors, and Refresh Diagnostics alongside Motion Detected and Movement Score.
- **ESPHome on-demand CSI diagnostics**. The diagnostics surface reports traffic, CSI callback, accepted, admitted, filtered, missing-slot, excess, stale, and out-of-order rates plus occupancy, Wi-Fi channel, and RSSI only when requested. Native MQTT Discovery and Micro-ESPectre MQTT now expose the same diagnostic sensors and Refresh Diagnostics button.
- **MQTT `commands` publishes a command catalog** on `commands/catalog`, derived from the same `supports_*` flags as `info`, so clients can populate help and completion without a local allowlist. `info` now also advertises `supports_ble` for Native MQTT `set_ble`.
- **Configurable SDK capability groups** for MQTT, BLE, provisioning, OTA, frontend support, and stream runtime when ESPectre is consumed as an ESP-IDF component.
- **Parallel detection profile inspection during collection** through `./espectre collect --detector lightweight,high_accuracy`.
- **Streamer discovery and collection improvements**, with collector-driven mDNS discovery, UDP record batching, and per-record PHY mode, LTF type, and channel-width metadata. Streamer credentials remain build-time `sdkconfig` values; the frontend does not expose a BLE control plane.
- **CSI amplitude heatmap generation** through `tools/plot_heatmap.py`.

### Changed

- **Detector choices and SDK classes now use product-facing detection profiles**: `lightweight` and `LightweightDetector` prioritize active CPU and working-memory headroom, while `high_accuracy` and `HighAccuracyDetector` prioritize detection quality, generalization, and startup without quiet-room threshold calibration. ML-specific names remain on model weights, features, training tools, and historical research evidence.
- **ESPHome now builds directly with ESP-IDF** through ESPHome 2026.7's native backend. The repository CLI explicitly selects the `esp-idf` toolchain, Native, Matter, and Streamer reuse its managed ESP-IDF installation when available, the external component registers the canonical shared sensing sources as a local ESP-IDF component, and CI plus release packaging consume native build artifacts.
- **PlatformIO integration was removed** from the repository CLI, firmware benchmarks, SDK bundles, and published install surfaces. C++ consumers now use the supported CMake source lists or vendored ESP-IDF component layout.
- **Native, Matter, and Streamer builds now select their environment automatically**: the repository CLI prefers local ESP-IDF, falls back to the pinned Docker image when local ESP-IDF is absent, asks before the first interactive image download, and exposes explicit backend and pull-policy controls for reproducible or non-interactive builds.
- **The website sitemap now omits ignored change-frequency hints** and receives source-accurate UTC `lastmod` dates from Git history and published SDK metadata during the Pages build.
- **Lightweight Detection now uses its final v3 two-feature model**: gain-invariant turbulence autocorrelation and robust IQR over `W=5` adjacent-bin aggregated turbulence feed a fixed weighted logistic fusion. The packet magnitude frame is shared across both streams; the additional state is one window-sized ring plus filter state, while the former complex frequency-coherence path is host-only. Startup calibration adapts the probability threshold to the current session and can recover once an unrepresentative startup period settles.
- **High-Accuracy Detection now uses the promoted DCT-backed subband eight-feature ML model** with 529 parameters. Guarded Kendall lag-excess joins the three existing physical-time trajectory inputs on the same eight-subband tracker, adding two 32-bit order masks per bin rather than a new history. Physical-time subband spread still replaces the full-band shape-spread history and shares that tracker, removing exactly `22,400` bytes of requested dynamic float storage at the default window (`24,720 -> 2,320` bytes, plus `104` bytes of Kendall masks). Host training, Python replay, C++ replay, and firmware inference use the same DCT-mode and pairwise-order arithmetic and exported feature order; the retired full-band feature remains available only in host research tooling. Current performance and parity evidence is published in [performance/README.md](performance/README.md); on-device CPU and peak-RAM measurements remain pending.
- **Threshold modes were removed**. Lightweight calibrates automatically at startup, High Accuracy uses its trained threshold, and runtime threshold changes apply only to the current session.
- **Motion activation now requires four consecutive evaluation hits by default**, corresponding to approximately one second at the default `250 ms` evaluation cadence.
- **Detector evaluation cadence is configured directly in milliseconds** with a `250 ms` default and advances only from packet timestamps, so confirmation timing no longer depends on the CSI packet rate and sources without usable timestamps no longer fall back to packet counting.
- **Detector windows now use fixed temporal CSI admission**: `csi_target_pps` and `segmentation_window_size_ms` define one stable slot grid, the candidate nearest each slot center is retained with a target-derived half-slot minimum spacing, a window-sized gap invalidates detector history immediately, missing slots remain invalid, and 70% occupancy is required for readiness. Arrival jitter no longer reconstructs detectors or restarts Lightweight calibration; Micro-ESPectre, collector sensing, replay, training, Python validation, and C++ integration replay reuse their production sampler, while Streamer preserves raw collector-paced transport.
- **Firmware runtime and Streamer collect benchmarks now pass on CSI occupancy at the 70% admitted-slot floor** instead of a 90–110 pps band. Occupancy is the valid fraction of the temporal detector window after admission. Matter smoke still checks boot and commissioning only.
- **ESPHome serial detection status is visible again.** ESPHome's ESP-IDF default log level is ERROR, which compiled out `ESP_LOGI` from the separate SDK component, so Home Assistant could update while `espectre monitor` saw no `IDLE | csi:` lines. The SDK now compiles INFO/DEBUG only in its own translation units and restores those `espectre` tags at runtime, without enabling Wi-Fi or lwIP debug.
- **Lightweight empty-room sequential tests allow at most one effective alarm per recording**, covering a single four-hit debounce burst after occupancy moved to 70%. High Accuracy remains zero-alarm, and two alarms on one short empty file still fail.
- **Periodic status logging now uses a monotonic `1000 ms` heartbeat** instead of counting 100 packets. Heartbeats report the packets accepted during the interval, sample diagnostics, publish zero-rate stalls in the status log, and never force detector evaluation or sensing telemetry.
- **Periodic sensing status now logs under `espectre.runtime`** from the shared runtime heartbeat, instead of each frontend's tag (`espectre`, `espectre.native`, `espectre.matter`).
- **Sensing status bars now map movement on a 0–1 scale**, with the threshold marker at the matching position, and no longer print a ratio percentage next to `mvmt` and `thr`.
- **Canonical MQTT telemetry and Home Assistant Movement Score now publish on every detector evaluation** (default `250 ms`) on Native, Micro-ESPectre, and ESPHome. Motion remains edge-published. Matter occupancy stays edge-only. `publish_interval_ms` remains the status-log heartbeat.
- **Home Assistant Intensity is removed**. Native and Micro unpublished the leftover discovery entity with an empty retained config so dashboards keep Motion Detected and Movement Score only.
- **Native and Micro Home Assistant entity IDs now match ESPHome slugs**, so the example dashboard works after replacing the `espectre_` prefix. Leftover discovery configs for the previous object IDs are unpublished with empty retained payloads. MQTT `ha/` state and command topic suffixes are unchanged.
- **The example Home Assistant dashboard is a two-column sections view**: motion state, a full-width Movement Score `bar-gauge`, a 30-minute movement-versus-threshold history, Detection Profile, Threshold, Trigger Calibration, and on-demand CSI diagnostics.
- **Native MQTT, Micro MQTT, and ESPHome now share one sensing-control family** for threshold, motion-hit debounce, recalibration, CSI traffic ownership, and traffic generator selection. Native and ESPHome persist the supported runtime settings, while Micro keeps them session-only. Sensing MQTT, Home Assistant, ESPHome, and the website expose `internal`, `external`, and `disabled` CSI traffic ownership; `pacing` is Streamer collector mode only. Native BLE does not carry those sensing writes.
- **Native BLE is setup and recovery only**: the radio starts automatically when Wi-Fi SSID or MQTT host is missing, pauses CSI while BLE is up, stays discoverable across nearby client disconnects, and stops only when `STOP_BLE` or MQTT `set_ble` with `ble=off` explicitly closes setup. Compile-time Kconfig Wi-Fi and MQTT defaults count as configured. MQTT `set_ble` or a three-second BOOT-button hold turns BLE back on for recovery. BLE exposes Wi-Fi, MQTT, identity, OTA, and read-only status. It does not publish live sensing or accept threshold, detector, traffic, or recalibrate commands.
- **Sensing now enforces one classifier-first HT20, HT-LTF, 64-subcarrier contract** across firmware, Micro-ESPectre, collection, datasets, training, and C++ replay. Format changes reset detector state, and rejected formats are exposed through diagnostics.
- **The Wi-Fi band is explicit while HT20 remains mandatory**. The validated default remains 2.4 GHz; ESP32-C5 integrations may select 5 GHz or automatic band choice, but detection quality on 5 GHz has not yet been characterized.
- **Device CSI traffic generation always uses the configured `csi_target_pps` cadence** across ESPHome, Native, Matter, and Micro-ESPectre. Occupancy-adaptive send-rate trials were removed after hardware A/B on ESP32-C3 and classic ESP32 failed to beat fixed cadence. Host `espectre collect` still slows on sustained firmware TX backpressure (15% steps, 70% floor, three-window settle, then recover toward `--pps`); occupancy remains telemetry. `--fixed` keeps a constant send rate and ignores that slowdown.
- **External UDP traffic can be unicast, or sent to the shared multicast group `239.255.0.1`.** ESP-IDF frontends join that group in `external` and `pacing`. Subnet and limited broadcast do not produce reliable HT20 CSI. ESPHome, Native, and Matter `external` mode listen on port `5555`. Streamer collection can pace several devices with `./espectre collect --target 239.255.0.1`.
- **Wi-Fi and CSI startup policy is shared across ESP-IDF frontends**: association, HT20 policy, CSI initialization, and traffic startup now follow one lifecycle.
- **Matter defaults to Lightweight Detection** and is documented as available with limited controller validation. ESPHome remains the most extensively validated Home Assistant integration.
- **Hardware gain locking was removed**. All maintained device paths keep AGC enabled and use scale-invariant detector inputs.
- **Published firmware separates installation and update artifacts**: releases provide 15 full-flash images for ESPHome, Native, and Matter, plus five application-only OTA images for Native.
- **ML training moved from TensorFlow/Keras to PyTorch** and now separates training data, model-selection replays, sealed holdouts, excluded recordings, and low-RSSI stress diagnostics. Training exports by default; use `--no-export` for candidate-only experiments. High Accuracy promotion now also requires occupancy-70% thinned reserved paired and quiet replays, keeping the production readiness floor, in addition to the clean reserved gates. The promoted `base` recipe uses packet-rate scale `0.7-1.0` with a 70 pps floor.
- **ML augmentation caching now keeps only the selected mixed rows**, caches lightweight source-admission metadata separately, avoids persisting complete intermediate seed views, and supports explicit age- and size-based cache pruning.
- **Host-side feature caching is now column-granular**: replay coordinates are stored once, each feature owns an independently versioned column, and adding a variant no longer invalidates sibling columns used by seed searches or model comparisons. Cold cache producers use per-key process locks, and augmented host views reuse the same granular artifacts. Long fills emit periodic `[npz-cache]` progress on stderr for hits, misses, in-progress builds, and writes.
- **Dataset validation is detector-independent** and evaluates shared scale-invariant feature evidence. Detector-specific promotion results remain in the performance report.
- **The dataset quality report now flags excluded idle captures** that produce no usable feature rows after temporal admission, listing them first with `n/a ⚠️`.
- **The generated detector performance report now publishes only reserved `selection + holdout` evidence** for both Lightweight and High Accuracy. Training-role recordings remain regression-tested but no longer appear in detector tables, input counts, or replay work.
- **The documentation and website now follow the modular platform structure**, with task-oriented setup, detection, hardware, embedded integration, and use case guides.
- **Website analytics now separates intent, transport connection, first valid data, and verified outcomes**. A shared route registry keeps SPA navigation and page-view metadata aligned, guide and documentation analytics derive stable parameters from route conventions, configuration writes are confirmed through sysinfo before reporting success, OTA and SDK outcomes have dedicated events, MQTT and BLE durations preserve their original entry point, and interrupted game sessions are explicit.
- **The website Flash tool detects the chip over USB** and no longer asks operators to pick a chip family. Connecting a chip without a published image for the selected firmware and channel reports a clear unsupported-board error that lists the chips with available binaries. Per-chip downloads live in the firmware panel. The page subtitle states that USB flashing requires desktop Chrome or Edge, and unsupported browsers see an explicit error with Install disabled. The next-step note follows the selected frontend: Native points to Configure and Monitor, ESPHome to the Wi-Fi setup methods in the setup guide, and Matter to controller commissioning plus an in-note USB QR reader.
- **The website now provides separate Configure and Monitor tools**. Configure provisions Wi-Fi, MQTT, and the device label over Bluetooth. Start sensing opens Monitor, connects over WebSockets, waits for MQTT `set_ble off` to be accepted (and sends `STOP_BLE` when a nearby session is still open), then shows live sensing after valid device telemetry arrives. Edit connectivity on Monitor publishes MQTT `set_ble on` and opens Configure plus the browser Bluetooth picker as soon as the device accepts. Live sensing and runtime inputs share one status surface, changes apply directly on field change, and diagnostics remain collapsed below live sensing and refresh once per second while that section is open, with command-catalog capability gating and accepted/rejected command acknowledgements. The device banner subtitle labels chip, device ID, and firmware version. A silent update check runs over Bluetooth after Configure connects and again over MQTT when Monitor goes live, so users who never open live sensing still see Latest, an Update action, or the check error. Connect with MQTT carries the browser WebSocket port, path, and TLS next to broker identity; Monitor copies host, credentials, topic prefix, and device ID from the device MQTT settings. Leave Device ID empty to scan `info` and `status` like `./espectre mqtt`: a single device is selected automatically, and several devices open a picker.
- **Monitor's movement chart now shows a five-minute time window** instead of a 120-sample buffer sized for 1 Hz publishes. The X axis is elapsed time on a fixed 0–1 score scale, so the default `250 ms` evaluation cadence no longer races the trace off-screen; Home Assistant movement and motion mirrors do not add extra samples while canonical telemetry is live.
- **Configure no longer shows a diagnostics or sensing-control surface**. The device banner and header dropdown label chip, device ID, and firmware version instead of the Native frontend name. Runtime sensing controls and diagnostics are available on Monitor only after MQTT is live.
- **Firmware, OTA, and SDK version labels now use `git describe`**. Channel names stay `preview` and `develop`; the GitHub release tags are `snapshot` and `snapshot-dev`. The version string baked into firmware, `ESPECTRE_SDK_VERSION_STRING`, OTA manifests, and the website is the numeric-tag describe identity (currently `2.8.0-237-g…` until a 3.x tag exists). OTA availability is then a same-string comparison. First-party CMake configure fails without git tags; published SDK bundles stamp the version into the header instead of falling back to a hardcoded `3.0.0`.

### Removed

- **The repository CLI no longer serves the website.** Browser tools live on [espectre.dev](https://espectre.dev); local previews use the website tree under `docs/web`. The MQTT shell `webui` and `web` utilities are removed with the same change.

### Fixed

- **MQTT discovery now shows the firmware currently on the device.** Native and Micro retain canonical `info` on connect and after an `info` command, so late subscribers and `./espectre mqtt` replace leftover retained identities. Older Streamer builds used to publish MQTT `info` with `frontend: streamer`; Streamer no longer uses MQTT, and those broker messages stayed until a sensing frontend overwrote them.
- **Native HTTPS OTA no longer fails with `ESP_FAIL` on GitHub Releases redirects.** GitHub's 302 responses include multi-kilobyte headers that overflowed the ESP-IDF default 512-byte HTTP buffer (`HTTP_CLIENT: Out of buffer`). The client now uses an 8 KiB receive buffer, and check, download, and failure progress is logged under `espectre.ota`.
- **Native BLE now treats compile-time Kconfig Wi-Fi and MQTT defaults as configured**, so lab images skip BLE at boot and leave CSI armed. Previously only NVS-provisioned Wi-Fi counted, which kept BLE in setup mode after an NVS erase even when `ESPECTRE_WIFI_SSID` and `ESPECTRE_MQTT_HOST` were baked in.
- **Native detection now resumes after BLE setup stops**. Disarming CSI no longer forgets the current Wi-Fi IP, so Stop BLE or MQTT `set_ble off` restarts capture without waiting for another GOT_IP event.
- **Monitor now clears the live chart, movement score, and diagnostics when the device ID changes**, and drops the previous MQTT session instead of keeping the old device's telemetry on screen.
- **ESPHome GitHub clones resolve the SDK version from `esphome.project.version` or `ESPECTRE_GIT_VERSION`**. ESPHome fetches a single commit and has no numeric tags, so `git describe` cannot run in that worktree. First-party CI already passes the git-describe identity as `project_version`.
- **ESPHome firmware assets now include the frontend name in published filenames**, aligning them with Native and Matter as `espectre-esphome-<channel-or-version>-<chip>.bin`. Existing direct GitHub asset URLs that omit `esphome` no longer match newly published files.
- **The OTA update dialog now completes when the device comes back online.** Monitor already receives retained `status`, `info`, and `ota/state`; after `reboot_scheduled` it waits for the next `online: true` or idle OTA snapshot, updates the current firmware version, closes the modal, and runs a silent update check. The previous dialog stayed on `Starting firmware update…` / `reboot_scheduled` until dismissed by hand.
- **Monitor and the header meter subscribe to the complete per-device MQTT topic tree**, including canonical telemetry, command acknowledgements, command catalog, info, OTA state, diagnostics, and Home Assistant state mirrors.
- **MQTT `info` and `stats` now share one schema across Native and Micro**. `info` includes CSI traffic ownership, generator mode, and target PPS when the frontend owns those settings. `stats` includes the CSI and Wi-Fi diagnostic fields from the cached periodic sample.
- **The interactive MQTT shell now forwards protocol commands to the device** instead of maintaining a local allowlist, annotates typed commands with `✓` or `✗` plus the reject reason, dumps `info`, `stats`, OTA, and command-catalog payloads instead of the raw command-ACK YAML, and builds help and tab completion from MQTT `commands`.
- **Native Home Assistant entities no longer remain unavailable after MQTT Discovery** because canonical online/offline status updates and the broker-published Last Will are retained for late availability subscribers.
- **Native Home Assistant Detection Profile now appears under Configuration** with the other runtime settings. MQTT Discovery publishes `entity_category: config`, matching ESPHome.
- **Home Assistant Configuration now lists CSI Traffic Ownership immediately before CSI Traffic Source**, with Trigger Calibration last. Discovery publishes those entities in that order, and the friendly names sort that way on the device page.
- **Firmware and SDK channels are now `release`, `preview`, and `develop`**. Official tagged builds use `release`. Rolling builds from `main` use channel `preview` on GitHub tag `snapshot`. Rolling builds from `develop` use channel `develop` on GitHub tag `snapshot-dev`. The browser flasher and SDK website pages expose all three channels.
- **MQTT and BLE OTA accept an optional `channel`**. `ota_check` and `ota_start` take `release`, `preview`, or `develop` and resolve the matching GitHub Releases manifest (`latest`, tag `snapshot`, or tag `snapshot-dev`). Omitting `channel` keeps the firmware's build-time default. Manifest, image, and version URL overrides remain rejected. The website always sends an explicit channel, defaulting to `release` even when the firmware was built for `preview` or `develop`. The connection menu and device banner show OTA status beside the firmware version and open the update dialog on click. Demo sessions keep Update device disabled.
- **Release and rolling publication now fail closed on stale or unvalidated sources**, reuse one verified Pages build, publish rolling tags atomically, and produce reproducible SDK archives with SHA-256 digests.
- **Wi-Fi channel changes no longer leave sensing or streaming in a stale CSI session**. Frontends now invalidate the session, reset the active detector or Streamer transport, and rearm capture outside the Wi-Fi callback.
- **Native Wi-Fi can reassociate correctly after BLE coexistence or protocol renegotiation**, including after a station stop event.
- **Native CSI occupancy can recover after BLE provisioning** because the Bluetooth controller is powered down during sensing unless MQTT `set_ble` requests setup mode again.
- **ESP-IDF frontends no longer attempt the unsupported 802.11n-only protocol configuration** before applying the shared Wi-Fi policy.
- **C++ and Python ML inference no longer diverge near the decision threshold** because compiler-dependent fused multiply-add contraction is disabled for the exported inference path.
- **Lightweight startup calibration can recover from a noisy or otherwise unrepresentative opening period** without requiring a reboot or manual threshold mode.
- **Streamer long-session handling is more resilient**, with PSRAM-backed staging where available, improved retry and duplicate telemetry, BLE suspension during sustained streaming, and chip-specific transport defaults.
- **C++ and Python detector replays now follow the same timing, cadence, calibration, and state-transition behavior**.
- **Python Lightweight report replay no longer scores occupancy holes as idle evaluations**, matching the C++ parity suites so missing slots are not counted as false negatives.
- **C++ Lightweight report calibration no longer spends the startup budget on occupancy holes**, matching firmware and Python so paired and quiet-room Lightweight metrics stay on the same threshold.
- **CPython High Accuracy inference now rounds like firmware ``float`` arithmetic**, closing residual Python/C++ decision flips near the 0.5 probability boundary.
- **Paired High Accuracy report replay now keeps both phases on the baseline temporal grid**, matching the C++ parity suites when the motion capture infers a slightly different packet rate.
- **Host training and collector live detectors now honor the same missing-slot and timestamp contract as firmware**: turbulence statistics skip invalid slots, trajectory bins use the admitted packet timestamp, and stale host feature columns are invalidated.
- **Lightweight settled-level recovery was recalibrated for temporal windows**, restoring the weak-link S3 recall floor without increasing the measured normal-link or quiet-room false-positive tails.
- **Cache and generated-artifact publication now fails safely under overlap or interruption**: nested provenance parameters are no longer dropped from feature-index identities, generated reports track implementation and capture revisions, related model outputs publish as a rollback-capable set, and seed-search rollback removes artifacts that were absent before the search.

### Breaking changes and migration

- **Detector identifiers and class names changed without compatibility aliases**. Replace `mvs` with `lightweight` and `ml` with `high_accuracy` in ESPHome YAML, Micro-ESPectre configuration, CLI arguments, BLE and MQTT commands, persisted selections, and protocol consumers. C++ users of the 2.8.0 ESPHome component must replace `DetectionAlgorithm::MVS`, `DetectionAlgorithm::ML`, `MVSDetector`, and `MLDetector` with `DetectionAlgorithm::LIGHTWEIGHT`, `DetectionAlgorithm::HIGH_ACCURACY`, `LightweightDetector`, and `HighAccuracyDetector`, respectively.
- **The command wrapper is now `./espectre`**. The previous `./me` name was removed, host commands such as `collect`, `ui`, and `mqtt` moved to the top level, and `micro` now contains only MicroPython device operations.
- **`collect` no longer saves without an explicit `--label`**. Omitting the label starts inspection-only mode.
- **Movement and threshold integrations must use the shared `0.0–1.0` probability scale**. The previous MVS and ML `0–10` amplitude assumptions are no longer supported.
- **ESP32-S2 support was removed** because it had no recorded hardware validation. The supported firmware targets are ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6.
- **The ESPHome baseline is now `2026.7.0`**, and the host and ML workflows now require Python `3.14`. PlatformIO-backed ESPHome builds and the former PlatformIO SDK manifest are no longer supported; use ESPHome's native ESP-IDF backend or the published CMake and ESP-IDF component surfaces.
- **Checked-in and collected dataset metadata use format `1.2`**. Consumers of 2.8.0 `.npz` metadata must migrate.
- **C++ integrators should include the supported SDK facade** through `espectre_sdk.h` and follow the v3 `core -> runtime -> frontend` dependency direction. The 2.8.0 `components/espectre/` layout is not a stable compatibility surface.
- **Window, evaluation, and publish timing now use milliseconds**. Migrate `segmentation_window_size` to `segmentation_window_size_ms` (default `1000`), `evaluation_interval` to `evaluation_interval_ms` (the legacy `25`-packet setting becomes timestamp-driven `250 ms`), and `publish_interval` to `publish_interval_ms` (the default changes from 100 packets to `1000 ms`).
- **CSI target configuration is now `csi_target_pps`**. Replace `traffic_generator_rate` with `csi_target_pps` plus an explicit `csi_traffic_mode`. A rate of zero is invalid and no longer disables traffic; use `csi_traffic_mode: external` or `disabled` instead.
- **Hardware gain locking was removed**. Drop `gain_lock` from ESPHome YAML and Micro-ESPectre configuration; all maintained device paths keep AGC enabled.
- **Threshold modes were removed**. Lightweight calibrates automatically at startup, High Accuracy uses its trained threshold, and runtime threshold changes apply only to the current session. Replace 2.8.0 `segmentation_threshold: auto` / `min` YAML with the shared runtime threshold controls.

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
