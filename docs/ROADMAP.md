# Roadmap

## Releases

| Version | Date | Status | Summary |
|---------|------|--------|---------|
| **v1.x** | 2025-11-09 | Released | First release demonstrating motion detection capabilities using a brand-new algorithm |
| **v2.x** | 2025-12-06 | Released | Home Assistant integration via ESPHome plus custom MicroPython-based firmware |
| **v3.x** | 2026-08 (target) | In progress | New detectors based on spectral features. Add Matter support with limited controller validation, native BLE/MQTT firmware, and an SDK-oriented foundation for OEM integrations |
| **v4.x** | 2026-12 (target) | Planned | Privacy-first web orchestration layer for multi-node sensing, secure onboarding, fleet visibility, history, alerting, and remote management |
| **v5.x** | Future | Exploratory | Standards-ready sensing platform prepared for practical IEEE 802.11bf / Wi-Fi Sensing hardware support when embedded vendors expose it |

---

## v3.x - Modular Sensing Platform

**Goal**: move from a single integration-focused firmware to a reusable Wi-Fi
sensing platform with shared sensing logic, a stable runtime contract, multiple
frontend paths, and an embeddable foundation for custom firmware and OEM products.

### Contains

| Area | Scope |
|------|-------|
| **Architecture** | Shared `core`, `runtime`, and `frontend` layers |
| **Runtime contract** | Stable frontend-oriented APIs such as `IEspectreRuntime`, snapshots, events, and capabilities |
| **ESPHome frontend** | Production Home Assistant path kept on top of the shared platform |
| **Native frontend** | Standalone custom GATT surface for generic BLE clients and web integrations, including runtime tuning and BLE-triggered HTTPS OTA |
| **ESPectre Protocol** | Shared BLE+MQTT Protocol baseline for provisioning, telemetry, status, info, commands, monitor integration, and reusable runtime protocol services |
| **Matter frontend** | Available Matter occupancy surface with limited controller validation |
| **Streamer frontend** | Standalone CSI UDP streamer for dataset collection, host tooling, and realtime fusion experiments |
| **SDK-oriented firmware path** | Ability to assemble alternate firmware targets from shared platform layers for custom devices and OEM products |
| **Practical sensing** | Presence and occupancy baselines, plus reusable inference/tooling foundations |
| **Host-side tooling** | Analysis tools, datasets, and training workflows that support the platform direction |

### Release Readiness

The v3 platform is approaching release-candidate state for the modular platform goal.
The shared architecture, protocol services, frontend paths, and host-side
validation workflows are present and covered by automated tests.
Remaining work is closing the Native BLE control gaps, release polish, and
clearly documenting current sensing characteristics.

| Area | State | Notes |
|------|-------|-------|
| **Shared architecture** | Ready | `core`, `runtime`, ESP-IDF runtime services, and frontend adapters are split and documented |
| **Frontend coverage** | Ready | ESPHome remains the most mature production Home Assistant path; native and streamer firmware paths are available on the shared platform, and Matter firmware is available with limited controller validation |
| **Firmware smoke coverage** | Ready | ESPHome dev config passes for C3/C5/C6/S3; ESPHome C3 build, native C3 Docker build, and Matter C3 Docker build pass; hardware flash/monitor smoke completed for the release targets |
| **Protocol baseline** | Ready | BLE+MQTT payloads, provisioning, telemetry, status, info, commands, and monitor tooling are documented in `ESPECTRE_PROTOCOL.md` |
| **Detection validation** | Ready | Current C++ and Python real-data and long-recording suites pass across supported chips; C5/C6 long-quiet false-positive rates remain below the 5% target |
| **Documentation** | Ready | Setup, architecture, protocol, tuning, performance, and frontend-specific READMEs describe the v3 surface |
| **Product polish** | Remaining | Native BLE OTA and hit-threshold controls, release notes, final binary artifact checks, and user-facing wording should be completed before tagging |

ESPectre v3 success criteria:

- [x] Keep C++ and Python real-data performance validation green
- [x] Keep C++ long-recording validation green
- [x] Keep Python long-recording validation green
- [x] Document multi-frontend setup, architecture, and protocol boundaries
- [x] Run local firmware smoke tests for ESPHome, native, and Matter C3 release paths
- [x] Run hardware flash/monitor smoke tests for the release targets, published factory images, and Native OTA images
- [x] Reduce long-recording false-positive caveats on C5/C6
- [x] Define local-first shared protocol baseline for BLE and MQTT derived telemetry
- [x] Implement BLE-assisted Wi-Fi and MQTT provisioning
- [x] Persist Wi-Fi and ESPectre Protocol settings on the native firmware path
- [x] Move ESPectre Protocol helpers and ESP-IDF protocol services into shared runtime layers
- [x] Keep the streamer firmware on a narrow Wi-Fi-only streaming path without a separate BLE, MQTT, or OTA control surface
- [x] Publish MQTT telemetry, status, info, stats, and command results from native firmware
- [x] Align `micro-espectre` MQTT payloads and commands with the ESPectre Protocol baseline
- [x] Adapt the existing web monitor into a protocol validation and MQTT dashboard client
- [x] Keep Classic and ML usable when RSSI drops into the roughly `-70` to `-80 dBm` range
  - [x] Add a session-centered L1 safeguard for Classic
  - [x] Add and validate an ML low-RSSI safeguard from real captures
  - [x] Gate Classic false positives on the empty-room recordings: the alarms on static-presence baselines were not weak-link noise but the stationary occupant's own micro-motion, and they occur on the strongest links as readily as on the weakest
- [x] Separate ML training data from reserved promotion replays, with lineage-grouped CV and a link-class stress policy for real weak-link captures
- [x] Promote a weak-link-robust ML feature set (Coherence-6: temporal-coherence features replace the two weakest Core-6 members; promoted end-to-end by the reserved-replay protocol with a novel-hardware holdout check). Now Coherence-7: the Classic lag ratio joined as a seventh input, taking reserved effective alarms from `8` to `3` and the worst reserved F1 from `92.92%` to `95.30%`
  - [x] Set the per-replay non-regression margin from measured seed noise instead of pinning it at one evaluation. Done 2026-07-27: `fp_rate` moves to five evaluations against a measured spread of four across fifteen seeds, `recall` keeps one evaluation because it did not move at all, and `effective_alarms` keeps a zero margin. See [2026-07-27-set-the-non-regression-margin-from-seed-noise.md](adr/2026-07-27-set-the-non-regression-margin-from-seed-noise.md)
    - Still open: the `low_rssi` exemption. Every weak pair sits in `train`, `holdout`, or `exclude`, and none in `selection`, so its dispersion can only be measured by contaminating training or by burning the holdout. `COLLECTION_PLAN.md` asks for a reserved weak selection pair; leave the exemption alone until one lands
    - Re-measure the margin once the C5 and S3 replacement captures land: it is a claim about this corpus, not about arithmetic
- [x] Test whether augmentation still helps at seven features: it does not. Twenty augmented seeds put worst-session recall between `43.8%` and `91.0%` against `84.3%` to `95.4%` unaugmented, failed the paired gate three times, and found no candidate worth promoting, while ten unaugmented seeds passed every time with zero alarms. Two of the seven features are scale-invariant by construction now, which is the obvious suspect for why a gain-perturbing augmentation stopped paying
- [x] Raise the ESP32 streamer sustained capture rate beyond the previous approximately 70 pps ceiling (stable ~80 pps via legacy broadcast pacing; L-LTF frames stay outside the HT20 sensing contract, so sensing datasets still come from HT captures)
- [x] Add a post-collect dataset consistency check for streamer captures that at least verifies there are no recording gaps and that class separation is decent
- [x] Make `segmentation_window_size`, detector feature windows, and `evaluation_interval` adapt automatically to the effective CSI packet rate, and keep Classic and ML features comparable across window sizes and different CSI packet rates
  - [ ] Verify the Classic coefficients off-nominal: they remain fitted at the nominal cadence, and a refit on the current corpus lost to them at matched false positives
- [x] Close the remaining detector recall gap. Every chip now clears the `95%` target, worst per-chip recall `97.7%`, after the settled-level rule recovered the ESP32 capture from `94.2%` to `98.0%`. Five bedroom pairs judged toxic are marked `[TO BE REPLACED]` and excluded pending re-collection, so this rests on a reduced corpus
  - [ ] Re-measure once the replacement captures land
  - [x] Remove the now-dead L1 noise-blend safeguard. It engaged only for the plain-mean L1 feature; once Classic moved to the lag ratio it no longer changed any outcome on the 27 paired and 12 empty-room recordings, so the extra startup L1 floor state, excursion gain, and blend branch were removed from both runtimes with parity re-checked
  - [ ] Revisit why Classic still pins its calibrated threshold near `1.0` on some captures, now that it no longer costs recall. Under the plain-mean L1 feature the two worst pinned captures managed `69%` while ML managed `100%`; with the lag ratio they calibrate at `0.980` and `0.996` and reach `96.8%` and `99.1%`, and five captures now sit above `0.90` without a recall penalty. So the symptom is unexplained but no longer harmful: the open question is whether a threshold that close to the ceiling leaves enough headroom on rooms the corpus does not cover, not whether it costs detections today
  - [ ] Explain the two excluded C3 bedroom pairs, which are the hardest Classic cases in the corpus and stayed hard after the lag ratio: `2026-07-22 19:58` reaches `82.5%` recall at `0.9852` separation and `2026-07-25 13:58` reaches `74.2%` at `0.9872`, both at `0.0%` false positives. Neither is a weak link, and the first sits at `-39/-38 dBm`, the strongest link in the corpus, so this is not the low-RSSI failure mode and not a separability failure either: the features separate, the detector does not follow. ML clears the same captures
  - [ ] Broaden ESP32 coverage: the chip still rests on a single capture, so nothing there distinguishes a chip characteristic from one recording
  - [x] Sweep the selected tone count at 16, 20, 24, and 32 with a refit per band: all regress against 12 once per-pair gates and the high-rate stress capture are included, so the count stays at 12 on detection evidence as well as channel statistics
  - [x] Revisit the startup threshold once a session has settled: measured threshold-free, the ESP32 features separate at `0.9999` AUC while the detector reaches `94.2%`, because the calibration prefix on that capture is `4.14x` noisier than the rest of the session. The settled-level rule now recovers it to `98.0%` and lifts the worst per-chip recall to `97.7%` at no cost to false positives or the empty-room gate
- [x] Restore Python/C++ performance-report parity, which had drifted on the post-reset interval fallback (mean against median), the replay timing seed, and the startup calibrator's weighted-sample accounting
- [ ] Add and validate broader PHY and band support, including Wi-Fi 6 / 802.11ax capabilities and, where supported by hardware and exposed APIs, 5 GHz operation
  - [x] Classify CSI formats before normalization and handle currently unsupported LLTF, HT40, and HE20 packets gracefully, with explicit drop-reason telemetry and detector resets on format-stream changes
- [ ] Set the runtime `motion_on_hits` and `motion_off_hits` thresholds through the Native BLE control surface
- [ ] Trigger Native firmware OTA from BLE, then resolve the manifest and download the update over HTTPS through the same OTA service used by MQTT
- [ ] Collect ESP32 data across all dataset environments
  - [ ] Retrain and validate the production model with the expanded ESP32 dataset
- [ ] Consent manager and cookies
- [x] Remove unused C++ and Python features once the detector experiments settled. Done 2026-07-27: exactly the seven Coherence-7 features exist in both runtimes, there is no candidate tier, and `ALL_FEATURES == DEFAULT_FEATURES` is asserted by a test. `turb_skewness`, `l1_delta_waveform_length`, and `l1_delta_cv` are gone with their helpers, ids, and struct members, along with three helpers no feature ever used (`calc_iqr`, `calc_l1_delta`, and a duplicate `normalize_features` that allocated where the device path reuses a buffer). Those three had survived an earlier scan because a helper no production path calls still looks alive when its own tests reference it. See [2026-07-27-reduce-the-feature-surface-to-the-production-set.md](adr/2026-07-27-reduce-the-feature-surface-to-the-production-set.md), which records every removed feature and the measurement that rejected it
- [ ] Simplify the training workflow by dropping options that are no longer useful. Not started: the feature cleanup above documented the three undocumented flags (`--experiment-output`, `--fp-weight-experiment-output`, `--hidden-layers`) rather than removing them, because all three work and one writes the report the dispersion analysis reads. `train_ml_model.py` still exposes `30` options, and the candidates worth assessing are the ones whose purpose narrowed when the feature surface did: `--ablation` (documented as unusable for promotion on its own), `--features` (now only subsets of seven), and `--positive-chip-boost`. Judge each against a run that used it, not against the help text
- [ ] Make a final review of code. Be dry, Check responsabilities and level (core, runtime, frontend). Performance security review. 
- [ ] Last check to doc. Do not repeat, simplify, every doc has his own responsibility.
  - [ ] Refresh the Home Assistant screenshots used by the documentation and website, replacing the current gauge with a more suitable visualization
- [ ] Finalize release notes and artifact checklist before tagging `v3.0.0`
  - [ ] Changelog review
- [ ] Re-enable the `CLA Signature Check` as a required status check in GitHub branch protection for `develop`
- [ ] Test the new GitHub issue and pull request templates end to end

### Planned v3.x Follow-Ups

These items belong to the v3 series but do not all need to block `v3.0.0`; they
may ship in later v3.x minor releases after the modular platform baseline is
tagged.

- [ ] Add Presence vs Empty detection
  - [ ] Find a feature that reads a stationary occupant's own micro-motion, because presence needs the signal that motion detection currently spends its effort suppressing. The evidence is already in the corpus: the `empty` recordings stay silent under every candidate, `quietMaxFP` holding at `0.00%` across a full seed search, while the static-presence captures activate in short scattered episodes, `17` of them on the S3 weak-link holdout with the longest running `4` evaluations. Those episodes are the occupant, not noise, which is why they were gated out of motion detection; see [2026-07-25-gate-classic-false-positives-on-empty-rooms.md](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md). Reading them as evidence rather than error needs a statistic tuned to brief low-amplitude excursions above a quiet floor, distinct from the window-level features both detectors use today
- [ ] Research whether breathing-related micro-motion can become a reliable local sensing signal, keeping the work explicitly non-medical and validating it separately from presence and motion detection
- [ ] Optimize Micro-ESPectre to exceed its current approximately 70 pps ceiling
- [ ] Evaluate how to improve detection quality at high CSI packet rates instead of relying on decimation as a temporary mitigation, so the platform can preserve short-timescale information for cases such as brief gesture recognition
  - [ ] Prototype brief gesture detection only after the higher-rate sensing path preserves enough short-timescale information, and define a validation corpus distinct from motion and presence
- [ ] Use a dedicated build directory for each chip instead of reusing the same directory across targets
- [ ] Add Native frontend support for local TFT/LCD status displays similar to `examples/espectre-s3-touch-lcd.yaml`

### Deferred Follow-Ups

- Evaluate LAN discovery for the streamer workflow via DNS-SD/mDNS so `./espectre collect` can browse reachable streamer nodes and optionally select a subset by `device_id`, while keeping explicit `--target` as the deterministic fallback and preserving CSI demultiplexing by `device_id`
- Evaluate a future Matter OTA design for a later 3.x or post-v3 release, including Requestor-plus-Provider ownership and release artifact expectations
- Validate and document Matter commissioning across additional controllers (Samsung SmartThings, Home Assistant Matter, and the Tuya app where occupancy sensors are supported), keeping a verified-controller matrix in the Matter frontend README
- Evaluate a Zigbee occupancy-sensor frontend on ESP32-C6 via `esp-zigbee-sdk`, starting with a coexistence spike to measure how 802.11 CSI capture behaves next to 802.15.4 time-slicing on the shared 2.4 GHz radio
- Evaluate Matter certification readiness for manufacturer-oriented builds, mapping the gap between the current Matter firmware and a CSA-certifiable product across vendor ID allocation, device attestation certificates, factory provisioning, and certification test coverage; commercial Apple Home and SmartThings reach flows through certified Matter rather than the non-commercial HomeKit ADK
- Evaluate a TuyaOpen reference integration that embeds the shared `core` and `runtime` into a TuyaOS application, aimed at manufacturers that already operate Tuya product pipelines, with per-device licensing and cloud coupling documented as integrator-side prerequisites
- Evaluate promoting the web BLE client (`docs/web/espectre-ble.js`) to a standalone integration artifact for third-party web apps; the Apache-2.0 licensing, event API, validated command builders, and unit tests are in place, and the remaining steps are dual ESM/IIFE packaging with npm publication and TypeScript definitions — this would also give the v4.x Web Bluetooth device claim flow a reusable foundation

---

## v4.x - Web Orchestration Layer

**Goal**: make multiple ESPectre devices behave like one coherent sensing
system through an optional, privacy-first web layer that can run locally,
self-hosted, or as a managed service, without requiring raw CSI or other
sensitive radio data to leave the user environment.

### Contains

| Area | Scope |
|------|-------|
| **Web orchestration** | Optional web layer for multi-device orchestration, built so local, self-hosted, and managed-service deployments remain viable |
| **Identity and tenancy** | User login, homes/locations, roles, and device ownership |
| **Secure device onboarding** | Physical-presence pairing, likely through Web Bluetooth, short-lived claim sessions, and per-device credentials |
| **Device visibility** | Sensor inventory, online/offline state, firmware version, runtime status, and fleet inspection |
| **Home map** | User-drawn home/office/location layout with devices placed in rooms or zones |
| **Realtime state** | Near-realtime movement score, motion state, and device health across the location |
| **Approximate room flow** | Best-effort room-to-room movement visualization from device transitions, without claiming precise localization |
| **Management** | Remote threshold updates, runtime settings, and signed firmware update workflows |
| **History** | Retained movement/status timeline with configurable privacy and retention policy |
| **Alerting** | Motion-triggered notifications through email first, then additional notification integrations as the web layer matures |
| **Privacy boundary** | Derived telemetry only; no raw CSI, no unnecessary Wi-Fi identifiers, no sensitive device logs by default |
| **Cross-frontend view** | Unified view across `ESPHome`, `Matter`, `Native`, streamer-derived tooling, and custom firmware nodes where applicable |
| **Deployment profiles** | Local web app, self-hosted service, and future managed ESPectre service built around the same privacy boundary |

### Implementation Checklist

- [x] Define web orchestration profiles, per-device service credentials, MQTT-over-TLS policy, and privacy boundary for device telemetry (documented in `ESPECTRE_PROTOCOL.md`)
- [ ] Design tenant, home/location, room, and device ownership model
- [ ] Implement social login and account management
- [ ] Implement secure Web Bluetooth assisted device claim flow
- [ ] Build telemetry ingestion path for derived sensing state and device status
- [ ] Evaluate privacy-preserving multisensing by combining Wi-Fi CSI motion detection with passive BLE presence and motion cues, without device pairing, identity binding, or tracking
- [ ] Build near-realtime dashboard with home map and device placement
- [ ] Add movement score, motion state, online/offline status, and firmware version views
- [ ] Add configurable threshold updates through the device control plane
- [ ] Add signed firmware artifact storage and OTA update workflow
- [ ] Add movement/status history with explicit retention controls
- [ ] Add alerting rules for motion detection, starting with email
- [ ] Evaluate additional notification integrations after the first alerting path is stable
- [ ] Add approximate room-to-room movement visualization from multi-device events
- [ ] Document open-source boundaries, self-hosting posture, and managed-service value
- [ ] Validate security, abuse resistance, privacy posture, and operational resilience before public launch

See [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) for the shared device protocol
and [ARCHITECTURE.md](ARCHITECTURE.md) for the local lab, self-hosted, and managed-service profiles.

---

## v5.x - Standards-Ready Wi-Fi Sensing

**Goal**: keep ESPectre aligned with the emerging IEEE 802.11bf / Wi-Fi Sensing ecosystem so future vendor-supported sensing chipsets can be integrated through the existing SDK-oriented architecture instead of requiring a new project shape.

IEEE 802.11bf is expected to make Wi-Fi sensing a first-class capability in future Wi-Fi products. ESPectre already validates the product and developer model around today's ESP32 CSI APIs: reusable sensing logic, runtime contracts, protocol semantics, dataset tooling, frontend adapters, and web-oriented orchestration boundaries.

When a microcontroller or embedded Wi-Fi platform exposes practical 802.11bf-style sensing APIs, the intended path is to add it as another runtime or hardware backend while preserving the higher-level ESPectre protocol, frontends, tooling, and device-maker integration model.

### Exploration Areas

- Track embedded vendor support for IEEE 802.11bf / Wi-Fi Sensing APIs
- Map standardized sensing measurements to ESPectre runtime snapshots and events
- Preserve compatibility with ESPectre Protocol telemetry and command semantics
- Keep frontend surfaces stable across ESPHome, Matter, native, web, and OEM firmware paths
- Validate whether standardized sensing improves calibration, false-positive control, and multi-node fusion
- Document the migration path from ESP32 CSI-based firmware to standards-backed hardware when available

---

## Roadmap Updates

Last update: **July 27, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
