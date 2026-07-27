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

- [ ] Finish post-promotion validation for the five-feature ML set
  - [x] Run a seed search over the five-feature set. The exported model now comes from a ten-trial search, not a force-promotion: seed `1876849819` ranked `robust_best` with `98.2%` blocked OOF F1, `92.39%` worst selection replay recall, `3.79%` max selection FP, and `0` selection alarms, then cleared the reserved holdout at `96.55%` worst replay recall, `2.19%` max FP, and `1` alarm. The open cost versus the seven-feature line remains the weaker worst-replay recall (`96.55%` versus `99.14%`)
  - [ ] Measure the `low_rssi` exemption across seeds now that one recovered weak-link selection pair exists. The first seed search kept the recovered S3 weak-link pair in family with the rest of the winners, but one recording is still not enough to tell a dispersion from that pair's own quirk, so collect a second weak selection pair first
  - [ ] Re-measure the non-regression margin once the specific holdout replacements in `COLLECTION_PLAN.md` land: the C5 bedroom normal-link holdout pair from `2026-07-24 12:59/13:05` is still the active C5 holdout, and the S3 bedroom weak-link holdout pair from `2026-07-22 17:20/17:23` is still the active S3 weak-link holdout. The report is current again, but the margin is still a claim about this corpus, not about arithmetic, so it should be re-derived on the replacement corpus rather than carried forward
- [x] Rework `--augment` around non-scale perturbations and re-validate it on the five-feature corpus. Gain scaling is no longer part of the recipe; the remaining question is whether feature jitter plus packet noise, packet loss, and timing-artifact stutter can earn a place back in promotion rather than stay experiment-only
- [ ] Finish detector follow-ups on the reduced corpus
  - [ ] Re-measure once the replacement captures land
  - [ ] Revisit why Classic still pins its calibrated threshold near `1.0` on some captures, now that it no longer costs recall. Under the plain-mean L1 feature the two worst pinned captures managed `69%` while ML managed `100%`; with the lag ratio they calibrate at `0.980` and `0.996` and reach `96.8%` and `99.1%`, and five captures now sit above `0.90` without a recall penalty. So the symptom is unexplained but no longer harmful: the open question is whether a threshold that close to the ceiling leaves enough headroom on rooms the corpus does not cover, not whether it costs detections today
  - [ ] Explain the two excluded C3 bedroom pairs, which are the hardest Classic cases in the corpus and stayed hard after the lag ratio: `2026-07-22 19:58` reaches `82.5%` recall at `0.9852` separation and `2026-07-25 13:58` reaches `74.2%` at `0.9872`, both at `0.0%` false positives. Neither is a weak link, and the first sits at `-39/-38 dBm`, the strongest link in the corpus, so this is not the low-RSSI failure mode and not a separability failure either: the features separate, the detector does not follow. ML clears the same captures
  - [ ] Broaden ESP32 coverage: the chip still rests on a single capture, so nothing there distinguishes a chip characteristic from one recording
- [ ] Add and validate broader PHY and band support, including Wi-Fi 6 / 802.11ax capabilities and, where supported by hardware and exposed APIs, 5 GHz operation
- [ ] Set the runtime `motion_on_hits` and `motion_off_hits` thresholds through the Native BLE control surface
- [ ] Trigger Native firmware OTA from BLE, then resolve the manifest and download the update over HTTPS through the same OTA service used by MQTT
- [ ] Collect ESP32 data across all dataset environments
  - [ ] Retrain and validate the production model with the expanded ESP32 dataset
- [ ] Consent manager and cookies
- [ ] Simplify the training workflow by dropping options that are no longer useful. Not started: the feature cleanup above documented the three undocumented flags (`--experiment-output`, `--fp-weight-experiment-output`, `--hidden-layers`) rather than removing them, because all three work and one writes the report the dispersion analysis reads. `train_ml_model.py` still exposes `30` options, and the candidates worth assessing are the ones whose purpose narrowed when the feature surface did: `--ablation` (documented as unusable for promotion on its own), `--features` (now only subsets of seven), and `--positive-chip-boost`. Judge each against a run that used it, not against the help text
- [ ] Evaluate removing the `test` label and moving the long recordings under `empty`, with an explicit role in training. All twelve captures in `data/test/` are quiet-room recordings of roughly `48k` packets each, and both long-recording suites already read them as the empty-room false-positive gate, so the label claims a mixed session while the content is `empty`; `ML_DATA_COLLECTION.md` reserves `test` for captures that are not label-homogeneous, and the only genuinely mixed captures were removed and live in history at `48c9cce^`, so the label currently has no user. The decision worth making is not the rename but the role: today these recordings are gate-only and never enter training, and the question is whether their length is a reason to keep them out, since one continuous session is one lineage group and admitting a whole recording shifts the group balance in CV, or whether a segmented share belongs in the idle class. Scope the loaders, the label list, dataset metadata, and the `data/test` paths in the long-recording suites, and re-run the quiet gate afterwards to confirm the empty-room floor is unchanged
- [ ] Evaluate three new host-side features, all scale-invariant, each reading a physical aspect the production five do not. The current set is five amplitude-domain time statistics over one window (turbulence dispersion, turbulence autocorrelation, turbulence zero-crossing rate, L1-delta autocorrelation, and the L1-delta lag ratio), so it describes how much the channel moves and how fast, and nothing else. Enrich that physical profile without restating it. Scale invariance is a membership rule, not a preference; see [2026-07-28-drop-the-absolute-l1-features.md](adr/2026-07-28-drop-the-absolute-l1-features.md). Keep the work in Python and host tooling only, and port to C++ only if a candidate is promoted
  - [ ] A spectral axis: where the movement energy sits in frequency rather than how large it is, as a normalized statistic over the movement series such as a band power ratio or a spectral centroid expressed in bins. This separates fast limb motion from slow whole-body motion, which the amplitude statistics conflate. `band_power_ratio` was measured before as the one noise-robust candidate of an earlier sweep, so start from that measurement rather than from scratch
  - [ ] A spatial axis across the selected tones: how coherently movement acts on the band, as a correlation across subcarriers rather than a per-tone magnitude. Multipath change and common-mode gain drift look identical in a per-tone amplitude statistic and separate here, which is the one thing the unrecorded per-packet gain makes hard to distinguish
  - [ ] A phase axis. Phase is untouched by the real int8 scaling factor, so it is scale-invariant by construction, but it is not usable raw: STO adds a linear ramp across subcarriers and CFO a common offset, both varying per packet. Sanitize first, by differencing across adjacent subcarriers or removing a linear fit, then build the statistic on the temporal behaviour of the residual. Confirm the sanitization holds across chips before trusting the feature, because the subcarrier layout differs between the shifted and unshifted families
  - [ ] Gate each candidate on incremental value, not standalone value: report its correlation against all five production features first, and accept only what survives the promotion protocol, lineage-grouped CV to lead, paired and quiet gates for safety, and the holdout sealed
- [ ] Make a final review of code. Be dry, Check responsabilities and level (core, runtime, frontend). Performance security review. 
- [ ] Last check to doc. Do not repeat, simplify, every doc has his own responsibility.
  - [ ] Refresh the Home Assistant screenshots used by the documentation and website, replacing the current gauge with a more suitable visualization
- [ ] Finalize release notes and artifact checklist before tagging `v3.0.0`
  - [ ] Changelog review
- [ ] Re-enable the `CLA Signature Check` as a required status check in GitHub branch protection for `develop`
- [ ] Test the new GitHub issue and pull request templates end to end

- [ ] Measure how much the unrecorded per-packet gain moves each production feature. Scale invariance is currently a way around the problem rather than a measurement of it. The gain-locked captures in history do not answer it: they carry no matched AGC-running capture of the same scene, so they show what locked-gain data looks like rather than what locking changes. This needs a deliberate collection, the same scene captured both ways
- [ ] Test detector behaviour across a scene change inside one continuous stream. Every capture in the corpus is one state end to end, so a transition is only ever observed across two files, which is not what a device experiences. Four mixed captures exist in history at `48c9cce^` and were measured on 2026-07-27: the transition is one evaluation wide, so `motion_on_hits` and not detector responsiveness owns the latency, and C6 shows Classic and ML disagreeing outright against a verified-quiet baseline. Recover them when the question comes up again; using them as corpus pairs first needs the motion onset recorded by the collector. See [2026-07-27-keep-the-mixed-transition-captures.md](adr/2026-07-27-keep-the-mixed-transition-captures.md)

### Planned v3.x Follow-Ups

These items belong to the v3 series but do not all need to block `v3.0.0`; they
may ship in later v3.x minor releases after the modular platform baseline is
tagged.

- [ ] Use a dedicated build directory for each chip instead of reusing the same directory across targets
- [ ] Evaluate LAN discovery for the streamer workflow via DNS-SD/mDNS so `./espectre collect` can browse reachable streamer nodes and optionally select a subset by `device_id`, while keeping explicit `--target` as the deterministic fallback and preserving CSI demultiplexing by `device_id`
- [ ] Evaluate promoting the web BLE client (`docs/web/espectre-ble.js`) to a standalone integration artifact for third-party web apps; the Apache-2.0 licensing, event API, validated command builders, and unit tests are in place, and the remaining steps are dual ESM/IIFE packaging with npm publication and TypeScript definitions. This would also give the v4.x Web Bluetooth device claim flow a reusable foundation
- [ ] Validate and document Matter commissioning across additional controllers (Samsung SmartThings, Home Assistant Matter, and the Tuya app where occupancy sensors are supported), keeping a verified-controller matrix in the Matter frontend README
- [ ] Evaluate a future Matter OTA design for a later 3.x or post-v3 release, including Requestor-plus-Provider ownership and release artifact expectations
- [ ] Evaluate Matter certification readiness for manufacturer-oriented builds, mapping the gap between the current Matter firmware and a CSA-certifiable product across vendor ID allocation, device attestation certificates, factory provisioning, and certification test coverage; commercial Apple Home and SmartThings reach flows through certified Matter rather than the non-commercial HomeKit ADK
- [ ] Optimize Micro-ESPectre to exceed its current approximately 70 pps ceiling
- [ ] Evaluate how to improve detection quality at high CSI packet rates instead of relying on decimation as a temporary mitigation, so the platform can preserve short-timescale information for cases such as brief gesture recognition
  - [ ] Prototype brief gesture detection only after the higher-rate sensing path preserves enough short-timescale information, and define a validation corpus distinct from motion and presence
- [ ] Add Presence vs Empty detection
  - [ ] Find a feature that reads a stationary occupant's own micro-motion, because presence needs the signal that motion detection currently spends its effort suppressing. The evidence is already in the corpus: the `empty` recordings stay silent under every candidate, `quietMaxFP` holding at `0.00%` across a full seed search, while the static-presence captures activate in short scattered episodes, `17` of them on the S3 weak-link holdout with the longest running `4` evaluations. Those episodes are the occupant, not noise, which is why they were gated out of motion detection; see [2026-07-25-gate-classic-false-positives-on-empty-rooms.md](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md). Reading them as evidence rather than error needs a statistic tuned to brief low-amplitude excursions above a quiet floor, distinct from the window-level features both detectors use today
- [ ] Research whether breathing-related micro-motion can become a reliable local sensing signal, keeping the work explicitly non-medical and validating it separately from presence and motion detection
- [ ] Add Native frontend support for local TFT/LCD status displays similar to `examples/espectre-s3-touch-lcd.yaml`
- [ ] Evaluate a Zigbee occupancy-sensor frontend on ESP32-C6 via `esp-zigbee-sdk`, starting with a coexistence spike to measure how 802.11 CSI capture behaves next to 802.15.4 time-slicing on the shared 2.4 GHz radio
- [ ] Evaluate a TuyaOpen reference integration that embeds the shared `core` and `runtime` into a TuyaOS application, aimed at manufacturers that already operate Tuya product pipelines, with per-device licensing and cloud coupling documented as integrator-side prerequisites

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

Last update: **July 28, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
