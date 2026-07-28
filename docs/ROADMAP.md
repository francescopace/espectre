# Roadmap

## Releases


| Version  | Date             | Status      | Summary                                                                                                                                                                        |
| -------- | ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **v1.x** | 2025-11-09       | Released    | First release demonstrating motion detection capabilities using a brand-new algorithm                                                                                          |
| **v2.x** | 2025-12-06       | Released    | Home Assistant integration via ESPHome plus custom MicroPython-based firmware                                                                                                  |
| **v3.x** | 2026-08 (target) | In progress | Scale-invariant Classic and ML detectors. Add Matter support with limited controller validation, native BLE/MQTT firmware, and an SDK-oriented foundation for OEM integrations |
| **v4.x** | 2026-12 (target) | Planned     | Privacy-first web orchestration layer for multi-node sensing, secure onboarding, fleet visibility, history, alerting, and remote management                                    |
| **v5.x** | Future           | Exploratory | Standards-ready sensing platform prepared for practical IEEE 802.11bf / Wi-Fi Sensing hardware support when embedded vendors expose it                                         |


---



## v3.x - Modular Sensing Platform

**Goal**: move from a single integration-focused firmware to a reusable Wi-Fi
sensing platform with shared sensing logic, a stable runtime contract, multiple
frontend paths, and an embeddable foundation for custom firmware and OEM products.

### Contains


| Area                           | Scope                                                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**               | Shared `core`, `runtime`, and `frontend` layers                                                                                                    |
| **Runtime contract**           | Stable frontend-oriented APIs such as `IEspectreRuntime`, snapshots, events, and capabilities                                                      |
| **ESPHome frontend**           | Production Home Assistant path kept on top of the shared platform                                                                                  |
| **Native frontend**            | Standalone custom GATT surface for generic BLE clients and web integrations, including runtime tuning and BLE-triggered HTTPS OTA                  |
| **ESPectre Protocol**          | Shared BLE+MQTT Protocol baseline for provisioning, telemetry, status, info, commands, monitor integration, and reusable runtime protocol services |
| **Matter frontend**            | Available Matter occupancy surface with limited controller validation                                                                              |
| **Streamer frontend**          | Standalone CSI UDP streamer for dataset collection, host tooling, and realtime fusion experiments                                                  |
| **SDK-oriented firmware path** | Ability to assemble alternate firmware targets from shared platform layers for custom devices and OEM products                                     |
| **Practical sensing**          | Presence and occupancy baselines, plus reusable inference/tooling foundations                                                                      |
| **Host-side tooling**          | Analysis tools, datasets, and training workflows that support the platform direction                                                               |




### Release Readiness

The v3 platform is approaching release-candidate state for the modular platform goal.
The shared architecture, protocol services, frontend paths, and host-side
validation workflows are present and covered by automated tests.
Remaining work is closing the Native BLE control gaps, release polish, and
clearly documenting current sensing characteristics.


| Area                        | State     | Notes                                                                                                                                                                                                        |
| --------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Shared architecture**     | Ready     | `core`, `runtime`, ESP-IDF runtime services, and frontend adapters are split and documented                                                                                                                  |
| **Frontend coverage**       | Ready     | ESPHome remains the most mature production Home Assistant path; native and streamer firmware paths are available on the shared platform, and Matter firmware is available with limited controller validation |
| **Firmware smoke coverage** | Ready     | ESPHome dev config passes for C3/C5/C6/S3; ESPHome C3 build, native C3 Docker build, and Matter C3 Docker build pass; hardware flash/monitor smoke completed for the release targets                         |
| **Protocol baseline**       | Ready     | BLE+MQTT payloads, provisioning, telemetry, status, info, commands, and monitor tooling are documented in `ESPECTRE_PROTOCOL.md`                                                                             |
| **Detection validation**    | Ready     | Current C++ and Python real-data and long-recording suites pass across supported chips; C5/C6 long-quiet false-positive rates remain below the 5% target                                                     |
| **Documentation**           | Ready     | Setup, architecture, protocol, tuning, performance, and frontend-specific READMEs describe the v3 surface                                                                                                    |
| **Product polish**          | Remaining | Native BLE OTA and hit-threshold controls, release notes, final binary artifact checks, and user-facing wording should be completed before tagging                                                           |


ESPectre v3 success criteria:

- [ ] Finish post-promotion validation for the five-feature ML set
  - [ ] Add a second weak-link selection pair before treating the current low-RSSI result as representative across seeds; see [COLLECTION_PLAN.md](../data/COLLECTION_PLAN.md)
  - [ ] Replace the fragile C6 normal-link and S3 weak-link holdouts, then re-derive the non-regression margin on the replacement corpus; see [COLLECTION_PLAN.md](../data/COLLECTION_PLAN.md) and [2026-07-27-set-the-non-regression-margin-from-seed-noise.md](adr/2026-07-27-set-the-non-regression-margin-from-seed-noise.md)
- [ ] Finish detector follow-ups on the reduced corpus
  - [ ] Re-measure once the replacement captures land: rerun `validate_dataset_quality.py`, the real-data and long-recording pytest suites, `generate_performance_report.py`, and, if the holdout swaps move the gate evidence, re-derive the non-regression margin from seed dispersion on that replacement corpus rather than carrying the current margin forward
- [ ] Add and validate broader PHY and band support, including Wi-Fi 6 / 802.11ax capabilities and, where supported by hardware and exposed APIs, 5 GHz operation
- [ ] Set the runtime `motion_on_hits` and `motion_off_hits` thresholds through the Native BLE control surface
- [ ] Trigger Native firmware OTA from BLE, then resolve the manifest and download the update over HTTPS through the same OTA service used by MQTT
- [ ] Collect ESP32 data across all dataset environments
  - [ ] Retrain and validate the production model with the expanded ESP32 dataset
- [ ] Consent manager and cookies
- [ ] Evaluate whether the Wi-Fi profile `scale` field can improve data stability. The current detector path treats scale invariance as a requirement because the capture metadata does not yet provide a trusted correction for packet-to-packet gain drift. If the profile-level `scale` term is exposed consistently enough across chips and collection paths, measure whether using it reduces session drift or cross-capture instability without breaking the current C++/Python alignment
- [ ] Test Igiene: remove redundant tests. Test only cpp, micropython and cli and cli dependencies. Avoid to test tools or scripts. Avoid to test configurations. keep coverage gate. Define this rule in agents.md
- [x] Complete the final C++ architecture, responsibility, duplication, and performance review; remaining fixes are tracked in [cpp-review-2026-07-28.md](review/cpp-review-2026-07-28.md)
- [ ] Complete the security review and encode the recurring review rule in `AGENTS.md`
- [x] Close the final documentation review and its generated-report work; all 15 findings are resolved in [documentation-review-2026-07-28.md](review/documentation-review-2026-07-28.md)
- [ ] Refresh the Home Assistant screenshots used by the documentation and website, replacing the current gauge with a more suitable visualization
- [ ] Finalize release notes and artifact checklist before tagging `v3.0.0`
  - [x] Changelog review
- [ ] Re-enable the `CLA Signature Check` as a required status check in GitHub branch protection for `develop`
- [ ] Test the new GitHub issue and pull request templates end to end



### Planned v3.x Follow-Ups

These items belong to the v3 series but do not all need to block `v3.0.0`; they
may ship in later v3.x minor releases after the modular platform baseline is
tagged.

- [x] Use dedicated per-chip build directories in Native, Matter, and Streamer CI/release builds
- [ ] Make local ESP-IDF CLI builds use per-chip build directories by default instead of requiring `ESPECTRE_IDF_BUILD_DIR`
- [ ] Evaluate LAN discovery for the streamer workflow via DNS-SD/mDNS so `./espectre collect` can browse reachable streamer nodes and optionally select a subset by `device_id`, while keeping explicit `--target` as the deterministic fallback and preserving CSI demultiplexing by `device_id`
- [ ] Evaluate promoting the web BLE client (`docs/web/espectre-ble.js`) to a standalone integration artifact for third-party web apps; the Apache-2.0 licensing, event API, validated command builders, and unit tests are in place, and the remaining steps are dual ESM/IIFE packaging with npm publication and TypeScript definitions. This would also give the v4.x Web Bluetooth device claim flow a reusable foundation
- [ ] Validate and document Matter commissioning across additional controllers (Samsung SmartThings, Home Assistant Matter, and the Tuya app where occupancy sensors are supported), keeping a verified-controller matrix in the Matter frontend README
- [ ] Evaluate a future Matter OTA design for a later 3.x or post-v3 release, including Requestor-plus-Provider ownership and release artifact expectations
- [ ] Evaluate Matter certification readiness for manufacturer-oriented builds, mapping the gap between the current Matter firmware and a CSA-certifiable product across vendor ID allocation, device attestation certificates, factory provisioning, and certification test coverage; commercial Apple Home and SmartThings reach flows through certified Matter rather than the non-commercial HomeKit ADK
- [ ] Optimize Micro-ESPectre to exceed its current approximately 70 pps ceiling
- [ ] Evaluate how to improve detection quality at high CSI packet rates instead of relying on decimation as a temporary mitigation, so the platform can preserve short-timescale information for cases such as brief gesture recognition
  - [ ] Prototype brief gesture detection only after the higher-rate sensing path preserves enough short-timescale information, and define a validation corpus distinct from motion and presence
- [ ] Add Presence vs Empty detection
  - [ ] Evaluate scale-invariant micro-motion and quiet-floor excursion features; keep measurements and candidate status in [FEATURES.md](FEATURES.md), source evidence in [LITERATURE.md](LITERATURE.md), and the motion-versus-presence boundary in [2026-07-25-gate-classic-false-positives-on-empty-rooms.md](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md)
- [ ] Research whether breathing-related micro-motion can become a reliable local sensing signal, keeping the work explicitly non-medical and validating it separately from presence and motion detection
  - [x] Separate ESP32-compatible evidence from wider-band, CIR, AoA, and range-Doppler work in [LITERATURE.md](LITERATURE.md), with explicit HT20 transfer limits
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


| Area                         | Scope                                                                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Web orchestration**        | Optional web layer for multi-device orchestration, built so local, self-hosted, and managed-service deployments remain viable |
| **Identity and tenancy**     | User login, homes/locations, roles, and device ownership                                                                      |
| **Secure device onboarding** | Physical-presence pairing, likely through Web Bluetooth, short-lived claim sessions, and per-device credentials               |
| **Device visibility**        | Sensor inventory, online/offline state, firmware version, runtime status, and fleet inspection                                |
| **Home map**                 | User-drawn home/office/location layout with devices placed in rooms or zones                                                  |
| **Realtime state**           | Near-realtime movement score, motion state, and device health across the location                                             |
| **Approximate room flow**    | Best-effort room-to-room movement visualization from device transitions, without claiming precise localization                |
| **Management**               | Remote threshold updates, runtime settings, and signed firmware update workflows                                              |
| **History**                  | Retained movement/status timeline with configurable privacy and retention policy                                              |
| **Alerting**                 | Motion-triggered notifications through email first, then additional notification integrations as the web layer matures        |
| **Privacy boundary**         | Derived telemetry only; no raw CSI, no unnecessary Wi-Fi identifiers, no sensitive device logs by default                     |
| **Cross-frontend view**      | Unified view across `ESPHome`, `Matter`, `Native`, streamer-derived tooling, and custom firmware nodes where applicable       |
| **Deployment profiles**      | Local web app, self-hosted service, and future managed ESPectre service built around the same privacy boundary                |




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
