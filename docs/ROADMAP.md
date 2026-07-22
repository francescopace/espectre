# Roadmap

## Releases

| Version | Date | Status | Summary |
|---------|------|--------|---------|
| **v1.x** | 2025-11-09 | Released | First release demonstrating motion detection capabilities using a brand-new algorithm |
| **v2.x** | 2025-12-06 | Released | Home Assistant integration via ESPHome plus custom MicroPython-based firmware |
| **v3.x** | 2026-08 (target) | In progress | New Detectors based on spectral features. Add Matter support, native BLE/MQTT firmware, and an SDK-oriented foundation for OEM integrations |
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
| **Matter frontend** | Matter occupancy surface proving a second ecosystem-facing frontend |
| **Streamer frontend** | Standalone CSI UDP streamer for dataset collection, host tooling, and realtime fusion experiments |
| **SDK-oriented firmware path** | Ability to assemble alternate firmware targets from shared platform layers for custom devices and OEM products |
| **Practical sensing** | Presence and occupancy baselines, plus reusable inference/tooling foundations |
| **Host-side tooling** | Analysis tools, notebooks, datasets, and training workflows that support the platform direction |

### Release Readiness

The v3 platform is approaching release-candidate state for the modular platform goal.
The shared architecture, protocol services, frontend paths, and host-side
validation workflows are present and covered by automated tests.
Remaining work is closing the Native BLE control gaps, release polish, hardware
smoke coverage, and clearly documenting current sensing characteristics.

| Area | State | Notes |
|------|-------|-------|
| **Shared architecture** | Ready | `core`, `runtime`, ESP-IDF runtime services, and frontend adapters are split and documented |
| **Frontend coverage** | Ready | ESPHome remains the production Home Assistant path; native, Matter, and streamer firmware paths are present on the shared platform |
| **Firmware smoke coverage** | Ready with caveats | ESPHome dev config passes for C3/C5/C6/S3; ESPHome C3 build, native C3 Docker build, and Matter C3 Docker build pass; hardware flash/monitor smoke remains open |
| **Protocol baseline** | Ready | BLE+MQTT payloads, provisioning, telemetry, status, info, commands, and monitor tooling are documented in `ESPECTRE_PROTOCOL.md` |
| **Detection validation** | Ready | Current C++ and Python real-data and long-recording suites pass across supported chips; C5/C6 long-quiet false-positive rates remain below the 5% target |
| **Documentation** | Ready | Setup, architecture, protocol, tuning, performance, and frontend-specific READMEs describe the v3 surface |
| **Product polish** | Remaining | Native BLE OTA and hit-threshold controls, hardware flash smoke, release notes, final binary artifact checks, and user-facing wording should be completed before tagging |

ESPectre v3 success criteria:

- [x] Keep C++ and Python real-data performance validation green
- [x] Keep C++ long-recording validation green
- [x] Keep Python long-recording validation green
- [x] Document multi-frontend setup, architecture, and protocol boundaries
- [x] Run local firmware smoke tests for ESPHome, native, and Matter C3 release paths
- [ ] Run hardware flash/monitor smoke tests for the release targets, published factory images, and Native OTA images
- [ ] Trigger Native firmware OTA from BLE, then resolve the manifest and download the update over HTTPS through the same OTA service used by MQTT
- [ ] Set the runtime `motion_on_hits` and `motion_off_hits` thresholds through the Native BLE control surface
- [x] Reduce long-recording false-positive caveats on C5/C6
- [ ] Re-enable the `CLA Signature Check` as a required status check in GitHub branch protection for `develop`
- [ ] Finalize release notes and artifact checklist before tagging `v3.0.0`

### Planned v3.x Follow-Ups

These items belong to the v3 series but do not all need to block `v3.0.0`; they
may ship in later v3.x minor releases after the modular platform baseline is
tagged.

- [ ] Add Presence vs Empty detection
- [ ] Raise the ESP32 streamer sustained capture rate beyond the current approximately 70 pps ceiling
  - [ ] Collect ESP32 data across all dataset environments
  - [ ] Retrain and validate the production model with the expanded ESP32 dataset
- [ ] Optimize Micro-ESPectre to exceed its current approximately 70 pps ceiling
- [ ] Refresh the Home Assistant screenshots used by the documentation and website, replacing the current gauge with a more suitable visualization
- [ ] Add and validate broader PHY and band support, including graceful handling for currently unsupported LLTF, HT40, and HE20 packets, plus Wi-Fi 6 / 802.11ax capabilities and, where supported by hardware and exposed APIs, 5 GHz operation
- [ ] Add low-RSSI safeguards so Classic and ML detector behavior remains usable when RSSI drops into the roughly `-70` to `-80 dBm` range
- [ ] Make `segmentation_window_size`, detector feature windows, and `evaluation_interval` adapt automatically to the effective CSI packet rate, and keep Classic and ML features comparable across window sizes
- [ ] Use a dedicated build directory for each chip instead of reusing the same directory across targets
- [ ] Test the new GitHub issue and pull request templates end to end

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

- [x] Define local-first shared protocol baseline for BLE and MQTT derived telemetry
- [x] Implement BLE-assisted Wi-Fi and MQTT provisioning
- [x] Persist Wi-Fi and ESPectre Protocol settings on the native firmware path
- [x] Move ESPectre Protocol helpers and ESP-IDF protocol services into shared runtime layers
- [x] Keep the streamer firmware on a narrow Wi-Fi-only streaming path without a separate BLE, MQTT, or OTA control surface
- [x] Publish MQTT telemetry, status, info, stats, and command results from native firmware
- [x] Align `micro-espectre` MQTT payloads and commands with the ESPectre Protocol baseline
- [x] Adapt the existing web monitor into a protocol validation and MQTT dashboard client
- [ ] Define web orchestration profiles, per-device service credentials, MQTT-over-TLS policy, and privacy boundary for device telemetry
- [ ] Design tenant, home/location, room, and device ownership model
- [ ] Implement social login and account management
- [ ] Implement secure Web Bluetooth assisted device claim flow
- [ ] Build telemetry ingestion path for derived sensing state and device status
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

Last update: **July 22, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
