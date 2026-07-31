# Roadmap

This roadmap records product outcomes, release gates, and sequencing. Detailed
experiments, replay selection, mutable corpus work, and benchmark measurements
live in their owning feature, dataset, and performance documents.

## Release Horizons

| Milestone | Target | Status | Outcome |
| --- | --- | --- | --- |
| **v3.0.0** | August 2026 | In progress | Release the modular sensing platform with shared runtime layers, stable protocol services, multiple firmware frontends, and validated Classic and ML detectors |
| **v3.x** | After v3.0.0 | Planned | Improve developer workflows, integrations, hardware coverage, and sensing performance without delaying the v3.0 platform baseline |
| **v4.x** | December 2026 | Planned | Add an optional privacy-first web orchestration layer for onboarding, multi-node visibility, management, history, and alerting |
| **v5.x** | Future | Exploratory | Adopt practical IEEE 802.11bf / Wi-Fi Sensing hardware through the existing runtime and protocol architecture when embedded vendor APIs become available |

## v3.0.0 - Modular Sensing Platform

**Goal**: release a reusable Wi-Fi sensing platform with shared sensing logic,
a stable runtime contract, multiple frontend paths, and an embeddable foundation
for custom firmware and OEM products.

### Current Baseline

| Area | State | Current outcome |
| --- | --- | --- |
| **Architecture** | Ready | Shared `core`, `runtime`, ESP-IDF services, and frontend adapters are split, reviewed, and documented |
| **Frontends** | Ready with limits | ESPHome is the mature Home Assistant path; Native and Streamer are available; Matter occupancy is available with limited controller validation |
| **Protocol** | Ready | BLE and MQTT provisioning, telemetry, status, info, commands, and reusable protocol services are documented in [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) |
| **Detection** | Ready pending final evidence | Current C++ and Python real-data, long-recording, low-RSSI, packet-rate, and parity gates pass across the maintained corpus |
| **Documentation** | Ready | Setup, architecture, protocol, tuning, performance, and frontend workflows describe the current v3 surface |

Completed implementation and review work is recorded in
[CHANGELOG.md](CHANGELOG.md) and the dated documents under `docs/review/`.

### v3.0.0 Release Gate

Every open item in this section blocks the `v3.0.0` tag unless it is explicitly
moved to the post-release v3.x backlog.

#### Detector Release Evidence

- [ ] Complete ClassicDetector and MLDetector tuning
- [ ] Complete the remaining Classic replay-cache migration by replacing the
  packet-summary `detector_replay` cache with the canonical time-aware path,
  and close the row-versus-packet parity evidence for native, high-rate, and
  decimated streams before retiring `detector_replay`
- [ ] Decide and document the `v3.0.0` validation claim for the original ESP32.
  If full cross-environment support is part of the release claim, complete the
  missing collection, retraining, and validation before tagging; otherwise move
  the expanded ESP32 corpus to the v3.x validation backlog

#### Product Surface

- [x] Allow the Native BLE control surface to set the runtime
  `motion_on_hits` and `motion_off_hits` thresholds
- [x] Allow BLE to trigger Native firmware OTA through the shared HTTPS OTA
  service and release manifest used by MQTT
- [x] Make local ESP-IDF CLI builds use per-chip build directories by default
  instead of requiring `ESPECTRE_IDF_BUILD_DIR`
- [x] Add optional DNS-SD/mDNS discovery to the Streamer collection workflow,
  while keeping explicit targets as the deterministic fallback and preserving
  CSI demultiplexing by `device_id`
- [ ] Review the embeddable `C++` SDK API and documentation, and align the
  published integration surface with standard `C++` SDK conventions where
  practical
- [ ] Move esphome examples in esphome frontend

#### Release Assurance

- [ ] Complete the doc review.
- [ ] Complete the security review.
- [ ] Complete the code review.
- [ ] Refresh the Home Assistant screenshots used by the documentation and
  website, replacing the current gauge with a suitable visualization
- [ ] Finalize release notes and verify the complete binary artifact checklist
- [ ] Re-enable the `CLA Signature Check` as a required status check for
  `develop`
- [ ] Test the GitHub issue and pull request templates end to end

## v3.x - Post-Release Outcomes

These outcomes belong to the v3 series but do not block `v3.0.0`.

### Developer Experience and Distribution


- [ ] Mature the published SDK distribution path beyond raw release assets:
  decide the supported install surfaces for `PlatformIO` and ESP-IDF component
  consumers, automate any registry publication that fits the project trust
  model, and keep the `stable`, `snapshot`, and `snapshot-dev` channels aligned
  across bundle manifests, website links, and release automation
- [ ] Evaluate publishing the web BLE client as a reusable third-party
  integration artifact with ESM and IIFE builds, npm packaging, and TypeScript
  definitions

### Product Integrations

- [ ] Validate Matter commissioning across additional controllers and maintain
  the verified-controller matrix in the Matter frontend README
- [ ] Define a future Matter OTA design, including Requestor and Provider
  ownership and release artifact requirements
- [ ] Assess Matter certification readiness for manufacturer builds, including
  vendor identity, device attestation, factory provisioning, and certification
  test coverage
- [ ] Add Native frontend support for local TFT/LCD status displays
- [ ] Run an ESP32-C6 Zigbee coexistence spike before deciding whether to add a
  Zigbee occupancy frontend
- [ ] Add direct Home Assistant integration over MQTT with MQTT Discovery for
  the standalone frontends and micro-espectre
- [ ] Evaluate a TuyaOpen reference integration that embeds the shared `core`
  and `runtime`, with licensing and cloud coupling documented as integrator-side
  prerequisites

### Runtime and Hardware Coverage

- [ ] Improve Micro-ESPectre beyond its current approximately 70 pps ceiling
- [ ] Improve detector quality at high CSI packet rates so short-timescale
  information does not depend on decimation as a permanent mitigation
- [ ] Evaluate broader PHY and band support, including Wi-Fi 6 / 802.11ax
  capabilities and 5 GHz operation where hardware and exposed APIs support it
- [ ] Expand the original ESP32 corpus across dataset environments if that work
  is not retained as a `v3.0.0` release gate

## Sensing Research

Research outcomes are not release promises. A measured rejection or deferral is
a valid completion when its evidence and verdict are retained in
[FEATURES.md](FEATURES.md); external evidence belongs in
[LITERATURE.md](LITERATURE.md).

- [x] Evaluate whether CSI capture metadata provides a trustworthy correction
  for packet-to-packet gain drift; the current ESP-IDF callback exposes
  scaling configuration but no measured per-packet scale, so production
  features remain scale invariant
- [ ] Validate the leading scale-invariant, multi-axis research model against
  new paired environments and firmware resource limits before deciding whether
  to promote its feature set
- [ ] Evaluate Presence versus Empty as a distinct task using scale-invariant
  micro-motion and quiet-floor evidence
- [ ] Evaluate brief gesture detection only after the high-rate sensing path
  preserves sufficient short-timescale information, using a corpus distinct
  from motion and presence
- [ ] Evaluate breathing-related micro-motion only after the Presence versus
  Empty boundary is measurable, keeping the work explicitly non-medical

## v4.x - Web Orchestration Layer

**Goal**: make multiple ESPectre devices behave like one coherent sensing
system through an optional privacy-first web layer that can run locally,
self-hosted, or as a managed service. Raw CSI and unnecessary radio identifiers
must remain outside the default service boundary.

The sequence is Foundation, Device Plane, Product MVP, and Launch Gate. Later
experiments do not block the first public orchestration release.

### Foundation

- [ ] Define tenant, home/location, room, device ownership, and role models
- [ ] Implement account management and the initial social login path
- [ ] Define local, self-hosted, and managed deployment profiles and their
  open-source boundaries
- [ ] Complete the privacy model, threat model, retention defaults, and
  consent/cookie posture for each deployment profile

### Device Plane

- [ ] Implement secure Web Bluetooth-assisted device claim with physical
  presence and short-lived credentials
- [ ] Build ingestion for derived sensing telemetry and device status
- [ ] Provide a unified inventory across supported firmware frontends
- [ ] Add remote threshold and supported runtime-setting updates through the
  shared control plane
- [ ] Add signed firmware artifact storage and OTA workflows

### Product MVP

- [ ] Build the home map and room/device placement workflow
- [ ] Show near-realtime movement score, motion state, online/offline status,
  firmware version, and device health
- [ ] Add movement and status history with explicit retention controls
- [ ] Add motion alert rules with email as the first notification path

### Launch Gate

- [ ] Validate security, abuse resistance, privacy posture, tenant isolation,
  operational resilience, backup, and recovery
- [ ] Document deployment, self-hosting, data retention, and managed-service
  responsibilities

### Later Experiments

- [ ] Evaluate privacy-preserving multi-sensing with passive BLE presence and
  motion cues, without pairing, identity binding, or tracking
- [ ] Evaluate approximate room-to-room movement visualization from multi-node
  events without claiming precise localization
- [ ] Evaluate a server-side Matter bridge or ecosystem integration path for
  Google Home, Apple Home, and similar partners on top of the orchestrated
  backend
- [ ] Evaluate additional notification integrations after the email path is
  stable

See [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) for the shared device protocol
and [ARCHITECTURE.md](ARCHITECTURE.md) for deployment profiles and orchestration
boundaries.

## v5.x - Standards-Ready Wi-Fi Sensing

**Trigger**: an embedded Wi-Fi platform exposes practical, documented
IEEE 802.11bf / Wi-Fi Sensing measurements suitable for the ESPectre runtime.

**Intended outcome**: add a standards-backed hardware/runtime backend while
preserving higher-level ESPectre protocol, frontend, tooling, and device-maker
contracts.

Exploration should:

- map standardized measurements to runtime snapshots and events
- preserve frontend and protocol compatibility
- measure whether the new hardware improves calibration, false-positive
  control, and multi-node fusion
- document migration from ESP32 CSI firmware without promising a delivery date
  before suitable hardware APIs exist

## Roadmap Updates

Last update: **July 30, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
