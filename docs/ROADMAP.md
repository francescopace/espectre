# Roadmap

This roadmap records product outcomes, release gates, and sequencing. Detailed
experiments, replay selection, mutable corpus work, and benchmark measurements
live in their owning feature, dataset, and performance documents.

## Release Horizons


| Milestone  | Target        | Status      | Outcome                                                                                                                                                  |
| ---------- | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v3.0.0-rc1** | August 2026   | Planned     | First release candidate for the v3 platform baseline, with frozen scope and release-artifact validation                                                   |
| **v3.0.0-rc2** | Early September 2026   | Planned     | Second release candidate for targeted fixes from `rc1`, without widening the v3.0 scope                                                                   |
| **v3.0.0**     | Mid September 2026   | Planned     | First stable v3 release with shared runtime layers, stable protocol services, multiple firmware frontends, and validated Classic and ML detectors        |
| **v3.0.x**     | After v3.0.0  | Planned     | Post-release hardening, compatibility fixes, and packaging or documentation corrections that preserve the v3.0 contract                                   |
| **v3.x**       | After v3.0.0  | Planned     | Improve developer workflows, integrations, hardware coverage, and sensing performance beyond the 3.0 baseline without redefining the platform contract    |
| **v4.x**   | December 2026 | Planned     | Add an optional privacy-first web orchestration layer for onboarding, multi-node visibility, management, history, and alerting                           |
| **v5.x**   | Future        | Exploratory | Adopt practical IEEE 802.11bf / Wi-Fi Sensing hardware through the existing runtime and protocol architecture when embedded vendor APIs become available |




## v3.0.0 - Release Track

**Outcome**: ship the first stable v3 platform release with shared sensing
logic, a stable runtime contract, multiple frontend paths, and an embeddable
foundation for custom firmware and OEM products.

### Candidate Baseline


| Area              | State             | Current outcome                                                                                                                                                                                |
| ----------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**  | Ready             | Shared `core`, `runtime`, ESP-IDF services, and frontend adapters are split, reviewed, and documented                                                                                          |
| **Frontends**     | Ready with limits | ESPHome has the most complete Home Assistant surface; Native MQTT Discovery, optional Micro-ESPectre discovery, and Streamer are available; Matter occupancy has limited controller validation |
| **Protocol**      | Ready             | BLE and MQTT provisioning, telemetry, status, info, commands, and reusable protocol services are documented in [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md)                                    |
| **Detection**     | Release candidate | C++ and Python real-data, long-recording, low-RSSI, packet-rate, and parity gates pass across the maintained release corpus                                                                    |
| **Documentation** | Ready             | Setup, architecture, protocol, tuning, performance, and frontend workflows describe the current v3 surface                                                                                     |


Completed implementation and review work is recorded in
[CHANGELOG.md](CHANGELOG.md) and the dated documents under `docs/review/`.

### Release Sequence

1. Cut `3.0.0-rc1` once the candidate baseline, documentation set, and release
   artifact inventory, and runtime/hardware coverage checks are internally
   consistent.
2. Cut `3.0.0-rc2` only for targeted fixes found during `rc1` validation,
   without widening the release scope.
3. Tag `3.0.0` after the release-readiness checklist and frozen-scope
   validation pass.

### Detector Evidence

- [ ] Complete ClassicDetector and MLDetector tuning
- [x] Persist canonical time-aware Classic replay rows, and close the
  row-versus-packet parity evidence for native, high-rate, and decimated
  streams
- [ ] Expand the original ESP32 corpus across environments, retrain where the
  evidence requires it, and document the broader v3.x validation claim



### Product Surface

- [x] Allow the Native BLE control surface to set the runtime
  `motion_on_hits` and `motion_off_hits` thresholds
- [x] Allow BLE to trigger Native firmware OTA through the shared HTTPS OTA
  service and release manifest used by MQTT
- [x] Make local ESP-IDF CLI builds use per-chip build directories by default
  instead of requiring `ESPECTRE_IDF_BUILD_DIR`
- [x] Add optional DNS-SD/mDNS discovery to the Streamer collection workflow,
  while keeping explicit targets as the deterministic fallback and preserving
  CSI demultiplexing by `device_id`
- [x] Add direct Home Assistant integration over MQTT with MQTT Discovery for
  the Native frontend and Micro-ESPectre, while preserving the canonical
  ESPectre MQTT protocol
- [x] Review the embeddable `C++` SDK API and documentation, and align the
  published integration surface with standard `C++` SDK conventions where
  practical
- [x] Move the ESPHome examples into the ESPHome frontend

### RC1 Runtime and Hardware Coverage

- [ ] Improve Micro-ESPectre beyond its current approximately 70 pps ceiling
- [ ] Improve detector quality at high CSI packet rates so short-timescale
  information does not depend on decimation as a permanent mitigation
- [ ] Benchmark `SIZE` versus `PERF` compiler optimization across the
  maintained firmware frontends, and adopt performance-oriented builds where
  device-side gains justify the binary-size and fit trade-offs
- [x] Replace the per-bin trigonometric derotation in delay-compensated
  coherence with a Horner evaluation of the same polynomial, rather than the
  incremental recurrence originally proposed, which drifts more for no gain.
  Adopted after the parity and detector performance validations held
- [x] Reduce the per-packet overhead in the shared runtime policy and the
  amplitude-profile helper
- [ ] Evaluate adjacent-subcarrier aggregation on the current 12-of-64 CSI path
  to reduce noise without hiding useful short-timescale or frequency-local
  structure, and adopt it only if the `C++`/Python parity and detector
  performance validations improve
- [ ] Evaluate broader PHY and band support, including Wi-Fi 6 / 802.11ax
  capabilities and 5 GHz operation where hardware and exposed APIs support it

### Release Readiness

- [ ] Complete the remaining documentation review of the 3.0 baseline.
- [ ] Complete the remaining security review of the 3.0 baseline.
- [ ] Complete the remaining code review of the 3.0 baseline.
- [ ] Complete the Google Analytics review.
- [ ] Refresh the Home Assistant screenshots used by the documentation and
  website, replacing the current gauge with a suitable visualization
- [ ] Audit the published release notes and complete binary artifact set
- [ ] Re-enable the `CLA Signature Check` as a required status check for
  `develop`
- [ ] Test the GitHub issue and pull request templates end to end



## v3.0.x - Post-Release Hardening

Version 3.0.x is reserved for post-release hardening that preserves the v3.0
contract, including compatibility fixes, packaging corrections, and
documentation or release-process follow-up discovered after the stable tag.
Concrete items land here after `3.0.0` ships.

## v3.x - Product Quality and Reach

These items start after `3.0.0` ships. They improve developer workflows,
integrations, and broader product reach beyond the 3.0 baseline without
redefining the v3 platform contract.

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
- [ ] Evaluate same-Wi-Fi peer discovery for nearby ESPectre nodes, potentially
  via ESP-NOW, and if reliable prototype multi-node broadcast coordination as
  an alternative to the classic traffic generator to reduce router airtime use
  without regressing CSI capture quality, latency, range, or interoperability
- [ ] Evaluate a TuyaOpen reference integration that embeds the shared `core`
  and `runtime`, with licensing and cloud coupling documented as integrator-side
  prerequisites



## Sensing Research

Research outcomes are not release promises. A measured rejection or deferral is
a valid completion when its evidence and verdict are retained in
[FEATURES.md](FEATURES.md); external evidence belongs in
[LITERATURE.md](LITERATURE.md).

- [x] Evaluate whether CSI capture metadata provides a trustworthy correction
  for packet-to-packet gain drift; the current ESP-IDF callback exposes
  scaling configuration but no measured per-packet scale, so production
  features remain scale invariant
- [x] Validate the leading scale-invariant, multi-axis research model against
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

Last update: **August 5, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
