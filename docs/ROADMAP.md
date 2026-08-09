# Roadmap

## Release Plan

| Milestone | Timing | Commitment | Product outcome |
| --- | --- | --- | --- |
| **v3.0.0-rc1** | August 2026 | Planned | Freeze the v3 platform scope and validate release artifacts |
| **v3.0.0-rc2** | September 2026 | Planned | Resolve targeted `rc1` findings without widening the baseline |
| **v3.0.0** | October 2026 | Planned | Ship the stable shared sensing platform and supported firmware frontends |
| **v3.1.0** | After v3.0.x triage | Planned | Improve sensing quality, high-rate behavior, and device-side performance |
| **v3.2.0** | After v3.1.0 | Planned | Make the SDK and web integration surfaces easier to consume |
| **v3.3.0** | After v3.2.0 | Planned | Establish a validated Matter integration and manufacturer-readiness position |
| **v3.4.0** | After v3.3.0 | Planned | Add local status surfaces and evidence-backed cooperative node behavior |
| **v3.5.0** | Before v4.0.0, if promoted | Conditional | Promote supported embedded ecosystem integrations when a candidate qualifies |
| **v4.0.0** | After v3.4.0 | Planned | Add optional privacy-first onboarding, fleet visibility, history, and alerting |
| **v5.0.0** | Hardware-triggered | Exploratory | Adopt practical standards-backed Wi-Fi sensing without breaking product contracts |

## v3.0.0 - Platform Baseline

**Product outcome**: ship the first stable v3 platform with shared sensing logic, a stable runtime and protocol contract, multiple firmware frontends, and an embeddable foundation for custom firmware and OEM products.

### Release Scope

| Area | Readiness | Release position |
| --- | --- | --- |
| **Architecture** | Ready | Shared `core`, `runtime`, ESP-IDF services, and frontend adapters are separated and documented |
| **Frontends** | Ready with known limits | ESPHome has the broadest Home Assistant surface; Native MQTT Discovery, optional Micro-ESPectre discovery, Matter occupancy, and Streamer are available; Matter controller coverage remains limited |
| **Protocol** | Ready | BLE and MQTT provisioning, telemetry, status, info, commands, and reusable services share the documented protocol contract |
| **Detection** | Ready with known limits | The promoted seven-feature ML model passes the maintained real-data, long-recording, low-RSSI, packet-rate, and C++/Python parity gates. Classic passes the maintained normal-link paired-data, packet-rate, and parity gates; weak-link and long quiet recordings remain report-only stress diagnostics because recall drops on weak links and the C6 long-quiet false-positive tail exceeds the published target |
| **Documentation** | Release candidate | Setup, architecture, protocol, tuning, performance, and frontend workflows describe the v3 surface; final review remains open |

Completed implementation, detector experiments, and dated reviews live in [CHANGELOG.md](CHANGELOG.md), [FEATURES.md](FEATURES.md), and [review/](review/).

**Exit criteria**:

- [x] Complete ClassicDetector and MLDetector tuning, pass their maintained detector-performance and C++/Python parity gates, and document the report-only Classic weak-link and long-recording limits
- [ ] Close the remaining documentation, security, and first-party code reviews
- [x] Audit the release notes
- [ ] Re-enable the `CLA Signature Check` as required on `develop`, and test the GitHub issue and pull request templates end to end
- [ ] Verify the GA4 property-side settings and live events, and capture a current Home Assistant visualization for the project overview

## v3.0.x - Stable-Line Maintenance

**Product outcome**: keep the stable v3.0 contract dependable while feature development proceeds in later minor releases.

**Scope**: compatibility and security fixes, packaging corrections, documentation fixes, and release-process follow-up discovered after `3.0.0`. New product capabilities begin in `3.1.0` or later.

**Exit criteria**: each patch has targeted regression evidence, preserves the published v3.0 contracts, and keeps release artifacts and documentation aligned.

## v3.x - Minor Release Plan

The v3 minor line improves the product without redefining the v3 platform contract. Releases and their activities are ordered by dependency. Calendar targets are assigned only after the preceding release clears its exit criteria. Research may feed a future minor, but it enters a numbered release only after a production promotion decision.

### v3.1.0 - Sensing Quality and Performance

**Product outcome**: make sensing more robust across environments and packet rates while improving embedded throughput on maintained device paths.

**Scope**:

- Expand the original ESP32 corpus across environments, and retrain when the evidence supports it
- Preserve short-timescale information at high CSI packet rates without using permanent decimation as the product solution
- Improve Micro-ESPectre beyond its current approximately 70 packets per second ceiling while preserving detector quality and runtime stability
- Compare `SIZE` and `PERF` compiler profiles across maintained firmware frontends, and adopt performance-oriented builds only where runtime gains justify binary-size and fit costs

**Exit criteria**: detector-performance and C++/Python parity gates pass on the expanded corpus, the high-rate path meets its quality target without permanent decimation, and every adopted build optimization passes binary-fit and runtime validation on affected frontends.

### v3.2.0 - Developer Distribution

**Product outcome**: provide explicit, reproducible ways to consume ESPectre from firmware projects and third-party web integrations.

**Scope**:

- Define the supported distribution surfaces for PlatformIO and ESP-IDF component consumers, including registry publication where it fits the project trust model
- Keep `stable`, `snapshot`, and `snapshot-dev` channels aligned across bundle manifests, website links, and release automation
- Package the web BLE client with ESM, IIFE, npm, and TypeScript surfaces, or retain it in-tree with a documented rationale if a reusable package does not meet the support bar

**Exit criteria**: every supported installation path works from a clean consumer project, every release channel resolves to its intended artifacts, and the web BLE client has a validated public package or a documented decision to keep it internal.

### v3.3.0 - Matter Product Readiness

**Product outcome**: replace limited Matter coverage with a clear, evidence-backed interoperability and manufacturer-readiness position.

**Scope**:

- Validate commissioning across selected additional controllers, and maintain the verified-controller matrix in the Matter frontend [README.md](../src/cpp/frontend/matter/README.md)
- Define Matter OTA ownership, Requestor and Provider responsibilities, and release-artifact requirements
- Assess manufacturer certification gaps, including vendor identity, device attestation, factory provisioning, and certification test coverage

**Exit criteria**: the selected controller matrix passes or records explicit limitations, and OTA plus certification work has a documented architecture, ownership model, and actionable gap list.

### v3.4.0 - Local and Cooperative Operation

**Product outcome**: improve standalone usability and determine whether nearby nodes can coordinate without depending on the future web orchestration layer.

**Scope**:

- Add a non-blocking Native frontend contract for local TFT or LCD status displays
- Evaluate same-Wi-Fi peer discovery, potentially through ESP-NOW
- If discovery is reliable, prototype multi-node broadcast coordination as an alternative to the classic traffic generator that reduces router airtime

**Exit criteria**: the local display surface is documented and validated, and cooperative-node work ends with either an implementation that preserves CSI quality, latency, range, and interoperability or a measured rejection retained in [FEATURES.md](FEATURES.md).

### v3.5.0 - Embedded Ecosystem Expansion

**Product outcome**: add an embedded ecosystem integration only when it can reuse the shared runtime without weakening licensing, maintainability, or sensing quality.

**Scope**:

- Run an ESP32-C6 Zigbee coexistence spike before considering a Zigbee occupancy frontend
- Evaluate a TuyaOpen reference integration that embeds the shared `core` and `runtime`, with licensing and cloud coupling documented as integrator-side prerequisites

**Exit criteria**: each candidate has a measured promotion or rejection decision. Release `3.5.0` only when at least one candidate passes its production gates; otherwise retain the findings in [FEATURES.md](FEATURES.md) without creating an empty feature release. This conditional minor does not block `4.0.0`.

## Research Pipeline

Research answers product questions; it does not reserve release scope. Work is sequenced by prerequisite, and a measured rejection or deferral is a valid outcome. Detailed experiments and internal evidence belong in [FEATURES.md](FEATURES.md); external evidence belongs in [LITERATURE.md](LITERATURE.md).

| Order | Track | Product question | Promotion gate |
| --- | --- | --- | --- |
| R1 | **5 GHz HT20** | Can ESP32-C5 deployments make a validated 5 GHz sensing claim? | Paired dual-band captures on the same hardware and environments validate both detectors |
| R2 | **VHT20** | Can the nearest PHY extension reuse the production sensing contract safely? | Proven capture provenance, safe normalization, representative detector results, and C++/Python parity after the 5 GHz HT20 baseline |
| R3 | **Stationary presence** | Can ESPectre distinguish an occupied quiet room from an empty room? | Paired same-session data supports a scale-invariant Presence-versus-Empty boundary |
| R4 | **Brief gestures** | Does preserved high-rate information support a distinct gesture product? | The `3.1.0` high-rate path is stable, and a gesture-specific corpus passes validation |
| R5 | **Breathing-related motion** | Are longer-window spectral features useful for non-medical micro-motion? | Stationary presence is measurable, paired recordings support longer windows, and host-side evidence justifies runtime work |
| Later | **HE20** | Can a substantially different subcarrier layout map into detector inputs? | Host-side mapping, representative corpus validation, and C++/Python parity justify a new runtime path |
| Later | **HT40 and wider layouts** | Does added bandwidth justify a separate sensing contract? | Each candidate proves value against the cost of its own grid, normalization, corpus, and detector validation |

Promotion follows the project workflow: prototype host-side, retain the verdict in the feature ledger, and add production C++ and device-side Python behavior only when the evidence justifies parity work.

## v4.0.0 - Web Orchestration Layer

**Product outcome**: make multiple ESPectre devices behave like one coherent sensing system through an optional web layer for onboarding, visibility, management, history, and alerting.

**Product boundary**: the service supports local, self-hosted, and managed deployment profiles. Raw CSI and unnecessary radio identifiers remain outside the default service boundary.

### Delivery Sequence

| Stage | Product scope | Completion condition |
| --- | --- | --- |
| **1. Foundation** | Tenant, location, room, device ownership, roles, accounts, initial social login, deployment profiles and open-source boundaries, privacy, threat models, retention, consent, and cookie posture | Identity and deployment boundaries are documented, testable, and consistent across supported profiles |
| **2. Device plane** | Physical-presence Web Bluetooth claim, short-lived credentials, derived telemetry and status ingestion, device inventory, remote supported settings, signed artifact storage, and OTA workflows | A device can be securely claimed, observed, configured, and updated without exporting raw CSI |
| **3. Product MVP** | Home and room map, device placement, near-real-time movement score, motion state, connectivity, firmware version, device health, history with retention controls, and email alerts | A user can onboard devices, understand current and historical state, and configure the first alert path |
| **4. Launch gate** | Security, abuse resistance, privacy, tenant isolation, resilience, backup, recovery, deployment, self-hosting, and service responsibilities | Operational and security reviews pass, and every deployment profile has complete operator documentation |

**Exit criteria**: all four stages meet their completion conditions, the privacy boundary is enforced by default, and local or self-hosted operation does not depend on the managed service.

### Post-Launch Candidates

- Privacy-preserving passive BLE motion or presence cues without pairing, identity binding, or tracking
- Approximate room-to-room movement views without claims of precise localization
- A server-side Matter bridge or partner integration above the orchestration backend
- Additional notification channels after email is stable

The shared device contract remains owned by [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md); deployment profiles and system boundaries remain owned by [ARCHITECTURE.md](ARCHITECTURE.md).

## v5.0.0 - Standards-Backed Wi-Fi Sensing

**Product outcome**: add a standards-backed sensing backend while preserving the protocol, frontend, tooling, and device-maker contracts established by v3 and v4.

**Activation trigger**: an embedded Wi-Fi platform exposes practical, documented IEEE 802.11bf or equivalent vendor measurements suitable for the ESPectre runtime. No delivery date is assigned before this trigger exists.

**Scope**:

- Map standardized measurements into runtime snapshots and events
- Preserve frontend and protocol compatibility
- Measure effects on calibration, false-positive control, and multi-node fusion
- Document migration from ESP32 CSI firmware

**Exit criteria**: supported hardware and APIs are available, the new backend passes its sensing and compatibility gates, and existing product integrations can adopt it without a parallel control plane.

## Ownership and Updates

This file owns product outcomes, release gates, and sequencing. Mutable details remain in their narrowest source of truth:

- [FEATURES.md](FEATURES.md) for feature experiments and promotion decisions
- [LITERATURE.md](LITERATURE.md) for external research
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) and [ML_TRAINING.md](ML_TRAINING.md) for corpus and training workflows
- [performance/](performance/) for current benchmark evidence
- [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for stable system contracts
- [CHANGELOG.md](CHANGELOG.md) for shipped behavior

Last update: **August 7, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
