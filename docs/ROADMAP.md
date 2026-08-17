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

## v3.0.0-rc1 - First Release Candidate

**Product outcome**: freeze the intended v3.0.0 platform contract, publish the first complete candidate artifacts, and finish whole-platform release validation.

### Release Scope

The candidate covers the shared sensing architecture, runtime and protocol contracts, supported firmware frontends, release artifacts, and embeddable SDK surface intended for v3.0.0. Other new product capabilities move to a later minor release unless they are required to correct a release blocker.

Completed implementation and detector experiments live in [CHANGELOG.md](CHANGELOG.md) and [FEATURES.md](FEATURES.md).

**Exit criteria**:

Before deploy:

- [ ] Test flashing firmware onto a device
- [ ] Test an OTA update (the `stable` versus `snapshot` channel is currently hardcoded; pass it with the OTA request)
- [ ] Test recalibration
- [ ] Review docs/web content, imagery, UX, and UI for release readiness
- [ ] Capture a current Home Assistant visualization for the project overview and docs
- [ ] Update devices performance report 

After deploy:

- [ ] Confirm `/documentation/setup/` falls through 404→home and `/guides/setup/` serves the static setup guide
- [ ] Test the GitHub issue and pull request templates end to end
- [ ] Confirm GA4 Realtime receives production events after consent
- [ ] Update the GitHub Discussion "ML Detector: architecture, training pipeline, and future direction"

## v3.0.0-rc2 - Second Release Candidate

**Product outcome**: resolve findings from the first candidate and prove that the frozen v3.0.0 contract is ready for stable release.

**Scope**: compatibility, correctness, security, packaging, documentation, and release-process fixes discovered after `v3.0.0-rc1`. The candidate does not widen the product baseline.

**Exit criteria**: every `rc1` release blocker is closed, required validation and release gates pass on the candidate commit, and firmware, SDK, web, and vendor artifacts are reproducible and aligned with the candidate documentation.

## v3.0.0 - Stable Release

**Product outcome**: publish the supported v3 platform baseline validated by the two release candidates.

**Scope**: release the contract and artifacts accepted in `v3.0.0-rc2`. Only fixes for stable-release blockers may land after the second candidate.

**Exit criteria**: no release blockers remain, every required gate passes on the release commit, release notes describe the final cumulative behavior and migration path, and published artifacts match the tagged source.

### v3.1.0 - Sensing Quality and Performance

**Product outcome**: make sensing more robust across environments and packet rates while improving embedded throughput on maintained device paths.

**Scope**:

- Expand the original ESP32 corpus across environments, and retrain when the evidence supports it
- Improve Micro-ESPectre beyond its current approximately 70 packets per second ceiling while preserving detector quality and runtime stability
- Compare `SIZE` and `PERF` compiler profiles across maintained firmware frontends, and adopt performance-oriented builds only where runtime gains justify binary-size and fit costs

**Exit criteria**: detector-performance and C++/Python parity gates pass on the expanded corpus, high-rate capture remains independent of detector admission, and every adopted build optimization passes binary-fit and runtime validation on affected frontends.

### v3.2.0 - Developer Distribution

**Product outcome**: provide explicit, reproducible ways to consume ESPectre from firmware projects and third-party web integrations.

**Scope**:

- Define the supported distribution surface for ESP-IDF component consumers, including registry publication where it fits the project trust model
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

## v4.0.0 - Web Orchestration Layer

**Product outcome**: make multiple ESPectre devices behave like one coherent sensing system through an optional web layer for onboarding, visibility, management, history, and alerting.

**Product boundary**: the service supports local, self-hosted, and managed deployment profiles. Raw CSI and unnecessary radio identifiers remain outside the default service boundary.

**Provider feasibility**: before locking the deployment architecture, evaluate low-cost MQTT deployment candidates, including self-hosted Eclipse Mosquitto, EMQX Cloud, HiveMQ Cloud, and AWS IoT Core, and assess full IoT platforms such as ESP RainMaker separately. Record a promotion or rejection decision based on representative fleet cost, ESPectre Protocol compatibility, TLS and per-device credentials, tenant isolation, regional availability, operational burden, licensing, portability, and migration risk.

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

## Ownership and Updates

This file owns product outcomes, release gates, and sequencing. Mutable details remain in their narrowest source of truth:

- [FEATURES.md](FEATURES.md) for feature experiments and promotion decisions
- [LITERATURE.md](LITERATURE.md) for external research
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) and [ML_TRAINING.md](ML_TRAINING.md) for corpus and training workflows
- [performance/](performance/) for current benchmark evidence
- [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for stable system contracts
- [CHANGELOG.md](CHANGELOG.md) for shipped behavior

Last update: **August 17, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
