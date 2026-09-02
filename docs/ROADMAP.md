# Roadmap

## Release Plan

| Milestone | Timing | Commitment | Product outcome |
| --- | --- | --- | --- |
| **v3.0.0-rc2** | After `rc1` findings | Planned | Resolve targeted findings without widening the frozen v3 baseline |
| **v3.0.0** | After `rc2` validation | Planned | Ship the stable shared sensing platform and supported firmware frontends |
| **v3.1.0** | After v3.0.x triage | Planned | Expand Matter support and validate it across more controllers |
| **v3.2.0** | After v3.1.0 | Planned | Bring ESPectre to Arduino projects through a supported SDK runtime |
| **v3.3.0** | After v3.2.0 | Planned | Add a dedicated Apple Home frontend based on Espressif's HomeKit SDK |
| **v3.4.0** | After v3.3.0 and the presence research gate | Research-gated | Add validated stationary-presence detection as a distinct sensing output |
| **v3.5.0** | After v3.4.0 and candidate evaluation | Research-gated | Evaluate brief gestures and non-medical breathing-related micro-motion |
| **v4.0.0** | After v3.5.0 | Planned | Coordinate sensing nodes locally, evaluate chip-specific acceleration, and manage deployments through an optional web layer |
| **v5.0.0** | Hardware-triggered | Exploratory | Adopt IEEE 802.11bf or equivalent sensing on practical future hardware |

## v3.0.0-rc2 - Second Release Candidate

**Product outcome**: resolve findings from the first candidate and prove that the frozen v3.0.0 contract is ready for stable release.

**Scope**: compatibility, correctness, security, packaging, documentation, and release-process fixes discovered after `v3.0.0-rc1`. The candidate does not widen the product baseline.

**Release tasks**:

- [ ] Run a classic ESP32/ESP32-S2 DNS/UDP A/B with Direct HTTPD task priorities 1 and 4 across ESPHome, Native, and Matter; remove the target-specific priority overrides if priority 1 preserves Direct availability, response latency, and CSI occupancy
- [ ] Investigate a driver-level CSI recovery after a managed Wi-Fi reconnect, validate it on every supported chip before adding it to the runtime, and keep the existing one-shot fallback reboot as the production watchdog
- [ ] Complete the v3 corpus collection backlog, including replacement `empty` captures for the low-occupancy recordings removed from the catalog and missing original ESP32 label and environment coverage; rerun the dataset-quality, training, and C++/Python parity gates on the final corpus.
- [ ] Benchmark the C++ Direct raw CSI queue with fixed 512-, 256-, and 128-byte payload bounds, then retain or reduce its internal fixed-slot size without changing the published raw-record contract or advertised capabilities.

**Exit criteria**: every `rc1` release blocker is closed, required validation and release gates pass on the candidate commit, and firmware, SDK, web, and vendor artifacts are reproducible and aligned with the candidate documentation.

## v3.0.0 - Stable Release

**Product outcome**: publish the supported v3 platform baseline validated by the two release candidates.

**Scope**: release the contract and artifacts accepted in `v3.0.0-rc2`. Only fixes for stable-release blockers may land after the second candidate.

**Exit criteria**: no release blockers remain, every required gate passes on the release commit, release notes describe the final cumulative behavior and migration path, and published artifacts match the tagged source.

## v3.1.0 - Matter Integration

**Product outcome**: make Matter commissioning and everyday operation dependable across a broader set of controllers, and define the path from the current integration to manufacturer-ready products.

**Scope**:

- Validate commissioning across selected additional controllers, and maintain the verified-controller matrix in the Matter frontend [README.md](../src/cpp/frontend/matter/README.md)
- Define Matter OTA ownership, Requestor and Provider responsibilities, and release-artifact requirements
- Assess manufacturer certification gaps, including vendor identity, device attestation, factory provisioning, and certification test coverage

**Exit criteria**: the selected controller matrix passes or records explicit limitations, and OTA plus certification work has a documented architecture, ownership model, and actionable gap list.

## v3.2.0 - Arduino SDK Runtime

**Product outcome**: let Arduino-ESP32 developers embed ESPectre through a supported SDK runtime while keeping control of their sketch, connectivity, and product behavior.

**Scope**:

- Add an Arduino-facing runtime adapter that reuses `RuntimeFrontendController`, `EspIdfRuntime`, and the shared detector implementation
- Keep Wi-Fi startup, reconnect policy, and product integration under the consuming sketch's control
- Reassess whether `RuntimeEventMailbox` should become public SDK API only after an external integration demonstrates the need and its event coverage, capacity, overflow, and threading semantics are stable
- Publish a clean installation path and focused examples for the supported Arduino-ESP32 target matrix

**Exit criteria**: a clean Arduino project can install, build, and run the SDK on every selected target through Arduino CLI, with sensing startup, events, reset behavior, and Wi-Fi reconnect lifecycle validated against the shared runtime contract.

## v3.3.0 - Apple Home Frontend

**Product outcome**: add a dedicated frontend that exposes ESPectre sensing in Apple Home through Espressif's `esp-homekit-sdk` without duplicating the shared runtime or detector stack.

**Scope**:

- Confirm the SDK's license, redistribution terms, maintained ESP-IDF compatibility, supported targets, and the boundary between the open-source and MFi product paths before adding the dependency
- Map ESPectre sensing state into supported HomeKit services and characteristics through a frontend adapter over the shared runtime
- Validate Wi-Fi provisioning, pairing, reconnect, reset, accessory identity, and recovery in Apple Home on the selected device matrix
- Document the certification, factory provisioning, credential, and OTA responsibilities that apply to commercial products

**Exit criteria**: the dependency and distribution path satisfy the project's dual-license policy, every selected target passes build and runtime validation, Apple Home behavior is recorded in a controller matrix, and the open-source and MFi product boundaries are explicit.

## v3.4.0 - Stationary Presence Detection

**Product outcome**: distinguish an occupied quiet room from an empty room as a sensing result separate from motion, without making identity, people-counting, or precise-location claims.

**Scope**:

- Validate stationary presence across representative hardware and environments using paired same-session evidence
- Promote a scale-invariant Presence-versus-Empty detector only if it generalizes across the required false-presence and missed-presence gates
- Add the validated presence state to the shared runtime, protocol, maintained frontends, and user-facing privacy guidance without changing the meaning of the existing motion state

**Exit criteria**: the detector passes its declared corpus and performance gates, C++ and Python behavior remain aligned, maintained frontends expose the same presence semantics, and unsupported inferences are explicit in current documentation. If the research gate fails, retain the result in [FEATURES.md](FEATURES.md) and defer the release scope.

## v3.5.0 - Gesture and Micro-Motion Research

**Product outcome**: determine whether ESPectre can support intentional brief gestures and non-medical breathing-related micro-motion beyond stationary presence.

**Scope**:

- Evaluate brief gestures only after the high-rate capture path preserves the required short-timescale information
- Evaluate breathing-related micro-motion only after stationary presence is measurable and paired recordings support longer analysis windows
- Keep candidates in host-side research until their evidence justifies production runtime work and C++/Python parity

**Exit criteria**: each candidate has a measured promotion, rejection, or deferral decision in [FEATURES.md](FEATURES.md). Release production behavior under `3.5.0` only if at least one candidate passes its declared sensing, resource, privacy, and parity gates.

## v4.0.0 - Cooperative Sensing, Hardware Acceleration, and Web Orchestration

**Product outcome**: make ESPectre nodes cooperate as one local sensing system, use validated chip-specific acceleration where it improves sensing capacity, and manage deployments through an optional web layer.

**Product boundary**: local sensing and node coordination must not depend on the managed service. The web layer supports local, self-hosted, and managed deployment profiles. Raw CSI and unnecessary radio identifiers remain outside the default service boundary, and cooperative nodes exchange only the minimum derived state required by the supported coordination contract.

**Relay boundary**: add an optional, protocol-documented, and self-hostable WebSocket relay. A device opens an authenticated outbound WSS connection, and the browser opens WSS to the same relay. The relay carries control, status, and derived sensing only, never raw CSI. Local Direct HTTP remains the default and must work without an account, relay, or Internet connection. `relay.espectre.dev` is the managed implementation, not a distinct protocol.

**Relay gates**: define per-device pairing and revocable credentials, tenant isolation, authorization, origin policy, bounded queues, heartbeat, reconnect and resume behavior, rate limits, abuse controls, observability, regional and retention policy, threat model, and credential recovery before enabling the portal's Relay mode. Validate clean failover without causing devices to expose inbound Internet ports or making local sensing depend on relay availability.

### Delivery Sequence

| Stage | Product scope | Completion condition |
| --- | --- | --- |
| **1. Coordination contract** | Node identity, room membership, capability discovery, peer discovery options, trust boundaries, failure behavior, and the minimum derived state shared between nodes | The architecture selects or rejects same-Wi-Fi, ESP-NOW, or other candidate mechanisms using measured latency, range, interoperability, airtime, and CSI-quality evidence |
| **2. Cooperative node plane** | Supported local discovery, coordinated traffic generation, derived event exchange, node health, and degraded operation when peers disappear | The supported coordination path improves multi-node operation or reduces airtime without weakening sensing quality, latency, standalone operation, or recovery |
| **3. Relay foundation** | Self-hostable WSS protocol, outbound device client, browser client, pairing, per-device credentials, revocation, tenant isolation, bounded queues, heartbeat, reconnect, rate limits, and threat model | Device and browser reconnect safely through authenticated WSS, revoked credentials stop working, tenants cannot cross boundaries, and local Direct HTTP remains independent |
| **4. Product plane** | `relay.espectre.dev`, tenant, location, room, device ownership, roles, accounts, derived telemetry and status ingestion, supported remote settings, signed artifact storage, OTA workflows, room views, history with retention controls, and email alerts | A user can onboard, observe, configure, and update a multi-node deployment without exporting raw CSI or requiring the managed relay for local operation |
| **5. Launch gate** | Security, abuse resistance, privacy, tenant isolation, resilience, backup, recovery, deployment, self-hosting, and service responsibilities | Operational and security reviews pass, and every deployment profile has complete operator documentation |

### Hardware Acceleration Gate

The portable sensing path remains the baseline, with ESP32-S3 as the first acceleration candidate.

- Profile the pipeline at declared CSI rates, and optimize only measured compute or memory bottlenecks
- Compare optimized and portable paths with the same captures, detector gates, traffic profiles, and benchmark method
- Preserve detector semantics, calibration, protocol and frontend compatibility, and a supported fallback path
- Keep accelerated backends within the existing dual-distribution model, without proprietary-only modules or chip-specific protocol variants
- Claim processing or airtime gains only when end-to-end measurements support them

**Track exit criteria**: the accelerated backend demonstrates a reproducible improvement in processing rate, detector capacity, analysis-window length, or operational headroom while passing the shared sensing, compatibility, and reliability gates. A failed gate is recorded in the owning ledger and does not block the portable v4 release.

**Release exit criteria**: all five delivery stages meet their completion conditions, cooperative sensing remains functional without the web layer, the privacy boundary is enforced by default, and local or self-hosted operation does not depend on the managed service.

### Post-Launch Candidates

- Privacy-preserving passive BLE motion or presence cues without pairing, identity binding, or tracking
- Approximate room-to-room movement views without claims of precise localization
- A server-side Matter bridge or partner integration above the orchestration backend
- Additional notification channels after email is stable

The shared device contract remains owned by [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md); deployment profiles and system boundaries remain owned by [ARCHITECTURE.md](ARCHITECTURE.md).

## v5.0.0 - Future Hardware and IEEE 802.11bf

**Product outcome**: add a standards-backed sensing backend on practical future hardware while preserving the protocol, frontend, tooling, and device-maker contracts established by v3 and v4.

**Activation trigger**: an embedded Wi-Fi platform exposes practical, documented IEEE 802.11bf or equivalent vendor measurements suitable for the ESPectre runtime. No delivery date is assigned before this trigger exists.

**Scope**:

- Map standardized measurements into runtime snapshots and events
- Preserve frontend and protocol compatibility
- Measure effects on calibration, false-positive control, and multi-node fusion
- Document migration from ESP32 CSI firmware

**Exit criteria**: supported hardware and APIs are available, the new backend passes its sensing and compatibility gates, and existing product integrations can adopt it without a parallel control plane.

## Research Pipeline

Dependencies set the research order. Evaluate HE20 and HT40 together first, then test higher CSI rates and longer analysis windows. The selected capture profile is the baseline for stationary presence and breathing-related motion. A rejection or deferral is a valid result. Detailed experiments and internal evidence belong in [FEATURES.md](FEATURES.md); external evidence belongs in [LITERATURE.md](LITERATURE.md).

| Order | Track | Product question | Promotion gate |
| --- | --- | --- | --- |
| R1 | **HE20 and HT40 sensing profiles** | Do HE20 or HT40 provide enough additional sensing detail to justify their capture and processing costs? | Both layouts map to canonical detector inputs, paired captures characterize their benefits and costs, and any promoted profile has a defined grid, normalization path, resource limits, and C++/Python parity |
| R2 | **Higher CSI rate** | Which sustained CSI rate preserves useful micro-motion information on supported hardware? | Rate sweeps select the highest useful rate within declared limits for loss, jitter, compute, memory, and transport load |
| R3 | **Longer and multi-scale windows** | Can longer windows expose slow micro-motion without weakening the current movement response? | The runtime analyzes short and long windows within declared latency and memory limits while preserving the movement detector's response time |
| R4 | **Stationary presence** | Can the selected capture profile distinguish an occupied quiet room from an empty room? | Paired same-session data supports a scale-invariant Presence-versus-Empty boundary for `v3.4.0` across the required hardware and environments |
| R5 | **Breathing-related motion** | Can the selected capture profile detect non-medical breathing-related micro-motion over longer windows? | Stationary presence is measurable, paired recordings cover the required observation period, and host-side evidence justifies runtime work for `v3.5.0` |
| R6 | **Brief gestures** | Does the higher-rate profile preserve enough short-timescale information for a distinct gesture product? | The high-rate capture path is stable, and a gesture-specific corpus passes validation for `v3.5.0` |

Complete R1 through R3 before starting stationary-presence research. Each track may end in promotion, rejection, or deferral; R4 uses the best supported profile. R5 also requires a validated presence baseline and the longer-window path. R6 can proceed once R2 is stable.

Prototype each candidate on the host and record its verdict in [FEATURES.md](FEATURES.md). Add production C++ and device-side Python behavior only when the evidence justifies parity work.

## Ownership and Updates

This file owns product outcomes, release gates, and sequencing. Mutable details remain in their narrowest source of truth:

- [FEATURES.md](FEATURES.md) for feature experiments and promotion decisions
- [LITERATURE.md](LITERATURE.md) for external research
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) and [ML_TRAINING.md](ML_TRAINING.md) for corpus and training workflows
- [performance/](performance/) for current benchmark evidence
- [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for stable system contracts
- [CHANGELOG.md](CHANGELOG.md) for shipped behavior

Last update: **September 2, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
