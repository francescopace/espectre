# Roadmap

## Release Plan

| Milestone | Timing | Commitment | Product outcome |
| --- | --- | --- | --- |
| **v3.0.0-rc1** | End of August 2026 | Planned | Remove first-party BLE, add local Direct HTTP/SSE, freeze the v3 contract, and validate release artifacts |
| **v3.0.0-rc2** | After `rc1` findings | Planned | Resolve targeted findings without widening the frozen v3 baseline |
| **v3.0.0** | After `rc2` validation | Planned | Ship the stable shared sensing platform and supported firmware frontends |
| **v3.1.0** | After v3.0.x triage | Planned | Expand Matter support and validate it across more controllers |
| **v3.2.0** | After v3.1.0 | Planned | Bring ESPectre to Arduino projects through a supported SDK runtime |
| **v3.3.0** | After v3.2.0 | Planned | Add a dedicated Apple Home frontend based on Espressif's HomeKit SDK |
| **v3.4.0** | After v3.3.0 and the presence research gate | Research-gated | Add validated stationary-presence detection as a distinct sensing output |
| **v3.5.0** | After v3.4.0 and candidate evaluation | Research-gated | Evaluate brief gestures and non-medical breathing-related micro-motion |
| **v4.0.0** | After v3.5.0 | Planned | Coordinate sensing nodes locally, evaluate chip-specific acceleration, and manage deployments through an optional web layer |
| **v5.0.0** | Hardware-triggered | Exploratory | Adopt IEEE 802.11bf or equivalent sensing on practical future hardware |

## v3.0.0-rc1 - First Release Candidate

**Product outcome**: complete the BLE-free Native transport cutover, freeze the intended v3.0.0 platform contract, publish the first complete candidate artifacts, and finish whole-platform release validation.

### Release Scope

The candidate covers the shared sensing architecture, runtime and protocol contracts, supported firmware frontends, release artifacts, and embeddable SDK surface intended for v3.0.0. Before the candidate is published, standard Improv Serial provisioning and the versioned local Direct HTTP API replace the first-party Native BLE surface across firmware, SDK, portal, tests, and current documentation. JSON commands use HTTP POST, processed events use SSE over streaming fetch, and ESPectre frontends expose bearer-bound raw CSI over a binary HTTP stream on supported chips. Other product capabilities move to a later minor release unless they are required to correct a release blocker.

Completed implementation and detector experiments live in [CHANGELOG.md](CHANGELOG.md) and [FEATURES.md](FEATURES.md).

**Release tasks**:

- [ ] Run a classic ESP32 DNS/UDP A/B with Direct HTTPD task priorities 1 and 4 across ESPHome, Native, and Matter; remove the target-specific priority overrides if priority 1 preserves Direct availability, response latency, and CSI occupancy
- [ ] Restore live CSI callbacks on ESP32-S3 after CSI suspend/resume across a managed Wi-Fi reconnect from a BSSID update so sensing resumes without a reboot, and keep the existing one-shot fallback reboot as a watchdog even after that path passes repeated hardware validation
- [ ] Re-run the on-device firmware benchmark on every supported chip and refresh the published performance reports

**Exit criteria**:

- [ ] Confirm `/documentation/setup/` falls through 404→home and `/guides/setup/` serves the static setup guide
- [ ] Test the GitHub issue and pull request templates end to end
- [ ] Confirm GA4 Realtime receives production events after consent
- [ ] Update the GitHub Discussion "ML Detector: architecture, training pipeline, and future direction"

## v3.0.0-rc2 - Second Release Candidate

**Product outcome**: resolve findings from the first candidate and prove that the frozen v3.0.0 contract is ready for stable release.

**Scope**: compatibility, correctness, security, packaging, documentation, and release-process fixes discovered after `v3.0.0-rc1`. The candidate does not widen the product baseline.

**Release tasks**:

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

Research answers product questions; it does not reserve release scope. Work is sequenced by prerequisite, and a measured rejection or deferral is a valid outcome. Detailed experiments and internal evidence belong in [FEATURES.md](FEATURES.md); external evidence belongs in [LITERATURE.md](LITERATURE.md).

| Order | Track | Product question | Promotion gate |
| --- | --- | --- | --- |
| R1 | **5 GHz HT20** | Can ESP32-C5 deployments make a validated 5 GHz sensing claim? | Paired dual-band captures on the same hardware and environments validate both detectors |
| R2 | **VHT20** | Can the nearest PHY extension reuse the production sensing contract safely? | Proven capture provenance, safe normalization, representative detector results, and C++/Python parity after the 5 GHz HT20 baseline |
| R3 | **Stationary presence** | Can ESPectre distinguish an occupied quiet room from an empty room? | Paired same-session data supports a scale-invariant Presence-versus-Empty boundary for `v3.4.0` |
| R4 | **Brief gestures** | Does preserved high-rate information support a distinct gesture product? | The high-rate capture path is stable, and a gesture-specific corpus passes validation for `v3.5.0` |
| R5 | **Breathing-related motion** | Are longer-window spectral features useful for non-medical micro-motion? | Stationary presence is measurable, paired recordings support longer windows, and host-side evidence justifies runtime work for `v3.5.0` |
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

Last update: **August 25, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
