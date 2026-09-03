# Roadmap

## Release Plan

| Milestone | Status | Starts when | Product outcome |
| --- | --- | --- | --- |
| **v3.0.0-rc1** | In validation | Current | Validate the first complete candidate of the shared sensing platform |
| **v3.0.0-rc2** | Planned | `rc1` findings are resolved | Close release blockers and validate the final v3 candidate |
| **v3.0.0** | Planned | `rc2` passes its release gates | Ship the supported shared sensing platform and firmware frontends |
| **v3.1.0** | Planned | v3.0.x triage is complete | Validate Matter with more controllers and define its production path |
| **v3.2.0** | Demand-gated | An external Arduino integration demonstrates the need | Bring ESPectre to Arduino projects through a supported SDK runtime |
| **v3.3.0** | Demand-gated | Apple Home over Matter leaves a documented product gap | Add a dedicated Apple Home frontend when HomeKit solves that gap |
| **v3.4.0** | Research-gated | Stationary presence passes its sensing and product gates | Add stationary presence as a distinct sensing output |
| **v3.5.0** | Research-gated | Presence is validated and at least one candidate earns promotion | Release a supported gesture or non-medical micro-motion capability |
| **v4.0.0** | Planned | The v3 sensing platform is stable | Coordinate sensing nodes on the local network |
| **v4.1.0** | Planned | The local coordination contract is stable | Add an optional, self-hostable relay for remote access |
| **v4.2.0** | Demand-gated | Multi-node deployments need managed operations | Add optional fleet, history, update, and alert workflows |
| **v5.0.0** | Exploratory | Practical sensing hardware exposes suitable measurements | Adopt IEEE 802.11bf or an equivalent sensing backend |

## v3.0.0-rc1 - First Release Candidate

**Product outcome**: validate the first complete candidate of the shared v3 sensing platform before freezing the stable release contract.

**Current state**: the shared C++ architecture, ESPHome, Native, and Matter frontends, browser tools, CLI, SDK packages, and release workflows are implemented in the current v3 candidate. The active [CHANGELOG.md](CHANGELOG.md) records the cumulative candidate behavior. Hardware, compatibility, security, corpus, and artifact findings remain release inputs until the candidate is tagged.

**Exit criteria**: publish one reproducible candidate commit and its aligned firmware, SDK, web, and compliance artifacts. Record every stable-release blocker against `v3.0.0-rc2` before changing the candidate status.

## v3.0.0-rc2 - Second Release Candidate

**Product outcome**: resolve findings from the first candidate and prove that the v3.0.0 contract is ready for stable release.

**Scope**: compatibility, correctness, security, packaging, documentation, and release-process fixes discovered during `v3.0.0-rc1`. New sensing outputs and frontends stay outside the release. Security work may narrow or protect an existing v3 surface when the current design cannot meet the stable-release boundary safely.

**Release tasks**:

- [ ] Run a classic ESP32/ESP32-S2 A/B with HTTPD task priorities 1 and 4; remove the target-specific priority overrides if priority 1 preserves Direct availability, response latency, and CSI occupancy
- [ ] Complete the v3 corpus collection backlog, including replacement `empty` captures for the low-occupancy recordings removed from the catalog and missing original ESP32 label and environment coverage; rerun the dataset-quality, training, and C++/Python parity gates on the final corpus.
- [ ] Benchmark the C++ Direct raw CSI queue with fixed 512-, 256-, and 128-byte payload bounds, then retain or reduce its internal fixed-slot size without changing the published raw-record contract or advertised capabilities.
- [ ] Authenticate Native OTA images independently of HTTPS: enable ESP-IDF signed-app verification during updates on every supported Native target, sign every channel artifact with release-managed keys, reject unsigned or invalid images, document key custody, rotation, and USB recovery, and validate upgrades, corrupt images, and rollback behavior. Record Secure Boot v2 and hardware anti-rollback as production provisioning requirements rather than silently enabling irreversible eFuse policy in general-purpose builds.
- [ ] Protect the existing `PATCH /mqtt` and `DELETE /mqtt` resources with per-device administrator pairing and encrypted Security2 sessions.

**Exit criteria**: every `rc1` release blocker is closed, required validation and release gates pass on the candidate commit, and firmware, SDK, web, and vendor artifacts are reproducible and aligned with the candidate documentation.

## v3.0.0 - Stable Release

**Product outcome**: publish the supported v3 platform baseline validated by the two release candidates.

**Scope**: release the contract and artifacts accepted in `v3.0.0-rc2`. Only fixes for stable-release blockers may land after the second candidate.

**Exit criteria**: no release blockers remain, every required gate passes on the release commit, release notes describe the final cumulative behavior and migration path, and published artifacts match the tagged source. The release evidence records first-use setup coverage, sensing readiness after Wi-Fi recovery, detector alarms and misses in the maintained replay gates, and OTA recovery for every supported path.

## v3.1.0 - Matter Compatibility and Production Readiness

**Product outcome**: make Matter commissioning and everyday operation dependable across a broader set of controllers, and define the path from the current integration to manufacturer-ready products.

**Current foundation**: the Matter frontend, commissioning flow, occupancy mapping, Direct controls, and target build and release paths exist in the v3 candidate. Controller validation remains limited, Matter OTA is not implemented, and published builds still use development identity and attestation material.

**Scope**:

- Validate commissioning across selected additional controllers, and maintain the verified-controller matrix in the Matter frontend [README.md](../src/cpp/frontend/matter/README.md)
- Define Matter OTA ownership, Requestor and Provider responsibilities, and release-artifact requirements
- Assess manufacturer certification gaps, including vendor identity, device attestation, factory provisioning, and certification test coverage

**Exit criteria**: the selected controller matrix passes or records explicit limitations, and OTA plus certification work has a documented architecture, ownership model, and specific gap list.

## v3.2.0 - Arduino SDK Runtime

**Product outcome**: let Arduino-ESP32 developers embed ESPectre through a supported SDK runtime while keeping control of their sketch, connectivity, and product behavior.

**Activation gate**: start implementation after an external Arduino integration identifies the runtime, lifecycle, packaging, and target support it needs. Until then, maintain the portable C++ core and ESP-IDF runtime as the reusable foundation without claiming Arduino support.

**Scope**:

- Add an Arduino-facing runtime adapter that reuses `RuntimeFrontendController`, `EspIdfRuntime`, and the shared detector implementation
- Keep Wi-Fi startup, reconnect policy, and product integration under the consuming sketch's control
- Reassess whether `RuntimeEventMailbox` should become public SDK API only after an external integration demonstrates the need and its event coverage, capacity, overflow, and threading semantics are stable
- Publish a clean installation path and focused examples for the supported Arduino-ESP32 target matrix

**Exit criteria**: a clean Arduino project can install, build, and run the SDK on every selected target through Arduino CLI, with sensing startup, events, reset behavior, and Wi-Fi reconnect lifecycle validated against the shared runtime contract.

## v3.3.0 - Apple Home Frontend

**Product outcome**: add a dedicated frontend that exposes ESPectre sensing in Apple Home through Espressif's `esp-homekit-sdk` without duplicating the shared runtime or detector stack.

**Activation gate**: first validate the existing Matter frontend with Apple Home under `v3.1.0`. Start a HomeKit-specific frontend only when that work identifies a user or product requirement that the standard Matter path cannot meet, and the dependency and distribution checks below pass.

**Scope**:

- Confirm the SDK's license, redistribution terms, maintained ESP-IDF compatibility, supported targets, and the boundary between the open-source and MFi product paths before adding the dependency
- Map ESPectre sensing state into supported HomeKit services and characteristics through a frontend adapter over the shared runtime
- Validate Wi-Fi provisioning, pairing, reconnect, reset, accessory identity, and recovery in Apple Home on the selected device matrix
- Document the certification, factory provisioning, credential, and OTA responsibilities that apply to commercial products

**Exit criteria**: the dependency and distribution path satisfy the project's dual-license policy, every selected target passes build and runtime validation, Apple Home behavior is recorded in a controller matrix, and the open-source and MFi product boundaries are explicit.

## v3.4.0 - Stationary Presence Detection

**Product outcome**: distinguish an occupied quiet room from an empty room as a sensing result separate from motion, without making identity, people-counting, or precise-location claims.

**Research timing**: a bounded HT20 feasibility study may run alongside v3 stabilization. It can reject or justify the longer research track, but it cannot promote a production detector. Production work still requires the capture-profile, rate, window, corpus, privacy, and parity gates in the Research Pipeline.

**Scope**:

- Validate stationary presence across representative hardware and environments using paired same-session evidence
- Promote a scale-invariant Presence-versus-Empty detector only if it generalizes across the required false-presence and missed-presence gates
- Add the validated presence state to the shared runtime, protocol, maintained frontends, and user-facing privacy guidance without changing the meaning of the existing motion state

**Exit criteria**: the detector passes its declared corpus and performance gates, C++ and Python behavior remain aligned, maintained frontends expose the same presence semantics, and unsupported inferences are explicit in current documentation. If the research gate fails, retain the result in [FEATURES.md](FEATURES.md) and defer the release scope.

## v3.5.0 - Gesture and Micro-Motion Research

**Product outcome**: release an intentional brief-gesture or non-medical breathing-related micro-motion capability only if research supports it; otherwise record a measured rejection or deferral without adding production behavior.

**Scope**:

- Evaluate brief gestures only after the high-rate capture path preserves the required short-timescale information
- Evaluate breathing-related micro-motion only after stationary presence is measurable and paired recordings support longer analysis windows
- Keep candidates in host-side research until their evidence justifies production runtime work and C++/Python parity

**Exit criteria**: each candidate has a measured promotion, rejection, or deferral decision in [FEATURES.md](FEATURES.md). Release production behavior under `3.5.0` only if at least one candidate passes its declared sensing, resource, privacy, and parity gates.

## v4.0.0 - Local Cooperative Sensing

**Product outcome**: make nearby ESPectre nodes operate as one local sensing system while each node remains useful on its own.

**Product boundary**: local sensing and coordination do not require an account, relay, or Internet connection. Raw CSI and unnecessary radio identifiers stay outside the coordination plane. Nodes exchange only the derived state and health data required by the selected coordination contract.

**Current foundation**: v3 devices advertise a stable identity and can perform bounded peer-assisted discovery for browser bootstrap. That service does not retain peer inventory, assign rooms, establish trust, coordinate traffic, or exchange sensing events.

**Scope**:

- Define node identity, room membership, capabilities, trust boundaries, failure behavior, and the minimum derived state shared between nodes
- Select or reject same-Wi-Fi, ESP-NOW, and other candidate paths using measured latency, range, interoperability, airtime, and CSI-quality evidence
- Implement the selected discovery and coordination path, including node health and degraded operation when peers disappear
- Coordinate traffic generation or derived events only when measurements show a benefit without weakening sensing quality, latency, standalone operation, or recovery

**Exit criteria**: the supported local path improves multi-node operation or reduces airtime under its declared tests. A node continues sensing when peers or optional management software disappear, and the protocol documents every shared field and failure state.

### Hardware Acceleration Track

Hardware acceleration is an independent, evidence-gated track. It does not block the portable `v4.0.0` release. ESP32-S3 is the first candidate.

- Profile the pipeline at declared CSI rates, and optimize only measured compute or memory bottlenecks
- Compare optimized and portable paths with the same captures, detector gates, traffic profiles, and benchmark method
- Preserve detector semantics, calibration, protocol and frontend compatibility, and a supported fallback path
- Keep accelerated backends within the existing dual-distribution model, without proprietary-only modules or chip-specific protocol variants
- Claim processing or airtime gains only when end-to-end measurements support them

**Track exit criteria**: the accelerated backend shows a reproducible improvement in processing rate, detector capacity, analysis-window length, or operational headroom while passing the shared sensing, compatibility, and reliability gates. Record a failed gate in the owning ledger and keep the portable path unchanged.

## v4.1.0 - Self-Hostable Relay

**Product outcome**: give operators remote access to their devices through an optional relay they can run themselves.

**Relay boundary**: a device opens an authenticated outbound WSS connection, and the browser opens WSS to the same relay. The relay carries control, status, and derived sensing, but never raw CSI. Local Direct HTTP remains the default and works when the relay is unavailable.

**Scope**:

- Publish one protocol for the device, browser, self-hosted relay, and any later managed implementation
- Define device pairing, revocable credentials, authorization, origin policy, bounded queues, heartbeat, reconnect and resume behavior, rate limits, and credential recovery
- Provide a self-hosted deployment with complete operator documentation and no dependency on `relay.espectre.dev`

**Exit criteria**: devices and browsers reconnect safely through authenticated WSS, revoked credentials stop working, queues remain bounded, and relay failure does not interrupt local sensing or Direct HTTP.

## v4.2.0 - Managed Deployment Operations

**Product outcome**: manage multi-node installations through optional hosted workflows for ownership, status, updates, history, and alerts.

**Activation gate**: begin the managed service after the `v4.1.0` protocol is stable and multi-node deployments show a need for remote fleet operations. `relay.espectre.dev` implements the published relay protocol; it does not define a private device protocol.

**Scope**:

- Add tenants, locations, rooms, device ownership, roles, and accounts
- Ingest derived telemetry and status, support approved remote settings, and store signed release artifacts for OTA workflows
- Add room views, history with retention controls, and email alerts
- Define tenant isolation, abuse controls, observability, regional and retention policy, backup, recovery, service responsibilities, and a reviewed threat model

**Exit criteria**: an operator can onboard, observe, configure, and update a multi-node deployment without exporting raw CSI. Security, privacy, tenant-isolation, resilience, backup, recovery, and deployment reviews pass before public launch. Local and self-hosted operation remain independent of the managed service.

### Post-Launch Candidates

- Privacy-preserving passive BLE motion or presence cues without pairing, identity binding, or tracking
- Approximate room-to-room movement views without claims of precise localization
- A server-side Matter bridge or partner integration above the orchestration backend
- Additional notification channels after email is stable

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

The first presence study uses the current HT20 path to decide whether deeper work is justified. It is a bounded feasibility test, not a production promotion path. The main pipeline then evaluates HE20 and HT40, higher CSI rates, and longer analysis windows before the formal stationary-presence gate. A rejection or deferral is a valid result. Detailed experiments and internal evidence belong in [FEATURES.md](FEATURES.md); external evidence belongs in [LITERATURE.md](LITERATURE.md).

| Order | Track | Product question | Promotion gate |
| --- | --- | --- | --- |
| R0 | **HT20 stationary-presence feasibility** | Does paired same-session HT20 evidence justify further presence research? | A small, predeclared corpus measures false presence and missed presence across more than one room and link condition; the result can continue or stop the research track, but cannot enter production |
| R1 | **HE20 and HT40 sensing profiles** | Do HE20 or HT40 provide enough additional sensing detail to justify their capture and processing costs? | Both layouts map to canonical detector inputs, paired captures characterize their benefits and costs, and any promoted profile has a defined grid, normalization path, resource limits, and C++/Python parity |
| R2 | **Higher CSI rate** | Which sustained CSI rate preserves useful micro-motion information on supported hardware? | Rate sweeps select the highest useful rate within declared limits for loss, jitter, compute, memory, and transport load |
| R3 | **Longer and multi-scale windows** | Can longer windows expose slow micro-motion without weakening the current movement response? | The runtime analyzes short and long windows within declared latency and memory limits while preserving the movement detector's response time |
| R4 | **Stationary presence** | Can the selected capture profile distinguish an occupied quiet room from an empty room? | Paired same-session data supports a scale-invariant Presence-versus-Empty boundary for `v3.4.0` across the required hardware and environments |
| R5 | **Breathing-related motion** | Can the selected capture profile detect non-medical breathing-related micro-motion over longer windows? | Stationary presence is measurable, paired recordings cover the required observation period, and host-side evidence justifies runtime work for `v3.5.0` |
| R6 | **Brief gestures** | Does the higher-rate profile preserve enough short-timescale information for a distinct gesture product? | The high-rate capture path is stable, and a gesture-specific corpus passes validation for `v3.5.0` |

R0 may run during v3 release work and may stop the presence track before the more expensive capture-profile studies. Complete R1 through R3 before promoting stationary presence through R4. Each track may end in promotion, rejection, or deferral; R4 uses the best supported profile. R5 also requires a validated presence baseline and the longer-window path. R6 can proceed once R2 is stable.

Prototype each candidate on the host and record its verdict in [FEATURES.md](FEATURES.md). Add production C++ and device-side Python behavior only when the evidence justifies parity work.

## Ownership and Updates

This file owns product outcomes, release gates, and sequencing. Mutable details remain in their narrowest source of truth:

- [FEATURES.md](FEATURES.md) for feature experiments and promotion decisions
- [LITERATURE.md](LITERATURE.md) for external research
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) and [ML_TRAINING.md](ML_TRAINING.md) for corpus and training workflows
- [performance/](performance/) for current benchmark evidence
- [API.md](API.md), [DISCOVERY.md](DISCOVERY.md), and [ARCHITECTURE.md](ARCHITECTURE.md) for stable system contracts
- [CHANGELOG.md](CHANGELOG.md) for shipped behavior

Last update: **September 3, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
