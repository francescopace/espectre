# Roadmap

## Release plan

| Milestone | Status | Starts when | Product outcome |
| --- | --- | --- | --- |
| **v3.0.0-rc1** | Released | September 5, 2026 | Publish the first complete candidate of the shared sensing platform |
| **v3.0.0-rc2** | Released | September 16, 2026 | Validate signed firmware, sensing fixes, and SDK packaging |
| **v3.0.0-rc3** | In progress | `rc2` is published | Validate Component Registry distribution, unified release delivery, and ESP-IDF 6.x SDK support |
| **v3.0.0** | Planned | `rc3` is published | Ship the supported shared sensing platform and firmware frontends, and publish the SDK on the ESP Component Registry |
| **v3.1.0** | Planned | v3.0.0 is published and no v3.0.x release blockers remain open | Validate Matter with more controllers and define its production path |
| **v3.2.0** | Demand-gated | An external Arduino integration demonstrates the need | Bring ESPectre to Arduino projects through a supported SDK runtime |
| **v3.3.0** | Demand-gated | Apple Home over Matter leaves a documented product gap | Add a dedicated Apple Home frontend when HomeKit solves that gap |
| **v3.4.0** | Research-gated | Stationary presence passes its sensing and product gates | Add stationary presence as a distinct sensing output |
| **v3.5.0** | Research-gated | Presence is validated and at least one candidate earns promotion | Release a supported gesture or non-medical micro-motion capability |
| **v4.0.0** | Planned | The v3 sensing platform is stable | Coordinate sensing nodes on the local network |
| **v4.1.0** | Planned | The local coordination contract is stable | Add an optional, self-hostable relay for remote access |
| **v4.2.0** | Demand-gated | Multi-node deployments need managed operations | Add optional fleet, history, update, and alert workflows |
| **v5.0.0** | Exploratory | Practical sensing hardware exposes suitable measurements | Adopt IEEE 802.11bf or an equivalent sensing backend |

Demand-gated and research-gated releases do not block later ones: `v4.0.0` does not wait for `v3.2.0` to `v3.5.0`.

## v3.0.0 - Stable release

**Product outcome**: release the stable v3 platform once the release candidates are validated, the remaining dataset work is done, and the rc-era compatibility code is gone.

**Scope**: fix what `rc2` found, pass the gates below, and freeze the API and SDK: later v3 releases may only add to them. New sensing features and frontends wait for later versions.

**Remaining release gates**:

- [ ] Declare the final v3 corpus: list the missing captures by chip, environment, and label in the data collection guide, collect them, and pass dataset-quality, training, and C++/Python parity gates on the result. See the [data collection guide](ML_DATA_COLLECTION.md) and the [ML training guide](ML_TRAINING.md).
- [ ] Remove the backward-compatibility code added during the release candidates, such as saved-setting migrations, guards for removed configuration keys, and cleanup of retired Home Assistant entities, together with its tests and documentation. v3.0.0 carries no rc-era compatibility paths.

**Exit criteria**:

- No release blockers remain, and all required checks pass on the release commit.
- Published files match the tagged source and documentation.
- Tests cover first-time setup, sensing after Wi-Fi drops, false alarms and missed motion, and OTA recovery, for every supported firmware.
- A new ESP-IDF project, without the ESPectre repository, can install the SDK from the Component Registry and build the example on every supported chip.

## v3.1.0 - Matter compatibility and production readiness

**Product outcome**: make Matter work reliably with more controllers, and define what a manufacturer would need to ship a certified product.

**Today**: the Matter firmware, pairing, occupancy sensor, and Direct controls are in v3. It has been tested with few controllers, has no OTA, and uses development certificates.

**Scope**:

- Validate commissioning across selected additional controllers, and maintain the verified-controller matrix in the [Matter guide](../src/cpp/frontend/matter/README.md)
- Define Matter OTA ownership, Requestor and Provider responsibilities, and release-artifact requirements
- Assess manufacturer certification gaps, including vendor identity, device attestation, factory provisioning, and certification test coverage

**Exit criteria**: each selected controller passes or has its limits documented, and OTA and certification have a documented design and a list of what is missing.

## v3.2.0 - Arduino SDK runtime

**Product outcome**: let Arduino-ESP32 developers add ESPectre to their sketches, while keeping control of Wi-Fi and their own code.

**Starts when**: a real Arduino project shows what it needs. Until then, Arduino is not supported.

**Scope**:

- Add an Arduino-facing runtime adapter that reuses the SDK's runtime controller, the ESP-IDF runtime, and the shared detector
- Keep Wi-Fi startup, reconnect policy, and product integration under the consuming sketch's control
- Reassess whether the runtime event mailbox should become public SDK API only after an external integration demonstrates the need and its event coverage, capacity, overflow, and threading semantics are stable
- Publish a clean installation path and focused examples for the supported Arduino-ESP32 target matrix

**Exit criteria**: a new Arduino project can install, build, and run the SDK on every selected chip with Arduino CLI, and sensing, events, resets, and Wi-Fi reconnects behave as in the ESP-IDF SDK.

## v3.3.0 - Apple Home frontend

**Product outcome**: a dedicated Apple Home frontend built on Espressif's `esp-homekit-sdk`, reusing the shared sensing code.

**Starts when**: testing Matter with Apple Home (`v3.1.0`) shows a need that Matter cannot meet, and the license checks below pass.

**Scope**:

- Confirm the SDK's license, redistribution terms, maintained ESP-IDF compatibility, supported targets, and the boundary between the open-source and MFi product paths before adding the dependency
- Map ESPectre sensing state into supported HomeKit services and characteristics through a frontend adapter over the shared runtime
- Validate Wi-Fi provisioning, pairing, reconnect, reset, accessory identity, and recovery in Apple Home on the selected device matrix
- Document the certification, factory provisioning, credential, and OTA responsibilities that apply to commercial products

**Exit criteria**: the dependency fits the project's dual license, every selected chip builds and runs, Apple Home results are documented, and the line between open-source and MFi products is clear.

## v3.4.0 - Stationary presence detection

**Product outcome**: tell a room with someone sitting still apart from an empty room, as a new result separate from motion. No identification, counting, or precise location.

**Research timing**: a small feasibility study can run during v3. It can decide whether to continue, but cannot ship a detector: that still needs the gates in the [research pipeline](#research-pipeline).

**Scope**:

- Ship the capture profile, CSI rate, and window changes that the presence detector depends on (R1 to R3 in the [research pipeline](#research-pipeline)), with C++/Python parity
- Validate stationary presence across representative hardware and environments using paired same-session evidence
- Promote a scale-invariant Presence-versus-Empty detector only if it generalizes across the required false-presence and missed-presence gates
- Add the validated presence state to the shared runtime, protocol, maintained frontends, and user-facing privacy guidance without changing the meaning of the existing motion state

**Exit criteria**: the detector passes its data and performance gates, C++ and Python agree, all frontends report presence the same way, and the documentation says clearly what presence cannot tell. If the research fails, record it in the [feature ledger](FEATURES.md) and postpone this release.

## v3.5.0 - Gesture and micro-motion research

**Product outcome**: ship short gestures or non-medical breathing detection only if research supports it; otherwise record the negative result and ship nothing.

**Scope**:

- Evaluate brief gestures only after the high-rate capture path preserves the required short-timescale information
- Evaluate breathing-related micro-motion only after stationary presence is measurable and paired recordings support longer analysis windows
- Keep candidates in host-side research until their evidence justifies production runtime work and C++/Python parity

**Exit criteria**: each candidate has a measured decision (promote, reject, or defer) in the [feature ledger](FEATURES.md). `3.5.0` ships a feature only if at least one candidate passes its sensing, resource, privacy, and parity gates.

## v4.0.0 - Local cooperative sensing

**Product outcome**: nearby ESPectre devices work together as one local system, while each still works on its own.

**Limits**: no account, relay, or Internet connection needed. Devices share only results and health, never raw CSI or unnecessary radio identifiers.

**Today**: v3 devices have a stable identity and can find each other to help the browser discover them. They do not share rooms, trust, traffic, or sensing events yet.

**Scope**:

- Define node identity, room membership, capabilities, trust boundaries, failure behavior, and the minimum derived state shared between nodes
- Select or reject same-Wi-Fi, ESP-NOW, and other candidate paths using measured latency, range, interoperability, airtime, and CSI-quality evidence
- Implement the selected discovery and coordination path, including node health and degraded operation when peers disappear
- Coordinate traffic generation or derived events only when measurements show a benefit without weakening sensing quality, latency, standalone operation, or recovery

**Exit criteria**: working together measurably improves multi-device setups or reduces airtime. A device keeps sensing when its peers or management software go away, and the protocol documents every shared field and failure.

## v4.1.0 - Self-hostable relay

**Product outcome**: reach your devices from outside the home through an optional relay you can host yourself.

**How it works**: the device and the browser both connect to the relay over authenticated WebSockets (WSS). The relay carries controls, status, and results, never raw CSI. Local access keeps working without it.

**Scope**:

- Publish one protocol for the device, browser, self-hosted relay, and any later managed implementation
- Define device pairing, revocable credentials, authorization, origin policy, bounded queues, heartbeat, reconnect and resume behavior, rate limits, and credential recovery
- Provide a self-hosted deployment with complete operator documentation and no dependency on `relay.espectre.dev`

**Exit criteria**: devices and browsers reconnect safely, revoked credentials stop working, queues stay bounded, and a relay failure never stops local sensing.

## v4.2.0 - Managed deployment operations

**Product outcome**: an optional hosted service to manage installations with many devices: ownership, status, updates, history, and alerts.

**Starts when**: the `v4.1.0` protocol is stable and real multi-device installations need it. `relay.espectre.dev` uses the public relay protocol, with no private extensions.

**Scope**:

- Add tenants, locations, rooms, device ownership, roles, and accounts
- Ingest derived telemetry and status, support approved remote settings, and store signed release artifacts for OTA workflows
- Add room views, history with retention controls, and email alerts
- Define tenant isolation, abuse controls, observability, regional and retention policy, backup, recovery, service responsibilities, and a reviewed threat model

**Exit criteria**: you can add, watch, configure, and update many devices without sending raw CSI anywhere. Security, privacy, isolation, backup, and recovery reviews pass before launch. Local and self-hosted use never depend on the service.

### Post-launch candidates

- Privacy-preserving passive BLE motion or presence cues without pairing, identity binding, or tracking
- Approximate room-to-room movement views without claims of precise localization
- A server-side Matter bridge or partner integration above the orchestration backend
- Additional notification channels after email is stable

## v5.0.0 - Future hardware and IEEE 802.11bf

**Product outcome**: support the IEEE 802.11bf Wi-Fi sensing standard on future hardware, keeping the protocol, frontends, and tools from v3 and v4.

**Starts when**: an affordable Wi-Fi chip exposes documented 802.11bf (or similar) measurements. There is no date until then.

**Scope**:

- Map standardized measurements into runtime snapshots and events
- Preserve frontend and protocol compatibility
- Measure effects on calibration, false-positive control, and multi-node fusion
- Document migration from ESP32 CSI firmware

**Exit criteria**: the hardware and APIs are available, the new backend passes its sensing and compatibility gates, and existing integrations can switch to it without a second control path.

## Hardware acceleration track

Hardware acceleration is a separate track that starts only with evidence, and does not block any release. ESP32-S3 is the first candidate.

- Profile the pipeline at declared CSI rates, and optimize only measured compute or memory bottlenecks
- Compare optimized and portable paths with the same captures, detector gates, traffic profiles, and benchmark method
- Preserve detector semantics, calibration, protocol and frontend compatibility, and a supported fallback path
- Keep accelerated backends within the existing dual-distribution model, without proprietary-only modules or chip-specific protocol variants
- Claim processing or airtime gains only when end-to-end measurements support them

**Track exit criteria**: the accelerated version is reproducibly faster or leaves more headroom, and passes the same sensing, compatibility, and reliability gates. If it fails, record why and keep the portable version.

## Research pipeline

Research runs in steps. A first small study on today's HT20 capture decides whether presence research is worth pursuing. Then come other capture formats (HE20, HT40), higher packet rates, and longer windows, before the real presence gate. Stopping or postponing is a valid outcome. Experiments are recorded in the [feature ledger](FEATURES.md), published research in the [literature review](LITERATURE.md).

| Order | Track | Product question | Promotion gate |
| --- | --- | --- | --- |
| R0 | **HT20 stationary-presence feasibility** | Does paired same-session HT20 evidence justify further presence research? | A small, predeclared corpus measures false presence and missed presence across more than one room and link condition; the result can continue or stop the research track, but cannot enter production |
| R1 | **HE20 and HT40 sensing profiles** | Do HE20 or HT40 provide enough additional sensing detail to justify their capture and processing costs? | Both layouts map to canonical detector inputs, paired captures characterize their benefits and costs, and any promoted profile has a defined grid, normalization path, resource limits, and C++/Python parity |
| R2 | **Higher CSI rate** | Which sustained CSI rate preserves useful micro-motion information on supported hardware? | Rate sweeps select the highest useful rate within declared limits for loss, jitter, compute, memory, and transport load |
| R3 | **Longer and multi-scale windows** | Can longer windows expose slow micro-motion without weakening the current movement response? | The runtime analyzes short and long windows within declared latency and memory limits while preserving the movement detector's response time |
| R4 | **Stationary presence** | Can the selected capture profile distinguish an occupied quiet room from an empty room? | Paired same-session data supports a scale-invariant Presence-versus-Empty boundary for `v3.4.0` across the required hardware and environments |
| R5 | **Breathing-related motion** | Can the selected capture profile detect non-medical breathing-related micro-motion over longer windows? | Stationary presence is measurable, paired recordings cover the required observation period, and host-side evidence justifies runtime work for `v3.5.0` |
| R6 | **Brief gestures** | Does the higher-rate profile preserve enough short-timescale information for a distinct gesture product? | The high-rate capture path is stable, and a gesture-specific corpus passes validation for `v3.5.0` |

Order:

- R0 can run during v3, starting from existing `static_presence` captures that have a same-session `empty` reference, and may stop the presence track early.
- R1 to R3 come before R4. R4 uses the best capture profile found, and the changes it needs ship with `v3.4.0`.
- R5 also needs validated presence (R4) and longer windows (R3).
- R6 can start once R2 is stable.

Each step can end in promotion, rejection, or postponement. Candidates are prototyped on the host first; C++ and device code come only when the evidence justifies them.

## Ownership and updates

This file covers goals, release gates, and order. Details live elsewhere:

- [feature ledger](FEATURES.md) for feature experiments and promotion decisions
- [literature review](LITERATURE.md) for external research
- [data collection guide](ML_DATA_COLLECTION.md) and the [ML training guide](ML_TRAINING.md) for corpus and training workflows
- [performance report](performance/README.md) for current results
- [API reference](API.md), [discovery reference](DISCOVERY.md), and [architecture overview](ARCHITECTURE.md) for system contracts
- [changelog](CHANGELOG.md) for shipped behavior

Last update: **September 23, 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)
