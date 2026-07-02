# Architecture Guide

This document explains the internal code layout introduced with the `core / runtime / frontend` refactor, why the project was split this way, and how the new structure enables multiple firmware targets without duplicating the motion-detection logic.

It also clarifies how the same split supports the current platform direction:

- reusable sensing logic in `core`
- multiple runtime implementations behind a stable contract
- multiple frontends such as `ESPHome`, `Matter`, and custom firmware adapters
- a future orchestration layer that can consume signals from multiple deployed devices

---

## Goals

The refactor was driven by five practical goals:

1. Keep the current ESPHome behavior stable for end users.
2. Make the motion-detection logic reusable by future frontends such as Matter.
3. Isolate ESP-IDF-specific orchestration from algorithmic code.
4. Make the shared core embeddable as a standalone library building block.
5. Create a clean foundation for multi-device product surfaces beyond a single firmware integration.

This started as an internal architectural split, not a user-visible product rename. The existing ESPHome component identity remains the same, but the same structure now also acts as the foundation for additional frontends and future orchestration layers.

---

## Source Layout

```text
src/
├── core/
├── runtime/
│   └── esp_idf/
└── frontend/
    ├── esphome/
    │   └── espectre/
    ├── native/
    │   ├── app/
    │   └── espectre/
    ├── matter/
    │   └── espectre/
    └── streamer/
        ├── app/
        └── espectre/
```

Dependency shape:

```text
┌────────────────────────────────────────────────────────────┐
│ FRONTEND                                                   │
│                                                            │
│  ESPHome frontend  native frontend  Matter frontend  Streamer frontend │
│  src/cpp/frontend/esphome/espectre  .../native/...  .../matter/...     │
└───────────────────────────┬────────────────────────────────┘
                            │ uses
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RUNTIME CONTRACT                                           │
│                                                            │
│  IEspectreRuntime + RuntimeSnapshot + events/capabilities  │
└───────────────────────────┬────────────────────────────────┘
                            │ implemented by
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RUNTIME IMPLEMENTATIONS                                    │
│                                                            │
│  ESP-IDF runtime (`EspIdfRuntime` today)                   │
│   ├─ CSIManager                                            │
│   ├─ WiFiLifecycleManager                                  │
│   ├─ Fixed-subcarrier threshold bootstrap                  │
│   ├─ TrafficGeneratorManager                               │
│   └─ UDPListener                                           │
└───────────────────────────┬────────────────────────────────┘
                            │ drives / embeds
                            ▼
┌────────────────────────────────────────────────────────────┐
│ CORE                                                       │
│                                                            │
│  BaseDetector / MVSDetector / MLDetector / filters / math  │
└────────────────────────────────────────────────────────────┘
```

This source tree describes the firmware-side architecture inside the repository.
A future local service or web orchestration layer would sit above deployed
devices and consume normalized state or events exposed through those frontend
surfaces rather than reaching into `core` directly.

### `src/cpp/core/`

`core` contains reusable detection logic and domain primitives:

- detectors such as `MVSDetector` and `MLDetector`
- signal-processing helpers such as filters and turbulence math
- threshold/domain constants
- model features and exported ML weights

Design rule: code in `core` should not depend on ESPHome-specific types and should avoid leaking platform orchestration concerns.

This is the part that can be embedded by other applications as a standalone library or SDK-style module.

### `src/cpp/runtime/`

`runtime` contains the runtime contract plus concrete execution environments:

- CSI ingress and normalization
- Wi-Fi lifecycle handling
- gain lock
- startup calibration orchestration
- traffic generation / UDP listener
- runtime facade and event contract

The shared runtime contract stays in `src/cpp/runtime/`, while the current ESP-IDF-specific implementation lives in `src/cpp/runtime/esp_idf/` and is currently implemented by `EspIdfRuntime`. Ecosystem-facing frontends share `RuntimeFrontendController`, which owns the common setup/loop/shutdown path, snapshot/capability cache, threshold updates, and runtime event bookkeeping. Frontends talk to that frontend-oriented runtime surface, not through `CSIManager`, `WiFiLifecycleManager`, or other low-level managers directly.

Shared runtime helpers also live here:

- `runtime_config_utils.*` for threshold validation and stable mode names
- `runtime_diagnostics.*` for common runtime diagnostic key/value fields
- `periodic_sensing_status_logger.*` for the shared progress-bar sensing status log used by `ESPHome`, `BLE`, and `Matter`
- `esp_idf/standalone_wifi_manager.*` for standalone ESP-IDF STA setup, CSI Wi-Fi policy, BSSID/channel fast scan, and retry behavior used by firmware targets that own their Wi-Fi stack
- `espectre_protocol.*`, `ble_protocol.h`, and `mqtt_transport.h` for the
  shared device protocol model, BLE GATT mapping constants, and transport
  boundary reused by ESP-IDF firmware targets
- `esp_idf/protocol/` for ESP-IDF protocol services such as NimBLE bindings,
  NVS-backed provisioning storage, MQTT transport implementation, and shared
  Wi-Fi provisioning command handling

### `src/cpp/frontend/esphome/espectre/`

This is the ESPHome adapter layer and external component root. It maps the
shared runtime into ESPHome/Home Assistant entities and owns the YAML schema,
codegen, and packaging metadata for the production-oriented frontend.

For frontend-specific details, see
[`src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md).

### `src/cpp/frontend/native/espectre/`

This is the standalone native adapter used by generic BLE clients, including a web
client as one example integration.

It reuses the same runtime contract as the other frontends, but maps runtime
events and controls to a custom GATT surface instead of Home Assistant entities
or Matter clusters.

The native adapter uses the shared `RuntimeFrontendController` for runtime
ownership and shared ESP-IDF protocol services for BLE transport, NVS-backed
Wi-Fi/device configuration, MQTT transport, and standalone Wi-Fi setup.

For the BLE protocol, payload shape, field semantics, and transport mapping,
see [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md). For native frontend firmware workflow
and frontend-specific operational notes, see
[`src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md).

### `src/cpp/frontend/matter/espectre/`

This is the Matter adapter and firmware entrypoint.

It maps the same runtime snapshot/events/capabilities into Matter endpoints without pulling Matter-specific concerns into `core` or the shared runtime contract.

The Matter adapter uses the shared `RuntimeFrontendController`, while Wi-Fi
ownership remains with `esp-matter`. Its app-level Wi-Fi hook only applies the
shared CSI policy when the Matter stack starts STA mode.

Architecturally, Matter is not a side experiment. It is the first explicit proof
that the new split can support a second ecosystem-facing frontend without
copying the detection pipeline or re-monolithizing the codebase.

For the Matter surface, commissioning notes, and firmware-specific workflow, see
[`src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md).

### `src/cpp/frontend/streamer/espectre/`

This is the standalone CSI streamer frontend.

Unlike the other frontends, it is not an ecosystem-facing adapter over
`IEspectreRuntime`. It uses lower-level `runtime/esp_idf` modules directly to
capture CSI and emit a compact UDP stream for host-side tools and data
collection workflows.

It still uses shared infrastructure where the behavior is identical, notably
the standalone Wi-Fi manager, CSI Wi-Fi policy, and BLE-assisted Wi-Fi
provisioning service. Its CSI capture and UDP streaming state machine stay
frontend-specific.

For the UDP packet format, frontend state machine, and Kconfig surface, see
[`src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md).

---

## Roadmap Alignment

This document is architecture-focused, but it maps directly to the current
roadmap direction:

- `v2.x`: the existing ESPHome-first motion-detection product remains stable
- `v3.x`: the project becomes a modular sensing platform where the same `core`
  can be reused across different runtimes and frontends, including `Matter`
- `v4.x`: a future local service or web orchestration layer can combine signals
  from multiple ESPectre devices, regardless of whether those nodes are exposed
  through `ESPHome`, `Matter`, or custom firmware

The important point is that the roadmap shift does not require another deep
refactor. The current code split is the enabling architecture for that
direction.

---

## Why Split Into Core, Runtime, and Frontend

Before the refactor, the ESPHome component owned nearly everything:

- detector selection
- CSI lifecycle
- calibration
- traffic generation
- BLE
- Home Assistant publishing

That made it hard to evolve the project without either:

- duplicating logic in another build target, or
- leaking ESPHome concerns into code that should have remained reusable.

The new split fixes that by drawing clearer boundaries:

- `core` answers "how do we detect motion?"
- `runtime` answers "how do we acquire/process CSI and drive the pipeline on this platform?"
- `frontend` answers "how do we expose the runtime to a specific ecosystem?"

---

## Why Multiple Runtimes Matter

Separating `runtime` from `core` is useful even when only one runtime exists today.

It gives us a stable place for platform-specific behavior that would otherwise pollute the detector layer:

- different ESP-IDF versions
- chip-family-specific CSI payload handling
- Wi-Fi API differences
- traffic generation and transport choices
- future alternate execution environments

Potential future runtimes include:

- another ESP-IDF runtime tuned for different SDK generations
- a runtime adapted to future CSI-capable Arduino support
- a Linux/Raspberry Pi runtime if CSI-capable drivers become practical
- stripped-down runtimes with fewer optional features

The important point is that frontends do not need to know which runtime implementation they are speaking to, as long as it satisfies the same runtime contract.

---

## Why Multiple Frontends Matter

The frontend layer is where ecosystem-specific integration belongs.

ESPHome and Matter solve different problems and have different constraints:

- ESPHome needs YAML/codegen, Home Assistant entities, and ESPHome component packaging.
- Matter needs endpoint/cluster mapping and must express only what the Matter model allows.

If frontend-specific logic stays in the frontend layer, the same `core` and `runtime` can power:

- the current ESPHome integration
- the current Matter frontend
- other adapters without duplicating the motion pipeline

This avoids the old pattern where a new frontend would have required copying orchestration logic out of the monolithic ESPHome component.

---

## Why This Split Also Enables Orchestration

The same boundaries that make multiple firmware targets possible also make
multi-device orchestration easier to build later.

If each deployed node shares:

- the same detector semantics from `core`
- the same runtime-facing state model
- frontend-specific adapters that translate those states into ecosystem surfaces

then a future local service can reason about devices at the level of motion,
presence, readiness, calibration state, and other normalized outputs instead of
having to understand firmware-internal implementation details per device type.

That is the architectural bridge between:

- today's firmware modularization work
- near-term multi-frontend support
- longer-term room-to-room event fusion across the home

---

## ESPectre Protocol In The Architecture

The firmware split gives each frontend a clean way to expose the same logical
device model without sharing ecosystem-specific code. ESPectre Protocol is that
logical device model at the integration boundary.

It defines message families such as telemetry, status, info, stats, commands,
and command results. It does not define separate local, self-hosted, or managed
service protocols. BLE, MQTT, MQTT over TLS, shadows, jobs, local services, and
managed services are deployment profiles or transports for the same message
semantics.

The source of truth for payload shape, topic shape, field semantics, and current
transport mapping is [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md).

---

## Local Lab Profile

The currently implemented self-hosted path is intentionally small:

```text
ESPectre device
  -> BLE for setup, recovery, and local diagnostics
  -> MQTT broker
  -> tools/web/espectre-mqtt.html
```

This profile already supports:

- BLE-assisted Wi-Fi provisioning
- BLE-assisted MQTT provisioning
- on-device persistence for Wi-Fi and MQTT settings
- MQTT telemetry, status, info, stats, and command results as defined in
  [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md)
- shared ESPectre Protocol payloads from native frontend firmware and `micro-espectre`
- `tools/web/espectre-ble.html` as the Web Bluetooth provisioning/test client, including subscription-driven live telemetry and runtime threshold tuning
- `tools/web/espectre-mqtt.html` as the browser MQTT monitor for realtime validation

A future local lab service can extend this into:

```text
ESPectre device
  -> BLE for setup, recovery, and local diagnostics
  -> MQTT broker
  -> local lab service
  -> SQLite or other lightweight local store
  -> MQTT dashboard or API
```

Practical local-lab scope:

- manual or BLE-assisted Wi-Fi provisioning
- single-site deployment
- latest device state plus recent history
- threshold updates
- local retention controls
- optional export of history

The local lab profile deliberately excludes first-wave local clones of:

- social login
- billing
- OTA fleet workflows
- complex queue orchestration
- multi-tenant account models
- image-based floor plans

---

## Web Orchestration Profile

The web orchestration profile builds on ESPectre Protocol and adds product
infrastructure around it. The same architecture should support local,
self-hosted, and future managed-service deployments:

```text
ESPectre device
  -> BLE for claim bootstrap, provisioning, and recovery
  -> MQTT over TLS for operational telemetry and commands
  -> device state mirror / shadow service
  -> ingestion and routing layer
  -> time-series telemetry store
  -> metadata store for users, homes, rooms, devices, and rules
  -> API backend and realtime UI
  -> OTA artifact and rollout service
  -> async alert workflow
```

Web-orchestration additions are profiles and services:

- user identity
- tenant/home/room/device ownership
- short-lived Web Bluetooth claim sessions
- per-device credentials and least-privilege MQTT policies
- retention controls
- alerting rules
- signed firmware artifact metadata
- OTA rollout and audit state

Candidate managed-service components remain implementation choices:

| Concern | Candidate |
|---------|-----------|
| Device MQTT ingress | AWS IoT Core |
| Device current state | AWS IoT Device Shadows |
| Device commands and OTA | AWS IoT Jobs |
| User identity | Cognito, Auth0, or Clerk |
| Application API | API Gateway + Lambda, ECS/Fargate, or AppSync |
| Near-realtime UI | AppSync subscriptions or API Gateway WebSocket |
| Time-series storage | Amazon Timestream or DynamoDB time-series pattern |
| Metadata storage | DynamoDB |
| Firmware artifacts | S3 + CloudFront |
| Async workflows | EventBridge + SQS + Lambda |
| Email alerts | SES |
| Telegram alerts | Telegram Bot API integration |
| WhatsApp alerts | WhatsApp Business Platform integration |
| Billing | Stripe |

### Device Connectivity And Policies

Devices expose two complementary connectivity surfaces:

- `BLE` for proximity-limited setup, claim, diagnostics, and recovery
- `MQTT` over TLS for normal telemetry and command flows

The normal steady-state operational path should use MQTT with per-device
credentials. Broker policies must restrict each device credential to its own
thing/shadow/jobs and MQTT topics. A device must not be able to publish as
another device or subscribe to tenant-wide topics.

Topic design should avoid human-readable tenant, user, or home names. Tenancy
belongs in credentials, policies, metadata, and backend authorization.

### Device State Mirror

The orchestration backend can keep a device state mirror derived from the current
ESPectre Protocol surfaces:

- `reported` should be built from the latest `status`, `telemetry`, `info`, and
  optional `stats` messages defined in [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md)
- `desired` can represent target configuration such as runtime threshold or
  firmware rollout intent

The device should acknowledge applied settings by copying accepted values from
`desired` to `reported` or by publishing ESPectre Protocol command results. For
the self-hosted local lab, an equivalent state mirror can be implemented by the
local service without requiring a provider-specific shadow product. The exact
payload fields should not be duplicated here; `docs/ESPECTRE_PROTOCOL.md`
remains the source of truth.

### Onboarding And Claim

Onboarding is split into two flows:

1. local provisioning
2. optional managed-service claim

Local provisioning must work without a hosted account:

1. User opens a local web app or desktop client.
2. Client connects to the ESPectre device over BLE.
3. Device exposes protocol version, firmware version, frontend, and basic
   health.
4. User provides Wi-Fi credentials and optional MQTT settings.
5. Device stores configuration and reconnects without losing BLE recovery.
6. Device connects to Wi-Fi and the configured MQTT broker.
7. Local tooling binds the device to a site or room if needed.

Managed-service claim builds on the same BLE surface:

1. User signs in to the web app.
2. User selects a home/location and starts "Add device".
3. Browser connects to the ESPectre device over Web Bluetooth.
4. Device exposes claim material such as firmware version, frontend capability,
   a device public key, and a nonce.
5. Backend creates a short-lived claim session bound to the authenticated user.
6. Browser passes claim material to the device over BLE.
7. Device exchanges the claim token for service credentials or receives a
   provisioned certificate bundle through the claim flow.
8. Device connects to the managed MQTT ingress and publishes first status.
9. Backend binds the service-side device record to the selected user, home, and room.

Security requirements:

- claim tokens must be short-lived and single-use
- pairing must require physical proximity
- long-lived service credentials must never be exposed as reusable browser secrets
- device credentials must be revocable and rotatable
- failed or abandoned claims must expire automatically
- stolen claim tokens must not allow claiming arbitrary devices

### Identity And Application Model

The managed-service profile should support social login early, because it
reduces account friction for a consumer-oriented service. Candidate providers
include Google, Microsoft, GitHub, Apple, Facebook, and LinkedIn.

The application model separates identity from tenancy:

```text
User
  -> Membership
  -> Tenant
  -> Home / Location
  -> Floor / Room / Zone
  -> Device
```

Core entities:

| Entity | Notes |
|--------|-------|
| `User` | Authenticated person |
| `Tenant` | Billing and ownership boundary |
| `Membership` | User role within a tenant |
| `Home` / `Location` | A physical site |
| `Floor` | Optional grouping for larger locations |
| `Room` / `Zone` | User-drawn area on the map |
| `Device` | Claimed ESPectre node |
| `DevicePlacement` | Device coordinates, room, label, orientation metadata |
| `TelemetrySample` | Time-series movement/status values |
| `AlertRule` | User-configured trigger |
| `AlertDelivery` | Delivery attempt and outcome |
| `FirmwareArtifact` | Signed firmware image metadata |
| `FirmwareRollout` | OTA targeting and rollout state |

### Home Map And Room Flow

The first managed-service map should avoid image uploads and keep the model
simple:

- user draws rooms/zones as rectangles or polygons
- user places devices on the map
- each device is assigned to one room or zone
- the UI shows live device state and room-level aggregate state

Approximate movement flow can be inferred from ordered room events:

```text
living_room motion -> hallway motion -> bedroom motion
```

This must be presented as best-effort activity flow, not precise localization.

### Realtime And History

Realtime view:

- online/offline status
- latest movement score
- motion state
- room-level active/idle state
- firmware/update status

History view:

- movement score over time
- motion events
- device availability
- firmware updates
- alert triggers and delivery status

Retention must be configurable and visible to users. Local lab dashboards can
start with polling; managed services can later add WebSocket or
subscription-style updates without changing the device payload model.

### Alerts

Initial alert rule:

```text
When motion is detected in selected room/device during selected schedule, send notification.
```

Delivery order:

1. Email through SES
2. Telegram bot integration
3. WhatsApp Business integration

Alerts should use asynchronous queues so notification provider failures do not
block telemetry ingestion.

### OTA And Remote Configuration

Firmware update requirements:

- firmware artifacts stored in object storage
- artifacts signed before publication
- device verifies signature before applying update
- rollout can target frontend, chip, firmware version, tenant, home, or device
- staged rollout and rollback metadata are supported
- update status is visible in the dashboard

Remote configuration requirements:

- threshold updates through desired state, command topics, or local BLE fallback
- device validates ranges before applying settings
- all remote changes are auditable
- user can restore defaults

Configuration ownership is split by channel:

- `BLE`: bootstrap, provisioning, recovery, and nearby diagnostics
- `MQTT`: routine online configuration and backend-issued commands

### Security Requirements

- per-device credentials
- least-privilege MQTT policies
- tenant isolation at every API boundary
- encrypted storage at rest
- TLS for all device and user traffic
- signed firmware artifacts
- audit log for ownership, settings, alerts, and OTA operations
- rate limiting for APIs and device ingestion
- abuse controls for alert delivery
- explicit deletion flow for accounts, homes, devices, and historical data

### Open Source Boundary

The open-source project should keep enough public surface to preserve trust and
avoid lock-in:

- firmware remains open source
- ESPectre Protocol payloads are documented
- provisioning protocol is documented
- device-side managed-service client code should be open source if shipped in
  firmware
- a minimal self-hosted local lab remains a supported option

The managed service can remain proprietary initially:

- managed backend implementation
- billing
- managed dashboard
- alert delivery orchestration
- managed OTA fleet workflows
- operational tooling

---

## Runtime Contract

The runtime contract is intentionally frontend-oriented, not platform-oriented.

Current key pieces:

- `runtime_interface.h`
- `runtime_snapshot.h`
- `runtime_events.h`
- `runtime_capabilities.h`

The frontend sees high-level operations such as:

- `setup()`
- `shutdown()`
- `loop()`
- `set_threshold_runtime()`
- `trigger_recalibration()`
- `get_snapshot()`
- `get_capabilities()`

And receives normalized events such as:

- motion state changes
- periodic updates
- threshold changes
- calibration start/finish
- runtime faults

The frontend does not manipulate low-level Wi-Fi/CSI details directly.

---

## Why The Core Can Now Be Used Standalone

The key architectural win is that `core` is no longer tied to the ESPHome component lifecycle.

That means the detector/calibration math can now be reused by:

- another firmware project
- a custom embedded application
- a future public SDK wrapper
- host-side analysis or simulation tools

In practice, "standalone SDK" here means:

- the reusable logic lives in a dedicated, stable subtree
- callers can embed it without carrying the full ESPHome adapter
- platform orchestration is optional and can be replaced

What is *not* implied yet:

- there is no separately versioned public SDK package today
- ABI/API stability outside this repository is not promised yet

So the refactor makes standalone reuse technically clean, even if packaging that reuse as a formal external SDK is still a future choice.

---

## ESPHome Packaging Note

ESPHome still expects a component-shaped directory under the external-components root.

For that reason, `src/cpp/frontend/esphome/espectre/` acts as both:

- the ESPHome frontend adapter, and
- the packaging entry point for the shared code now stored in `src/cpp/core/` and `src/cpp/runtime/`

This keeps ESPHome integration working while allowing the real source of truth to live in the product-first layout under `src/`.
It also keeps the component build reliable on Windows and in archives/checkouts where symlinks may be missing.

---

## Current Status

After the refactor:

- `components/` is no longer part of the active source tree
- the ESPHome build uses `src/cpp/frontend/esphome`
- native tests use the same frontend component root and shared `src/` layout
- the runtime contract is in place for future frontend expansion
- the repository has a clear path for custom firmware targets without copying the
  detector stack
- the same architectural split is now aligned with the roadmap's `Matter` and
  multi-device orchestration directions

Current implemented paths:

- `core`: shared detectors and math
- `runtime`: ESP-IDF runtime
- `frontend/esphome`: ESPHome adapter
- `frontend/native`: standalone native adapter + ESP-IDF firmware app
- `frontend/matter`: Matter adapter + esp-matter firmware app (experimental)

---

## Recommended Reading Order

If you are new to the project:

1. [README.md](../README.md) for product overview
2. [SETUP.md](SETUP.md) for the shared frontend chooser and install hub
3. this file for internal architecture
4. [test/cpp/README.md](../test/cpp/README.md) for validation strategy
5. [ALGORITHMS.md](ALGORITHMS.md) for algorithm details
