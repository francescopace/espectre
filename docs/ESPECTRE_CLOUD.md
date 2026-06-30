# ESPectre Cloud Architecture

This document defines the proposed technical direction for the ESPectre cloud-capable service layer planned for `v4.x`.

ESPectre remains an open-source Wi-Fi sensing project. The managed cloud service is a separate product layer that can add convenience, fleet visibility, history, alerting, and remote management without requiring raw CSI data or other sensitive radio data to leave the user's environment.

The same architecture should also support a minimal self-hosted local lab flow.
The local-first path is not a second-class fallback. It is the simplest way to
keep ESPectre useful without any hosted dependency while preserving a clean
upgrade path to managed cloud workflows later.

## Product Goals

- Provide a managed dashboard for homes, offices, labs, and other locations.
- Let users draw a location map and place ESPectre devices in rooms or zones.
- Show near-realtime device status, firmware version, movement score, and motion state.
- Show approximate room-to-room movement from multi-device event transitions.
- Provide historical movement/status timelines with explicit retention controls.
- Allow remote threshold updates and firmware updates.
- Provide configurable motion alerts through email, Telegram, and WhatsApp.
- Create a paid service that adds value while keeping the firmware and core protocol ecosystem open.

## Non-Goals

- Do not upload raw CSI samples to the cloud for normal product operation.
- Do not provide precise indoor localization in the first cloud version.
- Do not require cloud connectivity for the existing local/open-source workflows.
- Do not make the cloud backend design a blocker for local ESPHome, Matter, BLE or custom firmware usage.
- Do not store Wi-Fi identifiers, packet-level radio traces, or serial logs unless the user explicitly opts into a future diagnostic workflow.

## Privacy Boundary

Default cloud telemetry should be derived and minimal:

| Field | Purpose |
|-------|---------|
| `device_id` | Cloud-scoped opaque identifier, not a MAC address |
| `timestamp` | Event or sample time |
| `online` | Device availability |
| `firmware_version` | Fleet visibility and update eligibility |
| `frontend` | `esphome`, `matter`, `ble`, `custom`, or future frontend label |
| `motion_state` | Boolean or enum motion state |
| `movement_score` | Derived movement metric |
| `threshold` | Current runtime threshold |
| `health` | Minimal optional diagnostics such as heap, uptime, reset reason, or RSSI bucket |

The cloud service should not collect by default:

- raw CSI I/Q samples
- SSID, BSSID, access point MAC, or router identifiers
- local IP addresses surfaced to the user interface
- full serial logs
- packet captures
- room photos
- exact physical addresses unless needed for billing or explicitly provided by the user

Movement history can reveal occupancy habits, sleep patterns, and absences from home. 
Treat it as personal data even when it contains no raw CSI.

## Local-First Design Principle

The architecture should separate:

- a proximity and recovery plane
- an operational telemetry and command plane
- optional managed cloud services

Recommended split:

- `BLE` for local proximity workflows:
  - first-time setup
  - Wi-Fi provisioning
  - device claim or binding
  - local diagnostics
  - recovery when the device is not yet online
- `MQTT` for the normal operational plane once the device has network access:
  - derived telemetry
  - online/offline status
  - threshold and runtime commands
  - backend ingestion for history, dashboards, and alerts

This keeps the firmware useful in three modes:

1. standalone local BLE-only setup or diagnostics
2. self-hosted local lab with BLE plus local MQTT
3. managed cloud with BLE-assisted onboarding plus cloud MQTT

BLE should not be treated as the primary history transport. It is the
bootstrap, proximity, and fallback surface. MQTT should remain the normal path
for backend-visible telemetry and commands.

## Reference Local Lab Architecture

The simplest self-hosted deployment should avoid recreating a full managed cloud
stack locally:

```text
ESPectre device
  -> BLE for setup, recovery, and local diagnostics
  -> local MQTT broker
  -> local lab service
  -> SQLite or other lightweight local store
  -> local dashboard or API
```

Practical local-lab scope:

- manual or BLE-assisted Wi-Fi provisioning
- single-site deployment
- latest device state plus recent history
- threshold updates
- local retention controls
- optional export of history

This path deliberately excludes first-wave local clones of:

- social login
- billing
- OTA fleet workflows
- complex queue orchestration
- multi-tenant account models
- image-based floor plans

## Reference Managed Cloud Architecture

```text
ESPectre device
  -> BLE for claim bootstrap, local provisioning, and recovery
  -> MQTT over TLS for operational telemetry and commands
  -> device state mirror / shadow service
  -> ingestion and routing layer
  -> time-series telemetry store
  -> metadata store for users, homes, rooms, devices, and rules
  -> API backend for dashboard and control-plane operations
  -> realtime UI channel
  -> OTA artifact and rollout service
  -> async alert workflow
```

Candidate managed services:

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

## Device Connectivity

Devices should expose two complementary connectivity surfaces:

- `BLE` for proximity-limited setup and recovery
- `MQTT` over TLS for normal telemetry and command flows

The split should be intentional:

- BLE solves the "device is near me but not yet online" problem
- MQTT solves the "backend needs durable operational data" problem

The normal steady-state operational path should be MQTT using per-device
credentials.

Topic design should keep tenant boundaries explicit and should avoid exposing
human-readable user or home names:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/events
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
```

AWS IoT policies should restrict each device certificate to only its own thing,
shadow, jobs, and MQTT topics. A device must not be able to publish as another
device or subscribe to tenant-wide topics.

For the self-hosted local lab, the same topic shape should remain valid even if
the broker is local and the auth model is simpler. This keeps the local and
managed paths structurally aligned.

## Shared Device Protocol

ESPectre should converge on one shared logical device protocol that can be used
across:

- the standalone BLE frontend
- a future ESPectre MQTT-capable firmware path
- the existing `micro-espectre` MQTT workflows
- the self-hosted local lab
- the managed cloud backend

The important boundary is between:

- the logical message model
- the transport that carries it

The logical protocol should be transport-agnostic. BLE and MQTT can encode the
same message families even if they use different framing or delivery patterns.

Initial message families:

- `telemetry`
- `info`
- `stats`
- `command`
- `command_result`

Minimum shared semantics:

- protocol version
- device identity
- active frontend
- movement metric
- threshold
- motion state
- runtime health summary
- command acceptance or rejection

Practical transport mapping:

- `MQTT`:
  - topic-based routing for telemetry, commands, and command results
- `BLE`:
  - proximity transport for the same logical messages through GATT

The repository already contains a useful operational starting point in the
`micro-espectre` MQTT message flow. That flow should be treated as the starting
reference for message families and local dashboard behavior, but not as an
untouchable compatibility contract.

The target is not "BLE protocol" plus "separate MQTT protocol". The target is
one protocol model with multiple transports.

### Compatibility Direction

Strict backward compatibility is not required for the first protocol
consolidation pass.

If the current BLE or `micro-espectre` MQTT message shapes make the shared
protocol harder to define, they should be changed. It is acceptable to break the
current ad hoc protocol surfaces in order to establish a cleaner, versioned, and
reusable protocol baseline.

That means the project may intentionally replace:

- legacy command names tied to older internals
- inconsistent response payload shapes
- transport-specific naming that does not generalize
- message layouts that omit core state needed by other frontends

The priority is a coherent shared protocol, not preserving historical quirks.

### Protocol Cleanup Targets

The first unified protocol revision should address these issues explicitly:

- add a clear protocol version to all structured messages
- use one uniform command envelope
- use one uniform command-result envelope
- include device identity and frontend identity in structured payloads where useful
- avoid legacy naming that is too tied to older segmentation internals
- align the BLE telemetry surface with the MQTT telemetry surface at the logical level
- keep message semantics stable even if BLE and MQTT use different framing details

The consolidated protocol should prefer clear domain terms such as:

- `movement_score`
- `motion_state`
- `threshold`
- `detector`
- `health`

instead of transport-local or historically narrow names when a cleaner shared
term is available.

## BLE Frontend Role

The standalone BLE frontend should evolve from a lightweight transport-only
surface into the canonical proximity workflow surface for cloud-capable and
local-first deployments.

BLE responsibilities should include:

- advertising the device and protocol version
- exposing current runtime telemetry for nearby clients
- serving basic sysinfo and health diagnostics
- provisioning Wi-Fi credentials
- provisioning or updating MQTT broker settings
- allowing local threshold updates
- initiating claim or local binding flows
- supporting recovery when Wi-Fi or MQTT configuration is broken

BLE should remain intentionally small and should not become the primary transport
for history or backend analytics.

Reasonable next BLE additions include:

- Wi-Fi scan and selection
- Wi-Fi credential write and validation
- MQTT endpoint and topic-base configuration
- configuration summary readback
- reboot or reconnect commands
- explicit claim bootstrap payload exchange

Whether the BLE command encoding stays ASCII-oriented or moves to a compact
structured payload is an implementation choice, but provisioning commands should
be robust to SSIDs, passwords, and endpoint values that contain delimiters or
spaces.

For the shared protocol direction, BLE should move toward carrying the same
logical command and response model used by MQTT. A transport-specific compact
encoding is still acceptable, but it should map cleanly to the same message
families and field semantics.

## Device State Mirror Model

The managed backend can keep a device state mirror or shadow with the latest
runtime view:

```json
{
  "state": {
    "reported": {
      "firmware_version": "3.0.0",
      "frontend": "matter",
      "online": true,
      "motion_state": "idle",
      "movement_score": 0.18,
      "threshold": 5.0,
      "uptime_s": 3821
    },
    "desired": {
      "threshold": 4.5,
      "firmware_version": "3.1.0"
    }
  }
}
```

The device should acknowledge applied settings by copying accepted values from
`desired` to `reported` or by publishing command results.

For the self-hosted local lab, an equivalent state mirror can be implemented by
the local lab service without requiring a cloud-specific shadow product. The key
requirement is the state model, not a specific vendor feature.

## Onboarding And Device Claim

Web Bluetooth is a strong first option because it proves physical proximity and
can reuse patterns already explored by the BLE frontend and web game.

Onboarding should be split into two related but distinct flows:

1. local provisioning
2. optional managed-cloud claim

### Local provisioning flow

This should work even when no hosted cloud account exists yet:

1. User opens a local web app or desktop client.
2. Client connects to the ESPectre device over BLE.
3. Device exposes firmware version, frontend capability, and basic health.
4. User provides Wi-Fi credentials and optional local MQTT settings.
5. Device validates or tests network connectivity.
6. Device stores the local configuration and reboots or reconnects.
7. Device connects to the configured Wi-Fi and MQTT broker.
8. Local lab service binds the device to a room or site.

This flow is the minimum self-hosted story and should remain useful even if the
managed cloud never becomes part of the installation.

### Managed cloud claim flow

Once the device is locally reachable and provisionable, the managed-cloud claim
flow can build on the same BLE surface:

1. User signs in to the cloud web app.
2. User selects a home/location and starts "Add device".
3. Browser connects to the ESPectre device over Web Bluetooth.
4. Device exposes a claim service with a device public key, firmware version,
   frontend capability, and a nonce.
5. Backend creates a short-lived claim session bound to the authenticated user.
6. Browser passes claim material to the device over BLE.
7. Device exchanges the claim token for cloud credentials or receives a
   provisioned certificate bundle through the claim flow.
8. Device connects to AWS IoT Core and publishes its first status.
9. Backend binds the cloud thing to the selected user, home, and room.

Security requirements:

- claim tokens must be short-lived and single-use
- pairing must require physical proximity
- long-lived cloud credentials must never be exposed as reusable browser secrets
- device certificates must be revocable and rotatable
- failed or abandoned claims must expire automatically
- stolen claim tokens must not allow claiming arbitrary devices

Additional provisioning requirements:

- the local provisioning path must remain functional without a managed account
- Wi-Fi credentials should be writable only over the proximity channel
- BLE should remain available for recovery after broken MQTT or Wi-Fi updates
- local and managed onboarding should reuse as much protocol surface as possible

## Identity And Accounts

The cloud service should support social login early, because it reduces account friction for a consumer-oriented service.

Candidate identity providers:

- Google
- Microsoft
- GitHub
- Apple
- Facebook
- LinkedIn

The application model should separate identity from tenancy:

```text
User
  -> Membership
  -> Account / Tenant
  -> Home / Location
  -> Floor / Room / Zone
  -> Device
```

This allows future shared homes, family access, labs, offices, installers, and small business accounts without changing the device ownership model later.

## Application Data Model

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

## Home Map

The first version should avoid image uploads and keep the model simple:

- user draws rooms/zones as rectangles or polygons
- user places devices on the map
- each device is assigned to one room or zone
- the UI shows live device state and room-level aggregate state

Approximate movement flow can be inferred from ordered room events:

```text
living_room motion -> hallway motion -> bedroom motion
```

This should be presented as a best-effort activity path, not as precise localization.

## Realtime And History

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

Retention should be configurable and visible to users.

For the local lab, polling-based UI updates are acceptable for the first
version. Managed cloud can later add WebSocket or subscription-style updates
without changing the device payload model.

## Dashboard Starting Point

The first local lab dashboard should start from the existing MQTT-backed web
monitor already present in the repository rather than from a brand new frontend.

Current starting point:

- `tools/web/espectre-monitor.html`

Why start there:

- it already connects to an MQTT broker over WebSocket
- it already subscribes to telemetry and response topics
- it already publishes runtime commands
- it already visualizes movement, threshold, state, and runtime statistics
- it is a fast path for validating the self-hosted local lab before introducing
  a larger product UI

The initial dashboard roadmap should be:

1. align the page with the unified protocol
2. clean up any assumptions tied to the current ad hoc `micro-espectre` payloads
3. use it as the first local-lab control surface
4. later decide whether it evolves into the long-term dashboard or remains a
   development-oriented reference client

The dashboard should therefore be treated as:

- the first working lab UI
- a protocol validation client
- a reference consumer for the shared MQTT message model

It does not need to be the final product dashboard to be the correct starting
point.

## Alerts

Initial alert rule:

```text
When motion is detected in selected room/device during selected schedule, send notification.
```

Delivery order:

1. Email through SES
2. Telegram bot integration
3. WhatsApp Business integration

Alerts should use asynchronous queues so notification provider failures do not block telemetry ingestion.

## OTA And Remote Configuration

Firmware update requirements:

- firmware artifacts stored in S3
- artifacts signed before publication
- device verifies signature before applying update
- rollout can target frontend, chip, firmware version, tenant, home, or device
- staged rollout and rollback metadata are supported
- update status is visible in the dashboard

Remote configuration requirements:

- threshold updates through desired shadow state, command topics, or local BLE
  fallback
- device validates ranges before applying settings
- all remote changes are auditable
- user can restore defaults

Configuration ownership should be split by channel:

- `BLE`: bootstrap, provisioning, recovery, and nearby diagnostics
- `MQTT`: routine online configuration and backend-issued commands

This avoids overloading BLE with backend concerns while keeping the device
recoverable when it is not yet online.

## Security Requirements

- per-device credentials
- least-privilege IoT policies
- tenant isolation at every API boundary
- encrypted storage at rest
- TLS for all device and user traffic
- signed firmware artifacts
- audit log for ownership, settings, alerts, and OTA operations
- rate limiting for APIs and device ingestion
- abuse controls for alert delivery
- explicit deletion flow for accounts, homes, devices, and historical data

## Open Source Boundary

The open-source project should keep enough public surface to preserve trust and avoid lock-in:

- firmware remains open source
- cloud telemetry payloads are documented
- provisioning protocol should be documented
- device-side cloud client code should be open source if shipped in firmware
- a minimal self-hosted local lab should remain a supported option

The managed cloud service can remain proprietary initially:

- SaaS backend implementation
- billing
- managed dashboard
- alert delivery orchestration
- managed OTA fleet workflows
- operational tooling

This keeps ESPectre useful without the cloud while making the paid service a convenience and operations product.

## MVP Phases

### Phase 1: Local-First Foundation

- unified protocol definition across BLE and MQTT
- BLE-based local provisioning baseline
- MQTT ingestion
- device status and latest telemetry
- adapt existing MQTT web monitor into the first local dashboard
- self-hosted local lab service

### Phase 2: Managed Identity And Claim

- identity provider integration
- tenant/home/device data model
- Web Bluetooth claim flow
- BLE transport alignment with the shared protocol
- per-device certificate provisioning
- ownership binding
- room assignment during setup

### Phase 3: Realtime And History

- near-realtime movement dashboard
- home map editor
- device placement
- movement/status history
- retention settings

### Phase 4: Management And Alerts

- threshold updates
- signed firmware artifacts
- OTA workflow
- email alerts
- Telegram integration
- WhatsApp integration

### Phase 5: Productization

- billing and plans
- usage limits
- tenant roles
- export/delete account workflows
- security review
- operational monitoring and incident playbooks

## Open Questions

- Should the first self-hosted local lab use a lightweight broker plus SQLite, or
  should it start with a more cloud-shaped service stack?
- Should the first backend use Cognito directly, or use Auth0/Clerk for faster
  social-login product iteration?
- Should Timestream be the first time-series store, or is DynamoDB enough for
  the MVP retention and query patterns?
- Should firmware cloud connectivity be added to all frontends or introduced as
  a dedicated cloud-capable firmware profile first?
- Should Matter/ESPHome nodes connect directly to cloud, or should a local
  bridge/gateway mode be supported later?
- Should the first unified protocol encode BLE messages as JSON-shaped payloads,
  or use a compact binary framing with a documented field-equivalent schema?
- How much of the unified BLE plus MQTT device protocol should be frozen before
  the first paid beta?
- What minimum self-hosted story is needed to preserve community trust?

## Related Docs

- [ROADMAP.md](ROADMAP.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [SETUP.md](SETUP.md)
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md)
