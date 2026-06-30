# ESPectre Cloud Architecture

This document defines the proposed technical direction for the ESPectre Cloud service planned for `v4.x`.

ESPectre remains an open-source Wi-Fi sensing project. The cloud service is a separate managed product layer that can add convenience, fleet visibility, history, alerting, and remote management without requiring raw CSI data or other sensitive radio data to leave the user's environment.

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

## Reference AWS Architecture

```text
ESPectre device
  -> AWS IoT Core MQTT over mTLS
  -> Device Shadow for current/reported/desired state
  -> IoT Rules for telemetry routing
  -> ingestion Lambda or Kinesis stream
  -> Timestream for time-series telemetry
  -> DynamoDB for users, homes, rooms, devices, and rules
  -> API backend for dashboard and control-plane operations
  -> WebSocket/AppSync channel for near-realtime UI updates
  -> IoT Jobs plus S3 firmware artifacts for OTA
  -> EventBridge/SQS/Lambda for alert workflows
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

Devices should connect with MQTT over TLS using per-device credentials.

Topic design should keep tenant boundaries explicit and should avoid exposing
human-readable user or home names:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/events
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
```

AWS IoT policies should restrict each device certificate to only its own thing,
shadow, jobs, and MQTT topics. A device must not be able to publish as another
device or subscribe to tenant-wide topics.

## Device Shadow Model

The reported state can hold the latest device runtime view:

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

The device should acknowledge applied settings by copying accepted values from `desired` to `reported` or by publishing command results.

## Onboarding And Device Claim

Web Bluetooth is a strong first option because it proves physical proximity and can reuse patterns already explored by the BLE frontend and web game.

Proposed flow:

1. User signs in to the cloud web app.
2. User selects a home/location and starts "Add device".
3. Browser connects to the ESPectre device over Web Bluetooth.
4. Device exposes a claim service with a device public key, firmware version, frontend capability, and a nonce.
5. Backend creates a short-lived claim session bound to the authenticated user.
6. Browser passes claim material to the device over BLE.
7. Device exchanges the claim token for cloud credentials or receives a provisioned certificate bundle through the claim flow.
8. Device connects to AWS IoT Core and publishes its first status.
9. Backend binds the cloud thing to the selected user, home, and room.

Security requirements:

- claim tokens must be short-lived and single-use
- pairing must require physical proximity
- long-lived cloud credentials must never be exposed as reusable browser secrets
- device certificates must be revocable and rotatable
- failed or abandoned claims must expire automatically
- stolen claim tokens must not allow claiming arbitrary devices

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

- threshold updates through desired shadow state or command topics
- device validates ranges before applying settings
- all remote changes are auditable
- user can restore defaults

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
- a minimal self-hosted reference receiver can remain a future option

The managed cloud service can remain proprietary initially:

- SaaS backend implementation
- billing
- managed dashboard
- alert delivery orchestration
- managed OTA fleet workflows
- operational tooling

This keeps ESPectre useful without the cloud while making the paid service a convenience and operations product.

## MVP Phases

### Phase 1: Cloud Foundation

- identity provider integration
- tenant/home/device data model
- MQTT ingestion
- device status and latest telemetry
- minimal dashboard

### Phase 2: Secure Onboarding

- Web Bluetooth claim flow
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

- Should the first backend use Cognito directly, or use Auth0/Clerk for faster social-login product iteration?
- Should Timestream be the first time-series store, or is DynamoDB enough for the MVP retention and query patterns?
- Should firmware cloud connectivity be added to all frontends or introduced as a dedicated cloud-capable firmware profile first?
- Should Matter/ESPHome nodes connect directly to cloud, or should a local bridge/gateway mode be supported later?
- How much of the cloud protocol should be frozen before first paid beta?
- What minimum self-hosted story is needed to preserve community trust?

## Related Docs

- [ROADMAP.md](ROADMAP.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [SETUP.md](SETUP.md)
- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md)
