# Roadmap

## History

| Version | Purpose |
|---------|---------|
| **v1.x** | First release demonstrating motion detection capabilities using turbulence analysis |
| **v2.x** | Home Assistant integration using ESPHome plus the custom MicroPython `Micro-ESPectre` firmware |

## Current Release

| Version | Purpose | Status | Progress |
|---------|---------|--------|----------|
| **v3.x** | Turn ESPectre into a reusable platform across frontends and runtimes | Near Release | Platform split, multi-frontend firmware paths, room-state baselines, and dataset/training workflows are landed; remaining work is mostly polish, validation depth, and broader productization |

## Next Roadmap

| Version | Purpose | Status | Progress |
|---------|---------|--------|----------|
| **v4.x** | Build an optional cloud orchestration layer across multiple ESPectre nodes | Planned | Product and technical direction are defined in [ESPECTRE_CLOUD.md](ESPECTRE_CLOUD.md); implementation has not started yet |

---

## v3.x - Modular Sensing Platform

**Goal**: move from a single integration-focused firmware to a reusable platform
with shared sensing logic, a stable runtime contract, and multiple frontend
paths.

### Contains

| Area | Scope |
|------|-------|
| **Architecture** | Shared `core`, `runtime`, and `frontend` layers |
| **Runtime contract** | Stable frontend-oriented APIs such as `IEspectreRuntime`, snapshots, events, and capabilities |
| **ESPHome frontend** | Production Home Assistant path kept on top of the shared platform |
| **BLE frontend** | Standalone custom GATT surface for generic BLE clients and web integrations |
| **Matter frontend** | Matter occupancy and diagnostics surface proving a second ecosystem-facing frontend |
| **Streamer frontend** | Standalone CSI UDP streamer for dataset collection, host tooling, and realtime fusion experiments |
| **Custom firmware path** | Ability to assemble alternate firmware targets from shared platform layers |
| **Practical sensing** | Presence and occupancy baselines, plus reusable inference/tooling foundations |
| **Host-side tooling** | Analysis tools, notebooks, datasets, and training workflows that support the platform direction |

---

## v4.x - ESPectre Cloud Orchestration Layer

**Goal**: make multiple ESPectre devices behave like one coherent home sensing system through an optional, privacy-first cloud service that adds managed realtime visibility, history, alerting, fleet management, and firmware updates without requiring raw CSI or other sensitive radio data to leave the user environment.

### Contains

| Area | Scope |
|------|-------|
| **Cloud service** | Optional managed service for multi-device orchestration, built so local/open-source usage remains viable |
| **Identity and tenancy** | User login, homes/locations, roles, and device ownership |
| **Secure device onboarding** | Physical-presence pairing, likely through Web Bluetooth, short-lived claim sessions, and per-device credentials |
| **Device visibility** | Sensor inventory, online/offline state, firmware version, runtime status, and fleet inspection |
| **Home map** | User-drawn home/office/location layout with devices placed in rooms or zones |
| **Realtime state** | Near-realtime movement score, motion state, and device health across the location |
| **Approximate room flow** | Best-effort room-to-room movement visualization from device transitions, without claiming precise localization |
| **Management** | Remote threshold updates, runtime settings, and signed firmware update workflows |
| **History** | Retained movement/status timeline with configurable privacy and retention policy |
| **Alerting** | Motion-triggered notifications through email first, then Telegram and WhatsApp as integrations mature |
| **Privacy boundary** | Derived telemetry only; no raw CSI, no unnecessary Wi-Fi identifiers, no sensitive device logs by default |
| **Cross-frontend view** | Unified view across `ESPHome`, `Matter`, `BLE`, streamer-derived tooling, and custom firmware nodes where applicable |

### Implementation Checklist

- [ ] Define cloud protocol and privacy boundary for device telemetry
- [ ] Design tenant, home/location, room, and device ownership model
- [ ] Implement social login and account management
- [ ] Implement secure Web Bluetooth assisted device claim flow
- [ ] Build cloud ingestion path for derived telemetry and device status
- [ ] Build near-realtime dashboard with home map and device placement
- [ ] Add movement score, motion state, online/offline status, and firmware version views
- [ ] Add configurable threshold updates through the device control plane
- [ ] Add signed firmware artifact storage and OTA update workflow
- [ ] Add movement/status history with explicit retention controls
- [ ] Add alerting rules for motion detection, starting with email
- [ ] Add Telegram and WhatsApp notification integrations
- [ ] Add approximate room-to-room movement visualization from multi-device events
- [ ] Document open-source boundaries, self-hosting posture, and paid managed-service value
- [ ] Validate security, abuse resistance, privacy posture, and operational resilience before public launch

See [ESPECTRE_CLOUD.md](ESPECTRE_CLOUD.md) for the proposed architecture and
technical design details.

---

## Roadmap Updates

Last update: **July 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.
