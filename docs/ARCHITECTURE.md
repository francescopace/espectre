# Architecture

This reference is for contributors and firmware integrators who need the current code layout, dependency boundaries, and execution model. Use [SDK.md](SDK.md) for the integration contract and [SETUP.md](SETUP.md) for installation.

In this document, `core` means portable detector logic, `runtime` means the execution and event layer around it, and `frontend` means an ecosystem-specific adapter such as ESPHome or Matter. Historical rationale lives in the [README.md](adr/README.md); this page describes only the current structure.

## Source layout

```text
src/cpp/
├── core/
├── runtime/
│   └── esp_idf/
└── frontend/
    ├── esphome/
    ├── native/
    └── matter/
```

```text
Frontend -> Runtime -> Core
```

The portable runtime contracts and detector logic compile on a host without ESP-IDF. Platform implementations live under `runtime/esp_idf/` and implement the shared runtime boundaries; portable code must not depend on that subtree. Frontends select a backend through `RuntimeFrontendController` and do not call `core` directly.

## Layer responsibilities

### `src/cpp/core/`

`core` contains reusable sensing logic and domain primitives:

- `LightweightDetector` and `HighAccuracyDetector`
- `TemporalCsiSampler`, which admits at most one packet per configured slot
- feature extraction and detector math
- filters and helper utilities
- exported ML artifacts and related constants

`core` must stay independent of frontend schemas, network transports, and platform services.

Shared code uses the portable sink contract in `core/espectre_log.h`. The frontend registers its logging backend before runtime setup; with no sink, shared logging is silent. [SDK.md](SDK.md#logging) defines sink lifetime and callback requirements.

### `src/cpp/runtime/`

`runtime` owns the execution environment around the shared detectors:

- CSI ingestion, normalization, and temporal admission before detector input
- AGC-active sensing path
- startup calibration orchestration
- traffic generation or packet ingress hooks
- runtime snapshots, capabilities, and events
- common runtime-facing configuration validation

The shared runtime layer owns the frontend-facing contract. The ESP-IDF implementation under `src/cpp/runtime/esp_idf/` runs sensing and capability-gated raw collection for every maintained C++ frontend.

Shared runtime services also live here, including:

- `RuntimeFrontendController`
- standalone Wi-Fi helpers for non-ESPHome firmware
- shared diagnostics helpers
- ESPectre Protocol model and shared Direct HTTP/MQTT transport support
- NVS-backed provisioning helpers reused by ESP-IDF frontends

### Shared Wi-Fi and CSI lifecycle

`WiFiLifecycleManager` owns the shared CSI radio policy and coordinates association changes with traffic and capture startup. `csi_traffic_service` owns traffic selection, lifecycle, and counters through the portable `ICsiTrafficGenerator` and `ICsiTrafficIngress` boundaries. ESP-IDF adapters implement transmission and UDP ingress.

CSI callbacks validate and normalize frames before enqueueing them. The runtime loop admits samples and runs the detector; raw collection uses a separate bounded queue and HTTP worker. [CSI.md](CSI.md) describes the radio lifecycle, source selection, capture validation, and normalization. [ALGORITHMS.md](ALGORITHMS.md#detector-timing) defines temporal admission.

### Shared protocol and transport services

`FrontendCommandEngine` is the C++ command owner below the frontend adapters. Native MQTT, Native Direct, the shared Direct bridge, and ESPHome entities construct the same typed request and receive the same structured result and change set; Matter inherits the same path through the shared bridge. Commands execute serially on the existing frontend task. Queries return only through the requesting adapter, while accepted mutations publish the affected state families to every active adapter. MQTT and each Direct client keep independent outbound queues because transport backpressure is independent of command semantics.

`EspectreCapabilityProfile` is the single C++ catalog for executable Direct methods, published event families, and visible configuration sections. Serialization and command enforcement consume the same profile.

The shared Direct service owns HTTP request lifetime, SSE delivery, deferred responses, and the raw CSI session. Transport adapters retain their own connection and queue state. [SDK.md](SDK.md#transport-adapters) describes deferred-request lifetimes.

Peer-assisted discovery keeps orchestration out of `core`. `runtime/peer_discovery` owns bounded validation, deterministic deduplication, sorting, and serialization; `runtime/esp_idf/peer_discovery_service_esp_idf` owns the asynchronous DNS-SD browse; and `runtime/esp_idf/mdns_bootstrap_responder` owns the shared IPv4 bootstrap response through the existing Espressif responder. Frontend shutdown and Wi-Fi reconfiguration release pending discovery work without retaining a peer inventory. [`DISCOVERY.md`](DISCOVERY.md#dns-sd-and-mdns) owns the advertisement, bootstrap wire behavior, request and result schemas, limits, and compatibility rules.

### `src/cpp/frontend/`

`frontend` maps the runtime into a concrete ecosystem or firmware surface.

Frontend-specific schemas, transport bindings, and ecosystem integration belong here.

| Frontend | Responsibility | Local reference |
|----------|----------------|-----------------|
| ESPHome | Map YAML and entities to the shared runtime and Direct bridge; provide the external-component packaging root | [README.md](../src/cpp/frontend/esphome/README.md) |
| Native | Compose Direct, MQTT, provisioning, Home Assistant discovery, and frontend OTA adapters around the shared runtime | [README.md](../src/cpp/frontend/native/README.md) |
| Matter | Map runtime occupancy into Matter and expose the shared Direct bridge for detector controls | [README.md](../src/cpp/frontend/matter/README.md) |

Frontends own logger registration and keep the sink alive until the runtime shuts down. Shared code does not depend on ESPHome logging or ESP-IDF `esp_log`.

## Runtime contract

Frontends use `RuntimeFrontendController` and the contracts in `runtime_interface.h`, `runtime_snapshot.h`, `runtime_events.h`, and `runtime_capabilities.h`. They configure and drive the runtime through this interface and receive snapshots, motion and calibration events, and faults. They must not bypass it to control low-level Wi-Fi or CSI services.

[SDK.md](SDK.md#runtime-contract) documents the public lifecycle, capabilities, errors, and callback rules. Frontend READMEs describe which controls are exposed and persisted.

### Runtime performance diagnostics

The runtime owns cumulative capture counters, performance aggregation, and the shared diagnostic sample read by frontend and transport adapters. Keeping sampling below the frontends gives every adapter the same rates and observation window.

[SDK.md](SDK.md#diagnostics) describes the snapshots and sampling behavior. [API.md](API.md#diagnostics) defines public field names, units, and optionality; [README.md](performance/README.md) covers repeatable resource measurements.

## Protocol boundaries

ESPectre Protocol is the shared device-facing message model used by the standalone ESP-IDF frontends and related tools. [`API.md`](API.md) owns resources, operations, payloads, Direct HTTP and MQTT mappings, public limits, and version semantics. [`DISCOVERY.md`](DISCOVERY.md) owns DNS-SD, mDNS, browser bootstrap, and peer-result contracts.

For every maintained C++ frontend, protocol adapters sit at the boundary between the frontend and shared runtime layers.

## Related references

- Deployment and frontend selection: [SETUP.md](SETUP.md)
- Supported SDK surface: [SDK.md](SDK.md)
- CSI acquisition and traffic: [CSI.md](CSI.md)
- Detector behavior and troubleshooting: [ALGORITHMS.md](ALGORITHMS.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Measured detector results: [README.md](performance/README.md)
- Frontend operation: the relevant README under `src/cpp/frontend/`
