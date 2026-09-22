# Architecture

This page shows how the C++ code is organized today: the layers, what each one does, and how they depend on each other. It is for contributors and firmware integrators. For integrating the SDK, see the [SDK guide](SDK.md); for installing a device, the [setup guide](SETUP.md); for why things are this way, the [ADR index](adr/README.md).

Three layers:

- **`core`**: the detectors and signal processing. Portable, no platform code.
- **`runtime`**: runs the detectors: Wi-Fi, CSI capture, calibration, events.
- **`frontend`**: connects the runtime to one ecosystem, such as ESPHome or Matter.

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

Each layer may use only the one below it. The detectors and runtime interfaces compile on a computer without ESP-IDF; ESP-IDF code lives in `runtime/esp_idf/`, and portable code must not depend on it. Frontends go through `RuntimeFrontendController` and never call `core` directly.

## Layer responsibilities

### `src/cpp/core/`

`core` contains reusable sensing logic and domain primitives:

- `LightweightDetector` and `HighAccuracyDetector`
- `TemporalCsiSampler`, which admits at most one packet per configured slot
- feature extraction and detector math
- filters and helper utilities
- exported ML artifacts and related constants

`core` must stay independent of frontend schemas, network transports, and platform services.

Shared code uses the portable sink contract in `core/espectre_log.h`. The frontend registers its logging backend before runtime setup; with no sink, shared logging is silent. [SDK logging](SDK.md#logging) defines sink lifetime and callback requirements.

### `src/cpp/runtime/`

`runtime` owns the execution environment around the shared detectors:

- CSI ingestion, normalization, and temporal admission before detector input
- AGC-active sensing path
- startup calibration orchestration
- traffic generation or packet ingress hooks
- runtime snapshots, capabilities, and events
- common runtime-facing configuration validation

The ESP-IDF implementation in `src/cpp/runtime/esp_idf/` runs sensing and raw collection for every C++ frontend. Shared services also live here:

- `RuntimeFrontendController`
- standalone Wi-Fi helpers for non-ESPHome firmware
- shared diagnostics helpers
- ESPectre Protocol model and shared Direct HTTP/MQTT transport support
- NVS-backed provisioning helpers reused by ESP-IDF frontends

### Shared Wi-Fi and CSI lifecycle

`WiFiLifecycleManager` owns the shared CSI radio policy and coordinates association changes with traffic and capture startup. `csi_traffic_service` owns traffic selection, lifecycle, and counters through the portable `ICsiTrafficGenerator` and `ICsiTrafficIngress` boundaries. ESP-IDF adapters implement transmission and UDP ingress.

CSI callbacks validate and normalize frames before enqueueing them. The runtime loop admits samples and runs the detector; raw collection uses a separate bounded queue and HTTP worker. The [CSI guide](CSI.md) describes the radio lifecycle, source selection, capture validation, and normalization. [detector timing](ALGORITHMS.md#detector-timing) defines temporal admission.

### Shared protocol and transport services

**Commands.** Every command, from any transport, goes through `FrontendCommandEngine`. Native MQTT, Native Direct, the shared Direct bridge (also used by Matter), and ESPHome entities all build the same request and get the same result. Commands run one at a time on the frontend task. A query answers only the client that asked; an accepted change is published to every connected transport. MQTT and each Direct client have their own outgoing queue, so a slow client does not block the others.

**Capabilities.** `EspectreCapabilityProfile` is the one list of Direct methods, event types, and configuration sections a device offers. Both the JSON output and command checks use it.

**Direct service.** It handles HTTP requests, SSE, delayed responses, and the raw CSI session; each transport keeps its own connections and queues. The [integration reference](https://espectre.dev/sdk/api/?api=sdk_integration&member=integration_transport_adapters) explains delayed-request lifetimes.

**Discovery.** `runtime/peer_discovery` validates, de-duplicates, sorts, and serializes results; `runtime/esp_idf/peer_discovery_service_esp_idf` runs the DNS-SD browse; `runtime/esp_idf/mdns_bootstrap_responder` answers the browser's bootstrap name through Espressif's responder. Nothing is kept after a search, and shutdown or a Wi-Fi change cancels pending work. The protocol is in [DNS-SD and mDNS](DISCOVERY.md#dns-sd-and-mdns).

### `src/cpp/frontend/`

`frontend` connects the runtime to one ecosystem: its configuration format, transports, and integration.

| Frontend | Responsibility | Local reference |
|----------|----------------|-----------------|
| ESPHome | Map YAML and entities to the shared runtime and Direct bridge; provide the external-component packaging root | [ESPHome guide](../src/cpp/frontend/esphome/README.md) |
| Native | Compose Direct, MQTT, provisioning, Home Assistant discovery, and frontend OTA adapters around the shared runtime | [Native guide](../src/cpp/frontend/native/README.md) |
| Matter | Map runtime occupancy into Matter and expose the shared Direct bridge for detector controls | [Matter guide](../src/cpp/frontend/matter/README.md) |

- Frontends use only the public SDK headers. Native, Matter, and ESPHome list their own sources in `src/cpp/frontend/espectre_frontend_sources.cmake`, separate from the SDK.
- Micro-ESPectre links only the core and traffic code; MicroPython handles capture, calibration, and events.
- Improv Serial, console setup, firmware version, and OTA live in `frontend/`, outside the SDK. Native and Matter use the shared Improv service. Each frontend reports its version through `frontend_firmware_version()`.
- Each frontend registers the logger and keeps it alive until the runtime stops. Shared code depends on neither ESPHome logging nor `esp_log`.

To build a frontend against an extracted SDK, see [building against an SDK bundle](CLI.md#building-against-an-sdk-bundle).

## Runtime contract

Frontends control the runtime only through `RuntimeFrontendController` and the interfaces in `runtime_interface.h`, `runtime_snapshot.h`, `runtime_events.h`, and `runtime_capabilities.h`. They get back snapshots, motion and calibration events, and faults. They must never reach around it to Wi-Fi or CSI services.

[SDK runtime contract](SDK.md#runtime-contract) documents the public lifecycle, capabilities, errors, and callback rules. Each frontend guide lists which controls it exposes and saves.

### Runtime performance diagnostics

The runtime keeps the capture counters and performance statistics, and produces the one diagnostic sample all frontends and transports read, so they all report the same numbers over the same interval.

[SDK diagnostics](SDK.md#diagnostics) describes the snapshots and sampling behavior. [API diagnostics](API.md#diagnostics) defines public field names, units, and optionality; the [performance report](performance/README.md) covers repeatable resource measurements.

## Protocol boundaries

The ESPectre Protocol is the message format shared by the frontends and tools. The [API reference](API.md) defines messages, HTTP and MQTT mapping, limits, and versions; the [discovery reference](DISCOVERY.md) defines discovery. In every C++ frontend, the protocol adapters sit between the frontend and the runtime.

## Related references

- Deployment and frontend selection: [setup guide](SETUP.md)
- Supported SDK surface: [SDK guide](SDK.md)
- CSI acquisition and traffic: [CSI guide](CSI.md)
- Detector behavior and troubleshooting: [algorithms reference](ALGORITHMS.md) and the [troubleshooting guide](TROUBLESHOOTING.md)
- Measured detector results: [performance report](performance/README.md)
- Frontend operation: the relevant README under `src/cpp/frontend/`
