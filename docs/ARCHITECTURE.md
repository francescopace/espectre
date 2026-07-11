# Architecture Guide

This document describes the current firmware-side architecture of ESPectre.
Keep it as a present-state guide: code layout, layer boundaries, and runtime
surfaces that exist in the repository today.

For the decision history behind this structure, use the ADR index in
[`README.md` (ADR)](adr/README.md), especially:

- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2025-12-06-adopt-a-dual-platform-development-model.md`](adr/2025-12-06-adopt-a-dual-platform-development-model.md)
- [`2025-12-06-adopt-esphome-as-the-production-integration-surface.md`](adr/2025-12-06-adopt-esphome-as-the-production-integration-surface.md)
- [`2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`](adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md)

## Current Source Layout

```text
src/cpp/
├── core/
├── runtime/
│   └── esp_idf/
└── frontend/
    ├── esphome/
    ├── native/
    ├── matter/
    └── streamer/
```

High-level dependency shape:

```text
Frontend -> Runtime contract -> Runtime implementation -> Core
```

More concretely:

```text
ESPHome / Native / Matter / Streamer frontends
  -> IEspectreRuntime + snapshots/events/capabilities
  -> runtime backend selected by RuntimeFrontendController
  -> shared runtime services
```

## Layer Responsibilities

### `src/cpp/core/`

`core` contains reusable sensing logic and domain primitives:

- `ClassicDetector` and `MLDetector`
- feature extraction and detector math
- filters and helper utilities
- exported ML artifacts and related constants

Rule of thumb: code in `core` should stay free of frontend-specific concerns
such as ESPHome entities, Matter clusters, BLE transport details, or MQTT topic
handling.

### `src/cpp/runtime/`

`runtime` owns the execution environment around the shared detectors:

- CSI ingestion and normalization
- AGC-active sensing path
- startup calibration orchestration
- traffic generation or packet ingress hooks
- runtime snapshots, capabilities, and events
- common runtime-facing configuration validation

The frontend-oriented contract lives in the shared runtime layer. The current
ESP-IDF implementations under `src/cpp/runtime/esp_idf/` include both the
motion-oriented `EspIdfRuntime` and the transport-oriented
`StreamEspIdfRuntime`.

Shared runtime services also live here, including:

- `RuntimeFrontendController`
- standalone Wi-Fi helpers for non-ESPHome firmware
- shared diagnostics helpers
- ESPectre Protocol model and shared BLE/MQTT transport support
- NVS-backed provisioning helpers reused by ESP-IDF frontends

### `src/cpp/frontend/`

`frontend` maps the runtime into a concrete ecosystem or firmware surface.

Current frontends:

- `esphome`: Home Assistant-facing external component and packaging root
- `native`: standalone BLE/MQTT firmware surface
- `matter`: Matter-facing adapter and firmware path
- `streamer`: raw CSI UDP streamer for collection workflows

Rule of thumb: frontend-specific schemas, transport bindings, and ecosystem
integration belong here, not in `core`.

## Frontend Notes

### ESPHome

`src/cpp/frontend/esphome/` maps the shared runtime into ESPHome entities,
YAML/config-codegen, and external-component packaging.

Source of truth for ESPHome workflow:
[`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)

### Native

`src/cpp/frontend/native/` exposes the runtime through the standalone BLE/MQTT
surface and reuses the shared ESP-IDF frontend-support services for
provisioning, device configuration, and OTA-related control flows.

Sources of truth:

- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md)

### Matter

`src/cpp/frontend/matter/` maps runtime state and controls into the Matter
surface without pulling Matter-specific concerns into the shared detector or
runtime layers.

Source of truth:
[`README.md` (matter)](../src/cpp/frontend/matter/README.md)

### Streamer

`src/cpp/frontend/streamer/` is a dedicated CSI transport frontend. It now uses
the same controller/runtime contract as the other standalone frontends, but it
selects `StreamEspIdfRuntime` so the raw CSI transport path can stay focused and
detector-free.

Source of truth:
[`README.md`](../src/cpp/frontend/streamer/README.md)

## Runtime Contract

The shared runtime contract is intentionally frontend-oriented.

Current key pieces:

- `runtime_interface.h`
- `runtime_snapshot.h`
- `runtime_events.h`
- `runtime_capabilities.h`

Frontend-facing operations include:

- `setup()`
- `shutdown()`
- `loop()`
- `set_threshold_runtime()`
- `trigger_recalibration()`
- `get_snapshot()`
- `get_capabilities()`

Normalized runtime events include:

- motion-state changes
- threshold changes
- calibration start and finish
- periodic status updates
- runtime faults

Frontends should use this surface instead of reaching directly into low-level
Wi-Fi or CSI pipeline services.

## ESPectre Protocol In The Architecture

ESPectre Protocol is the shared device-facing message model used by the
standalone ESP-IDF frontends and related tools.

Use [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md) as the source of truth for:

- message families
- BLE and MQTT transport mapping
- payload semantics
- OTA-related command surfaces

This file only describes where the protocol sits architecturally: at the
frontend/runtime integration boundary for the non-ESPHome standalone firmware
paths.

## Packaging Note For ESPHome

ESPHome still expects a component-shaped entry point under the external
components root. For that reason, `src/cpp/frontend/esphome/espectre/` acts as
the ESPHome packaging root even though shared source of truth now lives under
`src/cpp/core/` and `src/cpp/runtime/`.

## Current Status

Implemented today:

- shared `core` sensing logic
- shared `runtime` contract
- current ESP-IDF runtime implementation
- `esphome`, `native`, `matter`, and `streamer` frontends
- shared ESPectre Protocol support across ESP-IDF standalone frontends

For performance and detector behavior, continue in:

- [`ALGORITHMS.md`](ALGORITHMS.md)
- [`PERFORMANCE.md`](PERFORMANCE.md)
- [`TUNING.md`](TUNING.md)

## Recommended Reading Order

1. [README.md](../README.md) for the project overview
2. [SETUP.md](SETUP.md) for the shared entry points
3. this file for the current internal structure
4. [ALGORITHMS.md](ALGORITHMS.md) for detector and pipeline details
5. the relevant frontend README for operational workflow
