# Architecture Guide

This reference is for contributors and firmware integrators who need the current code layout, dependency boundaries, and runtime contracts. It is not an installation guide; use [SETUP.md](SETUP.md) for deployment and [EMBEDDING.md](EMBEDDING.md) for the supported SDK path.

In this document, **core** means portable detector logic, **runtime** means the execution and event layer around it, and **frontend** means an ecosystem-specific adapter such as ESPHome or Matter. Historical rationale lives in the [ADR index](adr/README.md); this page describes only the current structure.

## Current Source Layout

```text
src/cpp/
├── core/
├── runtime/
│   └── esp_idf/
│       └── frontend_support/
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

- `LightweightDetector` and `HighAccuracyDetector`
- `TemporalCsiSampler`, which admits at most one packet per configured slot
- feature extraction and detector math
- filters and helper utilities
- exported ML artifacts and related constants

Rule of thumb: code in `core` should stay free of frontend-specific concerns such as ESPHome entities, Matter clusters, BLE transport details, or MQTT topic handling.

### `src/cpp/runtime/`

`runtime` owns the execution environment around the shared detectors:

- CSI ingestion, normalization, and temporal admission before detector input
- AGC-active sensing path
- startup calibration orchestration
- traffic generation or packet ingress hooks
- runtime snapshots, capabilities, and events
- common runtime-facing configuration validation

The frontend-oriented contract lives in the shared runtime layer. The current ESP-IDF implementations under `src/cpp/runtime/esp_idf/` include both the motion-oriented `EspIdfRuntime` and the transport-oriented `StreamEspIdfRuntime`.

Shared runtime services also live here, including:

- `RuntimeFrontendController`
- standalone Wi-Fi helpers for non-ESPHome firmware
- shared diagnostics helpers
- ESPectre Protocol model and shared BLE/MQTT transport support
- NVS-backed provisioning helpers reused by ESP-IDF frontends

### Shared Wi-Fi and CSI Lifecycle

`WiFiLifecycleManager` owns the CSI-specific ESP-IDF radio policy for every frontend. It applies the protocol and HT20 bandwidth policy synchronously on `WIFI_EVENT_STA_START`, before the first association, then completes the CSI prerequisites when `IP_EVENT_STA_GOT_IP` is drained from the runtime loop. ESPHome, Native, Matter, and Streamer must not apply these radio settings in their frontend code.

The frontend or SDK integrator explicitly selects `2g`, `5g`, or `auto`; `2g` is the validated default, while `5g` and `auto` are available only on dual-band targets. The lifecycle applies that band mode first, then pins an 802.11n protocol ceiling and HT20 bandwidth on the selected band or bands. Fixed-band policies use the single-band ESP-IDF APIs, while AUTO uses the per-band APIs. See [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

The `GOT_IP` payload is also the source of truth for the local address, netmask, and gateway used during service startup. The runtime passes that gateway directly to the internal traffic generator instead of querying the network interface again. Disconnect processing clears the shared ready state, so the same sequence is repeated after a genuine reconnect.

### `src/cpp/frontend/`

`frontend` maps the runtime into a concrete ecosystem or firmware surface.

Current frontends:

- `esphome`: Home Assistant-facing external component and packaging root
- `native`: standalone BLE/MQTT firmware surface
- `matter`: Matter-facing adapter and firmware path
- `streamer`: raw CSI UDP streamer for collection workflows

Rule of thumb: frontend-specific schemas, transport bindings, and ecosystem integration belong here, not in `core`.

## Frontend Notes

### ESPHome

`src/cpp/frontend/esphome/` maps the shared runtime into ESPHome entities, YAML/config-codegen, and external-component packaging.

For the ESPHome workflow, see [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md).

### Native

`src/cpp/frontend/native/` exposes the runtime through the standalone BLE/MQTT surface and reuses the shared ESP-IDF frontend-support services for provisioning, device configuration, and OTA-related control flows. Native refreshes the shared diagnostics sample from the existing sensing update that feeds its status log, but publishes the cached CSI and Wi-Fi values only after an explicit MQTT `stats` command. Micro-ESPectre uses the same rate-sampler contract on its publish heartbeat and exposes the cache on the same MQTT `stats` topic. ESPHome uses the same sampler and publishes its diagnostic entity states only after `Refresh Diagnostics` is pressed. These on-demand surfaces do not add a diagnostic timer and remain available in production builds independently of runtime debug logging.

For the native workflow and protocol surface, see:

- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md)

### Matter

`src/cpp/frontend/matter/` maps runtime state and controls into the Matter surface without pulling Matter-specific concerns into the shared detector or runtime layers.

For the Matter workflow, see [`README.md` (matter)](../src/cpp/frontend/matter/README.md).

### Streamer

`src/cpp/frontend/streamer/` is a dedicated CSI transport frontend. It now uses the same controller/runtime contract as the other standalone frontends, but it selects `StreamEspIdfRuntime` so the raw CSI transport path can stay focused and detector-free.

For the streamer workflow, see [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md).

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
- `set_detection_algorithm_runtime()`
- `trigger_recalibration()`
- `get_snapshot()`
- `get_capabilities()`

Normalized runtime events include:

- motion-state changes
- threshold changes, including Lightweight settled-level recovery
- detector changes
- calibration start and finish
- periodic status updates
- runtime faults

Frontends should use this surface instead of reaching directly into low-level Wi-Fi or CSI pipeline services.

Runtime detector selection is capability-gated. ESPHome and Native enable the shared ESP-IDF detector store, which persists `lightweight` or `high_accuracy` in NVS and restores it at boot. Matter can be built with either detector but has no writable runtime detector surface; published Matter firmware selects `lightweight`. Streamer remains detector-free.

### Shared Runtime Debug Telemetry

ESP-IDF runtime implementations reuse `RuntimeDebugTelemetry` for one `[telemetry]` log line approximately every 10 seconds at `DEBUG` level. Micro-ESPectre emits the same machine-readable timing keys when its default-off `DEBUG_TELEMETRY` benchmark switch is enabled. ESP-IDF reports current, minimum, and largest-block heap values plus configured CPU frequency; MicroPython reports current and sampled-minimum free heap. Both report runtime-loop load and timing plus sampled detector evaluation timing.

`runtime_load` measures wall time spent inside the ESPectre runtime loop, not whole-system CPU utilization. Wi-Fi callbacks only normalize and enqueue CSI; detector processing, inference, state transitions, and frontend callback delivery run in the owning loop task. MQTT, BLE, and OTA stacks may still perform transport work on private tasks, but their application events are drained by the frontend loop. Detector timing is sampled on an evaluation tick after approximately 1,000 detector packets. For High Accuracy, it covers ML feature extraction, inference, and state update.

This debug log is an implementation diagnostic, not part of ESPectre Protocol. Streamer also retains its separate live transport telemetry for pacing, CSI, and uplink health during collection.

## ESPectre Protocol In The Architecture

ESPectre Protocol is the shared device-facing message model used by the standalone ESP-IDF frontends and related tools.

See [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md) for:

- message families
- BLE and MQTT transport mapping
- payload semantics
- OTA-related command surfaces

Architecturally, the protocol sits at the frontend/runtime integration boundary for the non-ESPHome standalone firmware paths.

## Packaging Note For ESPHome

ESPHome still expects a component-shaped entry point under the external components root. For that reason, `src/cpp/frontend/esphome/components/espectre/` acts as the ESPHome packaging root even though the shared sources live under `src/cpp/core/` and `src/cpp/runtime/`.

## Current Status

Implemented today:

- shared `core` sensing logic
- shared `runtime` contract
- current ESP-IDF runtime implementation
- `esphome`, `native`, `matter`, and `streamer` frontends
- shared ESPectre Protocol support across ESP-IDF standalone frontends

For performance and detector behavior, continue in:

- [`ALGORITHMS.md`](ALGORITHMS.md)
- [`TUNING.md`](TUNING.md)
- [`docs/performance`](performance/README.md)

## Recommended Reading Order

1. [README.md](../README.md) for the project overview
2. [SETUP.md](SETUP.md) for the shared entry points
3. this file for the current internal structure
4. [ALGORITHMS.md](ALGORITHMS.md) for detector and pipeline details
5. the relevant frontend README for operational workflow
