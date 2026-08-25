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

```text
Frontend -> Runtime contracts -> ESP-IDF runtime services -> Core
```

The contracts and detector logic compile on a host without ESP-IDF. Frontends select a backend through `RuntimeFrontendController` and do not call `core` directly.

## Layer Responsibilities

### `src/cpp/core/`

`core` contains reusable sensing logic and domain primitives:

- `LightweightDetector` and `HighAccuracyDetector`
- `TemporalCsiSampler`, which admits at most one packet per configured slot
- feature extraction and detector math
- filters and helper utilities
- exported ML artifacts and related constants

Rule of thumb: code in `core` should stay free of frontend-specific concerns such as ESPHome entities, Matter clusters, HTTP transport details, or MQTT topic handling.

### `src/cpp/runtime/`

`runtime` owns the execution environment around the shared detectors:

- CSI ingestion, normalization, and temporal admission before detector input
- AGC-active sensing path
- startup calibration orchestration
- traffic generation or packet ingress hooks
- runtime snapshots, capabilities, and events
- common runtime-facing configuration validation

The shared runtime layer owns the frontend-facing contract. The ESP-IDF implementations under `src/cpp/runtime/esp_idf/` include the motion-oriented `EspIdfRuntime` and the transport-oriented `StreamEspIdfRuntime`.

Shared runtime services also live here, including:

- `RuntimeFrontendController`
- standalone Wi-Fi helpers for non-ESPHome firmware
- shared diagnostics helpers
- ESPectre Protocol model and shared Direct HTTP/MQTT transport support
- NVS-backed provisioning helpers reused by ESP-IDF frontends

### Shared Wi-Fi and CSI Lifecycle

`WiFiLifecycleManager` owns the CSI-specific ESP-IDF radio policy for every frontend. It applies the protocol and HT20 bandwidth policy synchronously on `WIFI_EVENT_STA_START`, before the first association, then completes the CSI prerequisites when `IP_EVENT_STA_GOT_IP` is drained from the runtime loop. ESPHome, Native, Matter, and Streamer must not apply these radio settings in their frontend code.

The frontend or SDK integrator explicitly selects `2g`, `5g`, or `auto`; `2g` is the validated default, while `5g` and `auto` are available only on dual-band targets. The lifecycle applies that band mode first, then pins an 802.11n protocol ceiling and HT20 bandwidth on the selected band or bands. Fixed-band policies use the single-band ESP-IDF APIs, while AUTO uses the per-band APIs. See [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

Supported first-party firmware uses an associated Wi-Fi station and does not enable promiscuous mode. Standalone ESP-IDF startup explicitly keeps promiscuous mode disabled, and the shared CSI pipeline filters frames against the local device identity where the relevant metadata is available. Micro-ESPectre likewise connects through `network.STA_IF` before starting CSI. This is an intentional responsible-use boundary: a protected network requires valid credentials, which raises the barrier against passive collection by an unaffiliated device. It is not an authorization mechanism or proof of consent; open networks need no password, credentials can be misused, and downstream open-source builds can change the radio policy.

The `GOT_IP` payload is also the source of truth for the local address, netmask, and gateway used during service startup. The runtime passes that gateway directly to the internal traffic generator instead of querying the network interface again. Disconnect processing clears the shared ready state, so the same sequence is repeated after a genuine reconnect.

### `src/cpp/frontend/`

`frontend` maps the runtime into a concrete ecosystem or firmware surface.

Current frontends:

- `esphome`: Home Assistant-facing external component, Direct runtime controls, and packaging root
- `native`: standalone Direct HTTP/MQTT firmware surface
- `matter`: Matter-facing adapter with a separate Direct tuning plane
- `streamer`: raw CSI UDP streamer with Direct status and diagnostics

Rule of thumb: frontend-specific schemas, transport bindings, and ecosystem integration belong here, not in `core`.

## Frontend Notes

### ESPHome

`src/cpp/frontend/esphome/` maps the shared runtime into ESPHome entities, YAML/config-codegen, external-component packaging, and the common Direct HTTP bridge. Direct mutations republish matching entity state rather than creating a second configuration owner.

For the ESPHome workflow, see [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md).

### Native

`src/cpp/frontend/native/` exposes the runtime through Improv Serial provisioning, local Direct HTTP, and optional MQTT. It reuses the shared ESP-IDF services for staged Wi-Fi configuration, device configuration, mDNS, transport-independent commands, and OTA. Native also owns the optional Direct raw CSI session: `EspIdfRuntime` transitions between `SENSING` and non-persistent `RAW_COLLECTION`, while the existing Direct server keeps the JSON control clients isolated from one binary collector. Native refreshes the shared diagnostics sample from the existing sensing update that feeds its status log and returns the same cache through correlated Direct or MQTT `diagnostics` queries. Micro-ESPectre uses the same rate-sampler contract on its publish heartbeat. ESPHome uses the same sampler and publishes its diagnostic entity states only after `Refresh Diagnostics` is pressed. These on-demand surfaces do not add a diagnostic timer and remain available in production builds independently of runtime debug logging.

`FrontendCommandEngine` is the C++ command owner below the frontend adapters. Native MQTT, Native Direct, the shared Direct bridge, and ESPHome entities all construct the same typed request and receive the same structured result and change set. Matter and Streamer inherit it through the shared bridge. Commands execute serially on the existing frontend task; there is no command worker. Queries return only through the requesting adapter, while accepted mutations publish their status, info, config, or OTA state change to every active adapter. MQTT and each Direct client deliberately keep separate outbound queues because transport backpressure is independent of command semantics. MicroPython mirrors the registry and dispatcher, with a host probe enforcing catalog parity.

Raw collection branches inside `CsiPipeline` before queueing, temporal sampling, feature extraction, calibration, and detector evaluation. The Wi-Fi callback only copies the candidate into preallocated slots and updates bounded counters. Entering raw mode saves whether sensing was armed, cancels calibration, stops generated or listener traffic, clears the pipeline, and leaves only capture active. Exiting disables and clears capture, restores the prior configuration, restarts sensing only when it was previously armed, and recalibrates from empty state. Runtime loss and recovery paths terminate the session rather than restoring raw mode.

Peer-assisted discovery keeps orchestration out of `core`. `runtime/peer_discovery` owns bounded validation, deterministic deduplication, sorting, and serialization for canonical Native, Streamer, ESPHome, and Matter records; `runtime/esp_idf/peer_discovery_service_esp_idf` owns one asynchronous query through the existing Espressif mDNS responder; and `frontend/native/native_mdns_bootstrap_responder` owns the IPv4 bootstrap response. The Native extension observes incoming Espressif mDNS actions before exact-name filtering, accepts only `espectre-devices-<24 hex>.local` class-IN A or AAAA questions, writes a shared 10-second A answer with an NSEC no-AAAA assertion, and answers standalone AAAA questions with that NSEC assertion through the existing socket before always allowing the original responder to continue. It never advertises an IPv6 bootstrap address. It does not register a hostname, retain a nonce, announce a record, send a goodbye, wrap outbound traffic, or create another socket. Native answers only while its station has a usable IPv4 address and is the validated bootstrap responder because it owns port 80, the common path and subprotocol, its responder lifecycle, and the peer query capability. This responder choice does not restrict discovery results: every accepted canonical frontend is selectable at its own advertised Direct port, and clients negotiate its exact capabilities after connection.

Direct transports may optionally accept a deferred handler. The ESP-IDF implementation assigns an opaque monotonically increasing token to each live connection, removes inbound work by token rather than file descriptor, and queues a completion only when that token still identifies the originating client. The default interface implementation reports deferred delivery as unsupported, preserving source compatibility for SDK transports that implement only synchronous requests. Native cancels all pending bootstrap answers before a requested Wi-Fi reconfiguration; an ordinary disconnect disables the bootstrap responder, and frontend shutdown releases any bounded query. No nonce or result survives as a hostname registration or peer inventory.

For the native workflow and protocol surface, see:

- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md)

### Matter

`src/cpp/frontend/matter/` maps occupancy into Matter without pulling Matter-specific concerns into the shared detector or runtime layers. Detector configuration is not represented by the standard occupancy clusters, so the frontend also exposes the shared Direct HTTP bridge as its local tuning plane.

For the Matter workflow, see [`README.md` (matter)](../src/cpp/frontend/matter/README.md).

### Streamer

`src/cpp/frontend/streamer/` is a dedicated CSI transport frontend. It uses the same controller/runtime contract and Direct HTTP bridge as the other C++ frontends, but selects `StreamEspIdfRuntime` so raw CSI remains on its collector-paced UDP path and the firmware stays detector-free.

For the streamer workflow, see [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md).

## Runtime Contract

The shared runtime contract is the interface used by frontends.

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

Runtime detector selection is capability-gated. ESPHome and Native enable the shared ESP-IDF detector store, which persists `lightweight` or `high_accuracy` in NVS and restores it at boot. Matter enables runtime detector selection through Direct HTTP because its standard clusters do not expose detector configuration. Streamer remains detector-free.

### Runtime Performance Diagnostics

C++ runtime implementations use `RuntimePerformanceDiagnostics` to aggregate runtime-loop load and timing plus sampled detector evaluation timing in bounded 10-second windows. `RuntimeDiagnosticsSnapshot` combines the latest complete window with current, minimum, and largest-block heap values and configured CPU frequency. Native, ESPHome, Matter, and Streamer expose these production fields through Direct `diagnostics`; collection is unconditional and does not emit a periodic debug log. Unsupported detector timing is explicit on Streamer.

Micro-ESPectre remains separate because it does not expose Direct HTTP. Its default-off `DEBUG_TELEMETRY` benchmark switch emits machine-readable serial timing, heap, garbage-collection, and packet-processing fields when enabled.

`runtime_load` measures wall time spent inside the ESPectre runtime loop, not whole-system CPU utilization. Wi-Fi callbacks only normalize and enqueue CSI; detector processing, inference, state transitions, and frontend callback delivery run in the owning loop task. MQTT, Direct HTTP, and OTA stacks may still perform transport work on private tasks, but their application events are drained by the frontend loop. Detector timing is sampled on an evaluation tick after approximately 1,000 detector packets. For High Accuracy, it covers ML feature extraction, inference, and state update.

The C++ field names and units are part of the additive Direct diagnostics contract in [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md#direct-http-v1). Streamer also retains its separate live transport telemetry for pacing, CSI, and uplink health during collection.

## ESPectre Protocol In The Architecture

ESPectre Protocol is the shared device-facing message model used by the standalone ESP-IDF frontends and related tools. [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md) owns its message families, Direct HTTP and MQTT mappings, payloads, and command surfaces.

For the non-ESPHome standalone firmware paths, the protocol sits at the boundary between the frontend and runtime layers.

## Packaging Note For ESPHome

ESPHome still expects a component-shaped entry point under the external components root. For that reason, `src/cpp/frontend/esphome/components/espectre/` acts as the ESPHome packaging root even though the shared sources live under `src/cpp/core/` and `src/cpp/runtime/`.

## Related References

- Deployment and frontend selection: [SETUP.md](SETUP.md)
- Supported SDK surface: [EMBEDDING.md](EMBEDDING.md)
- Detector behavior and tuning: [ALGORITHMS.md](ALGORITHMS.md) and [TUNING.md](TUNING.md)
- Measured detector results: [docs/performance](performance/README.md)
- Frontend operation: the relevant README under `src/cpp/frontend/`
