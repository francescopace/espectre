# Architecture Guide

This document explains the internal code layout introduced with the `core / runtime / frontend` refactor, why the project was split this way, and how the new structure enables multiple firmware targets without duplicating the motion-detection logic.

It also clarifies how the same split supports the current platform direction:

- reusable sensing logic in `core`
- multiple runtime implementations behind a stable contract
- multiple frontends such as `ESPHome`, `Matter`, and custom firmware adapters
- a future orchestration layer that can consume signals from multiple deployed devices

---

## Goals

The refactor was driven by four practical goals:

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
    ├── ble/
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
│  ESPHome frontend  BLE frontend  Matter frontend  Streamer frontend │
│  src/cpp/frontend/esphome/espectre  .../ble/...  .../matter/...     │
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
- `esp_idf/standalone_wifi_manager.*` for standalone ESP-IDF STA setup, CSI Wi-Fi policy, BSSID/channel fast scan, and retry behavior used by firmware targets that own their Wi-Fi stack

### `src/cpp/frontend/esphome/espectre/`

This is the ESPHome adapter layer and external component root. It maps the
shared runtime into ESPHome/Home Assistant entities and owns the YAML schema,
codegen, and packaging metadata for the production-oriented frontend.

For frontend-specific details, see
[`src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md).

### `src/cpp/frontend/ble/espectre/`

This is the standalone BLE adapter used by generic BLE clients, including a web
client as one example integration.

It reuses the same runtime contract as the other frontends, but maps runtime
events and controls to a custom GATT surface instead of Home Assistant entities
or Matter clusters.

The BLE adapter uses the shared `RuntimeFrontendController` for runtime
ownership and the shared standalone Wi-Fi manager for ESP-IDF STA setup.

For the BLE protocol, stability model, and firmware-specific surface, see
[`src/cpp/frontend/ble/README.md`](../src/cpp/frontend/ble/README.md).

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
the standalone Wi-Fi manager and CSI Wi-Fi policy. Its CSI capture and UDP
streaming state machine stay frontend-specific.

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
- `frontend/ble`: standalone BLE adapter + ESP-IDF firmware app
- `frontend/matter`: Matter adapter + esp-matter firmware app (experimental)

---

## Recommended Reading Order

If you are new to the project:

1. [README.md](../README.md) for product overview
2. [SETUP.md](SETUP.md) for the shared frontend chooser and install hub
3. this file for internal architecture
4. [test/cpp/README.md](../test/cpp/README.md) for validation strategy
5. [ALGORITHMS.md](ALGORITHMS.md) for algorithm details
