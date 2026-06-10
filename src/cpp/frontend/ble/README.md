# ESPectre BLE Frontend

This directory contains the standalone ESPectre BLE frontend.

Its role is to expose the shared ESPectre runtime through a lightweight custom
GATT surface that can be used by generic BLE clients, including web clients,
mobile apps, smart-device integrations, and other custom tooling.

This file is the source of truth for the BLE frontend protocol and firmware
workflow.

## Scope

The BLE frontend is intentionally separate from the ESPHome frontend:

- `ESPHome` remains focused on Home Assistant entities and YAML/codegen
- `BLE` exposes a transport-oriented integration surface
- `Matter` exposes the same runtime through Matter clusters

The current BLE frontend preserves the protocol already used by
`docs/web/game/`, but it is not tied to that specific client.

## Directory Layout

- `espectre/`:
  frontend adapter, protocol constants, and bindings interface
- `app/`:
  standalone ESP-IDF firmware app and NimBLE transport implementation
- `espectre/Kconfig.projbuild`:
  frontend-owned Wi-Fi configuration knobs

## Getting Started

If you arrived here from [`../../../../docs/SETUP.md`](../../../../docs/SETUP.md),
this README is the next step for the standalone BLE firmware path.

### Browser-Flashed Firmware

The web flasher can install published `BLE` images for supported chips. After
flashing, use a BLE client that understands this protocol, such as the example
web client documented in [`../../../../docs/web/game/README.md`](../../../../docs/web/game/README.md).

### Local ESP-IDF Workflow

Repository CLI:

```bash
./espectre ble build --chip c3
./espectre ble flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre ble monitor --chip c3 --port /dev/cu.usbmodemXXXX
```

The CLI is a thin wrapper over the ESP-IDF app in this directory.

## Wi-Fi Configuration

Unlike the ESPHome frontend, the standalone BLE firmware currently expects
Wi-Fi credentials at build time through Kconfig.

Frontend-owned options in [`espectre/Kconfig.projbuild`](espectre/Kconfig.projbuild):

| Option | Purpose |
|--------|---------|
| `ESPECTRE_WIFI_SSID` | Wi-Fi SSID |
| `ESPECTRE_WIFI_PASSWORD` | Wi-Fi password |
| `ESPECTRE_WIFI_BSSID` | Optional BSSID lock |

This means the current standalone BLE firmware is best suited for:

- local integration experiments
- custom client development
- controlled firmware deployments where build-time credentials are acceptable

It is not yet a provisioning-oriented end-user flow comparable to ESPHome.

## Current Protocol

Current protocol version: `1`

Transport model:

- one primary BLE service
- one notify characteristic for live telemetry
- one notify and read characteristic for line-based system information
- one write characteristic for control commands

## Stability Model

To help client implementations stay robust, treat the protocol in three layers:

### 1. Stable Transport Surface

These parts should be considered the most stable client contract:

- service and characteristic UUIDs
- telemetry payload shape
- control command syntax
- `proto_version`
- `END` terminator for sysinfo blocks

### 2. Current Operational and Diagnostic Surface

These fields are part of the current implementation and are useful for clients,
but should be treated as more flexible than the transport primitives above:

- the exact set of sysinfo keys
- the order of sysinfo lines
- human-readable formatting inside values such as `threshold=1.20 (auto)`
- optional diagnostic values that may expand over time

Clients should parse these conservatively and ignore unknown keys.

### 3. Future Extensions

These are not part of the current contract and should not be assumed by
clients until they are explicitly added:

- richer status and state fields
- firmware and build metadata
- extra health diagnostics
- capability discovery
- new control commands

## UUIDs

These values are defined in `espectre/ble_protocol.h`.

| Item | UUID | Direction | Notes |
|------|------|-----------|-------|
| Service | `d33ff46b-2203-4775-bc6f-b3a2c36af8f0` | - | ESPectre BLE service |
| Telemetry characteristic | `119d5cac-48da-4bd9-bfc3-169805868258` | device -> client (`notify`) | Binary payload |
| Sysinfo characteristic | `c8c89ffa-c401-461f-9ffc-942fa04adfe3` | device -> client (`notify`, `read`) | Text `key=value` lines |
| Control characteristic | `33ed9214-a8d7-40e8-82d1-c82747dcdc71` | client -> device (`write`) | ASCII commands |

Default device name:

- `ESPectre BLE`

## Telemetry Characteristic

Purpose:

- low-latency runtime telemetry for interactive clients

Encoding:

```text
[float32 movement][float32 threshold]
```

Field semantics:

| Field | Type | Description |
|------|------|-------------|
| `movement` | `float32` | Current movement metric from the runtime |
| `threshold` | `float32` | Current runtime threshold |

Notes:

- values are serialized as little-endian `float32`
- notifications are throttled in the frontend
- current default notify interval is `40 ms`

Example:

```text
00 00 40 3F  9A 99 99 3F
```

This corresponds to:

- `movement = 0.75`
- `threshold = 1.20`

## Sysinfo Characteristic

Purpose:

- expose textual runtime and configuration information
- provide a simple transport that remains easy to inspect from generic BLE tools

Framing:

- the device sends one `key=value` line per notification
- the block terminates with `END`

Example:

```text
proto_version=1
chip=esp32c6
threshold=1.20 (auto)
window=100
END
```

Current keys emitted by the frontend:

| Key | Class | Description |
|-----|-------|-------------|
| `proto_version` | stable | BLE protocol version |
| `chip` | operational | Chip target, for example `esp32c6` |
| `threshold` | operational | Current threshold plus mode suffix |
| `window` | operational | Segmentation window size |
| `detector` | operational | Active detector name |
| `subcarriers` | diagnostic | Current subcarrier source label |
| `lowpass` | operational | Low-pass filter state |
| `lowpass_cutoff` | operational | Low-pass cutoff in Hz, when enabled |
| `hampel` | operational | Hampel filter state |
| `hampel_window` | operational | Hampel window size, when enabled |
| `hampel_threshold` | operational | Hampel threshold, when enabled |
| `traffic_rate` | operational | Configured traffic generator rate |
| `publish_interval` | operational | Runtime publish interval |
| `evaluation_interval` | operational | Runtime evaluation interval |
| `motion_hits` | operational | Motion enter and exit hit counters |
| `best_pxx` | diagnostic | Current adaptive-threshold baseline metric |
| `END` | stable | End-of-block marker |

Legend:

- `stable`: clients can depend on it structurally
- `operational`: useful current runtime and configuration information
- `diagnostic`: primarily for visibility and debugging

Emission behavior:

- on client connect
- on explicit `REQ_SYSINFO`
- after threshold changes
- when calibration starts or finishes

## Control Characteristic

Purpose:

- allow lightweight runtime control from external clients

Encoding:

- ASCII command strings

Current commands:

| Command | Description | Notes |
|---------|-------------|-------|
| `REQ_SYSINFO` | Request a fresh sysinfo block | Exact string |
| `SET_THRESHOLD:X.XX` | Update the runtime threshold | Value must be finite and in range `0.0-10.0` |

Behavior notes:

- threshold updates are runtime and session only
- unknown commands are ignored and logged on the device
- invalid threshold writes are rejected and logged on the device

## Firmware Limits and Expectations

The current standalone BLE frontend intentionally stays small.

Important current limits:

- Wi-Fi provisioning is build-time, not end-user interactive
- threshold is the only runtime write command exposed today
- clients should not assume diagnostic sysinfo fields are stable forever

This keeps the transport simple while allowing external BLE clients to tune and
observe the runtime in real time.

## Compatibility Guidance

The current protocol is intentionally simple, but clients should still treat it
as versioned:

- always read and cache `proto_version`
- ignore unknown sysinfo keys
- ignore additional commands or features you do not understand
- do not assume the sysinfo key order is semantically important
- treat new keys as additive unless a future protocol version states otherwise
- avoid depending on diagnostic-only fields for core client behavior

## Possible Evolutions

The current protocol is intentionally minimal. The most likely future
extensions are:

- richer device status:
  - `ready_to_publish`
  - `calibrating`
  - `gain_locked`
  - `motion_state`
- firmware and build metadata:
  - firmware version
  - git ref
  - build id
- health diagnostics:
  - free heap
  - largest free block
  - uptime
  - reset reason
  - last runtime fault
- feature and capability discovery:
  - supported commands
  - optional diagnostics availability

These are intentionally ideas, not commitments.
They should not be interpreted as part of protocol version `1`.

## BLE-Specific Troubleshooting

### The client cannot control the device after connecting

Check these first:

1. the client writes exact ASCII commands
2. the value passed to `SET_THRESHOLD` is finite and inside `0.0-10.0`
3. the client does not depend on sysinfo ordering

### The firmware starts but never joins Wi-Fi

Verify the Kconfig values used at build time:

- `ESPECTRE_WIFI_SSID`
- `ESPECTRE_WIFI_PASSWORD`
- optional `ESPECTRE_WIFI_BSSID`

### The BLE firmware is not the right fit for the workflow

That can be expected. This frontend is optimized for custom BLE integrations,
not for Home Assistant-style provisioning or the Matter commissioning flow.

## Related Files

- `espectre/ble_protocol.h`:
  UUIDs and default device name
- `espectre/ble_frontend.cpp`:
  command parsing, sysinfo emission, telemetry serialization
- `app/main/ble_bindings_nimble.cpp`:
  NimBLE transport implementation
- `docs/web/game/README.md`:
  example client built on this protocol

## Related Docs

- [`../../../../docs/SETUP.md`](../../../../docs/SETUP.md):
  shared installation hub and frontend chooser
- [`../../../../docs/ARCHITECTURE.md`](../../../../docs/ARCHITECTURE.md):
  shared architecture and runtime contract
- [`../../../../docs/TUNING.md`](../../../../docs/TUNING.md):
  shared tuning guidance and parameter tradeoffs
- [`../matter/README.md`](../matter/README.md):
  Matter frontend
- [`../esphome/README.md`](../esphome/README.md):
  ESPHome frontend
