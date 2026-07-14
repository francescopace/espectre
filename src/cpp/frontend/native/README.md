# ESPectre Native Frontend

This directory contains the standalone ESPectre native frontend.

Its role is to expose the shared ESPectre runtime through a lightweight custom
GATT surface that can be used by generic BLE clients, including web clients,
mobile apps, smart-device integrations, and other custom tooling.

This README covers the native frontend firmware workflow and BLE-specific
operational notes. The shared protocol surface is documented in
[`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Scope

The native frontend is intentionally separate from the ESPHome frontend:

- `ESPHome` remains focused on Home Assistant entities and YAML/codegen
- `Native` exposes the standalone integration surface over BLE and MQTT
- `Matter` exposes the same runtime through Matter clusters

The native frontend now also supports HTTPS OTA triggered from its MQTT command
plane.

The current native frontend preserves the protocol already used by
`docs/web/game/`, but it is not tied to that specific client.

## Directory Layout

- `espectre/`:
  frontend adapter and runtime-to-BLE mapping
- `app/`:
  standalone ESP-IDF firmware app
- `../../runtime/` and `../../runtime/esp_idf/frontend_support/`:
  shared ESPectre Protocol serializer, BLE binding interface, NimBLE transport,
  NVS-backed device/Wi-Fi config store, MQTT transport boundary, and ESP-IDF
  provisioning helpers
- `espectre/Kconfig.projbuild`:
  frontend-owned Wi-Fi configuration knobs
- `../../runtime/esp_idf/espectre_config/Kconfig.projbuild`:
  shared sensing/runtime configuration knobs consumed by the standalone sensing
  frontends

The firmware app uses the shared standalone Wi-Fi manager for STA setup,
BSSID/channel fast scan, CSI Wi-Fi policy, and retry behavior. The frontend
adapter itself uses the shared runtime frontend controller and only owns the
BLE protocol mapping.

## Getting Started

If you arrived here from [`SETUP.md`](../../../../docs/SETUP.md),
this README is the next step for the standalone native firmware path.

### Browser-Flashed Firmware

The web flasher can install published `Native` images for supported chips. After
flashing, use a BLE client that understands this protocol, such as:

- [`espectre-ble.html`](../../../../tools/web/espectre-ble.html):
  local Web Bluetooth provisioning and protocol test client
- [`README.md`](../../../../docs/web/game/README.md):
  example interactive client built on the same BLE surface

### Local ESP-IDF Workflow

Before building locally, complete the shared
[`ESP-IDF Local Build Prerequisite`](../../../../docs/SETUP.md#esp-idf-local-build-prerequisite).
The repository CLI auto-detects a reusable ESP-IDF install, so the wrapper-first
workflow does not require a separate setup check before build.
See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side
tools, and wrapper behavior.

CI QEMU smoke currently covers `ESP32`, `ESP32-S3`, and `ESP32-C3` for the
native frontend. `ESP32-C5` and `ESP32-C6` remain build-only because the
current Espressif QEMU fork does not support them.

Repository CLI:

```bash
./espectre native build --chip c3 --clean
./espectre native flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

The CLI is a thin wrapper over the ESP-IDF app in this directory.
On Windows, use `.\espectre.cmd native ...` and `.\espectre.cmd monitor --port COM5`.
If the wrapper cannot find or validate ESP-IDF, run `.\espectre.cmd doctor`
or `./espectre doctor` to inspect the detected environment.

### Local Web Bluetooth Test Client

[`espectre-ble.html`](../../../../tools/web/espectre-ble.html)
is the reference browser client for local BLE validation, provisioning, and
live diagnostics.

Current capabilities:

- connect to the ESPectre BLE service from a desktop browser
- subscribe to telemetry and sysinfo notifications
- enable or disable the live telemetry subscription without disconnecting
- request a fresh sysinfo block with `REQ_SYSINFO`
- adjust the runtime threshold with `SET_THRESHOLD:X.XX`
- show a firmware-generated read-only `device_id`
- inspect the immutable firmware-derived `device_name`
- edit the human-facing `device_label`
- clear the persisted device-facing configuration without disconnecting
- expose the immutable BLE pairing name as the shared `device_name`
- provision or clear Wi-Fi credentials over BLE
- provision or clear MQTT configuration over BLE
- keep HTTPS OTA reachable through the always-on MQTT control plane

Requirements:

- desktop Chrome, Edge, or another Chromium-based browser with Web Bluetooth
- a secure context such as `http://localhost` or `https://`
- a BLE-capable ESP32 target supported by this frontend

Recommended local workflow from the repository root:

```bash
python3 -m http.server 8080 -d tools/web
```

Then open:

```text
http://localhost:8080/espectre-ble.html
```

Usage notes:

1. click `Connect` and select the ESPectre device
2. wait for the initial `REQ_SYSINFO` refresh after notifications start
3. disable live BLE telemetry from the test client when you only need provisioning or sysinfo
4. use `Save Wi-Fi` to send one atomic `SET_WIFI_CONFIG` update
5. use `Save Device` to persist the human-facing `device_label`
6. use `Clear Device` when you want to reset the persisted device-facing config while keeping the generated `device_id`
7. edit the `Threshold` box in the BLE client to send `SET_THRESHOLD` immediately with the current numeric value
8. use `Save MQTT` to send one atomic `SET_MQTT_CONFIG` update and enable MQTT transport

When telemetry notifications are disabled by the client, the standalone native
frontend keeps `sysinfo` and control commands active but deregisters the live
telemetry callback so BLE-only live telemetry is no longer produced in the
background. The shared protocol semantics remain documented in
[`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

The standalone native frontend uses the same shared periodic progress-bar sensing
status log helper used by the ESPHome and Matter frontends, so the serial log
shape stays aligned across those frontend surfaces.

## Wi-Fi Configuration

Unlike the ESPHome frontend, the standalone native firmware does not rely on YAML
or Home Assistant for setup. In the current local-lab profile, Wi-Fi can be
provisioned live over BLE and persisted in NVS.

Frontend-owned options in [`Kconfig.projbuild`](espectre/Kconfig.projbuild)
remain useful as firmware defaults for reproducible images or first boot.
Shared sensing options and their defaults now live in
[`SETUP.md`](../../../../docs/SETUP.md), and can be overridden per frontend in
[`sdkconfig.defaults`](app/sdkconfig.defaults).
Versioned transport defaults in [`sdkconfig.defaults`](app/sdkconfig.defaults)
also tune the standalone native firmware with the shared ESP-IDF Wi-Fi transport
baseline now used across the standalone frontends: AMPDU enabled, larger Wi-Fi
RX/TX buffers, plus lwIP mailbox and IRAM optimizations.

| Option | Purpose |
|--------|---------|
| `ESPECTRE_WIFI_SSID` | Wi-Fi SSID |
| `ESPECTRE_WIFI_PASSWORD` | Wi-Fi password |
| `ESPECTRE_WIFI_BSSID` | Optional BSSID lock |
| `ESPECTRE_WIFI_CHANNEL` | Optional channel lock (`0` = auto) |

When `ESPECTRE_WIFI_BSSID` is set, the firmware uses fast scan and pins the
association to that AP radio. Leave `ESPECTRE_WIFI_CHANNEL=0` unless you need
to force a known 2.4 GHz channel for repeatable CSI captures.

Runtime provisioning behavior:

- `SET_WIFI_CONFIG` persists the full Wi-Fi block in NVS and reconnects the
  station immediately without restarting BLE
- `CLEAR_WIFI` erases stored Wi-Fi values and disconnects the station
- `SET_MQTT_CONFIG` persists the full MQTT broker block in NVS and reinitializes
  the MQTT transport

This means the current standalone native firmware is best suited for:

- local integration experiments
- custom client development
- controlled deployments and recovery flows where BLE-assisted provisioning is acceptable

It is still a lab-oriented provisioning path, not a polished end-user flow
comparable to ESPHome.

## Protocol Reference

The shared BLE protocol surface is documented in:

- [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md)

That file defines:

- service and characteristic UUIDs
- telemetry payload format
- sysinfo framing and key semantics
- control command syntax
- nearby BLE client expectations

Local implementation anchors:

- [`ble_protocol.h`](../../runtime/ble_protocol.h):
  protocol constants such as UUIDs and default device name
- [`native_frontend.cpp`](espectre/native_frontend.cpp):
  command handling, sysinfo emission, and telemetry serialization
- [`espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp):
  shared MQTT topic, payload, and command serialization

## OTA

The native frontend uses the shared ESPectre MQTT command surface plus a shared
ESP-IDF HTTPS OTA implementation.

Operational model:

- MQTT remains the command plane
- `ota_check` checks a remote HTTPS manifest
- `ota_start` downloads an HTTPS application image into the inactive OTA slot
- successful OTA schedules an immediate reboot into the new slot

Artifact model:

- `factory` images remain the published full-flash binaries used by the browser
  flasher and manual recovery flows
- native OTA uses the published `espectre-native-...-ota.bin` payload together
  with its matching JSON manifest

BLE remains useful for local provisioning and recovery, but native OTA is not a
BLE-only workflow.

## Firmware Limits and Expectations

The current standalone native frontend intentionally stays small.

Important current limits:

- provisioning is intentionally lab-oriented and low-ceremony
- the BLE control surface is still ASCII commands rather than a structured schema
- clients should not assume diagnostic sysinfo fields are stable forever
- there is no capability discovery or negotiated feature set yet
- OTA uses HTTPS transport and dual OTA slots, so local recovery still starts
  from the published factory image when an image must be reflashed from USB

This keeps the transport simple while allowing external BLE clients to provision
Wi-Fi and MQTT, tune the runtime threshold, and observe the runtime in real
time.

## BLE-Specific Troubleshooting

### The client cannot control the device after connecting

Check these first:

1. the client writes exact ASCII commands
2. the value passed to `SET_THRESHOLD` is finite and inside the detector range (`classic`: `0.0-10.0`, `ml`: `0.0-1.0`)
3. the client does not depend on sysinfo ordering

### The firmware starts but never joins Wi-Fi

Check the active Wi-Fi values first:

1. request fresh sysinfo and inspect `wifi_ssid`, `wifi_bssid`,
   `wifi_channel`, and `wifi_connected`
2. if using `tools/web/espectre-ble.html`, press `Save Wi-Fi` and wait for the
   station reconnect after the atomic `SET_WIFI_CONFIG` update
3. if no provisioning has been stored yet, verify the Kconfig defaults used at
   build time:
   - `ESPECTRE_WIFI_SSID`
   - `ESPECTRE_WIFI_PASSWORD`
   - optional `ESPECTRE_WIFI_BSSID`

### The native firmware is not the right fit for the workflow

That can be expected. This frontend is optimized for the native standalone integration surface,
not for Home Assistant-style provisioning or the Matter commissioning flow.

## Related Files

- `../../runtime/ble_protocol.h`:
  UUIDs and default device name
- `../../runtime/espectre_protocol.cpp`:
  shared protocol payload and command helpers
- `../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`:
  shared ESP-IDF Wi-Fi provisioning command handling
- `espectre/native_frontend.cpp`:
  command parsing, sysinfo emission, telemetry serialization
- `../../runtime/esp_idf/frontend_support/ble_bindings_nimble.cpp`:
  NimBLE transport implementation
- `../../../../tools/web/espectre-ble.html`:
  local Web Bluetooth provisioning and protocol test client
- `../../../../docs/web/game/README.md`:
  example client built on this protocol
