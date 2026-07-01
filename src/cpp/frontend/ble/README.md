# ESPectre BLE Frontend

This directory contains the standalone ESPectre BLE frontend.

Its role is to expose the shared ESPectre runtime through a lightweight custom
GATT surface that can be used by generic BLE clients, including web clients,
mobile apps, smart-device integrations, and other custom tooling.

This file is the source of truth for the BLE frontend firmware workflow and
BLE-specific operational notes. The shared protocol surface is documented in
[`docs/ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Scope

The BLE frontend is intentionally separate from the ESPHome frontend:

- `ESPHome` remains focused on Home Assistant entities and YAML/codegen
- `BLE` exposes a transport-oriented integration surface
- `Matter` exposes the same runtime through Matter clusters

The current BLE frontend preserves the protocol already used by
`docs/web/game/`, but it is not tied to that specific client.

## Directory Layout

- `espectre/`:
  frontend adapter and runtime-to-BLE mapping
- `app/`:
  standalone ESP-IDF firmware app
- `../../runtime/` and `../../runtime/esp_idf/protocol/`:
  shared ESPectre Protocol serializer, BLE binding interface, NimBLE transport,
  NVS-backed device/Wi-Fi config store, MQTT transport boundary, and ESP-IDF
  provisioning helpers
- `espectre/Kconfig.projbuild`:
  frontend-owned Wi-Fi configuration knobs

The firmware app uses the shared standalone Wi-Fi manager for STA setup,
BSSID/channel fast scan, CSI Wi-Fi policy, and retry behavior. The frontend
adapter itself uses the shared runtime frontend controller and only owns the
BLE protocol mapping.

## Getting Started

If you arrived here from [`docs/SETUP.md`](../../../../docs/SETUP.md),
this README is the next step for the standalone BLE firmware path.

### Browser-Flashed Firmware

The web flasher can install published `BLE` images for supported chips. After
flashing, use a BLE client that understands this protocol, such as:

- [`tools/web/espectre-ble.html`](../../../../tools/web/espectre-ble.html):
  local Web Bluetooth provisioning and protocol test client
- [`docs/web/game/README.md`](../../../../docs/web/game/README.md):
  example interactive client built on the same BLE surface

### Local ESP-IDF Workflow

Repository CLI:

```bash
./espectre ble build --chip c3
./espectre ble flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre ble monitor --chip c3 --port /dev/cu.usbmodemXXXX
```

The CLI is a thin wrapper over the ESP-IDF app in this directory.

### Local Web Bluetooth Test Client

[`tools/web/espectre-ble.html`](../../../../tools/web/espectre-ble.html)
is the reference browser client for local BLE validation, provisioning, and
live diagnostics.

Current capabilities:

- connect to the ESPectre BLE service from a desktop browser
- subscribe to telemetry and sysinfo notifications
- enable or disable the live telemetry subscription without disconnecting
- request a fresh sysinfo block with `REQ_SYSINFO`
- adjust the runtime threshold with `SET_THRESHOLD:X.XX`
- show a firmware-generated read-only `device_id`
- edit the human-facing `device_name`
- clear the persisted device-facing configuration without disconnecting
- derive the BLE pairing name from `device_name`
- provision or clear Wi-Fi credentials over BLE
- provision or clear MQTT configuration over BLE

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
4. use `Save Wi-Fi` to write Wi-Fi values and `APPLY_WIFI` in one step
5. use `Save Device` to persist the human-facing `device_name`
6. use `Clear Device` when you want to reset the persisted device-facing config while keeping the generated `device_id`
7. use the threshold slider to send `SET_THRESHOLD` automatically when you release it
8. use `Save MQTT` to persist MQTT settings and enable MQTT transport
9. leave the Wi-Fi password field blank to keep an already stored password

When telemetry notifications are disabled by the client, the standalone BLE
frontend keeps `sysinfo` and control commands active but deregisters the live
telemetry callback so BLE-only live telemetry is no longer produced in the
background. The shared protocol semantics remain documented in
[`docs/ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

The standalone BLE frontend uses the same shared periodic progress-bar sensing
status log helper used by the ESPHome and Matter frontends, so the serial log
shape stays aligned across those frontend surfaces.

## Wi-Fi Configuration

Unlike the ESPHome frontend, the standalone BLE firmware does not rely on YAML
or Home Assistant for setup. In the current local-lab profile, Wi-Fi can be
provisioned live over BLE and persisted in NVS.

Frontend-owned options in [`espectre/Kconfig.projbuild`](espectre/Kconfig.projbuild)
remain useful as firmware defaults for reproducible images or first boot.
Versioned transport defaults in [`app/sdkconfig.defaults`](app/sdkconfig.defaults)
also tune the standalone BLE firmware for mixed BLE + Wi-Fi traffic, including
larger Wi-Fi RX/TX buffers plus lwIP mailbox and IRAM optimizations inherited
from the standalone streamer profile.

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

- `SET_WIFI_SSID`, `SET_WIFI_PASSWORD`, `SET_WIFI_BSSID`, and
  `SET_WIFI_CHANNEL` persist the working values in NVS
- `APPLY_WIFI` reconnects the station immediately without restarting BLE
- `CLEAR_WIFI` erases stored Wi-Fi values and disconnects the station
- the web client shows whether a password is already stored and lets you keep
  it by leaving the password field blank

This means the current standalone BLE firmware is best suited for:

- local integration experiments
- custom client development
- controlled deployments and recovery flows where BLE-assisted provisioning is acceptable

It is still a lab-oriented provisioning path, not a polished end-user flow
comparable to ESPHome.

## Protocol Reference

The shared BLE protocol surface is documented in:

- [`../../../../docs/ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md)

Use that file as the source of truth for:

- service and characteristic UUIDs
- telemetry payload format
- sysinfo framing and key semantics
- control command syntax
- compatibility expectations for nearby BLE clients

Local implementation anchors:

- [`../../runtime/ble_protocol.h`](../../runtime/ble_protocol.h):
  protocol constants such as UUIDs and default device name
- [`espectre/ble_frontend.cpp`](espectre/ble_frontend.cpp):
  command handling, sysinfo emission, and telemetry serialization
- [`../../runtime/espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp):
  shared MQTT topic, payload, and command serialization

## Firmware Limits and Expectations

The current standalone BLE frontend intentionally stays small.

Important current limits:

- provisioning is intentionally lab-oriented and low-ceremony
- the BLE control surface is still ASCII commands rather than a structured schema
- clients should not assume diagnostic sysinfo fields are stable forever
- there is no capability discovery or negotiated feature set yet

This keeps the transport simple while allowing external BLE clients to provision
Wi-Fi and MQTT, tune the runtime threshold, and observe the runtime in real
time.

## BLE-Specific Troubleshooting

### The client cannot control the device after connecting

Check these first:

1. the client writes exact ASCII commands
2. the value passed to `SET_THRESHOLD` is finite and inside `0.0-10.0`
3. the client does not depend on sysinfo ordering

### The firmware starts but never joins Wi-Fi

Check the active Wi-Fi values first:

1. request fresh sysinfo and inspect `wifi_ssid`, `wifi_bssid`,
   `wifi_channel`, and `wifi_saved`
2. if using `tools/web/espectre-ble.html`, press `Save Wi-Fi` and wait for the
   station reconnect after `APPLY_WIFI`
3. if no provisioning has been stored yet, verify the Kconfig defaults used at
   build time:
   - `ESPECTRE_WIFI_SSID`
   - `ESPECTRE_WIFI_PASSWORD`
   - optional `ESPECTRE_WIFI_BSSID`

### The BLE firmware is not the right fit for the workflow

That can be expected. This frontend is optimized for custom BLE integrations,
not for Home Assistant-style provisioning or the Matter commissioning flow.

## Related Files

- `../../runtime/ble_protocol.h`:
  UUIDs and default device name
- `../../runtime/espectre_protocol.cpp`:
  shared protocol payload and command helpers
- `../../runtime/esp_idf/protocol/wifi_provisioning_service.cpp`:
  shared ESP-IDF Wi-Fi provisioning command handling
- `espectre/ble_frontend.cpp`:
  command parsing, sysinfo emission, telemetry serialization
- `../../runtime/esp_idf/protocol/ble_bindings_nimble.cpp`:
  NimBLE transport implementation
- `../../../../tools/web/espectre-ble.html`:
  local Web Bluetooth provisioning and protocol test client
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
