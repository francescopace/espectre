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

- `ESPHome` remains focused on the most complete Home Assistant entity surface
  and YAML/codegen
- `Native` exposes the standalone integration surface over BLE and MQTT, plus
  an additive Home Assistant MQTT Discovery adapter
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

- [Configure](https://espectre.dev/configure/):
  Web Bluetooth provisioning and protocol test client
- [The Game](https://espectre.dev/game/):
  example interactive client built on the same BLE surface

Each release and snapshot publishes one full-flash native image and one
application-only OTA payload per supported chip. Both contain the same
application features; the smaller `-ota.bin` file omits the bootloader,
partition table, and other full-flash regions required only for USB recovery.
GitHub Pages stages only the full-flash image for the browser flasher.

### Local ESP-IDF Workflow

Before building locally, complete the shared
[`ESP-IDF Local Build Prerequisite`](../../../../docs/SETUP.md#esp-idf-local-build-prerequisite).
The repository CLI auto-detects a reusable ESP-IDF install, so the wrapper-first
workflow does not require a separate setup check before build.
See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side
tools, and wrapper behavior.

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

### Web Bluetooth Configuration Client

[Configure](https://espectre.dev/configure/) is the reference browser client
for BLE validation, provisioning, and live diagnostics. Run
`./espectre ui ble` to serve the same application from localhost.

Current capabilities:

- connect to the ESPectre BLE service from a desktop browser
- subscribe to telemetry and sysinfo notifications
- enable or disable the live telemetry subscription without disconnecting
- request a fresh sysinfo block with `REQ_SYSINFO`
- adjust the runtime threshold with `SET_THRESHOLD:X.XX`
- persist the runtime motion debounce thresholds with `SET_MOTION_HITS:on=4&off=3`
- select and persist the runtime detector with `SET_DETECTOR:classic` or `SET_DETECTOR:ml`
- select the same detector over MQTT with `{"command":"set_detector","detector":"ml"}`
- show a firmware-generated read-only `device_id`
- inspect the immutable firmware-derived `device_name`
- edit the human-facing `device_label`
- clear the persisted device-facing configuration without disconnecting
- expose the immutable BLE pairing name as the shared `device_name`
- provision or clear Wi-Fi credentials over BLE
- provision or clear MQTT configuration over BLE
- request OTA status, check for updates, and start HTTPS OTA over BLE

Requirements:

- desktop Chrome, Edge, or another Chromium-based browser with Web Bluetooth
- a secure context such as `http://localhost` or `https://`
- a BLE-capable ESP32 target supported by this frontend

Recommended local workflow from the repository root:

```bash
./espectre ui ble
```

Usage notes:

1. click `Connect` and select the ESPectre device
2. wait for the initial `REQ_SYSINFO` refresh after notifications start
3. disable live BLE telemetry from the test client when you only need provisioning or sysinfo
4. use `Save Wi-Fi` to send one atomic `SET_WIFI_CONFIG` update
5. use `Save Device` to persist the human-facing `device_label`
6. use `Clear Device` when you want to reset the persisted device-facing config while keeping the generated `device_id`
7. edit the runtime tuning controls to update the live sensing configuration
8. use `Save MQTT` to send one atomic `SET_MQTT_CONFIG` update and enable MQTT transport
9. use the OTA controls to request status, check the built-in release manifest, or start the update

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

## Home Assistant MQTT Discovery

The native frontend publishes a Home Assistant MQTT adapter surface on top of
the shared ESPectre MQTT protocol. It is enabled in the versioned firmware
defaults and can be disabled by clearing
`CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED` in `menuconfig`. When enabled, the
firmware:

- publishes retained MQTT Discovery config for motion, movement score, and the
  runtime detector select when detector switching is supported
- publishes plain HA state topics under the same device topic base used by
  ESPectre MQTT
- derives HA availability from the canonical ESPectre `status` topic, including
  its existing Last Will, so graceful and unexpected disconnects are reflected
  without replacing the ESPectre lifecycle contract
- subscribes to `homeassistant/status` and republishes discovery when Home
  Assistant announces `online`

This profile is additive. The canonical ESPectre topics under
`espectre/v1/devices/{device_id}/...` remain unchanged for standalone clients
and tooling.

## OTA

The native frontend uses the shared ESPectre MQTT command surface plus a shared
ESP-IDF HTTPS OTA implementation, and it exposes the same OTA service through
its BLE control characteristic.

Operational model:

- MQTT and BLE both call the same built-in OTA service
- `ota_check` checks the per-chip manifest embedded as a GitHub Releases URL
- `ota_start` resolves that manifest and downloads the application image into
  the inactive OTA slot
- MQTT clients cannot override the server, manifest, image, or target version
- successful OTA schedules an immediate reboot into the new slot

Stable builds use the latest GitHub release manifest by default. Snapshot
builds use the rolling `snapshot` release. The manifest filename is
`espectre-native-ota-<chip>.json`, and its `image_url` points to the matching
versioned `-ota.bin` release asset.

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
Wi-Fi and MQTT, tune the runtime threshold and motion-hit debounce, trigger OTA,
and observe the runtime in real time.

## BLE-Specific Troubleshooting

### The client cannot control the device after connecting

Check these first:

1. the client writes exact ASCII commands
2. the value passed to `SET_THRESHOLD` is finite and inside the shared detector range (`0.0-1.0`)
3. the values passed to `SET_MOTION_HITS:on=...&off=...` are integers inside the shared `1-20` range
4. the value passed to `SET_DETECTOR` is exactly `classic` or `ml`; accepted selections persist across reboot
5. the client does not depend on sysinfo ordering

### The firmware starts but never joins Wi-Fi

Check the active Wi-Fi values first:

1. request fresh sysinfo and inspect `wifi_ssid`, `wifi_bssid`,
   `wifi_channel`, and `wifi_connected`
2. in the Configure page, press `Save Wi-Fi` and wait for the
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
- `../../../../docs/web/configure/index.html`:
  unified Web Bluetooth provisioning and protocol test client
- [The Game](https://espectre.dev/game/):
  published example client built on this protocol
