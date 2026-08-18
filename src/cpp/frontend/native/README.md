# ESPectre Native Frontend

Use this guide for the standalone Native firmware, including BLE provisioning, MQTT operation, Home Assistant MQTT Discovery, and OTA. First-time users can jump to [Getting Started](#getting-started); client and firmware developers can continue through the protocol and implementation sections. The shared message model is documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Scope

The native frontend is intentionally separate from the ESPHome frontend:

- `ESPHome` remains focused on the most complete Home Assistant entity surface and YAML/codegen
- `Native` exposes the standalone integration surface over BLE and MQTT, plus an additive Home Assistant MQTT Discovery adapter
- `Matter` exposes the same runtime through Matter clusters

The native frontend now also supports HTTPS OTA triggered from its MQTT command plane.

The current native frontend preserves the protocol used by the browser BLE and MQTT tools in `docs/web/index.html`, but it is not tied to that specific client.

## Getting Started

If you arrived here from [`SETUP.md`](../../../../docs/SETUP.md), this README is the next step for the standalone native firmware path.

### Browser-Flashed Firmware

The web flasher can install published `Native` images for supported chips. After flashing, use a BLE client that understands this protocol, such as:

- [Configure](https://espectre.dev/#configure): Bluetooth provisioning for Wi-Fi, MQTT, and the device name
- [Monitor](https://espectre.dev/#monitor): MQTT telemetry, tuning, and diagnostics
- [The Game](https://espectre.dev/game/): example interactive client over MQTT after BLE setup

Each release and snapshot publishes one full-flash native image and one application-only OTA payload per supported chip. Both contain the same application features; the smaller `-ota.bin` file omits the bootloader, partition table, and other full-flash regions required only for USB recovery. GitHub Pages stages only the full-flash image for the browser flasher.

### Local ESP-IDF Workflow

Before building locally, complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites). The repository CLI prefers a reusable local ESP-IDF installation and falls back to the pinned Docker build environment when local ESP-IDF is absent; use [`CLI.md`](../../../../docs/CLI.md) for backend controls and command syntax.

Repository CLI:

```bash
./espectre native build --chip c3 --clean
./espectre native flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

The CLI is a thin wrapper over the ESP-IDF app in this directory. On Windows, use `.\espectre.cmd native ...` and `.\espectre.cmd monitor --port COM5`. Docker can replace local ESP-IDF for `build`; `flash` and `doctor` continue to use the local environment.

### Browser Configure and Monitor tools

[Configure](https://espectre.dev/#configure) is the reference browser BLE client. It uses Web Bluetooth only for nearby Wi-Fi, MQTT, and device-label setup. [Monitor](https://espectre.dev/#monitor) then uses MQTT over WebSockets for runtime controls, diagnostics, and recovery. To preview the same pages from this repository, serve `docs/web` locally as described in [docs/web/README.md](../../../../docs/web/README.md).

Current capabilities:

- connect to the ESPectre BLE service from a desktop browser
- subscribe to sysinfo notifications
- request a fresh sysinfo block with `REQ_SYSINFO`
- show a firmware-generated read-only `device_id`
- inspect the immutable firmware-derived `device_name`
- edit the human-facing `device_label`
- expose the immutable BLE pairing name as the shared `device_name`
- provision or clear Wi-Fi credentials over BLE
- select `2g`, `5g`, or `auto` over BLE when sysinfo reports `supports_wifi_5ghz=true`
- provision or clear MQTT configuration over BLE
- request OTA status, check for updates, and start HTTPS OTA over BLE
- stop BLE after Wi-Fi and MQTT are saved with `STOP_BLE`; disconnecting the Configure client leaves BLE advertising so setup can be reopened without another recovery action
- restart BLE later with MQTT `set_ble` (`ble on` in `./espectre mqtt`)
- restart BLE without MQTT by holding the board BOOT button for the configured recovery interval, 3 seconds by default
- use sensing controls over canonical MQTT with `commands`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate`, `set_csi_traffic_mode`, `set_traffic_generator_mode`, and `set_ble`

BLE does not carry live sensing, threshold or detector writes, CSI traffic control, or recalibration. Sensing pauses while BLE is up.

Use Lightweight Detection when the Native firmware must preserve more CPU time and working memory for MQTT, OTA, or product-specific services. Use High-Accuracy Detection when higher detection quality and calibration-free startup justify its additional feature state and inference work. Lightweight requires about 10 seconds of clean, ready quiet-room coverage after temporal warmup, so insufficient occupancy extends its wall-clock calibration; High Accuracy skips threshold calibration but still waits for CSI readiness and feature-window warmup. The selected profile persists across reboot and is changed over MQTT or Home Assistant, not BLE.

Requirements:

- desktop Chrome, Edge, or another Chromium-based browser with Web Bluetooth
- a secure context such as `http://localhost` or `https://`
- a BLE-capable ESP32 target supported by this frontend

Usage notes:

1. open [Configure](https://espectre.dev/#configure), click `Connect nearby device`, and select the ESPectre device
2. wait for the initial `REQ_SYSINFO` refresh after notifications start
3. use `Save Wi-Fi` to send one atomic `SET_WIFI_CONFIG` update
4. use `Save MQTT` to send one atomic `SET_MQTT_CONFIG` update and enable MQTT transport
5. use `Save Device` to persist the human-facing `device_label`
6. select `Start sensing`; Monitor connects to the broker, waits for MQTT `set_ble off` to be accepted, then opens live sensing and reports sensing as active after the first valid device telemetry
7. adjust MQTT-owned runtime settings directly in Monitor; changes apply when their fields change, while OTA status and on-demand diagnostics remain in the collapsed Diagnostics section below

The shared protocol semantics remain documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

The standalone native frontend uses the same shared periodic progress-bar sensing status log helper used by the ESPHome and Matter frontends, so the serial log shape stays aligned across those frontend surfaces.

## Wi-Fi Configuration

Unlike the ESPHome frontend, the standalone native firmware does not rely on YAML or Home Assistant for setup. In the current local-lab profile, Wi-Fi can be provisioned live over BLE and persisted in NVS.

Frontend-owned options in [`Kconfig.projbuild`](espectre/Kconfig.projbuild) remain useful as firmware defaults for reproducible images or first boot. Shared sensing options and their defaults now live in [`SETUP.md`](../../../../docs/SETUP.md), and can be overridden per frontend in [`sdkconfig.defaults`](app/sdkconfig.defaults). Versioned transport defaults in [`sdkconfig.defaults`](app/sdkconfig.defaults) also tune the standalone native firmware with the shared ESP-IDF Wi-Fi transport baseline now used across the standalone frontends: AMPDU enabled, larger Wi-Fi RX/TX buffers, plus lwIP mailbox and IRAM optimizations.

The shared menu keeps cadence and traffic ownership separate: `CONFIG_ESPECTRE_CSI_TARGET_PPS` is always positive, while the `CONFIG_ESPECTRE_CSI_TRAFFIC_MODE_*` choice selects internal, external, paced, or unmanaged traffic. The fixed target and detector-window duration define temporal slots; raw callback-rate jitter never reconstructs the detector.

| Option | Purpose |
|--------|---------|
| `ESPECTRE_WIFI_SSID` | Wi-Fi SSID |
| `ESPECTRE_WIFI_PASSWORD` | Wi-Fi password |
| `ESPECTRE_WIFI_BSSID` | Optional BSSID lock |
| `ESPECTRE_WIFI_BAND_2G`, `ESPECTRE_WIFI_BAND_5G`, `ESPECTRE_WIFI_BAND_AUTO` | Build-time band policy; exactly one selected |
| `ESPECTRE_WIFI_CHANNEL` | Optional channel hint (`0` = auto) |

`ESPECTRE_WIFI_BAND_2G` is the default. An ESP32-C5 integrator can instead select `ESPECTRE_WIFI_BAND_5G` or `ESPECTRE_WIFI_BAND_AUTO`; the sensing PHY remains HT20 in all three cases. When `ESPECTRE_WIFI_BSSID` is set, the firmware uses fast scan and pins the association to that AP radio. Leave `ESPECTRE_WIFI_CHANNEL=0` unless you need a channel hint for repeatable CSI captures. The channel must belong to the selected band.

Runtime provisioning behavior:

- BLE starts automatically when Wi-Fi or MQTT is unconfigured, and Native pauses CSI while BLE is up
- `SET_WIFI_CONFIG` persists the full Wi-Fi block in NVS; credential, BSSID, and channel changes reconnect immediately without restarting BLE, while a changed `band_policy` applies after restart so Wi-Fi and CSI use the same policy
- after Wi-Fi and MQTT are saved, BLE stays up across nearby client disconnects and resumes advertising; only `STOP_BLE` or MQTT `set_ble` with `ble=off` closes setup so sensing can use the radio alone
- MQTT `set_ble` with `ble=on` starts BLE again for recovery or reconfiguration; `ble=off` or `STOP_BLE` stops it only when Wi-Fi and MQTT are already configured
- holding BOOT for `ESPECTRE_BLE_RECOVERY_BUTTON_HOLD_MS` starts the same BLE recovery path and pauses sensing even when MQTT is unavailable; the default is 3000 ms
- `CLEAR_WIFI` erases stored Wi-Fi values, disconnects the station, and brings BLE back for provisioning
- `CLEAR_MQTT` or an empty MQTT host brings BLE back until a broker is saved again
- `SET_MQTT_CONFIG` persists the full MQTT broker block in NVS and reinitializes the MQTT transport

Physical BLE recovery is enabled by default when Bluetooth is built. `ESPECTRE_BLE_RECOVERY_BUTTON_GPIO` follows the BOOT strap used by the target family: GPIO0 on ESP32 and ESP32-S3, GPIO9 on ESP32-C3 and ESP32-C6, and GPIO28 on ESP32-C5. The input is active-low, is polled without blocking the runtime loop, and fires once per completed hold. Override the GPIO or disable `ESPECTRE_BLE_RECOVERY_BUTTON_ENABLED` when a custom board routes BOOT differently or owns that pin for another purpose.

This means the current standalone native firmware is best suited for:

- local integration experiments
- custom client development
- controlled deployments and recovery flows where BLE-assisted provisioning is acceptable

It is still a lab-oriented provisioning path, not a polished end-user flow comparable to ESPHome.

## Protocol Reference

The shared BLE protocol surface is documented in:

- [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md)

That file defines:

- service and characteristic UUIDs
- sysinfo framing and key semantics
- setup control command syntax
- nearby BLE client expectations

Local implementation anchors:

- [`ble_protocol.h`](../../runtime/ble_protocol.h): protocol constants such as UUIDs and default device name
- [`native_frontend.cpp`](espectre/native_frontend.cpp): command handling and sysinfo emission
- [`espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp): shared MQTT topic, payload, and command serialization

### On-Demand MQTT Diagnostics

The Native MQTT `stats` command always returns the base uptime, free-heap, and loop-time fields, plus traffic-generator, CSI, and Wi-Fi diagnostics. The frontend refreshes the cached diagnostic sample from the existing sensing update that also feeds the periodic status log. It adds that sample only to an explicitly requested `stats` response; it does not add a diagnostic timer or publish stats periodically. This on-demand surface is available in production and does not require runtime debug telemetry.

The field definitions and command topic are documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Home Assistant MQTT Discovery

The native frontend publishes a Home Assistant MQTT adapter surface on top of the shared ESPectre MQTT protocol. It is enabled in the versioned firmware defaults and can be disabled by clearing `CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED` in `menuconfig`. When enabled, the firmware:

- publishes retained MQTT Discovery config for Motion Detected, Movement Score, Threshold, Motion On Hits, Motion Off Hits, Detection Profile, CSI Traffic Ownership, CSI Traffic Source, Trigger Calibration, the ESPHome CSI diagnostic sensors, and Refresh Diagnostics when those runtime controls are supported
- publishes empty retained discovery payloads for leftover Intensity and previous Native object IDs so Home Assistant entity IDs match ESPHome slugs after a prefix swap
- publishes plain HA state topics under the same device topic base used by ESPectre MQTT
- derives HA availability from the retained canonical ESPectre `status` topic and retained Last Will, so late subscribers, graceful disconnects, and unexpected disconnects receive the current lifecycle state
- subscribes to `homeassistant/status` and republishes discovery when Home Assistant announces `online`

HA sensing cadences match ESPHome so the same Home Assistant dashboard can be reused after replacing entity ID prefixes:

| Entity | Topic suffix | Cadence |
|--------|--------------|---------|
| Motion Detected | `ha/motion/state` | Filtered state edges |
| Movement Score | `ha/movement/state` | Detector evaluation (`evaluation_interval_ms`, default 250 ms) |
| Threshold | `ha/threshold/state` and `ha/threshold/set` | On change, plus connect/birth snapshot; writable 0.0–1.0 number |
| Motion On Hits | `ha/motion_on_hits/state` and `ha/motion_on_hits/set` | On change, plus connect/birth snapshot; writable 1–20 number |
| Motion Off Hits | `ha/motion_off_hits/state` and `ha/motion_off_hits/set` | On change, plus connect/birth snapshot; writable 1–20 number |
| Detection Profile | `ha/detector/state` and `ha/detector/set` | On change, plus connect/birth snapshot; writable `lightweight` / `high_accuracy` configuration select |
| CSI Traffic Ownership | `ha/csi_traffic_mode/state` and `ha/csi_traffic_mode/set` | On change, plus connect/birth snapshot; writable `internal`, `external`, `pacing`, or `disabled` select |
| CSI Traffic Source | `ha/traffic_generator_mode/state` and `ha/traffic_generator_mode/set` | On change, plus connect/birth snapshot; writable `ping` / `dns` select |
| Trigger Calibration | `ha/calibrate/state` and `ha/calibrate/set` | ON while recalibrating; ON starts startup recalibration, OFF is ignored while a session is running |
| Traffic TX Rate, CSI rates, occupancy, Wi-Fi channel, Wi-Fi RSSI | `ha/traffic_tx_rate/state`, `ha/csi_callback_rate/state`, `ha/csi_accepted_rate/state`, `ha/csi_admitted_rate/state`, `ha/csi_filtered_rate/state`, `ha/csi_missing_rate/state`, `ha/csi_excess_rate/state`, `ha/csi_stale_rate/state`, `ha/csi_out_of_order_rate/state`, `ha/csi_occupancy/state`, `ha/wifi_channel/state`, `ha/wifi_rssi/state` | On demand after Refresh Diagnostics; diagnostic category |
| Refresh Diagnostics | `ha/diagnostics/set` | Button; publishes the latest cached diagnostic sample |

Entity IDs look like `sensor.native_0x0000111122223333_movement_score`. Copy the ESPHome dashboard from [`home-assistant-dashboard.yaml`](../esphome/examples/home-assistant-dashboard.yaml) and replace the `espectre_` prefix.

![ESPectre Home Assistant dashboard](../../../../docs/web/assets/images/guides/home-assistant-dashboard.png)

*Home Assistant dashboard from Native MQTT Discovery. ESPHome and Micro MQTT Discovery use the same cards after replacing the device prefix.*

This profile is additive. The canonical ESPectre topics under `espectre/v1/devices/{device_id}/...` remain unchanged for standalone clients and tooling.

## OTA

The native frontend uses the shared ESPectre MQTT command surface plus a shared ESP-IDF HTTPS OTA implementation, and it exposes the same OTA service through its BLE control characteristic.

Operational model:

- MQTT and BLE both call the same built-in OTA service
- `ota_check` checks the per-chip manifest embedded as a GitHub Releases URL
- `ota_start` resolves that manifest and downloads the application image into the inactive OTA slot
- MQTT clients cannot override the server, manifest, image, or target version
- successful OTA schedules an immediate reboot into the new slot

Stable builds use the latest GitHub release manifest by default. Snapshot builds use the rolling `snapshot` release. The manifest filename is `espectre-native-ota-<chip>.json`, and its `image_url` points to the matching versioned `-ota.bin` release asset.

## Firmware Limits and Expectations

The current standalone native frontend intentionally stays small.

Important current limits:

- provisioning is intentionally lab-oriented and low-ceremony
- the BLE control surface is still ASCII commands rather than a structured schema
- clients should not assume diagnostic sysinfo fields are stable forever
- there is no capability discovery or negotiated feature set yet
- OTA uses HTTPS transport and dual OTA slots, so local recovery still starts from the published factory image when an image must be reflashed from USB

This keeps the transport simple while allowing external BLE clients to provision Wi-Fi and MQTT, set device identity, trigger OTA, and inspect read-only status. Live sensing and runtime detector control stay on MQTT.

## BLE-Specific Troubleshooting

### The client cannot control the device after connecting

Check these first:

1. the client writes exact ASCII commands from the setup surface in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md)
2. sensing writes such as `SET_THRESHOLD` or `RECALIBRATE` are rejected over BLE; use MQTT
3. the client does not depend on sysinfo ordering

### CSI occupancy drops while BLE is on

On ESP32-C3, the Bluetooth controller and Wi-Fi coexistence starve CSI admission even at default NimBLE advertising intervals. Native therefore runs BLE only for setup and recovery: it starts automatically when Wi-Fi or MQTT is unconfigured, pauses sensing while BLE is up, keeps advertising across nearby client disconnects, and stops only when `STOP_BLE` or MQTT `set_ble` with `ble=off` explicitly closes setup. Use MQTT `set_ble` with `ble=on`, `ble on` in `./espectre mqtt`, or hold BOOT for 3 seconds to advertise again. The product decision is recorded in [`2026-08-17-keep-native-ble-as-setup-recovery.md`](../../../../docs/adr/2026-08-17-keep-native-ble-as-setup-recovery.md).

While BLE is up, Native uses the NimBLE default advertising and connection timings so nearby discovery stays fast. It does not publish live sensing over BLE.

### The firmware starts but never joins Wi-Fi

Check the active Wi-Fi values first:

1. request fresh sysinfo and inspect `wifi_ssid`, `wifi_bssid`, `wifi_channel`, `wifi_band_policy`, and `wifi_connected`
2. in nearby BLE setup, press `Save Wi-Fi` and wait for the station reconnect after the atomic `SET_WIFI_CONFIG` update
3. if no provisioning has been stored yet, verify the Kconfig defaults used at build time:
   - `ESPECTRE_WIFI_SSID`
   - `ESPECTRE_WIFI_PASSWORD`
   - optional `ESPECTRE_WIFI_BSSID`

### The native firmware is not the right fit for the workflow

That can be expected. This frontend is optimized for the native standalone integration surface, not for Home Assistant-style provisioning or the Matter commissioning flow.

## Implementation Map

This map is for frontend maintainers; it is not required for provisioning or ordinary operation.

- `espectre/`: frontend adapter and runtime-to-BLE mapping
- `app/`: standalone ESP-IDF firmware app
- `../../runtime/` and `../../runtime/esp_idf/frontend_support/`: shared ESPectre Protocol serializer, BLE binding interface, NimBLE transport, NVS-backed device/Wi-Fi config store, MQTT transport boundary, and ESP-IDF provisioning helpers
- `espectre/Kconfig.projbuild`: frontend-owned Wi-Fi configuration knobs
- `../../runtime/esp_idf/espectre_config/Kconfig.projbuild`: shared sensing/runtime configuration knobs consumed by the standalone sensing frontends

The firmware app uses the shared standalone Wi-Fi manager for station setup, BSSID/channel fast scan, CSI Wi-Fi policy, and retry behavior. The frontend adapter uses the shared runtime frontend controller and owns only the BLE protocol mapping.

## Related Files

- `../../runtime/ble_protocol.h`: UUIDs and default device name
- `../../runtime/ble_bindings_noop.h`: portable no-op BLE binding used when Bluetooth is disabled
- `../../runtime/espectre_protocol.cpp`: shared protocol payload and command helpers
- `../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`: shared ESP-IDF Wi-Fi provisioning command handling
- `espectre/native_frontend.cpp`: command parsing and sysinfo emission
- `../../runtime/esp_idf/frontend_support/ble_bindings_nimble.cpp`: NimBLE transport implementation
- `../../../../docs/web/index.html`: browser Configure and Monitor tools
- [The Game](https://espectre.dev/game/): published example client over MQTT after BLE setup
