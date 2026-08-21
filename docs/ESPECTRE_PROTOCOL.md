# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools, MQTT dashboards, and future web orchestration services.

ESPectre Protocol defines the message model. BLE, MQTT, MQTT over TLS, device shadows, jobs, and future bridges are transports or profiles that carry the same semantics across different trust boundaries.

This is an implementation reference for firmware, client, and integration developers. Read [Principles](#principles) and [Message Families](#message-families) when implementing a consumer; read the transport sections only for the connection mechanism you use. A **transport** carries messages, a **profile** adds deployment rules without changing their meaning, and a **retained** MQTT message is stored by the broker for future subscribers.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- BLE is for proximity, setup, recovery, and nearby identity or status.
- MQTT is the operational plane for telemetry, status, commands, dashboards, history, and alerts.
- Web orchestration profiles add identity, credentials, tenancy, retention, and fleet management; they do not redefine device telemetry.
- `device_id` is a logical protocol identifier. Native, Matter, Streamer, and Micro-ESPectre derive it once per boot as the first 64 bits of `SHA-256("espectre-device-id-v1" || station_mac_bytes)` and cache the result. This hides the MAC from routine inspection, but the stable pseudonym remains linkable and is not anonymous.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level radio traces, and serial logs must not be sent to managed services by default.

## Transports

### BLE

BLE is the proximity transport. It is used when a user is near a device or when network connectivity is not available.

Native firmware treats BLE as a setup and recovery radio. It advertises automatically when Wi-Fi SSID or MQTT host is missing, pauses CSI while BLE is up, and continues advertising after a nearby client disconnects. Compile-time Kconfig defaults count as configured, so images that already bake in Wi-Fi and MQTT skip BLE at boot. After nearby setup saves Wi-Fi and MQTT, only `STOP_BLE` or MQTT `set_ble` with `ble=off` closes setup so sensing can resume. MQTT `set_ble` starts BLE again when the device is already on the network. Native’s physical recovery input provides the same `ble=on` transition without a network: holding the board BOOT button for the configured recovery interval starts BLE and pauses sensing.

Current BLE responsibilities:

- advertise device availability
- expose protocol and sysinfo notifications
- expose the firmware-generated, read-only `device_id` and provision the mutable `device_label`
- provision Wi-Fi credentials
- provision MQTT endpoint settings
- trigger OTA status, checks, and updates through the shared HTTPS OTA service
- recover from broken Wi-Fi or MQTT configuration

Native does not expose live sensing, threshold or detector writes, CSI traffic control, or recalibration over BLE. Those commands stay on MQTT and Home Assistant Discovery. Sensing pauses while BLE is up. The product decision is recorded in [`2026-08-17-keep-native-ble-as-setup-recovery.md`](adr/2026-08-17-keep-native-ble-as-setup-recovery.md).

The dependency-free browser reference client is [`espectre-ble.js`](web/assets/js/espectre-ble.js). Its typed builders cover every command in the [Current BLE Control Surface](#current-ble-control-surface), enforce the firmware’s UTF-8 field and 512-byte control-write limits, serialize GATT writes, and accept only complete `proto_version=...` through `END` sysinfo snapshots.

Future BLE responsibilities:

- Wi-Fi scan and selection
- Wi-Fi credential validation
- web-service claim bootstrap
- reboot or reconnect commands
- structured command/result framing for the setup commands above

### MQTT

MQTT is the operational transport once the device has network access.

The same topic shape is valid for a local broker and for a managed broker, with auth and tenancy added by the deployment profile:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/stats
espectre/v1/devices/{device_id}/commands/catalog
espectre/v1/devices/{device_id}/ota/state
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
```

Managed-service MQTT should use TLS and per-device credentials. Local lab MQTT may use a simpler broker/auth model, but should keep the same message shape.

The dependency-free browser protocol layer is [`espectre-mqtt.js`](web/assets/js/espectre-mqtt.js). It is transport-policy agnostic and implements canonical topic construction, retained `info`/`status` discovery, protocol-version and JSON-object validation for every canonical message family above, generic command publication without a duplicated verb allowlist, correlation of `accepted`/`rejected` responses, timeouts, and pending-command cleanup. The website supplies the MQTT.js WebSocket transport and consumes the additive Home Assistant scalar topics separately.

### Home Assistant MQTT Adapter Profile

Native and Micro-ESPectre can publish an additive Home Assistant MQTT Discovery surface without changing the canonical ESPectre topics above. Discovery payloads use the standard `{discovery_prefix}/{component}/{object_id}/config` topic shape. Native also retains its canonical `status` payload so late subscribers receive the current availability; entity-shaped state topics remain non-retained under `espectre/v1/devices/{device_id}/ha/...`.

The HA adapter publishes sensing entities that match the ESPHome Home Assistant surface so one dashboard can be reused after replacing the device prefix: Motion Detected on filtered state edges, Movement Score on every detector evaluation (`evaluation_interval_ms`), writable Threshold on operator writes, calibration, and Lightweight settled-level recovery, Motion On Hits, and Motion Off Hits numbers, a Detection Profile select where the frontend supports runtime detector switching, CSI Traffic Ownership plus CSI Traffic Source selects where the frontend supports traffic control, a Trigger Calibration switch that starts startup recalibration, and the ESPHome CSI diagnostic sensors plus a Refresh Diagnostics button that publishes the latest cached sample on demand. Discovery `object_id` suffixes follow the ESPHome entity-ID slugs (`motion_detected`, `movement_score`, `trigger_calibration`, and so on); MQTT state and command topic suffixes under `ha/` stay unchanged. Canonical `telemetry` JSON keeps `movement_score` and `threshold` on that same evaluation cadence. Leftover Intensity and previous Native/Micro discovery object IDs are unpublished with empty retained configs.

Both adapters subscribe to `homeassistant/status` and republish discovery when Home Assistant announces `online`; this birth message is a recovery trigger, not the only discovery bootstrap. Native derives availability from the retained canonical `status` payload and its retained Last Will, while Micro-ESPectre uses a plain `ha/availability` topic. The Native adapter is enabled in the published firmware defaults and can be disabled at build time; Micro-ESPectre keeps the adapter opt-in. See [`README.md`](../src/cpp/frontend/native/README.md) for Native and [`README.md`](../src/python/micro_espectre/README.md) for Micro-ESPectre entity surfaces and configuration options.

## Message Families

### Telemetry

Published on:

```text
espectre/v1/devices/{device_id}/telemetry
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "frontend": "native",
  "timestamp_ms": 123456,
  "motion_state": "idle",
  "movement_score": 0.18,
  "threshold": 0.45,
  "detector": "lightweight",
  "health": {
    "uptime_s": 3821
  }
}
```

Native MQTT telemetry publishes on every detector evaluation once `ready_to_publish` is true, matching Micro-ESPectre. Filtered motion-state transitions update the Home Assistant motion entity immediately without a second telemetry publish. `publish_interval_ms` remains a monotonic-clock heartbeat for status logs and diagnostics sampling; it never publishes sensing telemetry and never forces detector evaluation. Native BLE does not carry live sensing telemetry.

### Status

Published on:

```text
espectre/v1/devices/{device_id}/status
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "online": true,
  "timestamp_ms": 123456
}
```

Native retains the latest status payload. A normal shutdown publishes retained `online: false`; after an unexpected disconnect, the broker publishes the retained Last Will with the same offline state. A later connection replaces it with retained `online: true`, allowing availability consumers that subscribe after discovery to recover the current state. MQTT connect also publishes retained `info` and, when OTA is present, the current `ota/state`, so a client that watched `reboot_scheduled` can treat the next `online: true` as the device having returned from the OTA reboot. Micro-ESPectre retains `info` the same way; its canonical `status` remains non-retained because HA availability uses the separate `ha/availability` topic.

### Info

Published on:

```text
espectre/v1/devices/{device_id}/info
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "device_name": "ESPectre C6 42bbac",
  "device_label": "Living Room",
  "frontend": "native",
  "firmware_version": "1.2.3",
  "chip": "esp32c6",
  "supports_info": true,
  "supports_stats": true,
  "supports_runtime_threshold": true,
  "supports_runtime_motion_hits": true,
  "supports_runtime_detector": true,
  "supports_manual_recalibration": true,
  "supports_traffic_control": true,
  "supports_ota": true,
  "supports_ble": true,
  "network": {
    "channel": {
      "primary": 6
    }
  },
  "detection": {
    "algorithm": "lightweight"
  },
  "csi_traffic_mode": "internal",
  "traffic_mode": "ping",
  "csi_target_pps": 100,
  "evaluation_interval_ms": 250,
  "publish_interval_ms": 1000
}
```

The `supports_*` fields are authoritative capability declarations for clients. Clients should not infer command support from `frontend`, telemetry fields, or other payload content. Native and Micro publish `info` retained on connect and after an `info` command so late subscribers, including `./espectre mqtt` discovery, see the current frontend identity instead of a previous retained payload for the same `device_id`. MQTT clients that need command names should send `commands` and read `commands/catalog` instead of reconstructing the list from these flags. `network` and `detection` are optional. Canonical MQTT `info` reports the active Wi-Fi channel when available, but does not serialize the local IP address or station MAC. `csi_traffic_mode`, `traffic_mode`, and `csi_target_pps` are included when the frontend owns CSI traffic configuration; omit them when those values are unset. `evaluation_interval_ms` and `publish_interval_ms` are the detector evaluation cadence and the status-log heartbeat; omit them when unset. Nearby setup and local logs may still expose configuration or link details, including SSID, BSSID, local IP, station MAC, broker host, or broker username. Managed services should not collect those values by default.

### Stats

Published on:

```text
espectre/v1/devices/{device_id}/stats
```

in response to an explicit `stats` command. Native and Micro include the CSI and Wi-Fi diagnostic fields below. A frontend that does not sample those counters omits the extra keys and keeps the shared core (`uptime`, `free_memory_kb`, `loop_time_ms`):

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "timestamp_ms": 123456,
  "uptime": 3821,
  "free_memory_kb": 182.4,
  "loop_time_ms": 0.31,
  "traffic_tx_pps": 100,
  "csi_callback_pps": 96,
  "csi_accepted_pps": 90,
  "csi_admitted_pps": 84,
  "csi_filtered_pps": 6,
  "csi_missing_slots_pps": 10,
  "csi_excess_pps": 6,
  "csi_stale_pps": 0,
  "csi_out_of_order_pps": 0,
  "csi_occupancy": 0.84,
  "wifi_channel": 10,
  "wifi_rssi_dbm": -55
}
```

Stats are diagnostic. Product dashboards should prefer telemetry/status/info for normal operation. When available, `free_memory_kb` reports current free heap and `loop_time_ms` reports the measured last loop-body cost in milliseconds, excluding the outer task sleep or idle delay. Motion state, movement score, threshold, detector selection, and turbulence belong to telemetry or live config/info surfaces instead of `stats`.

Native and Micro always include the CSI and Wi-Fi fields in a requested `stats` response. Both derive rates from the cumulative counters whenever the existing periodic sensing update runs, cache that completed sample, and do not add a diagnostic timer or publish it periodically. `traffic_tx_pps` is the traffic-generator transmit rate; `csi_callback_pps` is the raw CSI callback rate; `csi_accepted_pps` is the identity-accepted rate; `csi_admitted_pps` is the detector input rate after temporal admission; `csi_filtered_pps` is the capture-filter drop rate; the temporal drop fields distinguish missing slots, same-slot excess, stale packets, and out-of-order packets; and `csi_occupancy` is the valid fraction of the active detector window. Occupancy is diagnostic telemetry and does not change the device send rate. The extra CSI fields are additive on protocol `1.0`; consumers may ignore unknown keys. The SDK sample uses `csi_occupancy_ratio` for the same occupancy value. Before the first periodic sensing update completes, rate fields are zero.

ESPHome exposes the same cached measurements as diagnostic entities. Native MQTT Discovery and Micro-ESPectre MQTT match that surface: the diagnostic sensors stay unpublished until Home Assistant presses `Refresh Diagnostics`. These on-demand diagnostics are independent of the optional runtime debug logs.

### Command catalog

Published on:

```text
espectre/v1/devices/{device_id}/commands/catalog
```

in response to:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-catalog",
  "command": "commands"
}
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "commands": [
    "commands",
    "info",
    "stats",
    "set_threshold",
    "set_motion_hits",
    "set_detector",
    "recalibrate",
    "set_csi_traffic_mode",
    "set_traffic_generator_mode",
    "set_ble",
    "ota_status",
    "ota_check",
    "ota_start"
  ]
}
```

The list is derived from the same `supports_*` flags carried by `info`, plus `commands` itself. It is not retained. Clients should use it for help and completion instead of a local command allowlist. Firmware that does not implement `commands` rejects it, and clients should not reconstruct the list from `info`.

### Commands

Published to:

```text
espectre/v1/devices/{device_id}/commands/request
```

Set threshold:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-001",
  "command": "set_threshold",
  "threshold": 0.35
}
```

Select and persist the active detection profile on frontends that advertise runtime detector control:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-002",
  "command": "set_detector",
  "detector": "high_accuracy"
}
```

Accepted detector values are `lightweight` and `high_accuracy`. Switching to `lightweight` starts calibration automatically; switching to `high_accuracy` cancels any active calibration and follows the normal CSI-readiness and feature-window warmup path without threshold calibration.

Update the motion debounce thresholds on frontends that advertise runtime motion-hit control:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-003",
  "command": "set_motion_hits",
  "motion_on_hits": 4,
  "motion_off_hits": 3
}
```

Both values must stay inside the shared `1-20` range. Native persists accepted values across reboot.

Request a runtime recalibration on frontends that advertise manual recalibration:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-004",
  "command": "recalibrate"
}
```

Native and ESPHome run the shared runtime recalibration immediately. Micro-ESPectre queues the same recalibration work onto its main loop and keeps it session-only.

Update CSI traffic ownership on frontends that advertise traffic control:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-005",
  "command": "set_csi_traffic_mode",
  "csi_traffic_mode": "external"
}
```

Accepted values are `internal`, `external`, and `disabled`. Native persists the accepted value across reboot. Micro-ESPectre keeps the selection session-only. `pacing` is Streamer collector mode only and is rejected on sensing MQTT. On ESP-IDF sensing frontends, `external` opens the UDP listener on port `5555` and joins multicast group `239.255.0.1` unless `csi_traffic_multicast_group` is empty.

Update the internal traffic generator type on frontends that advertise traffic control:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-006",
  "command": "set_traffic_generator_mode",
  "traffic_generator_mode": "dns"
}
```

Accepted values are `ping` and `dns`. Native persists the accepted value across reboot. The selection is always stored, but only takes effect while `csi_traffic_mode` is `internal`.

Request an OTA manifest check. Omit `channel` to use the firmware's build-time default, or pass `release`, `preview`, or `develop`:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-ota-check",
  "command": "ota_check",
  "channel": "preview"
}
```

Start OTA from the selected or firmware-default channel:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-ota-start",
  "command": "ota_start",
  "channel": "release"
}
```

Native firmware resolves a per-chip GitHub Releases manifest URL from the channel. OTA commands do not accept server, manifest, image, or version parameters; payloads containing those overrides are rejected. When `channel` is omitted, release firmware uses the latest release, preview firmware uses the rolling `snapshot` tag, and develop firmware uses the rolling `snapshot-dev` tag. Frontends advertise support through `supports_ota`; Micro-ESPectre does not implement OTA commands.

Start or stop Native BLE setup mode. Sensing pauses while BLE is up. `off` is rejected until Wi-Fi is configured, so an unprovisioned device cannot drop its only setup radio:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-ble-1",
  "command": "set_ble",
  "ble": "on"
}
```

Accepted `ble` values are `on` and `off`. Micro-ESPectre rejects the command. After BLE starts, use nearby BLE setup. Disconnecting the nearby client keeps BLE advertising; writing `STOP_BLE` or sending `set_ble` with `ble=off` stops the radio when Wi-Fi SSID and MQTT host are already present from Kconfig defaults or NVS.

Publish OTA state on:

```text
espectre/v1/devices/{device_id}/ota/state
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "state": "update_available",
  "timestamp_ms": 123456,
  "busy": false,
  "update_available": true,
  "current_version": "1.2.2",
  "target_version": "1.2.3",
  "manifest_url": "https://github.com/francescopace/espectre/releases/latest/download/espectre-native-ota-esp32c6.json",
  "image_url": "https://github.com/francescopace/espectre/releases/download/1.2.3/espectre-native-1.2.3-esp32c6-ota.bin",
  "channel": "release",
  "message": "update available"
}
```

`ota/state` is not retained. Native publishes the current snapshot when MQTT connects, when an OTA command changes state, and when the HTTPS OTA worker reports progress. After a successful update the device reboots, so the next connect snapshot is `idle` with `current_version` set to the firmware now running.

Command result:

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "command_id": "cmd-001",
  "command": "set_threshold",
  "accepted": true,
  "message": "threshold updated"
}
```

## Current BLE Control Surface

The current BLE firmware still carries setup commands as ASCII control writes:

```text
REQ_SYSINFO
SET_DEVICE_CONFIG:device_label=Living Room
SET_MQTT_CONFIG:host=192.168.1.20&port=1883&username=mqtt&password=secret-password&topic_prefix=espectre%2Fv1%2Fdevices
CLEAR_MQTT_CONFIG
CLEAR_DEVICE_CONFIG
SET_WIFI_CONFIG:ssid=Lab%20Network&password=secret-password&channel=6&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&band_policy=2g
CLEAR_WIFI
OTA_STATUS
OTA_CHECK
OTA_CHECK:channel=preview
OTA_START
OTA_START:channel=develop
STOP_BLE
```

This is the current BLE framing, not a separate protocol. A future BLE framing can become more structured while preserving the same ESPectre Protocol command families and semantics.

Identity/config semantics for the current BLE control surface:

- `device_id` is the firmware-generated pseudonymous identity rendered as 16 lowercase hexadecimal characters without a `0x` prefix in BLE sysinfo, MQTT topics, MQTT payloads, Streamer discovery, and collector output; it is stable and linkable, and hashing the 48-bit MAC input does not make it anonymous or prevent a determined party from testing likely MAC values
- `device_name` is the immutable protocol/device name derived from chip and `device_id`
- `device_label` is the optional user-facing human-readable device label
- `SET_MQTT_CONFIG:...` replaces the full persisted MQTT broker block in one write
- `CLEAR_MQTT_CONFIG` clears only broker-related MQTT settings and disables the active MQTT transport
- `CLEAR_DEVICE_CONFIG` resets `device_label` to its build default, clears MQTT settings, disables the active MQTT transport, and keeps the firmware-generated `device_id`
- `SET_WIFI_CONFIG:...` replaces the full persisted Wi-Fi station block in one write; credentials, BSSID, and channel changes apply immediately, while a changed `band_policy` applies after restart so the Wi-Fi and CSI runtimes restart together
- `band_policy` accepts `2g`, `5g`, or `auto`; firmware rejects `5g` and `auto` unless the target reports `supports_wifi_5ghz=true`
- `CLEAR_WIFI` clears only persisted Wi-Fi station settings
- `STOP_BLE` stops BLE after Wi-Fi and MQTT are configured so CSI sensing can use the radio alone; it is rejected while either is unconfigured

## Current BLE Status Surface

The standalone native frontend exposes two GATT paths for setup:

- a line-based `sysinfo` characteristic for identity, configuration, and read-only diagnostics
- a control characteristic for Wi-Fi, MQTT, identity, OTA, and `STOP_BLE` writes

The telemetry characteristic UUID remains in the GATT table so older discovery still succeeds. Native does not notify on it, and `supports_live_telemetry` is `false`. Clients should not subscribe for live movement.

Sysinfo framing remains:

```text
key=value
...
END
```

`sysinfo` is a readable setup and status snapshot, not a live sensing transport. Runtime `motion_state` belongs on MQTT telemetry.

Current BLE `sysinfo` identity/config keys include:

| Key | Meaning |
|-----|---------|
| `device_id` | Current firmware-generated device identifier as 16 lowercase hexadecimal characters without a `0x` prefix |
| `device_name` | Current immutable protocol/device name derived from chip and `device_id` |
| `device_label` | Current human-readable device label |
| `mqtt_connected` | Whether the MQTT transport is currently connected |
| `mqtt_host` | Current MQTT broker host |
| `mqtt_port` | Current MQTT broker port |
| `mqtt_username` | Current MQTT username |
| `topic_prefix` | Current MQTT topic prefix |
| `wifi_connected` | Whether the Wi-Fi station is currently connected |
| `wifi_ssid` | Current persisted Wi-Fi SSID |
| `wifi_bssid` | Current persisted Wi-Fi BSSID lock |
| `wifi_channel` | Current persisted Wi-Fi channel hint (`0` = automatic scan) |
| `wifi_band_policy` | Persisted association policy: `2g`, `5g`, or `auto` |

Capability-oriented `sysinfo` keys may include:

| Key | Meaning |
|-----|---------|
| `frontend` | Firmware/frontend family currently exposing the BLE service |
| `supports_wifi_provisioning` | Whether BLE clients can edit and apply Wi-Fi settings |
| `supports_mqtt_config` | Whether BLE clients can edit MQTT broker settings |
| `supports_device_config` | Whether BLE clients can edit the user-facing `device_label` |
| `supports_runtime_threshold` | Native reports `false`; threshold writes belong to MQTT |
| `supports_runtime_motion_hits` | Native reports `false`; motion-hit writes belong to MQTT |
| `supports_runtime_detector` | Native reports `false`; detector selection belongs to MQTT |
| `supports_manual_recalibration` | Native reports `false`; recalibration belongs to MQTT |
| `supports_traffic_control` | Native reports `false`; CSI traffic writes belong to MQTT |
| `supports_live_telemetry` | Native reports `false`; BLE does not notify live sensing |
| `supports_extended_diagnostics` | Whether implementation-specific runtime diagnostics are exposed |
| `supports_ota` | Whether BLE clients can expose OTA-related controls |
| `supports_wifi_5ghz` | Whether the target radio can accept the `5g` and `auto` Wi-Fi band policies |
| `ble_active` | Whether Native currently has the BLE stack up |

Current BLE `sysinfo` diagnostic keys may include:

| Key | Meaning |
|-----|---------|
| `chip` | Target chip reported by the firmware, such as `esp32c3` |
| `firmware_version` | Running firmware version |
| `detector` | Active detection profile: `lightweight`, or `high_accuracy` |
| `window_ms` | Configured detection window duration in milliseconds |
| `lowpass` | Whether the low-pass stage is enabled |
| `lowpass_cutoff` | Low-pass cutoff in Hz |
| `hampel` | Whether the Hampel filter is enabled |
| `hampel_window` | Hampel window size |
| `hampel_threshold` | Hampel threshold in MAD units |
| `csi_traffic_mode` | CSI traffic ownership mode: `internal`, `external`, or `disabled` on sensing firmware; Streamer uses collector `pacing` internally |
| `traffic_mode` | Internal traffic generator mode such as `ping` or `dns` |
| `csi_target_pps` | Internal traffic generator target rate in packets per second |
| `publish_interval_ms` | Periodic status-log cadence in milliseconds |
| `evaluation_interval_ms` | Detector evaluation cadence in milliseconds |
| `motion_hits` | Motion-on/off consecutive hit thresholds |
| `ota_state` | Current OTA state reported by the shared HTTPS OTA service |
| `ota_busy` | Whether an OTA worker is active |
| `ota_update_available` | Whether the last OTA check found an update |
| `ota_current_version` | Firmware version compared against the manifest |
| `ota_target_version` | Version reported by the pending OTA target, when known |
| `ota_message` | OTA progress or error message |

These diagnostic keys are intentionally more implementation-oriented than the identity/config keys above. Nearby tools may display them, but clients should not treat the full diagnostic set or its formatting as a stable contract. BLE sysinfo reports the current detector settings as status; changing them requires MQTT.

Wi-Fi provisioning values are persisted in NVS by ESP-IDF firmware targets that use the shared provisioning service. `SET_WIFI_CONFIG:...` saves the full Wi-Fi block. Credential, BSSID, and channel changes update the station configuration and reconnect Wi-Fi without restarting the BLE transport. A changed `band_policy` is saved but takes effect after restart, because Wi-Fi association and the CSI runtime must start with the same policy. `CLEAR_WIFI` erases provisioned values and disconnects the station without rebooting unless it also restores a different build-default band policy, which likewise takes effect after restart.

MQTT settings are also persisted in NVS as one block. `SET_MQTT_CONFIG:...` replaces the saved MQTT broker settings and reinitializes the active MQTT transport. `CLEAR_MQTT_CONFIG` erases only the saved MQTT broker settings, stops any active MQTT client, and preserves the current device identity. `CLEAR_DEVICE_CONFIG` resets the persisted `device_label` to its build default, clears MQTT settings, stops the active MQTT client, and keeps the firmware-generated `device_id`; a later `SET_DEVICE_CONFIG` command can change only the label.

## Deployment Profiles

ESPectre Protocol can be carried by multiple deployment profiles. The currently implemented profile is the local lab path: [Configure](https://espectre.dev/configure) uses BLE for connectivity setup, and [Monitor](https://espectre.dev/monitor) uses MQTT over WebSockets for live sensing, runtime controls, diagnostics, and BLE recovery. Hosted HTTPS pages should use `wss://`; a local static preview of the website can still use an insecure `ws://` listener.

Web orchestration profiles add identity, tenancy, device claim, state mirrors, history, alerts, and OTA around the same protocol. Those system-level concerns belong to [ARCHITECTURE.md](ARCHITECTURE.md), not to this message schema.

## Web Orchestration Privacy Boundary

Default web-orchestration telemetry should be derived and minimal:

| Field | Purpose |
|-------|---------|
| `device_id` | Service-scoped opaque identifier |
| `timestamp_ms` | Event or sample time |
| `online` | Device availability |
| `firmware_version` | Fleet visibility and update eligibility |
| `frontend` | `esphome`, `matter`, `ble`, `micro`, `custom`, or future frontend label |
| `motion_state` | Motion state |
| `movement_score` | Derived movement metric |
| `threshold` | Current runtime threshold |
| `health` | Minimal optional diagnostics such as uptime, reset reason, or RSSI bucket |

Managed services should not collect by default:

- raw CSI I/Q samples
- SSID, BSSID, access point MAC, or router identifiers
- local IP addresses
- full serial logs
- packet captures
- room photos
- exact physical addresses unless needed for billing or explicitly provided

Movement history can reveal occupancy habits, sleep patterns, and absences from home. Treat it as personal data even when it contains no raw CSI.

## Protocol Improvements

- Evaluate structured BLE command formats such as JSON, TLV, CBOR, or compact binary framing instead of ad hoc ASCII strings, without adding sensing commands to BLE
