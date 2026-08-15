# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools, MQTT dashboards, and future web orchestration services.

ESPectre Protocol defines the message model. BLE, MQTT, MQTT over TLS, device shadows, jobs, and future bridges are transports or profiles that carry the same semantics across different trust boundaries.

This is an implementation reference for firmware, client, and integration developers. Read [Principles](#principles) and [Message Families](#message-families) when implementing a consumer; read the transport sections only for the connection mechanism you use. A **transport** carries messages, a **profile** adds deployment rules without changing their meaning, and a **retained** MQTT message is stored by the broker for future subscribers.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- BLE is for proximity, setup, recovery, and nearby diagnostics.
- MQTT is the operational plane for telemetry, status, commands, dashboards, history, and alerts.
- Web orchestration profiles add identity, credentials, tenancy, retention, and fleet management; they do not redefine device telemetry.
- `device_id` is a logical protocol identifier. Current firmware derives it from the station MAC, so it must be treated as a persistent hardware identifier rather than anonymous data.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level radio traces, and serial logs must not be sent to managed services by default.

## Transports

### BLE

BLE is the proximity transport. It is used when a user is near a device or when network connectivity is not available.

Current BLE responsibilities:

- advertise device availability
- expose protocol and sysinfo notifications
- publish live movement, threshold, and motion-state telemetry to subscribed nearby clients
- expose the firmware-generated, read-only `device_id` and provision the mutable `device_label`
- provision Wi-Fi credentials
- provision MQTT endpoint settings
- allow local threshold updates
- allow local motion-hit debounce updates
- trigger OTA status, checks, and updates through the shared HTTPS OTA service
- recover from broken Wi-Fi or MQTT configuration

Future BLE responsibilities:

- Wi-Fi scan and selection
- Wi-Fi credential validation
- web-service claim bootstrap
- reboot or reconnect commands
- structured command/result framing equivalent to MQTT commands

### MQTT

MQTT is the operational transport once the device has network access.

The same topic shape is valid for a local broker and for a managed broker, with auth and tenancy added by the deployment profile:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/stats
espectre/v1/devices/{device_id}/ota/state
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
```

Managed-service MQTT should use TLS and per-device credentials. Local lab MQTT may use a simpler broker/auth model, but should keep the same message shape.

### Home Assistant MQTT Adapter Profile

Native and Micro-ESPectre can publish an additive Home Assistant MQTT Discovery surface without changing the canonical ESPectre topics above. Discovery payloads use the standard `{discovery_prefix}/{component}/{object_id}/config` topic shape. Native also retains its canonical `status` payload so late subscribers receive the current availability; entity-shaped state topics remain non-retained under `espectre/v1/devices/{device_id}/ha/...`.

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
  "device_id": "0x00007c2c6742bbac",
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

Native MQTT telemetry uses a hybrid cadence. Filtered motion-state transitions are published immediately once `ready_to_publish` is true, while updates at the configured `publish_interval_ms` remain as a monotonic-clock heartbeat and current-metrics snapshot. Edge publishes occur only on state transitions, not on every detector evaluation, and heartbeat deadlines never force detector evaluation. Native BLE live telemetry remains opt-in and low-latency for nearby interactive clients.

### Status

Published on:

```text
espectre/v1/devices/{device_id}/status
```

```json
{
  "protocol_version": "1.0",
  "device_id": "0x00007c2c6742bbac",
  "online": true,
  "timestamp_ms": 123456
}
```

Native retains the latest status payload. A normal shutdown publishes retained `online: false`; after an unexpected disconnect, the broker publishes the retained Last Will with the same offline state. A later connection replaces it with retained `online: true`, allowing availability consumers that subscribe after discovery to recover the current state.

### Info

Published on:

```text
espectre/v1/devices/{device_id}/info
```

```json
{
  "protocol_version": "1.0",
  "device_id": "0x00007c2c6742bbac",
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
  "supports_ota": true,
  "network": {
    "ip_address": "192.168.1.28",
    "mac_address": "7C:2C:67:42:BB:AC",
    "channel": {
      "primary": 6
    }
  },
  "detection": {
    "algorithm": "lightweight"
  }
}
```

The `supports_*` fields are authoritative capability declarations for clients. Clients should not infer command support from `frontend`, telemetry fields, or other payload content. `network` and `detection` are optional. Local tools may display local IP and MAC values. Managed services should not collect local IP addresses, SSIDs, BSSIDs, access point MACs, or router identifiers by default.

### Stats

Published by Native only in response to an explicit `stats` command. Other clients may expose the same schema on request or at a low rate:

```json
{
  "protocol_version": "1.0",
  "device_id": "0x00007c2c6742bbac",
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

Native always includes the CSI and Wi-Fi fields in a requested `stats` response. It derives rates from the cumulative counters whenever the existing periodic sensing update runs, caches that completed sample, and does not add a diagnostic timer or publish it periodically. `traffic_tx_pps` is the traffic-generator transmit rate; `csi_callback_pps` is the raw CSI callback rate; `csi_accepted_pps` is the identity-accepted rate used by adaptive traffic control; `csi_admitted_pps` is the detector input rate after temporal admission; `csi_filtered_pps` is the capture-filter drop rate; the temporal drop fields distinguish missing slots, same-slot excess, stale packets, and out-of-order packets; and `csi_occupancy` is the valid fraction of the active detector window. The extra CSI fields are additive on protocol `1.0`; consumers may ignore unknown keys. The SDK sample uses `csi_occupancy_ratio` for the same occupancy value. Before the first periodic sensing update completes, rate fields are zero.

ESPHome exposes the same cached measurements as diagnostic entities. Their states are published only when the `Refresh Diagnostics` button is pressed. These on-demand diagnostics are independent of the optional runtime debug logs.

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

Request an OTA manifest check using the firmware's built-in release URL:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-ota-check",
  "command": "ota_check"
}
```

Start OTA using the built-in manifest:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-ota-start",
  "command": "ota_start"
}
```

Native firmware embeds a per-chip GitHub Releases manifest URL. OTA commands do not accept server, manifest, image, or version parameters; payloads containing those overrides are rejected. Stable firmware is pinned to the latest release channel, and snapshot firmware is pinned to the rolling snapshot release. Frontends advertise support through `supports_ota`; Micro-ESPectre does not implement OTA commands.

Publish OTA state on:

```text
espectre/v1/devices/{device_id}/ota/state
```

```json
{
  "protocol_version": "1.0",
  "device_id": "0x00007c2c6742bbac",
  "state": "update_available",
  "timestamp_ms": 123456,
  "busy": false,
  "update_available": true,
  "current_version": "1.2.2",
  "target_version": "1.2.3",
  "manifest_url": "https://github.com/francescopace/espectre/releases/latest/download/espectre-native-ota-esp32c6.json",
  "image_url": "https://github.com/francescopace/espectre/releases/download/1.2.3/espectre-native-1.2.3-esp32c6-ota.bin",
  "message": "update available"
}
```

Command result:

```json
{
  "protocol_version": "1.0",
  "device_id": "0x00007c2c6742bbac",
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
SET_THRESHOLD:0.35
SET_MOTION_HITS:on=4&off=3
SET_DETECTOR:high_accuracy
OTA_STATUS
OTA_CHECK
OTA_START
SET_DEVICE_CONFIG:device_label=Living Room
SET_MQTT_CONFIG:host=192.168.1.20&port=1883&username=mqtt&password=secret-password&topic_prefix=espectre%2Fv1%2Fdevices
CLEAR_MQTT_CONFIG
CLEAR_DEVICE_CONFIG
SET_WIFI_CONFIG:ssid=Lab%20Network&password=secret-password&channel=6&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&band_policy=2g
CLEAR_WIFI
```

This is the current BLE framing, not a separate protocol. A future BLE framing can become more structured while preserving the same ESPectre Protocol command families and semantics.

Identity/config semantics for the current BLE control surface:

- `device_id` is the firmware-generated station-MAC-packed identity rendered as a stable `0x...` hex string in BLE sysinfo, MQTT topics, and MQTT payloads; managed or privacy-sensitive profiles must pseudonymize or replace it before exposing it outside the local trust boundary
- `device_name` is the immutable protocol/device name derived from chip and `device_id`
- `device_label` is the optional user-facing human-readable device label
- `SET_MQTT_CONFIG:...` replaces the full persisted MQTT broker block in one write
- `CLEAR_MQTT_CONFIG` clears only broker-related MQTT settings and disables the active MQTT transport
- `CLEAR_DEVICE_CONFIG` resets `device_label` to its build default, clears MQTT settings, disables the active MQTT transport, and keeps the firmware-generated `device_id`
- `SET_WIFI_CONFIG:...` replaces the full persisted Wi-Fi station block in one write; credentials, BSSID, and channel changes apply immediately, while a changed `band_policy` applies after restart so the Wi-Fi and CSI runtimes restart together
- `band_policy` accepts `2g`, `5g`, or `auto`; firmware rejects `5g` and `auto` unless the target reports `supports_wifi_5ghz=true`
- `CLEAR_WIFI` clears only persisted Wi-Fi station settings

## Current BLE Telemetry Surface

The standalone native frontend currently exposes two data paths:

- a binary low-latency telemetry characteristic for interactive clients
- a line-based `sysinfo` characteristic for configuration and diagnostics

Telemetry delivery is subscription-driven:

- clients opt in by enabling notifications on the telemetry characteristic
- clients may stop notifications when only provisioning or diagnostics are needed
- when no client is subscribed, the standalone native frontend disables its live telemetry callback instead of continuing to generate BLE-only live telemetry
- `sysinfo` and BLE control writes remain available even when live telemetry is not subscribed

Current telemetry payload:

```text
[float32 movement][float32 threshold][uint8 motion_state?]
```

Field semantics:

| Field | Type | Description |
|-------|------|-------------|
| `movement` | `float32` | Current movement metric |
| `threshold` | `float32` | Current runtime threshold |
| `motion_state` | `uint8` | Optional trailing state byte: `0 = idle`, `1 = motion` |

Receiver notes:

- the first 8 bytes carry the fixed header fields required by the current protocol
- receivers may ignore trailing bytes they do not understand
- clients that want live movement updates must explicitly subscribe to the telemetry characteristic

Sysinfo framing remains:

```text
key=value
...
END
```

`sysinfo` is intended for readable configuration and diagnostics, not for the highest-rate live state transport. Runtime `motion_state` is therefore carried in telemetry, not as a `sysinfo` key.

Current BLE `sysinfo` identity/config keys include:

| Key | Meaning |
|-----|---------|
| `device_id` | Current firmware-generated device identifier in canonical `0x...` hex form |
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
| `supports_device_config` | Whether BLE clients can edit device identity settings |
| `supports_runtime_threshold` | Whether BLE clients can change the live motion threshold |
| `supports_runtime_motion_hits` | Whether BLE clients can change the persisted motion-on/off hit thresholds |
| `supports_runtime_detector` | Whether BLE clients can select and persist `lightweight` or `high_accuracy` |
| `supports_live_telemetry` | Whether BLE telemetry notifications are exposed |
| `supports_extended_diagnostics` | Whether implementation-specific runtime diagnostics are exposed |
| `supports_ota` | Whether BLE clients can expose OTA-related controls |
| `supports_wifi_5ghz` | Whether the target radio can accept the `5g` and `auto` Wi-Fi band policies |

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
| `traffic_mode` | Internal traffic generator mode such as `ping` or `dns` |
| `traffic_rate` | Internal traffic generator target rate in packets per second |
| `traffic_adaptive` | Whether adaptive traffic-rate control is enabled |
| `publish_interval_ms` | Periodic publish cadence in milliseconds |
| `evaluation_interval_ms` | Detector evaluation cadence in milliseconds |
| `motion_hits` | Motion-on/off consecutive hit thresholds |
| `ota_state` | Current OTA state reported by the shared HTTPS OTA service |
| `ota_busy` | Whether an OTA worker is active |
| `ota_update_available` | Whether the last OTA check found an update |
| `ota_current_version` | Firmware version compared against the manifest |
| `ota_target_version` | Version reported by the pending OTA target, when known |
| `ota_message` | OTA progress or error message |

These diagnostic keys are intentionally more implementation-oriented than the identity/config keys above. Nearby tools may display them, but clients should not treat the full diagnostic set or its formatting as a stable contract.

Wi-Fi provisioning values are persisted in NVS by ESP-IDF firmware targets that use the shared provisioning service. `SET_WIFI_CONFIG:...` saves the full Wi-Fi block. Credential, BSSID, and channel changes update the station configuration and reconnect Wi-Fi without restarting the BLE transport. A changed `band_policy` is saved but takes effect after restart, because Wi-Fi association and the CSI runtime must start with the same policy. `CLEAR_WIFI` erases provisioned values and disconnects the station without rebooting unless it also restores a different build-default band policy, which likewise takes effect after restart. The standalone BLE firmware uses this surface for its full runtime frontend.

MQTT settings are also persisted in NVS as one block. `SET_MQTT_CONFIG:...` replaces the saved MQTT broker settings and reinitializes the active MQTT transport. `CLEAR_MQTT_CONFIG` erases only the saved MQTT broker settings, stops any active MQTT client, and preserves the current device identity. `CLEAR_DEVICE_CONFIG` resets the persisted `device_label` to its build default, clears MQTT settings, stops the active MQTT client, and keeps the firmware-generated `device_id`; a later `SET_DEVICE_CONFIG` command can change only the label.

## Deployment Profiles

ESPectre Protocol can be carried by multiple deployment profiles. The currently implemented profile is the local lab path: BLE provisioning and diagnostics via [Configure](https://espectre.dev/configure/), plus telemetry inspection through the [MQTT Monitor](https://espectre.dev/monitor/). The same pages are served from localhost by `./espectre ui` when an insecure local `ws://` listener cannot be reached reliably from the public HTTPS site.

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

- Evaluate structured BLE command formats such as JSON, TLV, CBOR, or compact binary framing instead of ad hoc ASCII strings
