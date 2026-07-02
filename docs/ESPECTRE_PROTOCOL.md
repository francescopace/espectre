# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools,
MQTT dashboards, and future managed cloud services.

ESPectre Protocol defines the message model. BLE, MQTT, MQTT over TLS,
device shadows, jobs, and future bridges are transports or profiles that carry
the same semantics across different trust boundaries.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- BLE is for proximity, setup, recovery, and nearby diagnostics.
- MQTT is the operational plane for telemetry, status, commands, dashboards,
  history, and alerts.
- Cloud profiles add identity, credentials, tenancy, retention, and fleet
  management; they do not redefine device telemetry.
- Device identifiers are opaque protocol identifiers, not MAC addresses.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level
  radio traces, and serial logs must not be sent to managed cloud services by
  default.

## Transports

### BLE

BLE is the proximity transport. It is used when a user is near a device or when
network connectivity is not available.

Current BLE responsibilities:

- advertise device availability
- expose protocol and sysinfo notifications
- publish live movement, threshold, and motion-state telemetry to subscribed
  nearby clients
- provision device identity (`device_id`, `device_name`)
- provision Wi-Fi credentials
- provision MQTT endpoint settings
- allow local threshold updates
- recover from broken Wi-Fi or MQTT configuration

Future BLE responsibilities:

- Wi-Fi scan and selection
- Wi-Fi credential validation
- managed-cloud claim bootstrap
- reboot or reconnect commands
- structured command/result framing equivalent to MQTT commands

### MQTT

MQTT is the operational transport once the device has network access.

The same topic shape is valid for a local broker and for a managed broker, with
auth and tenancy added by the deployment profile:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/stats
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
```

Managed cloud MQTT should use TLS and per-device credentials. Local lab MQTT may
use a simpler broker/auth model, but should keep the same message shape.

## Message Families

### Telemetry

Published on:

```text
espectre/v1/devices/{device_id}/telemetry
```

```json
{
  "protocol_version": "1.0",
  "device_id": "espectre-7c2c6742bbac",
  "frontend": "native",
  "timestamp_ms": 123456,
  "motion_state": "idle",
  "movement_score": 0.18,
  "threshold": 5.0,
  "detector": "MVS",
  "health": {
    "uptime_s": 3821,
    "gain_locked": true
  }
}
```

### Status

Published retained on:

```text
espectre/v1/devices/{device_id}/status
```

```json
{
  "protocol_version": "1.0",
  "device_id": "espectre-7c2c6742bbac",
  "online": true,
  "timestamp_ms": 123456
}
```

### Info

Published retained on:

```text
espectre/v1/devices/{device_id}/info
```

```json
{
  "protocol_version": "1.0",
  "device_id": "espectre-7c2c6742bbac",
  "device_name": "Living Room",
  "frontend": "native",
  "firmware_version": "unknown",
  "chip": "esp32c6",
  "network": {
    "ip_address": "192.168.1.28",
    "mac_address": "7C:2C:67:42:BB:AC",
    "channel": {
      "primary": 6
    }
  },
  "detection": {
    "algorithm": "MVS"
  }
}
```

`network` and `detection` are optional. Local tools may display local IP and MAC
values. Managed cloud services should not collect local IP addresses, SSIDs,
BSSIDs, access point MACs, or router identifiers by default.

### Stats

Published on request or at low rate by clients that expose runtime diagnostics:

```json
{
  "protocol_version": "1.0",
  "device_id": "espectre-7c2c6742bbac",
  "timestamp_ms": 123456,
  "uptime": 3821,
  "free_memory_kb": 182.4,
  "loop_time_ms": 0.31
}
```

Stats are diagnostic. Product dashboards should prefer telemetry/status/info for
normal operation. When available, `free_memory_kb` reports current free heap and
`loop_time_ms` reports the measured last loop-body cost in milliseconds,
excluding the outer task sleep or idle delay. Motion state, movement score,
threshold, detector selection, turbulence, and gain-lock state belong to
telemetry or retained config/info surfaces instead of `stats`.

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
  "threshold": 4.5
}
```

Command result:

```json
{
  "protocol_version": "1.0",
  "device_id": "espectre-7c2c6742bbac",
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
SET_THRESHOLD:4.5
SET_DEVICE_CONFIG:device_name=Living Room
SET_DEVICE_CONFIG:mqtt_host=192.168.1.20
SET_DEVICE_CONFIG:mqtt_port=1883
SET_DEVICE_CONFIG:mqtt_username=mqtt
SET_DEVICE_CONFIG:mqtt_password=secret-password
SET_DEVICE_CONFIG:topic_prefix=espectre/v1/devices
SET_DEVICE_CONFIG:mqtt_enabled=true
CLEAR_MQTT_CONFIG
CLEAR_DEVICE_CONFIG
SET_WIFI_SSID:Lab Network
SET_WIFI_PASSWORD:secret-password
SET_WIFI_CHANNEL:6
SET_WIFI_BSSID:aa:bb:cc:dd:ee:ff
APPLY_WIFI
CLEAR_WIFI
```

This is the current BLE framing, not a separate protocol. A future BLE framing
can become more structured while preserving the same ESPectre Protocol command
families and semantics.

Identity/config semantics for the current BLE control surface:

- `device_id` is the firmware-generated protocol identity used by BLE sysinfo, MQTT topics, and MQTT payloads
- `device_name` is the user-facing human-readable device name
- the standalone native frontend exposes a separate pairing/display name
  (`ble_device_name`) for nearby clients, derived from `device_name`
- `CLEAR_MQTT_CONFIG` clears only broker-related MQTT settings and disables the
  active MQTT transport
- `CLEAR_DEVICE_CONFIG` resets device-facing naming and MQTT settings while keeping the firmware-generated `device_id`
- `CLEAR_WIFI` clears only persisted Wi-Fi station settings

## Current BLE Telemetry Surface

The standalone native frontend currently exposes two data paths:

- a binary low-latency telemetry characteristic for interactive clients
- a line-based `sysinfo` characteristic for configuration and diagnostics

Telemetry delivery is subscription-driven:

- clients opt in by enabling notifications on the telemetry characteristic
- clients may stop notifications when only provisioning or diagnostics are needed
- when no client is subscribed, the standalone native frontend disables its live
  telemetry callback instead of continuing to generate BLE-only live telemetry
- `sysinfo` and BLE control writes remain available even when live telemetry is
  not subscribed

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

Compatibility notes:

- the first 8 bytes remain the stable legacy surface for nearby clients
- clients that only read `movement` and `threshold` continue to work unchanged
- clients may ignore trailing bytes they do not understand
- clients that want live movement updates must explicitly subscribe to the
  telemetry characteristic

Sysinfo framing remains:

```text
key=value
...
END
```

`sysinfo` is intended for readable configuration and diagnostics, not for the
highest-rate live state transport. Runtime `motion_state` is therefore carried
in telemetry, not as a `sysinfo` key.

Current BLE `sysinfo` identity/config keys include:

| Key | Meaning |
|-----|---------|
| `device_id` | Current firmware-generated protocol device identifier |
| `device_name` | Current human-readable device name |
| `ble_device_name` | Current BLE pairing/display name derived from `device_name` |
| `mqtt_enabled` | Whether MQTT transport is currently enabled |
| `mqtt_host` | Current MQTT broker host |
| `mqtt_port` | Current MQTT broker port |
| `mqtt_username` | Current MQTT username |
| `topic_prefix` | Current MQTT topic prefix |
| `wifi_saved` | Whether Wi-Fi settings are persisted |
| `wifi_ssid` | Current persisted Wi-Fi SSID |
| `wifi_bssid` | Current persisted Wi-Fi BSSID lock |
| `wifi_channel` | Current persisted Wi-Fi channel lock |

Capability-oriented `sysinfo` keys may include:

| Key | Meaning |
|-----|---------|
| `frontend` | Firmware/frontend family currently exposing the BLE service |
| `supports_wifi_provisioning` | Whether BLE clients can edit and apply Wi-Fi settings |
| `supports_mqtt_config` | Whether BLE clients can edit MQTT broker settings |
| `supports_device_config` | Whether BLE clients can edit device identity settings |
| `supports_runtime_threshold` | Whether BLE clients can change the live motion threshold |
| `supports_live_telemetry` | Whether BLE telemetry notifications are exposed |
| `supports_extended_diagnostics` | Whether implementation-specific runtime diagnostics are exposed |

Current BLE `sysinfo` diagnostic keys may include:

| Key | Meaning |
|-----|---------|
| `chip` | Target chip reported by the firmware, such as `esp32c3` |
| `detector` | Active detector name |
| `gain_locked` | Whether CSI gain lock is currently active |
| `window` | Detection window size in packets |
| `lowpass` | Whether the low-pass stage is enabled |
| `lowpass_cutoff` | Low-pass cutoff in Hz |
| `hampel` | Whether the Hampel filter is enabled |
| `hampel_window` | Hampel window size |
| `hampel_threshold` | Hampel threshold in MAD units |
| `traffic_mode` | Internal traffic generator mode such as `ping` or `dns` |
| `traffic_rate` | Internal traffic generator target rate in packets per second |
| `publish_interval` | Periodic publish cadence in packets |
| `evaluation_interval` | Detector evaluation cadence in packets |
| `motion_hits` | Motion-on/off consecutive hit thresholds |
| `best_pxx` | Current detector-specific diagnostic metric |

These diagnostic keys are intentionally more implementation-oriented than the
identity/config keys above. Nearby tools may display them, but clients should
not treat the full diagnostic set or its formatting as a hard compatibility
contract.

Wi-Fi provisioning values are persisted in NVS by ESP-IDF firmware targets that
use the shared provisioning service. `APPLY_WIFI` saves the current values,
updates the station configuration, and reconnects Wi-Fi without restarting the
BLE transport. `CLEAR_WIFI` erases provisioned values and disconnects the
station without rebooting. The standalone BLE firmware uses the same surface for
its full runtime frontend, while the streamer firmware exposes Wi-Fi
provisioning, device naming, and a reduced sysinfo subset.

MQTT settings are also persisted in NVS after each
`SET_DEVICE_CONFIG:key=value` command. `CLEAR_MQTT_CONFIG` erases only the saved
MQTT broker settings, stops any active MQTT client, and preserves the current
device identity. `CLEAR_DEVICE_CONFIG` resets the persisted `device_name` and
MQTT settings and returns the live BLE session to the generated/default device
identity state until it is reprovisioned or rebooted.

## Deployment Profiles

ESPectre Protocol can be carried by multiple deployment profiles. The currently
implemented profile is the local lab path: BLE provisioning and diagnostics via
`tools/web/espectre-ble.html`, plus MQTT telemetry inspection via
`tools/web/espectre-mqtt.html`.

The managed cloud profile adds identity, tenancy, device claim, state mirrors,
history, alerts, and OTA around the same protocol. Those system-level concerns
belong to [ARCHITECTURE.md](ARCHITECTURE.md), not to this message schema.

## Cloud Privacy Boundary

Default managed-cloud telemetry should be derived and minimal:

| Field | Purpose |
|-------|---------|
| `device_id` | Cloud-scoped opaque identifier |
| `timestamp_ms` | Event or sample time |
| `online` | Device availability |
| `firmware_version` | Fleet visibility and update eligibility |
| `frontend` | `esphome`, `matter`, `ble`, `micro`, `custom`, or future frontend label |
| `motion_state` | Motion state |
| `movement_score` | Derived movement metric |
| `threshold` | Current runtime threshold |
| `health` | Minimal optional diagnostics such as uptime, reset reason, or RSSI bucket |

The managed cloud service should not collect by default:

- raw CSI I/Q samples
- SSID, BSSID, access point MAC, or router identifiers
- local IP addresses
- full serial logs
- packet captures
- room photos
- exact physical addresses unless needed for billing or explicitly provided

Movement history can reveal occupancy habits, sleep patterns, and absences from
home. Treat it as personal data even when it contains no raw CSI.

## Future Protocol Work

- Should the first self-hosted local lab use a lightweight broker plus SQLite,
  or start with a more cloud-shaped service stack?
- Should firmware cloud connectivity be added to all frontends or introduced as
  a dedicated cloud-capable firmware profile first?
- Should Matter/ESPHome nodes connect directly to cloud, or should a local
  bridge/gateway mode be supported later?
- Should BLE commands remain ASCII-oriented or move to structured framing now
  that the logical protocol is documented?
- How much of ESPectre Protocol should be frozen before the first paid beta?

## Related Docs

- [ROADMAP.md](ROADMAP.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [SETUP.md](SETUP.md)
- [MICRO_ESPECTRE.md](MICRO_ESPECTRE.md)
