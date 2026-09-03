# ESPectre API

This document owns the public ESPectre application contract carried by Direct HTTP, Server-Sent Events (SSE), and MQTT. Discovery is specified in [DISCOVERY.md](DISCOVERY.md).

## Contract principles

- The application version is `1.0`, and the Direct base path is `/espectre/v1`.
- HTTP and MQTT share resource payloads, operation names, validation, and result codes. Transport framing and delivery policy are not part of a resource payload.
- `protocol_version` appears only in `capabilities` and discovery metadata.
- `device_id` appears only in `device`, multi-device discovery results, and the established binary CSI records.
- This is a clean replacement for the former RPC and explicit CSI-session APIs. There are no legacy routes, commands, topics, or aliases.

## Direct HTTP

Direct HTTP listens on TCP port `62587`. Successful resource reads return the resource object directly. Successful synchronous mutations return HTTP `200`; accepted asynchronous or disruptive operations return HTTP `202`. Mutations return:

```json
{"accepted":true,"code":"ok","message":"operation accepted","data":{}}
```

`data` is optional. A rejected operation sets `accepted` to `false` and uses the status mapping in [Errors](#errors).

| Method | Resource | Support |
| --- | --- | --- |
| `GET` | `/health`, `/device`, `/capabilities`, `/sensing`, `/wifi`, `/wifi/access-points`, `/diagnostics` | C++ frontends and Micro |
| `GET` | `/devices` | Native, ESPHome, and Matter |
| `GET` | `/mqtt`, `/ota` | Native only |
| `PATCH` | `/device`, `/sensing` | Native, ESPHome, and Matter |
| `PATCH` | `/mqtt` | Native only |
| `POST` | `/sensing/calibrations`, `/wifi/scans` | C++ frontends; Micro supports recalibration only |
| `POST` | `/ota/checks`, `/ota/updates` | Native only |
| `PUT` | `/wifi/bssid` | Native, ESPHome, and Matter |
| `DELETE` | `/wifi/bssid`, `/wifi/credentials` | Native, ESPHome, and Matter |
| `DELETE` | `/mqtt` | Native only |
| `GET` | `/events` | All frontends |
| `GET` | `/csi` | Native, ESPHome, and Matter |

Unsupported resources or method combinations return `404`. BSSID changes and credential removal send their `202` response before the station is disconnected.

`PATCH /sensing` accepts any non-empty supported subset of `enabled`, `detector`, `threshold`, `motion_on_hits`, `motion_off_hits`, `csi_traffic_mode`, and `traffic_generator_mode`. The whole request is validated before changes are applied. `motion_on_hits` and `motion_off_hits` must be supplied together. Calibration remains the separate `POST /sensing/calibrations` action.

`POST /wifi/scans` starts a scan. `GET /wifi/access-points` returns `scanning`, `message`, and the most recent `access_points` array.

## Resources

### `health`

```json
{"status":"ok","online":true,"uptime_s":42,"timestamp_ms":42000}
```

Calibration and intentional CSI collection do not make health degraded.

### `device`

`device` contains `device_id`, the effective `name`, the configured `label`, `firmware`, `chip`, `frontend`, and `csi_profile`. Empty labels are represented as an empty string; `name` then uses the generated stable display name.

### `capabilities`

`capabilities` contains `protocol_version`, `resources`, `operations`, `events`, and `features`. The raw-stream feature is named `csi`. When supported, the `csi` object publishes the binary format and traffic-source parameters.

### `sensing`

`sensing` contains desired `enabled`, `ready`, `calibrating`, `mode`, `derived_events_paused`, `detector`, `threshold`, motion-hit tuning, and traffic-source tuning. During raw collection, `mode` is `csi_collection`, `ready` is false, and `derived_events_paused` is true.

### `wifi`

The Direct resource contains configuration and connection state, SSID, BSSID, IP address, band, channel, RSSI, and staged-apply state. Its MQTT representation omits SSID, BSSID, IP address, and MAC address.

### `mqtt`, `ota`, and `diagnostics`

`mqtt` exposes non-secret broker configuration state and never returns passwords. `ota` exposes the current OTA state. `diagnostics` is an on-demand, non-retained diagnostic snapshot; it is not streamed through `/events` or a dedicated MQTT topic.

## Events

`GET /espectre/v1/events` is the only JSON SSE connection. Event names are `health`, `device`, `sensing`, `wifi`, `ota`, `motion`, and `fault`, filtered by the frontend capability catalog. Resource events carry complete resource snapshots. MQTT configuration, diagnostics, discovery results, and CSI do not appear in this stream.

A `motion` event is emitted for each detector evaluation:

```json
{"timestamp_ms":42000,"state":"idle","score":0.0123}
```

Threshold and detector metadata remain in `sensing`. Future derived events, such as presence and gesture, use this same event plane.

## CSI collection

`GET /espectre/v1/csi` opens the single exclusive binary CSI collection session. No setup request, bearer token, session deletion, or bind timeout exists. Closing the TCP response ends collection.

Each CSI V8 record retains the existing 60-byte HTTP prefix. The client adopts the 16-byte session identifier from the first frame and rejects a change within the same connection. The producer preserves order; fixed-ring drops remain observable in the transport counters.

While CSI is active, sensing reports `csi_collection`, readiness is false, and motion plus all present or future derived events are paused on every transport. Control and resource events remain available. A second `/csi` request and sensing, Wi-Fi, or OTA mutations return `409`. On close, the runtime restores its prior state, recalibrates when required, and resumes derived events only after readiness returns. When external traffic is configured, the host traffic generator must start before opening `/csi`.

## MQTT

The base topic is `espectre/v1/devices/{device_id}`.

| Suffix | Retained | Purpose |
| --- | --- | --- |
| `health` | yes | Health, availability, and Last Will |
| `device` | yes | Device identity and build |
| `capabilities` | yes | Negotiation and supported surface |
| `sensing` | yes | Sensing state and tuning |
| `wifi` | yes | Redacted radio state |
| `ota` | yes | OTA state |
| `motion` | no | Per-evaluation motion event |
| `fault` | no | Runtime fault |
| `commands/result` | no | Correlated command result |

There is no extra application heartbeat. The retained `health` topic is also the Home Assistant availability topic, and the broker publishes the retained offline health Last Will after an ungraceful disconnect.

Requests are published to `commands/request`. Every request has a top-level `command_id` and `command`; parameters are also top-level. Neither `protocol_version` nor `device_id` is present. Supported MQTT commands are `update_device`, `update_sensing`, `recalibrate`, `read_diagnostics`, `check_ota`, and `start_ota`. Results echo `command_id` and `command`, then carry `accepted`, `code`, `message`, and optional `data`.

There are no MQTT topics for MQTT configuration, diagnostics, discovery, or CSI. `read_diagnostics` returns the snapshot in its correlated command result. Home Assistant Discovery remains a separate adapter profile and uses `health` for availability.

## Errors

| HTTP status | Stable result codes | Meaning |
| --- | --- | --- |
| `400` | `invalid_params` | Invalid JSON, fields, or values |
| `403` | `forbidden` | Origin or transport policy rejected the request |
| `404` | `unsupported` | Resource, operation, or method is not supported |
| `409` | `busy`, `conflict`, `busy_raw_collection` | Exclusive or disruptive operation conflicts with active work |
| `413` | `invalid_params` | Request exceeds the bounded frame size |
| `415` | `invalid_params` | JSON content type is required |
| `429` | `rate_limited` | Request or mutation rate limit reached |
| `503` | `unavailable`, `internal_error` | Resource, queue, or service is unavailable |

## Security and versioning

Direct HTTP is a trusted-LAN surface. Firmware enforces exact browser Origin allowlists, Private Network Access preflight, bounded bodies, queues, clients, and request rates. It binds to the station interface and does not expose stored Wi-Fi or MQTT passwords. `/mqtt` can be protected independently in a future additive security extension.

Clients negotiate once through `capabilities.protocol_version`. Compatible additions may add resources, fields, operations, or events within `v1`; consumers must ignore unknown additive fields. An incompatible contract requires a new base-path major version and discovery protocol version.
