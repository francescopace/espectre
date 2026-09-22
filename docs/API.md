# ESPectre API

This reference defines the ESPectre messages and operations used over Direct HTTP, Server-Sent Events (SSE), and MQTT. Discovery is specified in the [discovery reference](DISCOVERY.md).

## Contract principles

- The protocol version is `1.0`, and the Direct base path is `/espectre/v1`.
- HTTP and MQTT use the same payloads, operation names, validation, and result codes. Only framing and delivery differ.
- `protocol_version` appears only in `capabilities` and discovery data.
- `device_id` appears only in `device`, in discovery results, and in binary CSI records.
- There are no legacy routes, commands, topics, or aliases.

## Direct HTTP

Direct HTTP listens on TCP port `62587`. Clients start with `GET /espectre/v1/capabilities` and use the `resources` and `operations` it returns, rather than guessing from the frontend name.

### Support matrix

| Method | Path | Native | ESPHome | Matter | Micro |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/health`, `/device`, `/capabilities`, `/sensing`, `/wifi`, `/diagnostics`, `/events` | yes | yes | yes | yes |
| `GET` | `/wifi/access-points`, `/devices`, `/csi` | yes | yes | yes | no |
| `GET` | `/mqtt`, `/ota` | yes | no | no | no |
| `PATCH` | `/device`, `/sensing` | yes | yes | yes | no |
| `PATCH` | `/mqtt` | yes | no | no | no |
| `POST` | `/sensing/calibrations` | yes | yes | yes | yes |
| `POST` | `/wifi/scans` | yes | yes | yes | no |
| `POST` | `/ota/checks`, `/ota/updates` | yes | no | no | no |
| `PUT` | `/wifi/bssid` | yes | yes | yes | no |
| `DELETE` | `/wifi/bssid` | yes | yes | yes | no |
| `DELETE` | `/wifi/credentials`, `/mqtt` | yes | no | no | no |

Unsupported combinations return HTTP `404`. A custom build can disable features listed here, so always check `capabilities`. Micro-ESPectre limits are in its [Direct HTTP surface](../src/python/micro_espectre/README.md#direct-http-surface) section.

### Request and response framing

- A successful `GET` returns the resource object itself.
- A request body must be a JSON object; an empty body means `{}`. Send `Content-Type: application/json` with any non-empty body.
- C++ frontends accept at most 2,048 request bytes.

Mutations return a result object:

```json
{"accepted":true,"code":"ok","message":"operation accepted","data":{}}
```

`data` is optional. Immediate changes return HTTP `200`. Changes that run in the background or disconnect the device return `202` once accepted. Errors use the codes in [Errors](#errors).

## Resources

### `health`

```json
{"status":"ok","online":true,"uptime_s":42,"timestamp_ms":42000}
```

`status` is `ok` while the device is online, and `offline` in the MQTT Last Will. Times use the device's monotonic clock. Calibration and CSI collection do not affect health.

### `device`

```json
{
  "device_id": "3cf79180d3a0aca4",
  "name": "Kitchen",
  "label": "Kitchen",
  "frontend": "native",
  "firmware": "3.0.0-rc1",
  "chip": "esp32c6",
  "csi_profile": "ht20"
}
```

- `device_id`: stable identity, 16 lowercase hexadecimal characters.
- `label`: the name set by the user; may be empty. `name` is the label, or a generated name when the label is empty.
- `csi_profile`: the capture profile in use (`lltf20`, `ht20`, or `vht20`), when known. Read-only.

### `capabilities`

`capabilities` contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `protocol_version` | string | Application contract version, currently `1.0` |
| `resources` | array of strings | Resource names accepted by `GET`, without the base path |
| `operations` | array of objects | Supported operations as `{name, method, path}` |
| `events` | array of strings | Event names published by this frontend |
| `features.csi` | boolean | Whether `GET /csi` is available |
| `csi` | object, optional | CSI endpoint, framing versions, queue parameters, and external traffic settings |

Clients must ignore resources, operations, events, and fields they do not know.

### `sensing`

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | boolean | Desired sensing state |
| `ready` | boolean | Detector output may be consumed: calibration is complete, the detector window is valid, and admitted input is newer than one detector window |
| `calibrating` | boolean | Calibration is active |
| `mode` | string | `sensing` or `csi_collection` |
| `derived_events_paused` | boolean | Motion and other derived events are suspended |
| `detector` | string | `lightweight` or `high_accuracy` |
| `threshold` | number | Current detector threshold in `[0.0, 1.0]` |
| `motion_on_hits`, `motion_off_hits` | integer | Consecutive evaluations required for each state transition |
| `csi_traffic_mode` | string | `internal` or `external` |
| `traffic_generator_mode` | string | `ping`, `dns`, `dns_tcp`, or `wifi_raw` (experimental; see [compatibility limits](CSI.md#compatibility-limits)) |
| `csi_target_pps` | integer | Configured CSI traffic target in packets per second |
| `csi_traffic_udp_port` | integer, optional | External CSI traffic UDP port |
| `csi_traffic_multicast_group` | string, optional | External CSI traffic multicast group |

During CSI collection, `mode` is `csi_collection`, `ready` is false, and `derived_events_paused` is true.

### `wifi`

The Direct resource contains `configured`, `connected`, `ssid`, `bssid`, `band`, `channel`, and `rssi_dbm`. `band` is `2g`, `5g`, or empty when unknown; unknown RSSI is `null`. Native also returns `ip`, `apply_state`, and `apply_message`. The MAC address and credentials are never returned.

The MQTT `wifi` payload leaves out SSID, BSSID, and IP address.

### `wifi/access-points`

```json
{
  "scanning": false,
  "message": "scan complete",
  "access_points": [
    {"bssid":"aa:bb:cc:dd:ee:ff","rssi_dbm":-51,"channel":10}
  ]
}
```

`access_points` holds the last completed scan. While a new scan runs, `scanning` is true; poll this resource until it finishes.

### `mqtt`

Native returns `configured`, `scheme`, `host`, `port`, `username_configured`, and `topic_prefix`. `configured` is true only when scheme, host, and port are valid. The username and password are never returned.

### `ota`

OTA is available only on Native. It is a frontend extension: the sensing SDK does not update firmware.

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | string | `idle`, `checking`, `update_available`, `up_to_date`, `downloading`, `applying`, `reboot_scheduled`, `error`, or `unknown` |
| `timestamp_ms` | integer | Monotonic snapshot timestamp |
| `busy` | boolean | An OTA check or update is running |
| `update_available` | boolean | The latest completed check found a newer image |
| `current_version`, `target_version` | string | Running and candidate firmware versions |
| `manifest_url`, `image_url` | string | Resolved release artifacts; empty before resolution |
| `default_channel`, `channel` | string | Build default and channel used by the current or latest attempt |
| `message` | string | Current status or failure detail |

### `diagnostics`

Diagnostics are read on request only; they are never pushed over SSE or a retained MQTT topic.

**Catalog.** `GET /diagnostics` with no `fields`, or `fields: []`, returns the list of fields this frontend supports: `{"fields":[{"name":"traffic_tx_pps","type":"number","unit":"pps"}, ...]}`. The list comes from `src/cpp/runtime/diagnostic_fields.h`.

**Values.** Pass the fields you want:

```json
{"fields":["traffic_tx_pps","csi_hw_error_total","direct_http.send_failures"]}
```

- The response contains the selected fields plus `timestamp_ms` and `uptime`.
- A dotted path selects one nested value; a group name such as `raw_csi` selects the whole group. Duplicates are returned once.
- `["*"]` returns everything. `*` cannot be combined with other names.
- Unknown names, names outside the catalog, and wrong types return `invalid_params`.
- A supported field that has no value right now returns `null`.

**Transport.** Over HTTP, send the object as the GET body. Browsers can use a query instead: `GET /espectre/v1/diagnostics?fields=%5B%22traffic_tx_pps%22%5D`, with `fields` as a URL-encoded JSON array and no other parameters or body. Body and URI limits still apply, so use a body for long lists. Over MQTT, send `fields` with the `read_diagnostics` command; the values come back in `commands/result.data`.

The CLI requests `["*"]` by default; pass `fields: []` to see the catalog. Selecting fields only affects what is returned, not what is measured.

**Network rates** count station packets passed through the IP stack: UDP, ICMP, TCP, ARP, and so on. They do not count 802.11 ACKs, management frames, retries, or raw `wifi_raw` frames. TX counts packets the driver accepted, not packets delivered over the air. A rate of zero means no traffic; `null` means the counter is not available, for example on NPZ replay or older Micro firmware.

| Field group | Meaning |
| --- | --- |
| `timestamp_ms`, `uptime` | Monotonic timestamp in milliseconds and uptime in whole seconds |
| `free_memory_kb`, `minimum_free_memory_kb`, `largest_free_memory_kb` | Current heap, cumulative low-water heap, and current largest free block |
| `cpu_frequency_mhz`, `loop_time_ms` | CPU frequency and the most recent frontend loop-body cost |
| `performance_window_ready`, `performance_window_ms`, `runtime_load_percent` | Availability, duration, and runtime-loop load for the latest complete performance window |
| `loop_samples`, `loop_avg_us`, `loop_max_us` | Runtime loop timing for that window |
| `detection_timing_supported`, `detection_samples`, `detection_sum_us`, `detection_avg_us`, `detection_min_us`, `detection_max_us` | Detector timing support and aggregates |
| `generator_pps` | Successful internal generator sends per second, including `wifi_raw`; zero in external mode |
| `traffic_tx_pps`, `traffic_rx_pps` | Wi-Fi station network packets per second accepted by the driver for TX or delivered by the driver for RX |
| `csi_callback_pps`, `csi_accepted_pps` | All CSI callbacks and capture-valid CSI packets per second, respectively |
| `csi_callbacks_total`, `csi_provenance_rejected_total`, `csi_accepted_total`, `csi_admitted_total`, `csi_filtered_total` | Cumulative CSI pipeline counters |
| `csi_rx_error_total`, `csi_rx_end_error_total`, `csi_invalid_estimate_total`, `csi_invalid_first_word_total` | Cumulative capture-quality rejections; one reason per rejected callback |
| `csi_hw_error_total` | Cumulative hardware-quality rejections, aggregated on the device |
| `csi_hw_error_pps` | Hardware-quality rejection rate over the same elapsed interval as the other CSI rates |
| `csi_sanitized_first_word_total` | Frames whose hardware-invalid source pairs were zeroed; classic DC/+1 or centered guards, with the default turbulence band preserved |
| `csi_pending_frame_drops_total`, `csi_missing_slots_total`, `csi_excess_total`, `csi_stale_total`, `csi_out_of_order_total` | Cumulative queue and temporal-admission drop counters |
| `csi_occupancy_slots`, `csi_window_slots`, `csi_pending_frames`, `csi_pending_frame_capacity` | Detector-window and callback-queue occupancy |
| CSI and traffic fields ending in `_pps`, plus `csi_occupancy` | Cached packet rates and detector-window occupancy ratio |
| `wifi_channel`, `wifi_rssi_dbm` | Current channel and RSSI; unavailable RSSI is `null` |
| `runtime_motion_event_drops_total` | Runtime-to-frontend motion publications overwritten in the bounded mailbox |
| `task_stack_high_water_bytes` | Native frontend task stack headroom; `null` where unavailable |
| `direct_http` | SSE client and queue budgets plus connection, request, delivery, and `dropped_motion_events` counters |
| `raw_csi` | CSI session state, drops, send backpressure, delivered records, and stream sequence |
| `mqtt` | Native MQTT connection, queue, outbox, drop, failure, and reconnect counters |

Units: memory in KiB, times in microseconds unless the name ends in `_ms`, rates in packets per second. Fields that need a complete performance window are `null` until it is ready. A frontend may leave out fields it cannot measure; clients must not treat a missing value as zero.

The periodic log line shows a subset of these values (`tx`, `cb`, `accepted`, `hwerr`, `occ`, `ch`, `rssi`); see [check the sensing input](TROUBLESHOOTING.md#check-the-sensing-input). `hwerr` is the sum of RX errors, RX end errors, invalid estimates, and unusable invalid first words.

## Operations

Operations reject unknown fields. Routes described as taking no parameters accept an empty body or `{}` only.

### Device update

`PATCH /device` accepts:

```json
{"label":"Kitchen"}
```

`label` is required: a single line of at most 32 bytes. An empty string clears it. Success returns HTTP `200` and publishes `device`.

### Sensing update and calibration

`PATCH /sensing` accepts any non-empty supported subset of these fields:

| Field | Type and constraint |
| --- | --- |
| `enabled` | boolean |
| `detector` | `lightweight` or `high_accuracy` |
| `threshold` | finite number in `[0.0, 1.0]` |
| `motion_on_hits`, `motion_off_hits` | integers from `1` through `20`; both must be present together |
| `csi_traffic_mode` | `internal` or `external` |
| `traffic_generator_mode` | `ping`, `dns`, `dns_tcp`, or `wifi_raw` (experimental; see [compatibility limits](CSI.md#compatibility-limits)) |

All fields are checked before anything changes: either the whole request applies or none of it does. Success returns HTTP `200` and publishes `sensing`.

`POST /sensing/calibrations` takes no parameters. It returns `202` when calibration starts, or `409` with code `busy` if one is already running.

### Wi-Fi scan and BSSID selection

`POST /wifi/scans` takes no parameters and returns HTTP `202` when the scan starts. Read progress and results from `GET /wifi/access-points`.

`PUT /wifi/bssid` accepts:

```json
{"bssid":"aa:bb:cc:dd:ee:ff","force":false}
```

`bssid` is required, as six hexadecimal octets separated by colons. If the device is already on that BSSID, the pin is saved without reconnecting; set `force: true` to reconnect anyway.

`DELETE /wifi/bssid` clears the pin. `DELETE /wifi/credentials` (Native only) erases the Wi-Fi credentials and returns the device to provisioning. Neither takes parameters.

These three routes return `202` before the device disconnects. `data` holds the BSSID in use before the change:

```json
{"accepted":true,"code":"ok","message":"Wi-Fi BSSID update accepted","data":{"current_bssid":"11:22:33:44:55:66"}}
```

### MQTT configuration

`PATCH /mqtt` accepts:

```json
{
  "scheme": "mqtts",
  "host": "broker.example.net",
  "port": 8883,
  "username": "sensor",
  "password": "secret",
  "topic_prefix": "espectre/v1/devices"
}
```

| Field | Rule |
| --- | --- |
| `scheme` | Required. `mqtt` (plain TCP, for a trusted local broker) or `mqtts` (TLS, verified against the public certificate bundle and hostname) |
| `host` | Required. Hostname, IPv4, or IPv6 without brackets. No scheme, credentials, port, path, or spaces |
| `port` | Required. `1` to `65535` |
| `username`, `password`, `topic_prefix` | Optional, single line, at most 128, 256, and 128 bytes. Omitted fields keep their saved value. An empty `topic_prefix` restores `espectre/v1/devices` |

Success returns HTTP `200`. `DELETE /mqtt` takes no parameters, clears the broker and credentials, and returns `200`. MQTT over WebSocket is not supported.

### OTA actions

`POST /ota/checks` and `POST /ota/updates` accept an optional channel:

```json
{"channel":"release"}
```

`channel` is `release`, `preview`, or `develop`; the default is the firmware's build channel. Clients cannot pass a URL or version. A started check or update returns `202`; if one is already running, the result is `409` with code `busy`. Progress is published as `ota` events and on the retained MQTT `ota` topic.

## Events

`GET /espectre/v1/events` is the SSE stream. C++ frontends publish `health`, `device`, `sensing`, `wifi`, `ota`, `motion`, and `fault`, as listed in `capabilities`. Resource events carry the full resource. MQTT settings, diagnostics, discovery, and CSI never appear here.

A `motion` event is produced for each detector evaluation:

```json
{"timestamp_ms":42000,"state":"idle","score":0.0123}
```

`state` is `idle` or `motion`, and `score` is the detector output; the threshold is in `sensing`. If the client is slow, motion events may be dropped; the count is in `direct_http.dropped_motion_events`. Future events such as presence and gestures will use the same stream.

A `fault` event reports a runtime error without changing a resource payload:

```json
{"timestamp_ms":42000,"message":"runtime fault"}
```

Each connection gets a `: heartbeat` comment every 10 seconds. Missed events are not replayed. C++ frontends accept at most two clients, and close a stream that keeps failing.

## CSI collection

`GET /espectre/v1/csi` opens the binary CSI collection session. Only one session can be open, and closing the connection ends it. There is no setup request or token.

- **Before opening:** sensing must be enabled and running. Otherwise the request fails and nothing is restarted. With external traffic, start the host traffic generator first.
- **While open:** `sensing.mode` is `csi_collection`, `ready` is false, and motion and other derived events pause on every transport. Capture and traffic keep running, and resource events are still published. A second `/csi` request, and any sensing, Wi-Fi, or OTA change, returns `409`.
- **After closing:** the runtime restores its previous state, recalibrates if needed, and resumes events once it is ready again. If the client disconnects while no records are flowing, the session also ends.

Records use CSI format V8: a 60-byte little-endian prefix per record and a 16-byte session ID that must stay the same for the whole connection. Order is preserved; records dropped by the device queue show up in the diagnostics counters. NPZ datasets store decoded arrays, so they do not depend on the record version.

Packets with hardware errors are dropped before collection (see [capture quality](CSI.md#capture-quality)). If the source produces only invalid estimates, the stream can be empty; this is not a quiet room, and the runtime does not switch source. Hardware metadata is not stored in datasets.

### External CSI traffic

In `external` mode, ESPHome, Native, and Matter measure CSI on:

- **UDP markers** on port `5555`, sent to the device IP or to `csi_traffic_multicast_group` (default `239.255.0.1`; empty disables multicast). The payload must be exactly the four bytes `F0 9F 91 BB` (`"👻".encode("utf-8")`).
- **ICMP Echo Requests** sent by unicast to the device. The device replies normally.

The sending host sets the pace. Some access points convert multicast to unicast; both forms are accepted, as long as the destination IP, port, and marker match. See the [`collect` command](CLI.md#collect) and [external sources](CSI.md#external-sources).

## MQTT topics

The base topic is `espectre/v1/devices/{device_id}` unless Native is configured with another `topic_prefix`.

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

Liveness uses MQTT keepalive only. The retained `health` topic doubles as the Home Assistant availability topic; if the device drops off, the broker publishes its offline Last Will there.

Send commands to `commands/request`. `command_id`, `command`, and the parameters all sit at the top level:

```json
{"command_id":"cmd-42","command":"update_sensing","threshold":0.5}
```

`command_id` has 1 to 64 characters from ASCII letters, digits, `_`, `-`, `.`, and `:`. A command is at most 2,048 bytes. The result repeats `command_id` and `command`, and adds `accepted`, `code`, `message`, and optional `data`.

| Command | Parameters | HTTP-equivalent validation |
| --- | --- | --- |
| `update_device` | `label` | `PATCH /device` |
| `update_sensing` | Any non-empty supported sensing subset | `PATCH /sensing` |
| `recalibrate` | none | `POST /sensing/calibrations` |
| `read_diagnostics` | optional `fields` array | `GET /diagnostics`; snapshot returned in `data` |
| `check_ota` | optional `channel` | `POST /ota/checks` |
| `start_ota` | optional `channel` | `POST /ota/updates` |

MQTT settings, discovery, and CSI are not available over MQTT; diagnostics only through `read_diagnostics`. Home Assistant Discovery is a separate layer on top.

## Errors

Application results are JSON result objects. On C++ frontends, codes map to HTTP status like this:

| HTTP status | Result codes | Meaning |
| --- | --- | --- |
| `400` | `invalid_params` | The JSON object, field set, type, or value is invalid |
| `404` | `unsupported` | The resource or operation is absent from the frontend capability set |
| `409` | `busy`, `conflict`, `busy_raw_collection` | The request conflicts with active calibration, discovery, CSI, or OTA work |
| `200` or `202` | `unavailable`, `internal_error` | Dispatch reached the operation, but its backend could not complete it; the route's synchronous or asynchronous status is retained |

Over MQTT, read `accepted` and `code` in `commands/result`. A command not in the MQTT list returns `forbidden`.

The HTTP server can also reject a request before it reaches the application. These errors are plain text (`text/plain; charset=utf-8`), not result objects, so check the status and `Content-Type` before parsing JSON.

| HTTP status | Pre-dispatch failure |
| --- | --- |
| `400` | Malformed JSON, invalid route framing, or incomplete body |
| `403` | Origin or transport policy rejected the request |
| `404` | No route exists for the method and path |
| `409` | The HTTP service detected a CSI collection conflict |
| `413` | The body exceeds the frontend request limit |
| `415` | A non-empty body does not declare JSON content |
| `429` | The request or mutation rate limit was reached |
| `503` | The service is stopping, its queue is full, or request handling cannot start |

Rate limiting is reported only as HTTP `429` (there is no `rate_limited` result code) and is counted in diagnostics.

## Security and versioning

Direct HTTP is meant for a trusted local network. The firmware accepts only listed browser origins, requires the Private Network Access preflight, and limits body size, queues, clients, and request rate. It listens only on the Wi-Fi station interface and never returns stored passwords. Any extra protection for `/mqtt` would come as an additive extension.

Clients check `capabilities.protocol_version` once. Version `1.0` can gain resources, fields, operations, and events; clients must ignore what they do not know. After the stable release, a breaking change needs a new base path and discovery protocol version.

During the 3.0.0 release candidates the version stays `1.0` (`/espectre/v1`, MQTT prefix `espectre/v1/devices`, `protovers=1.0`). Diagnostics changed within this phase: a request without fields now returns the catalog, and `["*"]` returns all values. Use matching firmware and clients.
