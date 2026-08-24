# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools, MQTT dashboards, and future web orchestration services.

ESPectre Protocol defines the message model. Direct WebSocket, MQTT, MQTT over TLS, device shadows, jobs, and future bridges are transports or profiles that carry the same semantics across different trust boundaries.

This is an implementation reference for firmware, client, and integration developers. Read [Principles](#principles) and [Message Families](#message-families) when implementing a consumer; read the transport sections only for the connection mechanism you use. A **transport** carries messages, a **profile** adds deployment rules without changing their meaning, and a **retained** MQTT message is stored by the broker for future subscribers.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- Direct WebSocket is the common local configuration, control, and monitoring plane for the first-party C++ frontends.
- MQTT is the operational plane for telemetry, status, commands, dashboards, history, and alerts.
- Web orchestration profiles add identity, credentials, tenancy, retention, and fleet management; they do not redefine device telemetry.
- `device_id` is a logical protocol identifier. Native, Matter, Streamer, and Micro-ESPectre derive it once per boot as the first 64 bits of `SHA-256("espectre-device-id-v1" || station_mac_bytes)` and cache the result. This hides the MAC from routine inspection, but the stable pseudonym remains linkable and is not anonymous.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level radio traces, and serial logs must not be sent to managed services by default.

## Transports

### mDNS/DNS-SD Discovery

ESPectre uses mDNS/DNS-SD as local connection bootstrap, not as a message transport. Native, Streamer, ESPHome, and Matter publish the same first-party service type, `_espectre._tcp.local.`, for their Direct WebSocket endpoint. The `frontend` TXT field identifies the firmware surface without relying on an upstream discovery schema.

A browse starts from the service-type PTR record, which lists matching service instances. Each instance then resolves through an SRV record for hostname and port, a TXT record set for metadata, and an address record for the hostname. The current host CLI accepts IPv4 A records; an advertisement that resolves only through AAAA is not included in its results.

| Frontend | Service type | Direct SRV port | Additional frontend transport |
| --- | --- | --- | --- |
| Native | `_espectre._tcp.local.` | `80` | Optional MQTT |
| Streamer | `_espectre._tcp.local.` | `80` | UDP pacing and raw CSI collection |
| ESPHome | `_espectre._tcp.local.` | `6054` | ESPHome native API |
| Matter | `_espectre._tcp.local.` | `80` | Matter operational and commissioning services |
| Micro-ESPectre | No service advertisement | — | MQTT endpoints are configured explicitly |

ESPHome continues to publish `_esphomelib._tcp.local.`, and Matter continues to publish its standard operational and commissioning records. Those upstream services are not inputs to `./espectre devices`; the CLI browses only `_espectre._tcp.local.`. Micro-ESPectre can resolve DNS and mDNS hostnames, but it does not yet publish a Direct endpoint or discovery service.

#### Canonical advertisement

Every publishing frontend uses the following TXT contract. `<device_id>` is the canonical 16-character lowercase hexadecimal ESPectre device ID. The SRV port and `path` locate the Direct WebSocket endpoint; frontend-native services and Streamer's UDP ports remain separate.

| TXT key | Published value | Meaning |
| --- | --- | --- |
| `device_id` | `<device_id>` | Stable ESPectre protocol identity |
| `name` | Frontend-owned display name | User-facing display name; Native uses the saved label when present |
| `frontend` | `native`, `streamer`, `esphome`, or `matter` | Frontend discriminator |
| `txtvers` | `1` | TXT schema version |
| `protovers` | `1` | Direct protocol generation |
| `path` | `/espectre/v1/ws` | WebSocket endpoint path |
| `firmware` | Current firmware build identity | Running firmware version |
| `chip` | Active ESP-IDF target, such as `esp32c3` | Hardware target |
| `tls` | `0` | `0` selects `ws://`; `1` is reserved for `wss://` advertisements |
| `capabilities` | Comma-separated values | Coarse discovery capabilities; clients still negotiate exact Direct methods after connecting |
| `traffic_port` | Streamer only | UDP pacing target used by `./espectre collect` |

The CLI accepts a record only when it resolves to IPv4, has a valid SRV port and `device_id`, reports a supported `frontend`, provides an absolute `path` without spaces, uses `tls=0` or `tls=1`, and declares `txtvers=1` and `protovers=1`. A Streamer record also requires a valid `traffic_port`, because collection must not mistake the Direct SRV port for its UDP pacing target. Unknown TXT keys are ignored so the contract can grow additively. `name`, `firmware`, `chip`, and `capabilities` enrich the normalized result but do not identify the device.

#### Frontend-specific behavior

- Native owns its responder, uses the stable hostname `espectre-<device_id>.local`, and updates the TXT `name` after a saved label change.
- Streamer advertises Direct on SRV port `80`. `./espectre collect` reads `traffic_port` from TXT and validates the announced `device_id` against the first CSI packets before saving a capture. Raw CSI remains on the Streamer UDP data path and is never carried by Direct WebSocket.
- ESPHome adds `_espectre._tcp` to the responder already owned by ESPHome and advertises Direct on port `6054`. Direct mutations update the shared runtime first, then republish the corresponding ESPHome number and select entities so Home Assistant stays aligned.
- Matter adds `_espectre._tcp` to the responder already owned by the Matter stack and advertises Direct on port `80`. The service remains available after commissioning and provides detector selection and tuning that the standard Matter occupancy surface does not expose.

Services are enabled only while the station interface has a usable IPv4 address. A frontend that owns its responder sends a best-effort goodbye on a clean disconnect and reannounces after reconnect or an IP-address change; ESPHome and Matter retain responder lifecycle ownership and ESPectre only adds or removes its own service.

DNS-SD enumeration is not a browser guarantee. Configure and Monitor therefore continue to accept manual Native IP or `.local` entry, remembered endpoints, and credential-free QR or share links without an extension.

#### Peer-assisted browser discovery

Native publishes the shared IPv4 A-record name `espectre-devices.local` on port `80`, path `/espectre/v1/ws`, and subprotocol `espectre.v1`, using the same exact Origin policy as its unique Direct endpoint. Shared alias answers use a 10-second RR TTL; this bounds stale cache lifetime after abrupt loss without the continuous remove-and-add churn observed with a one-second TTL. A clean removal uses the distinct TTL-zero goodbye semantics. The record exists only while the station interface has an IPv4 address. Streamer, ESPHome, Matter, and Micro-ESPectre do not publish the alias, but canonical Streamer, ESPHome, and Matter records are accepted discovery results and retain their advertised Direct port. IPv6 remains outside the supported peer-assisted discovery boundary.

After the normal capability handshake, an eligible responder advertises the read-only `discover_peers` method. The request accepts only an empty object. It runs one asynchronous PTR browse for `_espectre._tcp.local.` with a fixed 3,000 ms query window; a second request while that operation is active receives `conflict`, and a start failure receives `unavailable`. The operation is associated with the requesting connection's opaque token and request ID. A disconnect prevents later delivery but does not create a waiter or persistent peer inventory. Existing synchronous Direct transports remain source-compatible because deferred request support is optional.

The production boundary is IPv4-only and includes the requesting Native device even when the Espressif query API omits its own advertisement. Results are deduplicated by the canonical 16-character lowercase hexadecimal `device_id`. Records for the same identity and endpoint merge and sort their addresses; conflicting endpoints reject that identity. Identities sort lexicographically. Returned IPv4 addresses must be unicast and on-link under the active station netmask; unspecified, network, broadcast, loopback, multicast, and off-link addresses are rejected. Discovery TXT capabilities are presentation hints only. After selecting an endpoint, a client must perform the normal Direct `capabilities` handshake and use the returned method catalog to expose or suppress configuration, sensing, tuning, traffic-control, and OTA operations.

The fixed limits are eight accepted devices, two IPv4 addresses per device, eight unique capability tokens, 32 characters per capability token, 128 characters for the capability list, 63 characters each for service instance, hostname, and display name, 48 characters for firmware, 16 characters for frontend and chip, and 3,584 bytes for the result object. `path` must equal `/espectre/v1/ws`; `txtvers`, `protovers`, and `tls` must equal `1`, `1`, and `0`, respectively. Frontend must be `native`, `streamer`, `esphome`, or `matter`, and the SRV port must be non-zero. Invalid records increment `rejected_results`; device, address, or serialization limits set `truncated` and retain deterministic leading results.

```json
{
  "schema_version": 1,
  "elapsed_ms": 3019,
  "status": "complete",
  "truncated": false,
  "rejected_results": 0,
  "devices": [
    {
      "device_id": "0123456789abcdef",
      "instance": "ESPectre 0123456789abcdef",
      "hostname": "espectre-0123456789abcdef",
      "name": "ESPectre 0123456789abcdef",
      "frontend": "native",
      "txt_version": 1,
      "protocol_version": 1,
      "path": "/espectre/v1/ws",
      "firmware": "3.0.0-rc1",
      "chip": "esp32c3",
      "tls": false,
      "port": 80,
      "capabilities": ["config", "monitor", "ota", "peer_discovery"],
      "addresses": ["192.168.1.29"]
    }
  ]
}
```

The portal validates the complete result again before rendering or constructing an endpoint, remembers only the selected unique address, and never stores the shared alias or peer list. Alias resolution, handshake, query, and selection failures return to the existing manual and remembered endpoint paths within the client timeout.

### Direct WebSocket v1

This section defines the Native local transport. The durable direction is recorded in `docs/adr/2026-08-23-replace-native-ble-with-direct-websocket.md`.

Native exposes Direct on `/espectre/v1/ws` and requires the `espectre.v1` WebSocket subprotocol. A client that cannot negotiate that exact subprotocol must stop before sending a request. The server accepts text frames only, with one complete JSON envelope per frame and a maximum frame size of 4,096 bytes. Binary frames, incompatible subprotocols, fragmented messages when the server cannot reassemble them within the same bound, and invalid UTF-8 are rejected.

Every client request uses this envelope:

```json
{"v":1,"type":"request","id":"req-42","method":"set_threshold","params":{"threshold":0.42}}
```

`v` is the integer envelope version. `type` is `request` for every client frame. `id` is a non-empty client-generated correlation identifier of at most 64 ASCII letters, digits, `.`, `_`, `-`, or `:`. `method` is a non-empty identifier of at most 64 ASCII letters, digits, `.`, `_`, or `-`. `params` is an object and defaults to `{}` when omitted. Unknown envelope fields are ignored so clients can add optional metadata within v1; duplicate fields, wrong field types, malformed JSON, and unsupported versions are rejected.

A successful response echoes the request identifier:

```json
{"v":1,"type":"response","id":"req-42","ok":true,"result":{"threshold":0.42}}
```

A rejected request uses the same correlation identifier when it was valid:

```json
{"v":1,"type":"response","id":"req-42","ok":false,"error":{"code":"invalid_request","message":"threshold must be between 0.0 and 1.0"}}
```

Stable v1 error codes are `invalid_request`, `invalid_params`, `unsupported_version`, `unsupported_method`, `unsupported_capability`, `not_ready`, `unavailable`, `conflict`, `rate_limited`, `apply_failed`, and `internal_error`. Human-readable `message` text is diagnostic and may change without a protocol-version bump. An envelope that cannot yield a valid request identifier may be answered with an empty `id` before the server closes the connection.

Unsolicited state uses an event envelope:

```json
{"v":1,"type":"event","event":"telemetry","data":{"movement_score":0.18,"threshold":0.42,"motion":false}}
```

The event names are `capabilities`, `info`, `status`, `telemetry`, `diagnostics`, `config`, `ota_status`, and `command_result`. Their `data` objects reuse the corresponding ESPectre message-family fields rather than MQTT topic names. Native emits `status` when its MQTT connection changes so Direct clients can update broker state without polling or reconnecting. Direct never carries raw CSI.

Direct v1 methods are grouped by capability:

| Capability | Methods | Behavior |
| --- | --- | --- |
| Base reads | `capabilities`, `info`, `status`, `config` | Available to every compatible client. `config` may report SSID, associated or pinned BSSID, channel, band policy, MQTT configured state, and non-secret endpoint fields, but never Wi-Fi or MQTT passwords. |
| Diagnostics | `diagnostics` | Returns the latest bounded runtime and transport diagnostics sample. |
| Device configuration | `set_device_label`, `set_wifi_config`, `clear_wifi_config`, `set_mqtt_config`, `clear_mqtt_config` | Uses the same validation and persistence owner as MQTT or other adapters. Wi-Fi changes stage and verify a candidate before replacing the last-known-good configuration. Secrets are write-only. |
| Sensing | `start_sensing`, `stop_sensing`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate` | Available only when advertised. `start_sensing` does not require MQTT. |
| CSI traffic | `set_csi_traffic_mode`, `set_traffic_generator_mode` | Available only when the runtime advertises traffic control. |
| OTA | `ota_status`, `ota_check`, `ota_start` | Uses the same channel and no-override policy as MQTT OTA commands. |
| Peer discovery | `discover_peers` | Advertised by Native through its bounded peer-assisted discovery service and deferred transport. |

The additive `diagnostics` result is the production performance boundary for every C++ frontend. Memory values use KiB, timing values use microseconds unless the field ends in `_ms`, rates use packets per second, and `runtime_load_percent` is runtime-loop wall time divided by the complete aggregation window. The shared runtime owns one bounded 10-second window and keeps its latest complete snapshot available between window boundaries; no build option or periodic debug logger changes whether these fields are collected.

| Field | Meaning |
| --- | --- |
| `timestamp_ms`, `uptime` | Monotonic device time in milliseconds and whole seconds |
| `free_memory_kb`, `minimum_free_memory_kb`, `largest_free_memory_kb` | Current heap, cumulative low-water heap, and current largest free block |
| `cpu_frequency_mhz` | Resolved firmware CPU frequency |
| `performance_window_ready`, `performance_window_ms` | Whether a complete window is available and its measured duration; duration is `null` before the first complete window |
| `runtime_load_percent` | Runtime-loop wall-time load over the complete window, or `null` before that window exists |
| `loop_samples`, `loop_avg_us`, `loop_max_us` | Runtime loop sample count, average duration, and maximum duration for the complete window |
| `detection_timing_supported` | Whether the selected runtime evaluates a detector; false for Streamer |
| `detection_samples`, `detection_sum_us`, `detection_avg_us`, `detection_min_us`, `detection_max_us` | Detector evaluation aggregates for the complete window, or `null` when unsupported or not ready |
| `csi_admitted_pps`, `csi_occupancy` | Native sampled detector-input rate and occupancy ratio; shared-bridge frontends expose the cumulative totals and slot counts from which a client derives the same values |
| `task_stack_high_water_bytes` | Native frontend-task stack headroom; omitted by frontends that do not own an equivalent task measurement |
| `direct` | Client and queue budgets plus accepted, rejected, malformed, oversized, rate-limited, dropped-telemetry, send-failure, and slow-client-disconnect counters |

Unsupported values are `null` or are omitted only where the owning frontend does not expose that optional measurement. Clients must not synthesize zero for a missing measurement. Direct transport counters are cumulative, so a health window compares its first and last samples.

Native accepts at most two Direct clients. Both clients may read, receive events, and issue mutations. Mutations enter one serialized dispatcher in receive order; the last accepted mutation becomes current state, and every requester receives its own correlated result. State transitions are broadcast after the mutation commits. This is an explicit multi-writer policy, not a lease hidden in the portal.

Each Direct client has a fixed-capacity outbound queue and at most one asynchronous send in flight. Telemetry coalesces to the newest value, while command responses and state transitions are never overwritten by telemetry. A client that repeatedly fails to drain is closed. MQTT has a separate 16-message frontend queue and an 8 KiB ESP-IDF outbox; only canonical telemetry is replaceable, and command results and state transitions retain their order. An outbox-full result leaves the oldest queued message in place for a later retry. Runtime callbacks copy the latest numeric sensing snapshot into frontend-owned storage; JSON serialization and transport enqueueing happen after detector evaluation returns, and socket I/O remains owned by the transport tasks.

The server accepts exact portal Origins `https://espectre.dev`, `https://www.espectre.dev`, and `https://test.espectre.dev`. A development-only Kconfig option additionally accepts HTTP Origins on any port only when the host is exactly `localhost`, `127.0.0.1`, or `[::1]`; lookalike hosts, paths, userinfo, invalid ports, and HTTPS loopback Origins remain rejected. Published firmware disables the loopback exception. Requests without an `Origin` header are rejected by default; a non-browser integration requires an explicit build-time policy. The server limits connection count, frame size, mutation rate, and queue depth, binds only after the station interface has a usable address, and stops on address loss.

Native, Streamer, ESPHome, and Matter advertise this endpoint through the [mDNS/DNS-SD discovery contract](#mdnsdns-sd-discovery). Native implements the complete device, Wi-Fi, MQTT, sensing, diagnostics, and OTA surface. The shared bridge used by Streamer, ESPHome, and Matter exposes runtime capabilities, information, status, configuration, diagnostics, sensing controls, detector tuning, and CSI traffic controls when the selected runtime supports them. Clients must use the returned capability catalog instead of assuming that every frontend implements every method.

Direct v1 is compatible only with the `espectre.v1` subprotocol. Additive object fields and advertised methods may appear during v1 and must be ignored when unknown. Removing or reinterpreting an existing field, changing envelope semantics, or accepting a different required type needs a new WebSocket subprotocol version. Home Assistant Discovery remains MQTT-only.

The ESP-IDF WebSocket stack handles RFC 6455 Ping, Pong, and Close control frames; Direct v1 adds no JSON heartbeat message. Each request has a client-side timeout. The portal treats a socket close as loss of liveness, rejects pending requests, and attempts reconnect after 500 ms, 1.5 seconds, and 3 seconds before returning to manual connection. A reconnect repeats capability negotiation and refreshes `info`, `status`, and `config` before resuming the session.

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

Native publishes telemetry over Direct and MQTT on every detector evaluation once `ready_to_publish` is true, matching Micro-ESPectre's MQTT cadence. Filtered motion-state transitions update the Home Assistant motion entity immediately without a second telemetry publish. `publish_interval_ms` remains a monotonic-clock heartbeat for status logs and diagnostics sampling; it never publishes sensing telemetry and never forces detector evaluation.

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
  "supports_device_config": true,
  "supports_runtime_threshold": true,
  "supports_runtime_motion_hits": true,
  "supports_runtime_detector": true,
  "supports_manual_recalibration": true,
  "supports_traffic_control": true,
  "supports_ota": true,
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
  "minimum_free_memory_kb": 151.8,
  "task_stack_high_water_bytes": 2876,
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
  "wifi_rssi_dbm": -55,
  "direct": {
    "clients": 1,
    "client_limit": 2,
    "queue_capacity": 8,
    "queued_messages": 0,
    "dropped_telemetry_events": 3,
    "send_failures": 0,
    "slow_client_disconnects": 0
  },
  "mqtt": {
    "connected": true,
    "queue_capacity": 16,
    "outbox_capacity_bytes": 8192,
    "queued_publishes": 0,
    "dropped_publishes": 0,
    "publish_failures": 0,
    "reconnects": 1
  }
}
```

Stats are diagnostic. Product dashboards should prefer telemetry/status/info for normal operation. When available, `free_memory_kb` reports current free heap and `loop_time_ms` reports the measured last loop-body cost in milliseconds, excluding the outer task sleep or idle delay. Motion state, movement score, threshold, detector selection, and turbulence belong to telemetry or live config/info surfaces instead of `stats`.

Native and Micro always include the CSI and Wi-Fi fields in a requested `stats` response. Native additionally includes the minimum observed free heap, the current frontend task's stack high-water mark in ESP-IDF bytes, and `direct` and `mqtt` transport diagnostics. The transport objects report their fixed queue and client or outbox budgets beside current occupancy and cumulative drop or failure counters; Micro omits those fields and objects. Both derive rates from the cumulative counters whenever the existing periodic sensing update runs, cache that completed sample, and do not add a diagnostic timer or publish it periodically. `traffic_tx_pps` is the traffic-generator transmit rate; `csi_callback_pps` is the raw CSI callback rate; `csi_accepted_pps` is the identity-accepted rate; `csi_admitted_pps` is the detector input rate after temporal admission; `csi_filtered_pps` is the capture-filter drop rate; the temporal drop fields distinguish missing slots, same-slot excess, stale packets, and out-of-order packets; and `csi_occupancy` is the valid fraction of the active detector window. Occupancy is diagnostic telemetry and does not change the device send rate. The extra CSI and transport fields are additive on protocol `1.0`; consumers may ignore unknown keys. The SDK sample uses `csi_occupancy_ratio` for the same occupancy value. Before the first periodic sensing update completes, rate fields are zero.

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
    "set_device_label",
    "set_threshold",
    "set_motion_hits",
    "set_detector",
    "recalibrate",
    "set_csi_traffic_mode",
    "set_traffic_generator_mode",
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

Set or clear the persisted user-facing label on frontends that advertise device configuration support. An empty string clears the label without changing the immutable `device_id` or derived `device_name`:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-label",
  "command": "set_device_label",
  "device_label": "Living Room"
}
```

Native republishes its retained `info` payload and Home Assistant discovery after accepting the change. Micro-ESPectre reports `supports_device_config: false` and does not advertise this command.

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

Accepted values are `ping` and `dns`. `ping` selects stateless ICMP echo traffic. `dns` selects length-prefixed DNS queries over one persistent, non-blocking TCP connection to gateway port `53`, so the gateway must accept DNS over TCP. Native persists the accepted value across reboot. The selection is always stored, but only takes effect while `csi_traffic_mode` is `internal`. Streamer does not advertise this command because collector pacing owns its traffic source.

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

Native firmware resolves a per-chip GitHub Releases manifest URL from the channel. OTA commands do not accept server, manifest, image, or version parameters; payloads containing those overrides are rejected. When `channel` is omitted, release firmware uses the latest release, preview firmware uses the rolling `snapshot` tag, and develop firmware uses the rolling `snapshot-dev` tag. Native orders numeric release tags, SemVer prereleases, and rolling `git describe` identities and advertises or applies only a strictly newer target. An older target is reported as `up_to_date`, while an unrecognized version or a divergent Git identity at the same commit distance is an error. Frontends advertise support through `supports_ota`; Micro-ESPectre does not implement OTA commands.

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
  "default_channel": "release",
  "channel": "release",
  "message": "update available"
}
```

`default_channel` is the firmware build-time default, while `channel` is the channel resolved for the current or latest attempt. `ota/state` is not retained. Native publishes the current snapshot when MQTT connects, when an OTA command changes state, and when the HTTPS OTA worker reports progress. Direct returns the same snapshot from `ota_status` and broadcasts subsequent worker snapshots as `ota_status` events. After a successful update the device reboots, so the next connect snapshot is `idle` with `current_version` set to the firmware now running.

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

## Deployment Profiles

ESPectre Protocol can be carried by multiple deployment profiles. In the local Native path, [Configure](https://espectre.dev/configure) hands a newly flashed device to standard Improv Serial for initial Wi-Fi provisioning, then uses Direct WebSocket for configuration and recovery. [Monitor](https://espectre.dev/monitor) supports Direct WebSocket for broker-free local sensing and MQTT over WebSockets for broker-backed monitoring. Direct is a trusted-LAN transport; browser support depends on the browser's mixed-content and local-network access policy.

Web orchestration profiles add identity, tenancy, device claim, state mirrors, history, alerts, and OTA around the same protocol. Those system-level concerns belong to [ARCHITECTURE.md](ARCHITECTURE.md), not to this message schema.

## Web Orchestration Privacy Boundary

Default web-orchestration telemetry should be derived and minimal:

| Field | Purpose |
|-------|---------|
| `device_id` | Service-scoped opaque identifier |
| `timestamp_ms` | Event or sample time |
| `online` | Device availability |
| `firmware_version` | Fleet visibility and update eligibility |
| `frontend` | `esphome`, `matter`, `native`, `streamer`, `micro`, `custom`, or future frontend label |
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

- Evaluate an authenticated secure-local Direct transport if the supported browser matrix cannot sustain the current trusted-LAN WebSocket profile without flags or certificate exceptions.
