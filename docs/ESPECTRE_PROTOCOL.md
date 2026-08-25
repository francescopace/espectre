# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools, MQTT dashboards, and future web orchestration services.

ESPectre Protocol defines the message model. Direct HTTP, MQTT, MQTT over TLS, device shadows, jobs, and future bridges are transports or profiles that carry the same semantics across different trust boundaries.

This is an implementation reference for firmware, client, and integration developers. Read [Principles](#principles) and [Message Families](#message-families) when implementing a consumer; read the transport sections only for the connection mechanism you use. A **transport** carries messages, a **profile** adds deployment rules without changing their meaning, and a **retained** MQTT message is stored by the broker for future subscribers.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- Direct HTTP is the common local configuration, control, and monitoring plane for the first-party C++ frontends.
- MQTT is the operational plane for telemetry, status, commands, dashboards, history, and alerts.
- Web orchestration profiles add identity, credentials, tenancy, retention, and fleet management; they do not redefine device telemetry.
- `device_id` is a logical protocol identifier. Native, Matter, Streamer, and Micro-ESPectre derive it once per boot as the first 64 bits of `SHA-256("espectre-device-id-v1" || station_mac_bytes)` and cache the result. This hides the MAC from routine inspection, but the stable pseudonym remains linkable and is not anonymous.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level radio traces, and serial logs must not be sent to managed services by default.

## Transports

### mDNS/DNS-SD Discovery

ESPectre uses mDNS/DNS-SD as local connection bootstrap, not as a message transport. Native, Streamer, ESPHome, and Matter publish the same first-party service type, `_espectre._tcp.local.`, for their Direct HTTP endpoint. The `frontend` TXT field identifies the firmware surface without relying on an upstream discovery schema.

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

Every publishing frontend uses the following TXT contract. `<device_id>` is the canonical 16-character lowercase hexadecimal ESPectre device ID. The SRV port and `path` locate the Direct HTTP endpoint; frontend-native services and Streamer's UDP ports remain separate.

| TXT key | Published value | Meaning |
| --- | --- | --- |
| `device_id` | `<device_id>` | Stable ESPectre protocol identity |
| `name` | Frontend-owned display name | User-facing display name; Native uses the saved label when present |
| `frontend` | `native`, `streamer`, `esphome`, or `matter` | Frontend discriminator |
| `txtvers` | `2` | TXT schema version |
| `protovers` | `1` | Direct protocol generation |
| `transport` | `http` | Local Direct transport |
| `path` | `/espectre/v1/request` | JSON request endpoint |
| `events` | `/espectre/v1/events` | SSE event endpoint |
| `firmware` | Current firmware build identity | Running firmware version |
| `chip` | Active ESP-IDF target, such as `esp32c3` | Hardware target |
| `capabilities` | Comma-separated values | Coarse discovery capabilities; clients still negotiate exact Direct methods after connecting |
| `traffic_port` | Streamer only | UDP pacing target used by `./espectre collect` |

The CLI accepts a record only when it resolves to IPv4, has a valid SRV port and `device_id`, reports a supported `frontend`, declares `txtvers=2`, `protovers=1`, and `transport=http`, and publishes the exact request and event paths above. A Streamer record also requires a valid `traffic_port`, because collection must not mistake the Direct SRV port for its UDP pacing target. Unknown TXT keys are ignored so the contract can grow additively. `name`, `firmware`, `chip`, and `capabilities` enrich the normalized result but do not identify the device.

#### Frontend-specific behavior

- Native owns its responder, uses the stable hostname `espectre-<device_id>.local`, and updates the TXT `name` after a saved label change.
- Streamer advertises Direct on SRV port `80`. `./espectre collect` reads `traffic_port` from TXT and validates the announced `device_id` against the first CSI packets before saving a capture. Raw CSI remains on the Streamer UDP data path and is never carried by Direct HTTP.
- ESPHome adds `_espectre._tcp` to the responder already owned by ESPHome and advertises Direct on port `6054`. Direct mutations update the shared runtime first, then republish the corresponding ESPHome number and select entities so Home Assistant stays aligned.
- Matter adds `_espectre._tcp` to the responder already owned by the Matter stack and advertises Direct on port `80`. The service remains available after commissioning and provides detector selection and tuning that the standard Matter occupancy surface does not expose.

Services are enabled only while the station interface has a usable IPv4 address. A frontend that owns its responder sends a best-effort goodbye on a clean disconnect and reannounces after reconnect or an IP-address change; ESPHome and Matter retain responder lifecycle ownership and ESPectre only adds or removes its own service.

DNS-SD enumeration is not a browser guarantee. Configure and Monitor therefore continue to accept manual Native IP or `.local` entry, remembered endpoints, and credential-free QR or share links without an extension.

#### Peer-assisted browser discovery

For each Auto-discovery attempt, the portal generates 96 bits with Web Crypto and resolves one lowercase bootstrap hostname in the form `espectre-devices-<24 hexadecimal characters>.local`. Native answers only valid, uncompressed class-IN A or AAAA questions matching that form. An A response repeats the requested owner name, contains the current station IPv4 address, uses a 10-second TTL, clears the cache-flush bit so simultaneous Native responders contribute shared records, and adds an NSEC record whose bitmap declares A but not AAAA. A standalone AAAA question receives only the same NSEC assertion, so dual-stack resolvers can proceed to A without waiting for an IPv6 timeout; the responder never advertises an IPv6 address. The browser then sends `discover_peers` to port `80` at `/espectre/v1/request` with the same exact Origin policy as the unique Direct endpoint and a 10-second client timeout. The responder is stateless: it does not register, retain, announce, or send a goodbye for the nonce hostname. It accepts multicast, QU, and legacy-unicast queries, keeps at most four delayed multicast answers at 25, 50, 75, and 100 ms, and schedules at most eight answers per second. Pending answers are discarded on an IPv4 change, Wi-Fi disconnect, or reconfiguration.

The static `espectre-devices.local` alias is intentionally unsupported, and there is no automatic compatibility fallback between portal and firmware versions using different bootstrap contracts. Streamer, ESPHome, Matter, and Micro-ESPectre do not implement the nonce responder, but canonical Streamer, ESPHome, and Matter records remain accepted discovery results and retain their advertised Direct port. IPv6 remains outside the supported peer-assisted discovery boundary; manual private IP, unique hostname, and device-ID entry are unchanged.

After the normal capability handshake, an eligible responder advertises the read-only `discover_peers` method. The request accepts only an empty object. It runs one asynchronous PTR browse for `_espectre._tcp.local.` with a fixed 3,000 ms query window; a second request while that operation is active receives `conflict`, and a start failure receives `unavailable`. The operation is associated with the requesting connection's opaque token and request ID. A disconnect prevents later delivery but does not create a waiter or persistent peer inventory. Existing synchronous Direct transports remain source-compatible because deferred request support is optional.

The production boundary is IPv4-only and includes the requesting Native device even when the Espressif query API omits its own advertisement. Results are deduplicated by the canonical 16-character lowercase hexadecimal `device_id`. Records for the same identity and endpoint merge and sort their addresses; conflicting endpoints reject that identity. Identities sort lexicographically. Returned IPv4 addresses must be unicast and on-link under the active station netmask; unspecified, network, broadcast, loopback, multicast, and off-link addresses are rejected. Discovery TXT capabilities are presentation hints only. After selecting an endpoint, a client must perform the normal Direct `capabilities` handshake and use the returned method catalog to expose or suppress configuration, sensing, tuning, traffic-control, and OTA operations.

The fixed limits are eight accepted devices, two IPv4 addresses per device, eight unique capability tokens, 32 characters per capability token, 128 characters for the capability list, 63 characters each for service instance, hostname, and display name, 48 characters for firmware, 16 characters for frontend and chip, and 3,584 bytes for the result object. `txtvers`, `protovers`, and `transport` must equal `2`, `1`, and `http`; `path` and `events` must equal `/espectre/v1/request` and `/espectre/v1/events`. Frontend must be `native`, `streamer`, `esphome`, or `matter`, and the SRV port must be non-zero. Invalid records increment `rejected_results`; device, address, or serialization limits set `truncated` and retain deterministic leading results.

```json
{
  "schema_version": 2,
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
      "schema_version": 2,
      "txt_version": 2,
      "protocol_version": 1,
      "transport": "http",
      "path": "/espectre/v1/request",
      "events": "/espectre/v1/events",
      "firmware": "3.0.0-rc1",
      "chip": "esp32c3",
      "port": 80,
      "capabilities": ["config", "monitor", "ota", "peer_discovery"],
      "addresses": ["192.168.1.29"]
    }
  ]
}
```

The portal validates the complete result again before rendering or constructing an endpoint, remembers only the selected unique address, and never stores the shared alias or peer list. Alias resolution, handshake, query, and selection failures return to the existing manual and remembered endpoint paths within the client timeout.

### Direct HTTP v1

This section defines the common local transport used by the C++ frontends. Native owns the complete local profile; Streamer, ESPHome, and Matter use the shared bridge and advertise a filtered command intersection. The durable direction is recorded in `docs/adr/2026-08-25-replace-local-direct-websocket-with-http.md` and `docs/adr/2026-08-24-use-one-command-engine-across-frontends.md`.

Direct exposes `POST /espectre/v1/request` with `application/json`, `GET /espectre/v1/events` with `text/event-stream`, and, on supported Native C3 builds, `GET /espectre/v1/csi` with `application/octet-stream`. Requests are limited to 4,096 bytes, and correlated JSON responses and SSE event envelopes are limited to 8,192 bytes. The server dispatches commands only from the frontend owner task, uses fixed queues and rate limits, and reports malformed input, unsupported media types, disallowed Origins, oversize input, saturation, and internal failures through an appropriate HTTP status. A syntactically valid Direct request receives a correlated response envelope even when the command is rejected.

Every Direct response sets `Cache-Control: no-store`. Browser calls use `targetAddressSpace: "local"`, and the server handles CORS preflight and Private Network Access. `Access-Control-Allow-Origin` echoes only an exact configured Origin and is paired with `Vary: Origin`; no wildcard Origin is accepted. ESPHome serves Direct independently on port `6054` and does not depend on the ESPHome web server or native API.

Every client request uses this envelope:

```json
{"v":1,"type":"request","id":"req-42","method":"set_threshold","params":{"threshold":0.42}}
```

`v` is the integer envelope version. `type` is `request` for every client message. `id` is a non-empty client-generated correlation identifier of at most 64 ASCII letters, digits, `.`, `_`, `-`, or `:`. `method` is a non-empty identifier of at most 64 ASCII letters, digits, `.`, `_`, or `-`. `params` is an object and defaults to `{}` when omitted. Unknown envelope fields are ignored so clients can add optional metadata within v1; duplicate fields, wrong field types, malformed JSON, and unsupported versions are rejected.

A successful response echoes the request identifier and wraps the transport-neutral engine result:

```json
{"v":1,"type":"response","id":"req-42","ok":true,"result":{"command":"set_threshold","code":"ok","message":"threshold updated"}}
```

A rejected request uses the same correlation identifier when it was valid:

```json
{"v":1,"type":"response","id":"req-42","ok":false,"error":{"code":"invalid_params","message":"threshold must be between 0.0 and 1.0"}}
```

The command engine's stable v1 error codes are `invalid_params`, `unsupported`, `forbidden`, `busy`, `busy_raw_collection`, `not_raw_session_owner`, `conflict`, `unavailable`, and `internal_error`; `ok` identifies an accepted command. Envelope parsing may additionally report `invalid_request` or `unsupported_version` before a command reaches the engine. Human-readable `message` text is diagnostic and must not drive client behavior. An envelope that cannot yield a valid request identifier may be answered with an empty `id` before the server closes the connection.

Unsolicited state uses an event envelope:

```json
{"v":1,"type":"event","event":"telemetry","data":{"movement_score":0.18,"threshold":0.42,"motion":false}}
```

The canonical event names are `telemetry`, `status`, `info`, `config`, `ota_status`, and `fault`. Each SSE message uses the name in `event:` and the complete event envelope in `data:`. Diagnostics and command results are correlated responses, not events. The service emits a heartbeat comment every 10 seconds, supports at most two subscribers, coalesces replaceable telemetry in a fixed per-client queue, and disconnects slow clients. There is no replay. After reconnecting with the 500 ms, 1.5 s, and 3 s retry sequence, the web client repeats capability negotiation and refreshes current state.

Direct v1 methods are grouped by capability:

| Capability | Methods | Behavior |
| --- | --- | --- |
| Base reads | `capabilities`, `info`, `status`, `config` | Available to every compatible client. `config` may report SSID, associated or pinned BSSID, channel, band policy, MQTT configured state, and non-secret endpoint fields, but never Wi-Fi or MQTT passwords. |
| Diagnostics | `diagnostics` | Returns the latest bounded runtime and transport diagnostics sample. |
| Device configuration | `set_device_label`, `set_wifi_config`, `clear_wifi_config`, `set_mqtt_config`, `clear_mqtt_config` | Uses the same validation and persistence owner as MQTT or other adapters. Wi-Fi changes stage and verify a candidate before replacing the last-known-good configuration. Secrets are write-only. |
| Sensing | `set_sensing`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate` | Available only when advertised. `set_sensing` carries the required Boolean `enabled` parameter and does not require MQTT. |
| CSI traffic | `set_csi_traffic_mode`, `set_traffic_generator_mode` | Available only when the runtime advertises traffic control. |
| OTA | `ota_status`, `ota_check`, `ota_start` | Uses the same channel and no-override policy as MQTT OTA commands. |
| Peer discovery | `discover_peers` | Advertised by Native through its bounded peer-assisted discovery service and deferred transport. |
| Raw CSI | `start_raw_stream`, `stop_raw_stream` | Advertised only by a Native build whose runtime and Direct transport support an owner-bound binary raw session. |

#### Native Direct raw CSI

Native Direct raw CSI is an additive capability of the HTTP service. Capability negotiation reports `features.raw_csi=true` and a `raw_csi` object containing endpoint `/espectre/v1/csi`, transport `http`, raw protocol version `1`, record version `8`, a 76-byte frame prefix, and target PPS range `1–500`. The initial rollout advertises this capability only from ESP32-C3 Native firmware; other chips and frontends report it as unsupported until their hardware gates pass.

`start_raw_stream` accepts only `target_pps`, creates a random 128-bit session ID, and moves the runtime from `sensing` to `raw_collection`. `GET /espectre/v1/csi` and `stop_raw_stream` require `Authorization: Bearer <session-id>`. Another start receives `busy_raw_collection`, and a stop with the wrong bearer receives `not_raw_session_owner`. Reads remain available during the session. Wi-Fi, OTA, detector, calibration, traffic, and sensing mutations receive `busy_raw_collection`. MQTT stays connected but does not publish motion, sensing telemetry, or Home Assistant state until sensing is restored. Stream abort, Wi-Fi loss, channel or BSSID change, timeout, reboot, stop, or fault terminates the session without persisting `raw_collection`.

The raw endpoint accepts one bearer-bound collector. A dedicated device worker paces output at `target_pps`, selects the freshest unconsumed AP/BSSID candidate whose age does not exceed `clamp(2_000_000 / target_pps, 10_000, 100_000)` microseconds, and retains only bounded state. A missing candidate emits a `no_sample` heartbeat at most once per second. There is no credit message or credit window.

Each record starts with a packed 76-byte little-endian HTTP prefix containing magic `ESPR`, protocol version, status, prefix length, session ID, stream sequence, record version, record length, and cumulative fresh, no-sample, replacement, drop, and send-backpressure counters. A fresh prefix is followed immediately by one CSI V8 record; a no-sample prefix has no payload. Clients must reconstruct frames across arbitrarily split or aggregated HTTP chunks and reject invalid magic, version, length, session, sequence, status, or record data.

Version 8 raw records retain the 64-byte packed header used by Streamer V7. `device_ticks_us` is the capture timestamp, and the final counters are `transport_backpressure_total`, `fresh_record_total`, and `request_accepted_total`. Native Direct emits V8 only. Streamer remains byte-for-byte V7 during the migration, while the host parser accepts both versions for comparison and historical captures.

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
| `direct_http` | SSE subscriber and queue budgets plus accepted, rejected, malformed, oversized, rate-limited, dropped-telemetry, send-failure, and slow-client-disconnect counters |

Unsupported values are `null` or are omitted only where the owning frontend does not expose that optional measurement. Clients must not synthesize zero for a missing measurement. Direct transport counters are cumulative, so a health window compares its first and last samples.

Native accepts at most two Direct SSE subscribers and one independent raw collector. POST requests use a separate fixed request queue. Outside raw collection, any authorized requester may issue mutations. Direct, MQTT, and frontend-native controls enter the same `FrontendCommandEngine` serially on the frontend task; no command worker is added. The last accepted mutation becomes current state, and every requester receives its own correlated result. A query responds only to its requesting transport. State transitions caused by a mutation are broadcast after it commits. This is an explicit multi-writer policy, with the documented raw-session ownership exception.

Shared MQTT and Direct event payloads are serialized once and fanned out to every active transport. The transports retain independent backpressure because a slow broker must not delay local clients, and one slow SSE subscriber must not delay MQTT or another subscriber. Each SSE subscriber therefore has a fixed-capacity outbound queue. Telemetry coalesces to the newest value, while state transitions are never overwritten by telemetry. A subscriber that repeatedly fails to drain is closed. Correlated command responses use their originating POST request rather than the event queue. MQTT has a separate 16-message freshness queue and an 8 KiB ESP-IDF network outbox: the first coalesces replaceable topics and orders command results, while the second transfers bytes to the esp-mqtt task and socket. The Native build uses a 20 ms esp-mqtt poll timeout and allows replaceable publications into the network outbox below a 2 KiB high-water mark, rather than serializing them behind an empty-outbox gate. An outbox-full result leaves the oldest frontend-queued message in place for a later retry. Runtime callbacks copy the latest numeric sensing snapshot into frontend-owned storage; JSON serialization and transport enqueueing happen after detector evaluation returns, and socket I/O remains owned by the transport tasks.

The server accepts exact portal Origins `https://espectre.dev`, `https://www.espectre.dev`, and `https://test.espectre.dev`. A development-only Kconfig option additionally accepts HTTP Origins on any port only when the host is exactly `localhost`, `127.0.0.1`, or `[::1]`; lookalike hosts, paths, userinfo, invalid ports, and HTTPS loopback Origins remain rejected. Published firmware disables the loopback exception. Requests without an `Origin` header are rejected by default; a non-browser integration requires an explicit build-time policy. The server limits connection count, frame size, mutation rate, and queue depth, binds only after the station interface has a usable address, and stops on address loss.

Native, Streamer, ESPHome, and Matter advertise this endpoint through the [mDNS/DNS-SD discovery contract](#mdnsdns-sd-discovery). Native implements the complete device, Wi-Fi, MQTT, sensing, diagnostics, and OTA surface. The shared bridge used by Streamer, ESPHome, and Matter exposes runtime capabilities, information, status, configuration, diagnostics, sensing controls, detector tuning, and CSI traffic controls when the selected runtime supports them. Clients must use the returned capability catalog instead of assuming that every frontend implements every method.

Direct v1 is identified by the envelope `v` field and `protovers=1` discovery metadata. Additive object fields and advertised methods may appear during v1 and must be ignored when unknown. Removing or reinterpreting an existing field, changing envelope semantics, or accepting a different required type needs a new Direct protocol version. Home Assistant Discovery remains MQTT-only.

The SSE stream emits a comment heartbeat every 10 seconds; Direct v1 adds no JSON heartbeat message. Each POST has a client-side timeout. The portal treats an ended SSE response as loss of liveness, aborts pending requests, and attempts reconnect after 500 ms, 1.5 seconds, and 3 seconds before returning to manual connection. A reconnect repeats capability negotiation and refreshes `info`, `status`, and `config` before resuming the session.

### MQTT

MQTT is the operational transport once the device has network access.

The same topic shape is valid for a local broker and for a managed broker, with auth and tenancy added by the deployment profile:

```text
espectre/v1/devices/{device_id}/telemetry
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/config
espectre/v1/devices/{device_id}/capabilities
espectre/v1/devices/{device_id}/ota_status
espectre/v1/devices/{device_id}/fault
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/result
```

`status`, `info`, `config`, `capabilities`, and `ota_status` are retained. `telemetry`, `fault`, and `commands/result` are not retained. There is no diagnostics topic: a `diagnostics` query returns its correlated sample in `commands/result.data`. A query never publishes a second side response and never crosses into Direct; accepted mutations publish the relevant retained state events to every active transport. Managed-service MQTT should use TLS and per-device credentials. Local lab MQTT may use a simpler broker/auth model, but should keep the same message shape.

The dependency-free browser protocol layer is [`espectre-mqtt.js`](web/assets/js/espectre-mqtt.js). It is transport-policy agnostic and implements canonical topic construction, retained discovery, protocol-version and JSON-object validation, generic command publication from the retained capability schema, `commands/result` correlation, timeouts, and pending-command cleanup. The website supplies the MQTT.js WebSocket transport and consumes the additive Home Assistant scalar topics separately.

### Home Assistant MQTT Adapter Profile

Native and Micro-ESPectre can publish an additive Home Assistant MQTT Discovery surface without changing the canonical ESPectre topics above. Discovery payloads use the standard `{discovery_prefix}/{component}/{object_id}/config` topic shape. Native also retains its canonical `status` payload so late subscribers receive the current availability; entity-shaped state topics remain non-retained under `espectre/v1/devices/{device_id}/ha/...`.

The Native HA adapter publishes sensing entities that match the ESPHome Home Assistant surface so one dashboard can be reused after replacing the device prefix: Motion Detected on filtered state edges, Movement Score on every detector evaluation (`evaluation_interval_ms`), writable Threshold on operator writes, calibration, and Lightweight settled-level recovery, Motion On Hits and Motion Off Hits numbers, a Detection Profile select where the frontend supports runtime detector switching, CSI Traffic Ownership plus CSI Traffic Source selects where the frontend supports traffic control, a configuration-category Recalibrate button that starts startup recalibration, a diagnostic-category Calibration Active binary sensor that reports the authoritative runtime state, and the ESPHome CSI diagnostic sensors plus a Refresh Diagnostics button that publishes the latest cached sample on demand. Native discovery `object_id` suffixes follow the ESPHome entity-ID slugs (`motion_detected`, `movement_score`, `recalibrate`, `calibration_active`, and so on); MQTT state and command topic suffixes under `ha/` stay unchanged. Micro-ESPectre currently retains its combined Trigger Calibration switch while matching the remaining applicable scalar and diagnostic entities. Canonical `telemetry` JSON keeps `movement_score` and `threshold` on that same evaluation cadence. Leftover Intensity and previous Native/Micro discovery object IDs are unpublished with empty retained configs.

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
  "timestamp_ms": 123456,
  "sensing_enabled": true,
  "ready_to_publish": true,
  "calibrating": false
}
```

Every MQTT frontend retains the latest status payload. A normal shutdown publishes retained `online: false`; after an unexpected disconnect, the broker publishes the retained Last Will with the same offline state. A later connection replaces it with retained `online: true`, allowing availability consumers that subscribe after discovery to recover the current state. MQTT connect also publishes retained `info`, `config`, `capabilities`, and, when OTA is present, `ota_status`. Motion and runtime tuning are deliberately absent from `status`; they belong to `telemetry` and `config`.

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

`info` contains identity, firmware, chip, frontend, timing, and non-sensitive descriptive data. Capability booleans are not duplicated here; clients consume the `capabilities` schema. Native and Micro publish `info` retained on connect and after an accepted label change so late subscribers, including `./espectre mqtt` discovery, see the current frontend identity. `network` and `detection` are optional. Canonical MQTT `info` reports the active Wi-Fi channel when available, but does not serialize the local IP address or station MAC. `csi_traffic_mode`, `traffic_mode`, and `csi_target_pps` are included when the frontend owns CSI traffic configuration; omit them when those values are unset. `evaluation_interval_ms` and `publish_interval_ms` are the detector evaluation cadence and the status-log heartbeat; omit them when unset. Nearby setup and local logs may still expose configuration or link details, including SSID, BSSID, local IP, station MAC, broker host, or broker username. Managed services should not collect those values by default.

### Config

Published retained on:

```text
espectre/v1/devices/{device_id}/config
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "runtime": {
    "threshold": 0.45,
    "motion_on_hits": 4,
    "motion_off_hits": 3,
    "detector": "lightweight",
    "csi_traffic_mode": "internal",
    "traffic_generator_mode": "ping"
  }
}
```

`runtime` is the uniform cross-frontend configuration section. `device`, `wifi`, and `mqtt` are optional and are included only when both the frontend and requesting transport authorize them. Passwords and other secrets are write-only and never appear in a response. Runtime tuning changes publish one `config` state transition; a label change publishes `info`; sensing state publishes `status`; OTA state publishes `ota_status`; and recalibration publishes `status`, followed by `config` only when the resulting threshold changes.

### Diagnostics

Returned only as `data` in the correlated response to an explicit `diagnostics` query. Native and Micro include the CSI and Wi-Fi diagnostic fields below. A frontend that does not sample those counters omits the extra keys and keeps the shared core (`uptime`, `free_memory_kb`, and `loop_time_ms`):

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

Diagnostics are on-demand. Product dashboards should prefer telemetry, status, and info for normal operation. When available, `free_memory_kb` reports current free heap and `loop_time_ms` reports the measured last loop-body cost in milliseconds, excluding the outer task sleep or idle delay. Motion state, movement score, threshold, detector selection, and turbulence belong to telemetry or live config and info surfaces instead of diagnostics.

Native and Micro include the CSI and Wi-Fi fields in a requested `diagnostics` response. Native additionally includes the minimum observed free heap, the current frontend task's stack high-water mark in ESP-IDF bytes, and `direct` and `mqtt` transport diagnostics. The transport objects report their fixed queue and client or outbox budgets beside current occupancy and cumulative drop or failure counters; Micro omits those fields and objects. Both derive rates from the cumulative counters whenever the existing periodic sensing update runs, cache that completed sample, and do not add a diagnostic timer or publish it periodically. `traffic_tx_pps` is the traffic-generator transmit rate; `csi_callback_pps` is the raw CSI callback rate; `csi_accepted_pps` is the identity-accepted rate; `csi_admitted_pps` is the detector input rate after temporal admission; `csi_filtered_pps` is the capture-filter drop rate; the temporal drop fields distinguish missing slots, same-slot excess, stale packets, and out-of-order packets; and `csi_occupancy` is the valid fraction of the active detector window. Occupancy is diagnostic telemetry and does not change the device send rate. The extra CSI and transport fields are additive on protocol `1.0`; consumers may ignore unknown keys. The SDK sample uses `csi_occupancy_ratio` for the same occupancy value. Before the first periodic sensing update completes, rate fields are zero.

ESPHome exposes the same cached measurements as diagnostic entities. Native MQTT Discovery and Micro-ESPectre MQTT match that surface: the diagnostic sensors stay unpublished until Home Assistant presses `Refresh Diagnostics`. These on-demand diagnostics are independent of the optional runtime debug logs.

### Capabilities

Published on:

```text
espectre/v1/devices/{device_id}/capabilities
```

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "commands": [
    {
      "name": "status",
      "kind": "query",
      "access": "read",
      "params": {
        "additionalProperties": false
      },
      "result": "status"
    },
    {
      "name": "set_threshold",
      "kind": "mutation",
      "access": "control",
      "params": {
        "type": "object",
        "properties": {"threshold": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["threshold"],
        "additionalProperties": false
      }
    }
  ],
  "events": ["telemetry"],
  "config_sections": ["runtime"],
  "features": {"raw_csi": false}
}
```

The example is abbreviated. The retained payload contains every command executable through that transport and frontend, the canonical event names, available configuration sections, and feature flags. Each command declares `name`, `kind` (`query`, `mutation`, or `action`), `access`, a constrained JSON Schema subset (`type`, `properties`, `required`, `additionalProperties`, `enum`, `minimum`, and `maximum`), and a named `result` schema only when it returns data. Because the transport envelope already requires an object, no-parameter schemas contain only `additionalProperties: false`; empty `properties` and `required` members are omitted. The complete minified catalog must remain below 4 KiB. Clients use it for rendering, validation, help, and completion instead of maintaining verb allowlists.

Access classes are `read`, `control`, `device_admin`, `network_admin`, `firmware_update`, and `discovery`. Native Direct may expose every implemented class. Native MQTT exposes read, control, device administration, and firmware update, including `set_sensing`; Wi-Fi and MQTT configuration and `discover_peers` remain Direct-local. Other frontends publish only the intersection they can execute. C++ and MicroPython keep independent registries, with a host probe enforcing normalized catalog parity for the shared profile.

### Commands

Published to:

```text
espectre/v1/devices/{device_id}/commands/request
```

Canonical reads are `capabilities`, `info`, `status`, `config`, `diagnostics`, and `ota_status`. Canonical mutations and actions are `set_sensing`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate`, `set_csi_traffic_mode`, `set_traffic_generator_mode`, `set_device_label`, the advertised Wi-Fi and MQTT configuration methods, `ota_check`, `ota_start`, and `discover_peers`. The removed `commands`, `stats`, `start_sensing`, and `stop_sensing` names have no v1 aliases because this contract has not shipped in a stable v3 release.

Enable or pause sensing on frontends that advertise sensing control:

```json
{"protocol_version":"1.0","command_id":"cmd-sensing","command":"set_sensing","enabled":true}
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

Native firmware resolves a per-chip GitHub Releases manifest URL from the channel. OTA commands do not accept server, manifest, image, or version parameters; payloads containing those overrides are rejected. When `channel` is omitted, release firmware uses the latest release, preview firmware uses the rolling `snapshot` tag, and develop firmware uses the rolling `snapshot-dev` tag. Native orders numeric release tags, SemVer prereleases, and rolling `git describe` identities and advertises or applies only a strictly newer target. An older target is reported as `up_to_date`, while an unrecognized version or a divergent Git identity at the same commit distance is an error. Support is declared only through the `capabilities` catalog; Micro-ESPectre does not advertise OTA commands.

Publish OTA state on:

```text
espectre/v1/devices/{device_id}/ota_status
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

`default_channel` is the firmware build-time default, while `channel` is the channel resolved for the current or latest attempt. `ota_status` is retained. Native publishes the current snapshot when MQTT connects, when an OTA command changes state, and when the HTTPS OTA worker reports progress. Direct returns the same snapshot from `ota_status` and broadcasts subsequent worker snapshots as `ota_status` events. After a successful update the device reboots, so the next connect snapshot is `idle` with `current_version` set to the firmware now running.

Command result:

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "command_id": "cmd-001",
  "command": "set_threshold",
  "accepted": true,
  "code": "ok",
  "message": "threshold updated"
}
```

Every accepted or rejected MQTT request produces exactly one non-retained `commands/result`. A query adds `data` containing its schema payload; mutations and actions omit `data`. `command_id` is the caller's correlation key. Command results are non-replaceable in the frontend freshness queue, while all human-readable decisions remain subordinate to the stable `accepted` and `code` fields.

## Deployment Profiles

ESPectre Protocol can be carried by multiple deployment profiles. In the local Native path, [Configure](https://espectre.dev/configure) hands a newly flashed device to standard Improv Serial for initial Wi-Fi provisioning, then uses Direct HTTP for configuration and recovery. [Monitor](https://espectre.dev/monitor) supports Direct HTTP for broker-free local sensing and MQTT over WebSockets for broker-backed monitoring. Direct is a trusted-LAN transport; browser support depends on the browser's mixed-content and local-network access policy.

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

- Keep the hosted-HTTPS-to-local-HTTP browser matrix current, and offer the optional authenticated WSS relay where local fetch is unavailable without weakening browser security.
