# ESPectre Protocol

ESPectre Protocol is the shared logical protocol for ESPectre devices, tools, MQTT dashboards, and future web orchestration services.

ESPectre Protocol defines the message model. Direct HTTP, MQTT, MQTT over TLS, device shadows, jobs, and future bridges are transports or profiles that carry the same semantics across different trust boundaries.

This is an implementation reference for firmware, client, and integration developers. Read [Principles](#principles) and [Message Families](#message-families) when implementing a consumer; read [mDNS/DNS-SD Discovery](#mdnsdns-sd-discovery) and the transport sections only for the connection mechanisms you use. A **transport** carries messages, a **profile** adds deployment rules without changing their meaning, and a **retained** MQTT message is stored by the broker for future subscribers.

## Principles

- Derived telemetry only; raw CSI is not part of the normal protocol surface.
- One message model, multiple transports.
- Direct HTTP is the common local configuration, control, and monitoring plane for the first-party C++ frontends.
- MQTT is the operational plane for telemetry, status, commands, dashboards, history, and alerts.
- Web orchestration profiles add identity, credentials, tenancy, retention, and fleet management; they do not redefine device telemetry.
- `device_id` is a logical protocol identifier. Native, ESPHome, Matter, and Micro-ESPectre derive it once per boot as the first 64 bits of `SHA-256("espectre-device-id-v1" || station_mac_bytes)` and cache the result. This hides the MAC from routine inspection, but the stable pseudonym remains linkable and is not anonymous.
- `device_name` is the immutable generated name `ESPectre <chip> <last-six-device-id-characters>` on every frontend. `device_label` is empty by default and contains only an optional user-provided label; clients display the label when non-empty and otherwise fall back to the generated name. The separate `frontend` field identifies Native, ESPHome, Matter, or Micro-ESPectre without changing device identity.
- Privacy-sensitive values such as SSID, BSSID, local IP address, packet-level radio traces, and serial logs must not be sent to managed services by default.

## mDNS/DNS-SD Discovery

ESPectre uses mDNS/DNS-SD as local connection bootstrap, not as a message transport. Native, ESPHome, Matter, and Micro-ESPectre publish the same first-party service type, `_espectre._tcp.local.`, for their Direct HTTP endpoint. The `frontend` TXT field identifies the firmware surface without relying on an upstream discovery schema.

A browse starts from the service-type PTR record, which lists matching service instances. Each instance then resolves through an SRV record for hostname and port, a TXT record set for metadata, and an address record for the hostname. The current host CLI accepts IPv4 A records; an advertisement that resolves only through AAAA is not included in its results.

| Frontend | Service type | Direct SRV port | Additional frontend transport |
| --- | --- | --- | --- |
| Native | `_espectre._tcp.local.` | `62587` | Optional MQTT |
| ESPHome | `_espectre._tcp.local.` | `62587` | ESPHome native API |
| Matter | `_espectre._tcp.local.` | `62587` | Matter operational and commissioning services |
| Micro-ESPectre | `_espectre._tcp.local.` | `62587` | No secondary service |

Direct uses TCP port `62587` (`0xF47B`), the low 16 bits of Unicode `U+1F47B` GHOST (`👻`), as the shared ESPectre service port. A manually entered endpoint may still specify another explicit port, but clients do not probe legacy ports automatically.

ESPHome continues to publish `_esphomelib._tcp.local.`, and Matter continues to publish its standard operational and commissioning records. Those upstream services are not inputs to `./espectre devices`; the CLI browses only `_espectre._tcp.local.`. Micro-ESPectre publishes its read-only Direct endpoint and unique `.local` hostname through the same service.

### Canonical advertisement

Every publishing frontend uses the following TXT contract. `<device_id>` is the canonical 16-character lowercase hexadecimal ESPectre device ID. The SRV port and `path` locate the Direct HTTP endpoint; frontend-native services remain separate.

| TXT key | Published value | Meaning |
| --- | --- | --- |
| `device_id` | `<device_id>` | Stable ESPectre protocol identity |
| `name` | Frontend-owned display name | User-facing display name; Native uses the saved label when present |
| `frontend` | `native`, `esphome`, `matter`, or `micro` | Frontend discriminator |
| `txtvers` | `1` | DNS-SD TXT schema version |
| `protovers` | `1.0` | ESPectre application protocol version |
| `transport` | `http` | Local Direct transport |
| `path` | `/espectre/v1/request` | JSON request endpoint |
| `events` | `/espectre/v1/events` | SSE event endpoint |
| `firmware` | Current firmware build identity | Running firmware version |
| `chip` | Active ESP-IDF target, such as `esp32c3` | Hardware target |
| `capabilities` | Comma-separated values | Coarse discovery capabilities; clients still negotiate exact Direct methods after connecting |

### Version ownership

The DNS-SD wire keys follow RFC 6763. `txtvers` versions only the key/value profile of this TXT record, while `protovers` advertises the application protocol implemented by the discovered service. `protovers` must therefore be serialized from the same canonical version constant exposed as `protocol_version` in ESPectre JSON messages; it is not an independent Direct version. Internal constant names remain descriptive even though the DNS-SD keys stay compact. Binary raw-CSI framing has its own independent version because it is not part of the canonical JSON message model.

No earlier ESPectre DNS-SD profile was released. The first public baseline is `txtvers=1` and application protocol `1.0`; superseded development values do not consume compatibility versions or require aliases.

The CLI accepts a record only when it resolves to IPv4, has a valid SRV port and `device_id`, reports a supported `frontend`, declares `txtvers=1`, `protovers=1.0`, and `transport=http`, and publishes the exact request and event paths above. Unknown TXT keys are ignored so the contract can grow additively. `name`, `firmware`, `chip`, and `capabilities` enrich the normalized result but do not identify the device.

### Frontend-specific behavior

- Native owns its responder, uses the stable hostname `espectre-<device_id>.local`, publishes the generated name when its label is empty, and updates the TXT `name` after a saved label change.
- ESPHome adds `_espectre._tcp` to the responder already owned by ESPHome, publishes the generated name when its ESPectre-only label override is empty, and advertises Direct on the shared port `62587`. Direct mutations update the shared runtime first, then republish the corresponding ESPHome number and select entities so Home Assistant stays aligned.
- Matter adds `_espectre._tcp` to the responder already owned by the Matter stack only after a fabric has been commissioned, initializes the Basic Information `NodeLabel` as empty unless configured otherwise, and advertises Direct on the shared port `62587`. Removing the last fabric removes the ESPectre service and stops Direct; commissioning it again restores both. Direct provides detector selection and tuning that the standard Matter occupancy surface does not expose.
- Micro-ESPectre owns a single `_espectre._tcp` advertisement for its bounded, read-only Direct surface. It exposes monitoring queries and telemetry, but no configuration mutations, raw CSI stream, or peer-discovery responder.

Services are enabled only while the station interface has a usable IPv4 address. A frontend that owns its responder sends a best-effort goodbye on a clean disconnect and reannounces after reconnect or an IP-address change; ESPHome and Matter retain responder lifecycle ownership and ESPectre only adds or removes its own service.

Ordinary web applications cannot enumerate DNS-SD services, even when the host resolver can resolve individual `.local` names. Configure and Monitor therefore resolve a one-shot hostname to reach one eligible device, which performs the DNS-SD browse on the browser's behalf. The portal still accepts manual private IP or `.local` entry, remembered endpoints, and credential-free QR or share links.

### Peer-assisted browser discovery

For each Auto-discovery attempt, the portal generates 96 bits with Web Crypto and resolves one lowercase bootstrap hostname in the form `espectre-devices-<24 hexadecimal characters>.local`. Because each attempt uses a distinct DNS cache key, cached positive or negative answers for an earlier bootstrap name cannot satisfy the new lookup. Native, Matter, and ESPHome answer only valid, uncompressed class-IN A or AAAA questions matching that form. An A response repeats the requested owner name, contains the current station IPv4 address, uses a 10-second TTL, clears the cache-flush bit so simultaneous responders contribute shared records, and adds an NSEC record whose bitmap declares A but not AAAA. A standalone AAAA question receives only the same NSEC assertion, so dual-stack resolvers can proceed to A without waiting for an IPv6 timeout; the responder never advertises an IPv6 address. The browser then sends `discover_peers` to port `62587` at `/espectre/v1/request` with the same exact Origin policy as the unique Direct endpoint and a 10-second client timeout. The responder is stateless: it does not register, retain, announce, or send a goodbye for the nonce hostname. It accepts multicast, QU, and legacy-unicast queries, keeps at most four delayed multicast answers at 25, 50, 75, and 100 ms, and schedules at most eight answers per second. Pending answers are discarded on an IPv4 change, Wi-Fi disconnect, or reconfiguration.

The static `espectre-devices.local` alias is intentionally unsupported, and there is no automatic compatibility fallback between portal and firmware versions using different bootstrap contracts. Micro-ESPectre does not implement the nonce responder; another ESPectre frontend may still return its DNS-SD record, while manual private IP and unique `.local` hostname entry work directly. IPv6 remains outside the supported peer-assisted discovery boundary; manual private IP, unique hostname, and device-ID entry are unchanged.

After the normal capability handshake, an eligible responder advertises the read-only `discover_peers` method. The request accepts only an empty object. It runs one asynchronous PTR browse for `_espectre._tcp.local.` with a fixed 3,000 ms query window; a second request while that operation is active receives `conflict`, and a start failure receives `unavailable`. The result is correlated to the requesting connection and request ID. A disconnect prevents later delivery and does not create a waiter or persistent peer inventory.

The production boundary is IPv4-only and includes the requesting Native, ESPHome, or Matter device even when the Espressif query API omits its own advertisement. Results are deduplicated by the canonical 16-character lowercase hexadecimal `device_id`. Records for the same identity and endpoint merge and sort their addresses; conflicting endpoints reject that identity. Identities sort lexicographically. Returned IPv4 addresses must be unicast and on-link under the active station netmask; unspecified, network, broadcast, loopback, multicast, and off-link addresses are rejected. Firmware accepts only canonical Native, ESPHome, Matter, and Micro records and returns no credentials, configuration secrets, telemetry, CSI, or broker details. Discovery TXT capabilities are presentation hints only. After selecting an endpoint, a client must perform the normal Direct `capabilities` handshake and use the returned method catalog to expose or suppress configuration, sensing, tuning, traffic-control, and OTA operations. The frontend label is not an authorization or UI feature gate.

The fixed limits are eight accepted devices, two IPv4 addresses per device, eight unique capability tokens, 32 characters per capability token, 128 characters for the capability list, 63 characters each for service instance, hostname, and display name, 48 characters for firmware, 16 characters for frontend and chip, and 3,584 bytes for the result object. `txtvers`, `protovers`, and `transport` must equal `1`, `1.0`, and `http`; `path` and `events` must equal `/espectre/v1/request` and `/espectre/v1/events`. Frontend must be `native`, `esphome`, `matter`, or `micro`, and the SRV port must be non-zero. Invalid records increment `rejected_results`; device, address, or serialization limits set `truncated` and retain deterministic leading results.

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
      "dns_sd_schema_version": 1,
      "protocol_version": "1.0",
      "transport": "http",
      "path": "/espectre/v1/request",
      "events": "/espectre/v1/events",
      "firmware": "3.0.0-rc1",
      "chip": "esp32c3",
      "port": 62587,
      "capabilities": ["config", "monitor", "ota", "peer_discovery"],
      "addresses": ["192.168.1.29"]
    }
  ]
}
```

The portal validates the complete result again before rendering or constructing an endpoint, remembers only the selected unique address, and never stores the shared alias or peer list. Alias resolution, handshake, query, and selection failures return to the existing manual and remembered endpoint paths within the client timeout.

## Transports

### One message model, multiple transports

ESPectre protocol `1.0` defines one JSON request, one correlated result shape for success and error, and one payload for each event family. A transport may add framing and delivery metadata, but it does not rename, nest, unwrap, or otherwise translate application fields.

| Message | Direct HTTP | MQTT |
| --- | --- | --- |
| Request | POST body at `/espectre/v1/request` | Payload on `commands/request` |
| Result or error | HTTP response body | Payload on `commands/result` |
| Event | SSE `event:` names the family and `data:` contains the payload | Topic suffix names the family and the payload is unchanged |

[`ARCHITECTURE.md`](ARCHITECTURE.md#shared-protocol-and-transport-services) owns implementation placement, command-engine ownership, task boundaries, and parity gates. A change to the shared message model must update every language implementation, while a transport mapping change updates only its owning adapter without changing the canonical application shape.

### Direct HTTP transport

This section defines the common local transport used by the C++ frontends. Native owns the complete local profile; ESPHome and Matter use the shared bridge and advertise a filtered command intersection. The durable direction is recorded in `docs/adr/2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md` and `docs/adr/2026-07-02-use-one-message-model-and-command-engine-across-transports.md`.

Direct exposes `POST /espectre/v1/request` with `application/json`, `GET /espectre/v1/events` with `text/event-stream`, and `GET /espectre/v1/csi` with `application/octet-stream`. Requests are limited to 4,096 bytes, and correlated JSON results and SSE event payloads are limited to 8,192 bytes. The production default accepts at most 20 Direct POST requests per one-second window before returning HTTP 429; the existing mutation limit applies independently. Malformed JSON framing, unsupported media types, disallowed Origins, oversize input, saturation, and internal failures produce an appropriate HTTP status. A syntactically valid JSON object reaches the canonical command validator, so unsupported versions, invalid identifiers, unknown parameters, and command rejection return the same correlated result object used on MQTT. Micro-ESPectre advertises only `capabilities`, `info`, `status`, `config`, and `diagnostics`, publishes only the `telemetry` event family, exposes the `runtime`, `device`, and `wifi` configuration sections, and accepts no mutations.

Every Direct response sets `Cache-Control: no-store`. Browser calls use `targetAddressSpace: "local"`, and the server handles CORS preflight and Private Network Access. `Access-Control-Allow-Origin` echoes only an exact configured Origin and is paired with `Vary: Origin`; no wildcard Origin is accepted. Native, Matter, and ESPHome serve Direct on port `62587`; ESPHome's listener remains independent of its port-80 web server and native API.

The POST body is the canonical request also published to MQTT `commands/request`. Command parameters are top-level fields:

```json
{"protocol_version":"1.0","command_id":"req-42","command":"set_threshold","threshold":0.42}
```

`protocol_version` is the string `1.0`. `command_id` is a non-empty client-generated correlation identifier of at most 64 ASCII letters, digits, `.`, `_`, `-`, or `:`. `command` selects one operation from the capability catalog. Each operation's `params` schema governs every remaining top-level field; duplicate fields, reserved-field collisions, unknown parameters, wrong types, malformed JSON, and unsupported versions are rejected.

A successful result has the same shape in an HTTP response body and on MQTT `commands/result`:

```json
{"protocol_version":"1.0","device_id":"0123456789abcdef","command_id":"req-42","command":"set_threshold","accepted":true,"code":"ok","message":"threshold updated"}
```

A rejected command changes `accepted` and `code` without changing the message shape:

```json
{"protocol_version":"1.0","device_id":"0123456789abcdef","command_id":"req-42","command":"set_threshold","accepted":false,"code":"invalid_params","message":"threshold must be between 0.0 and 1.0"}
```

Query results and actions that return structured output add a `data` object. The stable v1 error codes are `unsupported_version`, `invalid_params`, `unsupported`, `forbidden`, `busy`, `busy_raw_collection`, `not_raw_session_owner`, `conflict`, `unavailable`, and `internal_error`; `ok` identifies an accepted command. Transport-level HTTP failures can occur before a canonical result exists. Human-readable `message` text is diagnostic and must not drive client behavior.

Events are complete canonical payloads. MQTT publishes the payload to its event topic; SSE uses the same event name in `event:` and the unchanged JSON payload in `data:`:

```text
event: telemetry
data: {"protocol_version":"1.0","device_id":"0123456789abcdef","frontend":"native","timestamp_ms":1000,"motion_state":"motion","movement_score":0.18,"threshold":0.42,"detector":"lightweight","health":{"uptime_s":1}}
```

The canonical event names are `telemetry`, `status`, `info`, `config`, `ota_status`, and `fault`. Diagnostics and command results are correlated results, not events. The service emits a heartbeat comment every 10 seconds, supports at most two subscribers, coalesces replaceable telemetry in a fixed per-client queue, and disconnects slow clients. There is no replay. After reconnecting with the 500 ms, 1.5 s, and 3 s retry sequence, the web client repeats capability negotiation and refreshes current state.

Direct methods are grouped by capability:

| Capability | Methods | Behavior |
| --- | --- | --- |
| Base reads | `capabilities`, `info`, `status`, `config` | Available to every compatible client. Native, Matter, and ESPHome expose `runtime`, `device`, and read-only `wifi` sections; Native additionally exposes non-secret MQTT configuration. Passwords are never returned. |
| Diagnostics | `diagnostics` | Returns the latest bounded runtime and transport diagnostics sample. |
| Device configuration | `set_device_label` | Native persists its saved device label, Matter maps it to the Basic Information `NodeLabel`, and ESPHome persists an ESPectre-only override without changing its hostname, YAML, or entity IDs. The shared maximum is 32 UTF-8 bytes. |
| Wi-Fi access-point selection | `wifi_access_points`, `scan_wifi_access_points`, `set_wifi_bssid`, `clear_wifi_bssid` | Native, Matter, and ESPHome return BSSID, channel, and RSSI for scan results. `set_wifi_bssid` pins one BSSID; `clear_wifi_bssid` restores automatic access-point selection without removing the SSID or password. Native and ESPHome suspend sensing, verify the new association and address acquisition, persist the pin, reset the CSI session, and recalibrate without rebooting. Failed updates restore the last-known-good selection. ESPHome keeps its pin separate from YAML. Matter applies the pin to the current station session. |
| Native-owned configuration | `clear_wifi_config`, `set_mqtt_config`, `clear_mqtt_config` | Native alone owns removal of provisioned Wi-Fi credentials, MQTT settings, and write-only MQTT secrets. `clear_wifi_config` disconnects Direct HTTP and returns the device to Improv Serial provisioning. |
| Sensing | `set_sensing`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate` | Available only when advertised. `set_sensing` carries the required Boolean `enabled` parameter and does not require MQTT. |
| CSI traffic | `set_csi_traffic_mode`, `set_traffic_generator_mode` | Available only when the runtime advertises traffic control. |
| OTA | `ota_status`, `ota_check`, `ota_start` | Uses the same channel and no-override policy as MQTT OTA commands. |
| Peer discovery | `discover_peers` | Advertised by Native, Matter, and ESPHome through the shared bounded peer-assisted discovery service and deferred transport. |
| Raw CSI | `start_raw_stream`, `stop_raw_stream` | Advertised by ESPectre frontends when the runtime and Direct transport support an owner-bound binary raw session. |

#### Direct raw CSI v2

Raw CSI is an additive capability of the HTTP service shared across supported ESPectre frontends. Capability negotiation reports `features.raw_csi=true` and a `raw_csi` object containing endpoint `/espectre/v1/csi`, transport `http`, raw protocol version `2`, record version `8`, a 60-byte frame prefix, a 16-record ring, a four-record maximum chunk batch, the external UDP port, and `marker: "👻"`.

`start_raw_stream` accepts an empty object, creates a random 128-bit session ID, and moves the runtime from `sensing` to `raw_collection` without changing `csi_traffic_mode` or the active traffic generator. `GET /espectre/v1/csi` and `stop_raw_stream` require `Authorization: Bearer <session-id>`. Another start receives `busy_raw_collection`, and a stop with the wrong bearer receives `not_raw_session_owner`. Reads remain available during the session. Wi-Fi, OTA, detector, calibration, traffic, and sensing mutations receive `busy_raw_collection`. Stream abort, Wi-Fi loss, channel or BSSID change, the five-second initial bind timeout, reboot, stop, or fault terminates the session, restores sensing, and leaves persisted traffic configuration unchanged. Once bound, inactivity does not cause an application timeout; TCP keepalive owns inactive-connection detection.

The raw endpoint accepts one bearer-bound collector. The CSI callback classifies traffic before the sensing/raw split. In raw mode, every classified frame bypasses `TemporalCsiSampler` and is offered to the 16-record ring advertised by the capability catalog. The 64-bit stream sequence is assigned before enqueue. A full ring drops the newest record, preserving queued order and creating a visible sequence gap. The service drains continuously and sends up to four complete records in each HTTP chunk. There is no polling interval, credit message, timer, rate limiter, freshness replacement, or application heartbeat; inactive sessions rely on TCP keepalive.

Each record starts with a packed 60-byte little-endian HTTP prefix containing magic `ESPR`, raw protocol version `2`, CSI record version `8`, prefix length, session ID, stream sequence, record length, zero reserved flags, and cumulative sent-record, raw-drop, and send-backpressure counters. One CSI V8 record follows immediately. Ring overflow, records popped for a failed send, and records still queued when the session closes count toward `raw_drop_total`. After the queue is drained, `fresh_record_total + raw_drop_total == classified_frames_offered_to_raw`. Clients must reconstruct records across arbitrarily split or aggregated HTTP chunks and reject invalid magic, version, length, session, non-monotonic sequence, flags, mismatched V8 sequence, or record data.

Version 8 raw records use a transport-neutral 64-byte packed header. `device_ticks_us` is the capture timestamp, and the final counters are `transport_backpressure_total`, `fresh_record_total`, and `request_accepted_total`. Current HTTP sessions emit V8 only; the host parser retains V7 read compatibility for historical captures.

Native accepts at most two Direct SSE subscribers and one independent raw collector. Outside raw collection, any authorized requester may issue mutations, and the last accepted mutation becomes current state. A query responds only to its requesting transport, while state transitions caused by a mutation are broadcast after commit. This is an explicit multi-writer policy with the documented raw-session ownership exception.

MQTT and Direct retain independent backpressure. Telemetry may coalesce to the newest value, state transitions are never overwritten by telemetry, and a Direct subscriber that repeatedly fails to drain is closed. Correlated command responses use the originating request path rather than an event queue. The internal task, queue, and worker ownership that implements these rules is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md#shared-protocol-and-transport-services).

The server accepts exact portal Origins `https://espectre.dev`, `https://www.espectre.dev`, and `https://test.espectre.dev`. A development-only Kconfig option additionally accepts HTTP Origins on any port only when the host is exactly `localhost`, `127.0.0.1`, or `[::1]`; lookalike hosts, paths, userinfo, invalid ports, and HTTPS loopback Origins remain rejected. Published firmware disables the loopback exception. Requests without an `Origin` header are rejected by default; a non-browser integration requires an explicit build-time policy. The server limits connection count, frame size, total request and mutation rates, and queue depth, binds only after the station interface has a usable address, and stops on address loss.

ESPectre frontends advertise this endpoint through the [mDNS/DNS-SD discovery contract](#mdnsdns-sd-discovery). They share identity, Wi-Fi status, access-point scan and BSSID pinning, runtime controls, diagnostics, peer discovery, and raw CSI. Native additionally owns Wi-Fi credential reset, MQTT, and OTA. Clients must use the returned capability catalog instead of assuming that every frontend implements every method.

Direct carries ESPectre protocol `1.0`, advertised as `protovers=1.0` and serialized as `protocol_version: "1.0"`. Direct has no separate application version or JSON envelope. New optional fields require an updated command or event schema; removing or reinterpreting a field, changing correlation or result semantics, or changing a required type requires a new ESPectre protocol version. Home Assistant Discovery remains MQTT-only.

The SSE stream emits a comment heartbeat every 10 seconds; it adds no JSON heartbeat message. Each POST has a client-side timeout. The portal treats an ended SSE response as loss of liveness, aborts pending requests, and attempts reconnect after 500 ms, 1.5 seconds, and 3 seconds before returning to manual connection. A reconnect repeats capability negotiation and refreshes `info`, `status`, and `config` before resuming the session.

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

### Home Assistant MQTT Adapter Profile

Native can publish an additive Home Assistant MQTT Discovery surface without changing the canonical ESPectre topics above. Discovery payloads use the standard `{discovery_prefix}/{component}/{object_id}/config` topic shape. Native also retains its canonical `status` payload so late subscribers receive the current availability; entity-shaped state topics remain non-retained under `espectre/v1/devices/{device_id}/ha/...`.

The Native HA adapter publishes sensing entities that match the ESPHome Home Assistant surface so one dashboard can be reused after replacing the device prefix: Motion Detected on filtered state edges, Movement Score on every detector evaluation (`evaluation_interval_ms`), writable Threshold on operator writes, calibration, and Lightweight settled-level recovery, Motion On Hits and Motion Off Hits numbers, a Detection Profile select where the frontend supports runtime detector switching, CSI Traffic Ownership plus CSI Traffic Source selects where the frontend supports traffic control, a configuration-category Recalibrate button that starts startup recalibration, a diagnostic-category Calibration Active binary sensor that reports the authoritative runtime state, and the ESPHome CSI diagnostic sensors plus a Refresh Diagnostics button that publishes the latest cached sample on demand. Native discovery `object_id` suffixes follow the ESPHome entity-ID slugs (`motion_detected`, `movement_score`, `recalibrate`, `calibration_active`, and so on); MQTT state and command topic suffixes under `ha/` stay unchanged. Canonical `telemetry` JSON keeps `movement_score` and `threshold` on that same evaluation cadence. Leftover Intensity and previous Native discovery object IDs are unpublished with empty retained configs.

The adapter subscribes to `homeassistant/status` and republishes discovery when Home Assistant announces `online`; this birth message is a recovery trigger, not the only discovery bootstrap. Native derives availability from the retained canonical `status` payload and its retained Last Will. The adapter is enabled in the published firmware defaults and can be disabled at build time. See [`README.md`](../src/cpp/frontend/native/README.md) for its entity and configuration surfaces. Micro-ESPectre has no MQTT or Home Assistant adapter; its sensing state is available through Direct HTTP and SSE.

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

Every shipped frontend makes telemetry available on each detector evaluation once `ready_to_publish` is true, but suppresses the high-rate serialization path when it has no consumer. Native treats MQTT or a Direct SSE client as a consumer, ESPHome also treats its Movement Score entity as one, and Matter and Micro-ESPectre enable the stream for a Direct SSE client. Filtered motion-state transitions update the ecosystem motion entity immediately without forcing another detector evaluation. The C++ and Micro runtimes use a fixed one-second heartbeat for status logs and diagnostics sampling; it is not configurable and does not control sensing telemetry.

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
  "evaluation_interval_ms": 250
}
```

`info` contains identity, firmware, chip, frontend, timing, and non-sensitive descriptive data. Capability booleans are not duplicated here; clients consume the `capabilities` schema. Native publishes `info` retained over MQTT on connect and after an accepted label change so late subscribers, including `./espectre mqtt` discovery, see the current frontend identity. Direct clients request the current value from Native or Micro explicitly. `network` and `detection` are optional. Canonical MQTT `info` reports the active Wi-Fi channel when available, but does not serialize the local IP address or station MAC. `csi_traffic_mode`, `traffic_mode`, and `csi_target_pps` are included when the frontend owns CSI traffic configuration; omit them when those values are unset. `evaluation_interval_ms` is the detector and sensing-telemetry cadence; omit it when unset. Nearby setup and local logs may still expose configuration or link details, including SSID, BSSID, local IP, station MAC, broker host, or broker username. Managed services should not collect those values by default.

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

`runtime` is the uniform cross-frontend configuration section. `device`, `wifi`, and `mqtt` are optional and are included only when both the frontend and requesting transport authorize them. Native Direct limits `wifi` to `configured`, read-only `ssid`, read-only active `band` (`2g`, `5g`, or empty when unknown), `bssid`, `apply_state`, and `apply_message`; password, channel, and band policy are not part of the Direct configuration surface. Passwords and other secrets are write-only and never appear in a response. Runtime tuning changes publish one `config` state transition; a label change publishes `info`; sensing state publishes `status`; OTA state publishes `ota_status`; and recalibration publishes `status`, followed by `config` only when the resulting threshold changes.

### Diagnostics

Diagnostics are returned only as `data` in the correlated response to an explicit `diagnostics` query; there is no diagnostics event or MQTT topic. The following representative Native response includes the canonical runtime fields and its optional transport objects:

```json
{
  "protocol_version": "1.0",
  "device_id": "3cf79180d3a0aca4",
  "timestamp_ms": 123456,
  "uptime": 3821,
  "free_memory_kb": 182.4,
  "minimum_free_memory_kb": 151.8,
  "largest_free_memory_kb": 94.1,
  "cpu_frequency_mhz": 160,
  "loop_time_ms": 0.31,
  "performance_window_ready": true,
  "performance_window_ms": 10000,
  "runtime_load_percent": 2.5,
  "loop_samples": 10000,
  "loop_avg_us": 200,
  "loop_max_us": 800,
  "detection_timing_supported": true,
  "detection_samples": 40,
  "detection_sum_us": 4000,
  "detection_avg_us": 100,
  "detection_min_us": 80,
  "detection_max_us": 140,
  "csi_classified_total": 42100,
  "csi_provenance_rejected_total": 3900,
  "traffic_tx_pps": 100,
  "csi_callback_pps": 96,
  "csi_accepted_pps": 90,
  "csi_admitted_pps": 84,
  "csi_filtered_pps": 6,
  "csi_pending_frame_drops_total": 0,
  "csi_pending_frames": 0,
  "csi_pending_frame_capacity": 8,
  "csi_pending_frame_drop_pps": 0,
  "csi_missing_slots_pps": 10,
  "csi_excess_pps": 6,
  "csi_stale_pps": 0,
  "csi_out_of_order_pps": 0,
  "csi_occupancy": 0.84,
  "wifi_channel": 10,
  "wifi_rssi_dbm": -55,
  "runtime_motion_event_drops_total": 0,
  "task_stack_high_water_bytes": 2876,
  "direct_http": {
    "event_clients": 1,
    "event_client_limit": 2,
    "queue_capacity": 8,
    "queued_messages": 0,
    "accepted_connections": 4,
    "rejected_connections": 0,
    "malformed_requests": 0,
    "oversized_requests": 0,
    "rate_limited_requests": 0,
    "dropped_telemetry_events": 3,
    "send_failures": 0,
    "slow_client_disconnects": 0
  },
  "raw_csi": {
    "active": false,
    "binary_bound": false,
    "raw_drop_total": 0,
    "send_backpressure_total": 0,
    "fresh_record_total": 0,
    "stream_sequence": 0
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

| Field | Meaning |
| --- | --- |
| `timestamp_ms`, `uptime` | Monotonic device time in milliseconds and whole seconds |
| `free_memory_kb`, `minimum_free_memory_kb`, `largest_free_memory_kb` | Current heap, cumulative low-water heap, and current largest free block |
| `cpu_frequency_mhz` | Resolved firmware CPU frequency |
| `loop_time_ms` | Most recent frontend loop-body cost, excluding the outer task sleep or idle delay |
| `performance_window_ready`, `performance_window_ms` | Whether a complete aggregation window is available and its measured duration; duration is `null` before the first complete window |
| `runtime_load_percent` | Runtime-loop wall-time load over the complete window, or `null` before that window exists |
| `loop_samples`, `loop_avg_us`, `loop_max_us` | Runtime loop sample count, average duration, and maximum duration for the complete window |
| `detection_timing_supported` | Whether the selected runtime evaluates a detector |
| `detection_samples`, `detection_sum_us`, `detection_avg_us`, `detection_min_us`, `detection_max_us` | Detector evaluation aggregates for the complete window, or `null` when unsupported or not ready |
| `traffic_packets_total`, `csi_callbacks_total`, `csi_classified_total`, `csi_provenance_rejected_total`, `csi_accepted_total`, `csi_admitted_total`, `csi_filtered_total`, `csi_pending_frame_drops_total`, `csi_missing_slots_total`, `csi_excess_total`, `csi_stale_total`, `csi_out_of_order_total`, `csi_occupancy_slots`, `csi_window_slots` | Supported cumulative runtime counters and slot counts used to derive rates, provenance rejection, queue overflow, and window occupancy |
| `csi_pending_frames`, `csi_pending_frame_capacity` | Current occupancy and fixed capacity of the callback-to-runtime CSI queue |
| `runtime_motion_event_drops_total` | Cumulative ordered motion-state publications overwritten in the bounded runtime-to-frontend mailbox |
| CSI and traffic fields ending in `_pps`, plus `csi_occupancy` | Cached traffic and CSI rates in packets per second, plus the active detector-window occupancy ratio |
| `wifi_channel`, `wifi_rssi_dbm` | Current Wi-Fi channel and RSSI; unavailable RSSI is `null` |
| `task_stack_high_water_bytes` | Native frontend-task stack headroom; omitted by frontends without an equivalent measurement |
| `direct_http` | SSE client and queue budgets plus cumulative connection, request, delivery, and slow-client counters |
| `raw_csi` | Raw-session state plus cumulative drops, send backpressure, delivered records, and stream sequence |
| `mqtt` | Native MQTT connection, queue, outbox, drop, failure, and reconnect diagnostics |

Memory values use KiB, timings use microseconds unless the field ends in `_ms`, and rates use packets per second. `runtime_load_percent` is runtime-loop wall time divided by the complete aggregation window. C++ runtimes keep the latest complete bounded 10-second performance window available between boundaries; collection does not depend on a build option or periodic debug logger. Unsupported values are `null`, or are omitted when the owning frontend cannot expose that optional measurement. Clients must not synthesize zero for a missing measurement.

Rate fields derive from cumulative counters on the fixed one-second sensing heartbeat. `traffic_tx_pps` is the traffic-generator transmit rate; `csi_callback_pps` is the raw CSI callback rate; `csi_accepted_pps` is the identity-accepted rate; `csi_admitted_pps` is detector input after temporal admission; `csi_filtered_pps` is the capture-filter drop rate; and `csi_pending_frame_drop_pps` is callback-to-runtime queue overflow. The temporal drop fields distinguish missing slots, same-slot excess, stale packets, and out-of-order packets. `csi_occupancy` is the valid fraction of the active detector window and does not change the device send rate. Before the first sample completes, rate fields are zero.

Diagnostics are on-demand, and product dashboards should prefer telemetry, status, and info for normal operation. Motion state, movement score, threshold, detector selection, and turbulence belong to telemetry or live config and info surfaces. Native adds `task_stack_high_water_bytes`, `direct_http`, `raw_csi`, and `mqtt`; ESPHome and Matter return their supported shared measurements through Direct; and Micro-ESPectre returns its canonical subset without transport-specific objects. The extra fields are additive on protocol `1.0`, and consumers may ignore unknown keys. The SDK sample uses `csi_occupancy_ratio` for the same occupancy value exposed as `csi_occupancy` on the wire.

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

The example is abbreviated. The retained payload contains every command executable through that transport and frontend, the canonical event names, available configuration sections, and feature flags. Each command declares `name`, `kind` (`query`, `mutation`, or `action`), `access`, a constrained JSON Schema subset (`type`, `properties`, `required`, `additionalProperties`, `enum`, `minimum`, and `maximum`), and a named `result` schema only when it returns data. Because every request is an object, no-parameter schemas contain only `additionalProperties: false`; empty `properties` and `required` members are omitted. The complete minified catalog must remain below 4 KiB. Clients use it for rendering, validation, help, and completion instead of maintaining verb allowlists.

Access classes are `read`, `control`, `device_admin`, `network_admin`, `firmware_update`, and `discovery`. Native Direct may expose every implemented class. Native MQTT exposes read, control, device administration, and firmware update, including `set_sensing`; Wi-Fi and MQTT configuration and `discover_peers` remain Direct-local. Other frontends publish only the intersection they can execute. C++ and MicroPython keep independent registries; the host probe enforces the exact Micro capability profile and shared serialized-message parity.

### Commands

Published to:

```text
espectre/v1/devices/{device_id}/commands/request
```

Canonical queries are `capabilities`, `info`, `status`, `config`, `diagnostics`, `ota_status`, and, when advertised, `discover_peers`. Canonical mutations and actions are `set_sensing`, `set_threshold`, `set_motion_hits`, `set_detector`, `recalibrate`, `set_csi_traffic_mode`, `set_traffic_generator_mode`, `set_device_label`, the advertised Wi-Fi and MQTT configuration methods, `ota_check`, and `ota_start`. The removed `commands`, `stats`, `start_sensing`, and `stop_sensing` names have no v1 aliases because this contract has not shipped in a stable v3 release.

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

Accepted values are `internal` and `external`. ESPectre frontends persist the accepted value across reboot. Micro-ESPectre advertises no traffic-control mutation; its deployment configuration selects native ICMP traffic or external traffic for the current boot. Runtime requests using removed `pacing` or `disabled` values receive `invalid_params`; persisted legacy values migrate once to `internal`. On ESP-IDF sensing frontends, `external` opens the UDP listener on port `5555`, joins multicast group `239.255.0.1` unless `csi_traffic_multicast_group` is empty, and accepts only the exact four-byte UTF-8 marker `"👻".encode("utf-8")` (`F0 9F 91 BB`) addressed to the device or configured group. A period payload, truncated or malformed UTF-8, and any payload with additional bytes are rejected fail-closed.

Update the internal traffic generator type on frontends that advertise traffic control:

```json
{
  "protocol_version": "1.0",
  "command_id": "cmd-006",
  "command": "set_traffic_generator_mode",
  "traffic_generator_mode": "dns"
}
```

Accepted values are `ping` and `dns`. `ping` selects stateless ICMP echo traffic. `dns` selects length-prefixed DNS queries over one persistent, non-blocking TCP connection to gateway port `53`, so the gateway must accept DNS over TCP. Native persists the accepted value across reboot. The selection is always stored, but only takes effect while `csi_traffic_mode` is `internal`.

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

ESPectre Protocol can be carried by multiple deployment profiles. In the local Native path, [Configure](https://espectre.dev/tools/configure/) hands a newly flashed device to standard Improv Serial for initial Wi-Fi provisioning, then uses Direct HTTP for configuration and recovery. [Monitor](https://espectre.dev/tools/monitor/) uses Direct HTTP for broker-free local sensing. Direct is a trusted-LAN transport; browser support depends on the browser's mixed-content and local-network access policy. MQTT remains available to device integrations, Home Assistant, and the host CLI, but is not a browser Monitor transport.

Web orchestration profiles add identity, tenancy, device claim, state mirrors, history, alerts, and OTA around the same protocol. Current protocol semantics remain here; future product outcomes, relay sequencing, and release gates belong to [ROADMAP.md](ROADMAP.md), while deployed component boundaries belong to [ARCHITECTURE.md](ARCHITECTURE.md).

## Web Orchestration Privacy Boundary

Default web-orchestration telemetry should be derived and minimal:

| Field | Purpose |
|-------|---------|
| `device_id` | Service-scoped opaque identifier |
| `timestamp_ms` | Event or sample time |
| `online` | Device availability |
| `firmware_version` | Fleet visibility and update eligibility |
| `frontend` | `esphome`, `matter`, `native`, `micro`, `custom`, or future frontend label |
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
