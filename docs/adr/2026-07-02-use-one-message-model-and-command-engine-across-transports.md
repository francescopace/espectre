# ADR: use one message model and command engine across transports

- Status: Accepted
- Date: 2026-07-02
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26
- Implementation: Complete for protocol `1.0`; cross-language parity covers the message model, DNS-SD versions, and the Micro capability profile, while C++ separately verifies Direct/MQTT mapping

## Context

Native MQTT, Direct HTTP, ESPHome entities, Matter controls, and Micro-ESPectre originally evolved separate dispatchers and transport-specific application shapes. The first shared ESP-IDF services reduced frontend duplication, but shared helpers alone did not prevent operation names, validation, result shapes, or version fields from drifting at the wire boundary.

Transport isolation still matters. Native MQTT and each Direct client need independent bounded queues because broker and local-client backpressure are unrelated. That requirement does not justify separate message models, registries, or command workers.

## Decision

ESPectre follows **one message model, multiple transports**. MQTT, Direct HTTP, and future transports carry the same canonical JSON requests, results, and events, including one application protocol version, correlation fields, operation names, parameter objects, stable result codes, and data schemas. A transport adapter carries that model without renaming fields, adding another application envelope, or mapping results into a transport-specific JSON shape.

Transport framing and delivery policy remain independent. MQTT owns topics, QoS, retention, broker delivery, and its outbound queue. Direct HTTP owns methods, status, headers, request lifetime, SSE framing, local-origin policy, and its queues. Authentication, authorization, rate limits, backpressure, and capability filtering may differ without changing the canonical application model. DNS-SD `protovers` advertises the same application version; `txtvers` and binary raw-CSI framing retain independent version axes.

All C++ frontends use one `FrontendCommandEngine`. Each adapter supplies framing, origin, and access policy, then executes the canonical command serially on the frontend task. MicroPython maintains an equivalent registry and dispatcher because it cannot share the C++ implementation. Host parity gates compare canonical schemas and serialized public messages, not only normalized command names.

The engine owns operation names, parameter validation, access classes, stable result codes, capability filtering, and logical change sets. Canonical queries are `capabilities`, `info`, `status`, `config`, `diagnostics`, and `ota_status`. Mutations use `set_sensing` and the canonical tuning, device, network, OTA, and discovery actions documented in `ESPECTRE_PROTOCOL.md`. The unreleased `commands`, `stats`, `start_sensing`, and `stop_sensing` names have no aliases.

A query returns only to its requesting transport. An accepted mutation emits one logical change per affected state family and fans that state out to active transports. Diagnostics and command results are correlated responses rather than events. MQTT and each Direct client keep separate outbound queues, coalescing, and backpressure; no command queue, worker, or application task is added.

Capability discovery is a filtered, minified schema catalog below 4 KiB. It describes command kind, access, parameter schema, result schema, events, features, and visible configuration sections. Clients derive controls, help, completion, and validation from this catalog instead of maintaining duplicate allowlists.

Protocol `1.0` uses the former MQTT flat request and correlated result shape as the canonical model. Direct POST bodies carry that request unchanged, Direct responses carry the same `commands/result` object as MQTT, and SSE `data:` carries the same event payload published to the corresponding MQTT topic. DNS-SD advertises `txtvers=1` and `protovers=1.0`, with `protovers` serialized from the same constant as JSON `protocol_version`.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-02 | Extract BLE, MQTT, provisioning, telemetry, OTA status, and command helpers from Native into shared ESP-IDF runtime services | Established the first shared protocol baseline |
| 2026-08-24 | Use one C++ command engine across ESPectre frontends | Replaced frontend-local dispatchers while retaining transport-specific envelopes |
| 2026-08-25 | Carry Direct control over HTTP and processed events over SSE | Changed transport framing without changing command semantics |
| 2026-08-26 | Use one canonical message model and version across HTTP, MQTT, and MicroPython | Replaced the remaining transport-specific envelopes and added serialized cross-language parity |

## Alternatives Considered

### Keep protocol handling frontend-local

Rejected. It duplicates validation, state transitions, and public schemas across firmware targets.

### Keep one dispatcher and worker per transport

Rejected. Independent outbound queues already isolate transport backpressure; separate command owners would duplicate semantics.

### Publish every query result on all transports

Rejected. It breaks request correlation, discloses locally visible data across trust boundaries, and creates unsolicited traffic.

### Share one outbound queue for MQTT and Direct

Rejected. A slow broker must not delay Direct requests or SSE subscribers, and a slow Direct client must not delay MQTT.

### Keep transport-specific JSON envelopes

Rejected. Topics, HTTP, and SSE already provide transport framing. A second application envelope duplicates versioning and field mappings.

### Generate both language registries from one artifact

Rejected for now. Independent registries plus executable parity preserve local implementation style without adding a generated build-time dependency to constrained firmware paths.

## Consequences

- command semantics, JSON shapes, errors, and state changes converge across Native, ESPHome, Matter, and the supported Micro-ESPectre profile;
- MQTT and Direct keep transport-specific framing and queues, but not transport-specific application envelopes;
- ESPHome entity writes and Direct mutations converge on the same authoritative runtime state;
- consumers must use `capabilities`, `diagnostics`, `set_sensing`, and `commands/result`; and
- future operations require coordinated schema, registry, policy, engine, transport, consumer, test, and documentation changes.

## Related

- [`../ESPECTRE_PROTOCOL.md`](../ESPECTRE_PROTOCOL.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
