# ADR: use one message model and command engine across transports

- Status: Accepted
- Date: 2026-08-24
- Updated: 2026-08-26
- Implementation: Partial; command semantics are shared, but the MQTT and Direct JSON envelopes still require convergence

## Context

Native MQTT, Native Direct HTTP, the shared Direct bridge, ESPHome entities, Matter controls, and Micro-ESPectre evolved separate dispatchers. They described similar operations with different names, validation, result shapes, and side effects. Direct later added an RPC-style JSON envelope beside the existing MQTT message shape. Sharing only the command engine aligned behavior but left two application envelopes, two version representations, and transport adapters that can still drift at the wire boundary. In Native, independent outbound transport queues are required for backpressure, but independent message models, command workers, and registries are not.

## Decision

ESPectre follows **one message model, multiple transports**. MQTT, Direct HTTP, and future transports carry the same canonical JSON requests, results, and events, including one application protocol version, correlation fields, operation names, parameter objects, stable result codes, and data schemas. The canonical model is transport-neutral: an adapter carries the message without renaming fields, flattening or nesting parameters, adding another application envelope, or mapping the result into a transport-specific JSON shape.

Transport framing and delivery policy remain independent. MQTT owns topics, QoS, retention, broker delivery, and its outbound queue. Direct HTTP owns methods, status, headers, request lifetime, SSE framing, local-origin policy, and its queues. Authentication, authorization policy, rate limits, backpressure, and capability filtering may differ by transport without changing the canonical message model. DNS-SD `protovers` advertises the same canonical application protocol version; `txtvers` and binary raw-CSI framing retain independent version axes.

All C++ frontends use one `FrontendCommandEngine`. Each transport supplies framing, origin, and access policy, then executes the canonical command serially on the frontend task. MicroPython maintains an equivalent registry and dispatcher because it cannot share the C++ implementation; host parity gates compare the canonical schemas and serialized public messages, not only normalized command catalogs.

The engine owns command names, parameter validation, access classes, stable result codes, capability filtering, and logical change sets. Canonical queries are `capabilities`, `info`, `status`, `config`, `diagnostics`, and `ota_status`. The unreleased `commands`, `stats`, `start_sensing`, and `stop_sensing` names are removed without aliases. Mutations use `set_sensing` and the canonical tuning, device, network, OTA, and discovery actions described by the protocol.

A query returns only to its requesting transport. An accepted mutation emits one logical change per affected state family and fans that state out to active transports. Diagnostics and command results are correlated responses rather than events. MQTT and each Direct client retain separate outbound queues, coalescing, and backpressure so a slow broker or client cannot block another transport. No command queue, worker, or application task is added.

Capability discovery is a filtered, minified schema catalog below 4 KiB. It includes command kind, access, parameter schema, result schema, events, features, and visible configuration sections. Clients render controls, help, completion, and validation from this catalog rather than duplicated allowlists. The catalog and canonical message schema are the executable contract; adding an operation requires coordinated registry, transport, frontend, client, test, and documentation changes.

The current MQTT flat request/result shape and Direct `v`/`id`/`method` envelope are migration debt. Before the protocol is frozen for v3, the owning protocol document and implementations must select one canonical request, result, and event shape; migrate firmware, CLI, browser clients, MicroPython, discovery metadata, and tests atomically; and remove the redundant translation path without preserving unreleased aliases.

## Alternatives Considered

### Keep one dispatcher and worker per transport

Rejected. It duplicates semantics and does not improve transport isolation; independent outbound queues already provide that isolation.

### Publish every query result on all transports

Rejected. It breaks request correlation, discloses locally visible data across trust boundaries, and creates unsolicited MQTT traffic.

### Share one outbound queue for MQTT and Direct

Rejected. Backpressure and delivery guarantees differ by transport. A slow broker must not delay Direct requests or SSE subscribers, and a slow Direct client must not delay MQTT.

### Keep transport-specific JSON envelopes

Rejected. Topics, HTTP, and SSE already provide transport framing. A second application envelope duplicates versioning and field mappings, weakens executable parity, and contradicts one message model across transports.

### Generate both language registries from one artifact

Rejected for now. A generated registry would add a build-time source dependency to constrained firmware paths. Independent registries plus an executable parity gate preserve local implementation style while preventing drift.

## Consequences

- command semantics, JSON shapes, errors, and state changes converge across Native, ESPHome, and Matter;
- MicroPython advertises only operations it can execute and is checked against the shared schema and serialized-message parity gates;
- MQTT and Direct keep transport-specific framing and queues, but not transport-specific application envelopes;
- ESPHome entity writes and Direct mutations converge on the same authoritative runtime state;
- consumers must migrate to `capabilities`, `diagnostics`, `set_sensing`, and `commands/result`; and
- future command additions require schema, registry, policy, engine, transport, consumer, and parity updates as one protocol change.

## Related

- `docs/ESPECTRE_PROTOCOL.md`
- `docs/ARCHITECTURE.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-08-24-unify-frontend-discovery-and-direct-control.md`
