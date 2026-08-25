# ADR: use one command engine across frontends

- Status: Accepted
- Date: 2026-08-24

## Context

Native MQTT, Native Direct HTTP, the shared Direct bridge, ESPHome entities, Matter controls, Streamer controls, and Micro-ESPectre evolved separate dispatchers. They described similar operations with different names, validation, result shapes, and side effects. In Native, independent outbound transport queues are required for backpressure, but independent command workers and registries are not. The duplicate control paths made it possible for one transport to accept a value another rejected, for queries to leak side responses onto MQTT, and for clients to maintain stale verb allowlists.

## Decision

All C++ frontends use one `FrontendCommandEngine`. Each transport or ecosystem surface parses its own envelope into `EspectreCommand`, supplies an origin and access policy, executes serially on the frontend task, and maps the structured result back to its envelope. MicroPython maintains an equivalent registry and dispatcher because it cannot share the C++ implementation; a host C++ probe and Python gate compare the normalized catalogs.

The engine owns command names, parameter validation, access classes, stable result codes, capability filtering, and logical change sets. Canonical queries are `capabilities`, `info`, `status`, `config`, `diagnostics`, and `ota_status`. The unreleased `commands`, `stats`, `start_sensing`, and `stop_sensing` names are removed without aliases. Mutations use `set_sensing` and the canonical tuning, device, network, OTA, and discovery actions described by the protocol.

A query returns only to its requesting transport. An accepted mutation emits one logical change per affected state family and fans that state out to active transports. Diagnostics and command results are correlated responses rather than events. MQTT and each Direct client retain separate outbound queues, coalescing, and backpressure so a slow broker or client cannot block another transport. No command queue, worker, or application task is added.

Capability discovery is a filtered, minified schema catalog below 4 KiB. It includes command kind, access, parameter schema, result schema, events, features, and visible configuration sections. Clients render controls, help, completion, and validation from this catalog rather than duplicated allowlists.

## Alternatives Considered

### Keep one dispatcher and worker per transport

Rejected. It duplicates semantics and does not improve transport isolation; independent outbound queues already provide that isolation.

### Publish every query result on all transports

Rejected. It breaks request correlation, discloses locally visible data across trust boundaries, and creates unsolicited MQTT traffic.

### Share one outbound queue for MQTT and Direct

Rejected. Backpressure and delivery guarantees differ by transport. A slow broker must not delay local WebSocket clients, and a slow SSE subscriber must not delay MQTT.

### Generate both language registries from one artifact

Rejected for now. A generated registry would add a build-time source dependency to constrained firmware paths. Independent registries plus an executable parity gate preserve local implementation style while preventing drift.

## Consequences

- command semantics, errors, and state changes converge across Native, ESPHome, Matter, and Streamer;
- MicroPython advertises only operations it can execute and is checked against the shared schema;
- MQTT and Direct keep transport-specific envelopes and queues without owning command behavior;
- ESPHome entity writes and Direct mutations converge on the same authoritative runtime state;
- consumers must migrate to `capabilities`, `diagnostics`, `set_sensing`, and `commands/result`; and
- future command additions require registry, policy, engine, consumer, and parity updates as one protocol change.

## Related

- `docs/ESPECTRE_PROTOCOL.md`
- `docs/ARCHITECTURE.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-08-24-unify-frontend-discovery-and-direct-control.md`
