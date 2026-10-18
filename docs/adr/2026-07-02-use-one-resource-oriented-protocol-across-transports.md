# ADR: use one resource-oriented protocol across transports

- Status: Accepted
- Date: 2026-07-02
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-09-03
- Supersedes in part: `2026-07-03-unify-raw-csi-collection-over-http.md` and `2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`

## Context

Native MQTT, Direct HTTP, ESPHome entities, Matter, and Micro-ESPectre each grew their own dispatcher and message shapes. Shared ESP-IDF helpers cut some duplication, but operation names, validation, results, and version fields still drifted at the wire. Transports do need separate queues, because a slow broker and a slow local client are unrelated problems, but that is a reason for separate queues, not for separate message models.

The first unified API then sent every read, change, and action through one `POST /request` endpoint. Raw CSI needed separate start and stop commands, a bearer token, and a bind timeout before the client could open the stream. It worked on every transport, but it duplicated lifecycle concepts, repeated identity and version fields everywhere, made browser caching awkward, and turned transport mechanics into application commands. MQTT had also collected query commands and topics that did not match who owns each resource; diagnostics, discovery, configuration, availability, and high-rate sensing events need different retention and security rules.

## Decision

**One model, multiple transports.** MQTT, Direct HTTP, and any future transport carry the same resources, operations, parameters, result codes, and schemas, under application version `1.0`. An adapter carries the message as is: no renamed fields, no extra envelope. Each transport keeps its own framing and delivery: MQTT owns topics, QoS, retention, and its outbound queue; Direct HTTP owns methods, status codes, headers, SSE framing, local-origin policy, and its queues. Auth, rate limits, and backpressure can differ per transport. DNS-SD `protovers` advertises the application version from the same constant; `txtvers` and the binary raw-CSI framing are versioned separately.

**Resources.** Direct HTTP exposes resources under `/espectre/v1`, with no `/request` route or alias. `GET` reads, `PATCH` updates part of a resource, `PUT` replaces the BSSID pin, `DELETE` removes, and actions are `POST` to sub-resources. The version appears only in capabilities and discovery, and device identity only in the device resource, discovery results, and CSI metadata. On MQTT, `health` is the availability topic and Last Will; device state is retained per resource, while `motion`, faults, and command results are published without retention.

**CSI sessions.** `GET /csi` owns the whole collection session: opening the response starts collection, closing it stops it, and the client takes the session ID from the first frame. Derived sensing events pause on every transport during collection, control events keep flowing, and sensing and readiness come back before derived events resume.

**One engine.** All C++ frontends share one `FrontendCommandEngine`, which owns operation names, validation, access classes, result codes, capability filtering, and change sets. Adapters add framing and access policy, and commands run one at a time on the frontend task. A query answers only the transport that asked. An accepted mutation emits one change per affected state family, fanned out to every active transport. MQTT and each Direct client keep separate outbound queues; there is no extra command queue or worker. Capabilities list commands by name only, and clients build their controls from those names instead of keeping their own allowlists. MicroPython has its own equivalent registry and publishes the subset it supports; host parity tests compare the serialized messages, not just command names.

The public specification lives in [API.md](../API.md) and [DISCOVERY.md](../DISCOVERY.md). The C++ registry and engine are the executable contract.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-02 | Move BLE, MQTT, provisioning, telemetry, OTA, and command helpers from Native into shared runtime services | First shared protocol baseline |
| 2026-08-24 | One C++ command engine for all frontends | Replaced per-frontend dispatchers; envelopes still differed per transport |
| 2026-08-25 | Direct control over HTTP, events over SSE | Framing changed, semantics did not |
| 2026-08-26 | One message model and version for HTTP, MQTT, and MicroPython | Removed the last transport-specific envelopes, added cross-language parity |
| 2026-08-30 | Name-only command descriptors in capabilities | Dropped unused schema metadata |
| 2026-09-03 | Resource-oriented API replaces the `POST /request` RPC and the bearer-bound CSI session | Accepted; pre-release clients migrated with no compatibility period |

## Alternatives considered

- **Protocol handling inside each frontend.** Duplicates validation, state changes, and schemas on every target.
- **One dispatcher and worker per transport.** Separate outbound queues already isolate backpressure; separate command owners would duplicate semantics.
- **Publish every query result on all transports.** Breaks correlation, leaks local data across trust boundaries, and adds noise.
- **One shared outbound queue.** A slow broker would stall Direct clients, and the other way round.
- **Transport-specific JSON envelopes.** Topics, HTTP, and SSE already frame the message; a second envelope duplicates versioning.
- **Keep the single RPC endpoint.** Every resource becomes a command, caching and retention cannot follow resources, and CSI needs its own session protocol.
- **Generate both registries from one artifact.** Not for now: two registries plus parity tests avoid a code-generation step in constrained firmware builds.

## Consequences

Commands, JSON shapes, errors, and state changes behave the same on Native, ESPHome, Matter, and the supported Micro-ESPectre profile, and an ESPHome entity write and a Direct mutation end up in the same runtime state. Browser, CLI, benchmark, and SDK clients work on explicit resources, the three C++ frontends share one dispatcher and one CSI session controller, and MQTT retention, availability, and command security follow resource boundaries. A frontend that lacks a resource leaves it out of capabilities and returns `404`.

The cost is coordination: a new operation touches the schema, registry, engine, transports, clients, tests, and docs together.

## Related

- [API reference](../API.md), [discovery](../DISCOVERY.md), and [architecture](../ARCHITECTURE.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
- Commits `5f757b8a` (shared protocol), `86299569` (command engine), `49af058c` (HTTP and SSE), `901bc8c6` (name-only descriptors), `405122ef` (resource API), `99be41a3` (published contract)
