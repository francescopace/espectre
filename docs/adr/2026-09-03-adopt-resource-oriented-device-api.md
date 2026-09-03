# ADR: adopt a resource-oriented device API

- Status: Accepted
- Date: 2026-09-03
- Supersedes in part: `2026-07-02-use-one-message-model-and-command-engine-across-transports.md`, `2026-07-03-unify-raw-csi-collection-over-http.md`, and `2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`

## Context

The first Direct API carried every read, mutation, and action through one `POST /request` RPC endpoint. Raw CSI then required separate start and stop commands, a bearer, and a bind timeout before a client could open the binary response. The model worked across transports, but it duplicated lifecycle concepts, repeated identity and version fields, made browser resource caching awkward, and exposed transport mechanics as application commands.

MQTT also accumulated query commands and topic payloads that did not match their resource ownership. Diagnostics, discovery, configuration, availability, and high-rate sensing events need different retention and security policies.

## Decision

Keep application version `1.0`, but replace the unreleased API with resource-oriented Direct HTTP under `/espectre/v1`. Reads use `GET`, partial resource updates use `PATCH`, replacement-style BSSID selection uses `PUT`, removal uses `DELETE`, and actions use subordinate `POST` resources. There is no `/request` route or compatibility alias.

Keep one canonical resource and operation model across HTTP and MQTT. Publish version only in capabilities and discovery, and publish device identity only in the device resource, discovery results, and binary CSI metadata. Use `health` as MQTT availability and Last Will. Publish retained device state by resource; publish per-evaluation `motion`, faults, and correlated command results without retention.

Make `GET /csi` own the entire exclusive collection lifetime. The TCP response starts collection, its close stops collection, and the client adopts the first binary frame's session ID. Pause all derived sensing events on every transport during collection, keep control events active, and restore sensing plus readiness before derived events resume.

Split the public specification into [API.md](../API.md) and [DISCOVERY.md](../DISCOVERY.md). The C++ registry and command engine remain the executable contract, and Micro publishes its supported read-only intersection.

## Consequences

- browser, CLI, benchmark, and SDK clients operate on explicit resources;
- the three C++ frontends share one dispatcher and one CSI session controller;
- MQTT retention, availability, and command security follow resource boundaries;
- explicit raw-session commands, bearer binding, per-message version fields, and redundant identity fields disappear;
- unsupported frontend resources are absent from capabilities and return `404`; and
- pre-release clients using the earlier RPC model must migrate without a compatibility period.

## Related

- [`../API.md`](../API.md)
- [`../DISCOVERY.md`](../DISCOVERY.md)
- [`2026-07-02-use-one-message-model-and-command-engine-across-transports.md`](2026-07-02-use-one-message-model-and-command-engine-across-transports.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
