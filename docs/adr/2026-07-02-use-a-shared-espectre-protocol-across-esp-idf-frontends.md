# ADR: use a shared espectre protocol across esp-idf frontends

- Status: Accepted
- Date: 2026-07-02
- Recorded: 2026-07-09 (retrospective)

## Context

Once native firmware, Matter, and streamer started to coexist with ESPHome, the project needed a reusable device-facing protocol surface for BLE, MQTT, provisioning, telemetry, OTA status, and command handling. Keeping those pieces inside one frontend would duplicate transport logic and make behavior drift between firmware targets more likely.

The changelog and git history show an explicit extraction of BLE/MQTT protocol behavior from the native frontend into shared runtime services.

## Decision

Treat ESPectre Protocol as a shared runtime service for ESP-IDF frontends, rather than a native-only implementation detail.

Concretely:

- keep the protocol model and transport boundaries in shared runtime code
- share BLE, MQTT, provisioning, and protocol helpers across native and Matter
- let frontends map the shared protocol to their own ecosystem surfaces without re-implementing the transport core

## Alternatives Considered

### Keep BLE and MQTT handling frontend-local

Rejected. That would duplicate transport behavior and increase the risk of inconsistent provisioning, telemetry, and command semantics.

### Build one protocol per frontend

Rejected. Frontends expose different presentations and capability subsets, but share device identity, command, and telemetry semantics.

## Consequences

Benefits:

- BLE, MQTT, provisioning, and OTA-related flows stay aligned across firmware targets
- new ESP-IDF frontends can reuse the same protocol baseline
- frontend code can focus on presentation and integration specifics

Trade-offs:

- shared protocol changes require broader coordination
- protocol abstractions must stay general enough for multiple frontends

## Related

- `docs/adr/2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`
- git commits: `9cd50a48`, `556eda9d`, `2fb6fca0`
