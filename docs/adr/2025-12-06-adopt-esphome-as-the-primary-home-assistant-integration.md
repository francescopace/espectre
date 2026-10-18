# ADR: adopt ESPHome as the primary Home Assistant integration

- Status: Accepted
- Date: 2025-12-06
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

ESPectre started as standalone ESP-IDF firmware controlled over MQTT. That proved on-device CSI sensing worked, but it did not feel native to Home Assistant users. Release 2.0.0 ported the firmware to a native ESPHome component and split the project in two: ESPHome and C++ for production motion detection, Micro-ESPectre and Python for research.

## Decision

ESPHome is the primary Home Assistant integration. We ship a first-party ESPHome component and published firmware, expose motion, thresholds, and diagnostics as ESPHome entities, and rely on ESPHome for configuration and OTA.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-11-01 | Standalone ESP-IDF firmware with MQTT | First deployable product |
| 2025-12-06 | ESPHome as the primary end-user integration | Accepted |
| 2026-06-03 | Shared runtime with Native and Matter frontends | ESPHome stays primary for Home Assistant, not for every workflow |

## Alternatives considered

- **Keep standalone ESP-IDF as the main path.** It did not give Home Assistant users a native setup.
- **Offer ESPHome as an optional extra.** It is the path most users take, so it has to be first-class.

## Consequences

Home Assistant users get configuration, OTA, and entities through workflows they already know. In exchange, the ESPHome frontend has to follow ESPHome conventions, and later architecture work had to keep that surface intact while adding other frontends.

## Related

- `2.0.0:CHANGELOG.md`
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- Commit `6bfc035d`
