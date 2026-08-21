# ADR: adopt esphome as the production integration surface

- Status: Accepted
- Date: 2025-12-06
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-12

## Context

The `2.0.0` release moved the primary smart-home integration from standalone ESP-IDF firmware to a native ESPHome component.

This change also clarified the product split: ESPHome/C++ for production motion detection, and Micro-ESPectre/Python for experimentation and research work.

The first product shape was a standalone ESP-IDF firmware controlled through MQTT. That path established on-device CSI sensing and the host tooling, but the 2.0.0 direction moved the primary end-user integration to ESPHome. Later native, Matter, and streamer frontends complement this surface without replacing ESPHome's role for Home Assistant users.

## Decision

Adopt ESPHome as the primary production integration surface for end users.

Concretely:

- ship the main user-facing firmware path as an ESPHome external component
- map motion detection, thresholds, and diagnostics into ESPHome/Home Assistant entities and workflows
- treat Home Assistant integration and ESPHome OTA/configuration flows as core parts of the deployment experience

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-11-01 | Ship standalone ESP-IDF firmware with MQTT as the initial product | Established the first deployable product shape |
| 2025-12-06 | Make ESPHome the primary end-user integration | Accepted while later frontends reuse the shared runtime |

## Alternatives Considered

### Continue with standalone ESP-IDF firmware as the main production path

Rejected. The project wanted a more native smart-home deployment experience and closer alignment with Home Assistant users.

### Treat ESPHome as only an optional integration

Rejected. The release explicitly promoted ESPHome to the main production-facing path rather than a side integration.

## Consequences

Benefits:

- Home Assistant users gained configuration, OTA, and entities through established ESPHome workflows
- the project could separate production integration concerns from R&D work more clearly

Trade-offs:

- the project had to respect ESPHome conventions in the production-facing frontend
- later architectural work had to preserve the ESPHome surface while enabling additional frontends

## Related

- versioned changelog snapshot: `2.0.0:CHANGELOG.md`
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- git commit: `6bfc035d`
