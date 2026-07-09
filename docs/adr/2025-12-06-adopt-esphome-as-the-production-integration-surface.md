# ADR: adopt esphome as the production integration surface

- Status: Accepted
- Date: 2025-12-06
- Recorded: 2026-07-09 (retrospective)
- Supersedes: 2025-11-01-adopt-standalone-esp-idf-mqtt-firmware-as-the-initial-product-shape.md

## Context

The `2.0.0` release changelog describes a major architectural shift from
standalone ESP-IDF firmware to a native ESPHome component. That release did not
present ESPHome as a minor packaging tweak; it framed it as the new
production-facing integration surface for smart-home users.

This change also clarified the product split: ESPHome/C++ for production motion
detection, and Micro-ESPectre/Python for experimentation and research work.

## Decision

Adopt ESPHome as the primary production integration surface for end users.

Concretely:

- ship the main user-facing firmware path as an ESPHome external component
- map motion detection, thresholds, and diagnostics into ESPHome/Home Assistant
  entities and workflows
- treat Home Assistant integration and ESPHome OTA/configuration flows as core
  parts of the deployment experience

## Alternatives Considered

### Continue with standalone ESP-IDF firmware as the main production path

Rejected. The project wanted a more native smart-home deployment experience and
closer alignment with Home Assistant users.

### Treat ESPHome as only an optional integration

Rejected. The release explicitly promoted ESPHome to the main production-facing
path rather than a side integration.

## Consequences

Benefits:

- the deployment path became much more natural for Home Assistant users
- configuration, OTA, and entity exposure aligned with the target ecosystem
- the project could separate production integration concerns from R&D work more
  clearly

Trade-offs:

- the project had to respect ESPHome conventions in the production-facing
  frontend
- later architectural work had to preserve the ESPHome surface while enabling
  additional frontends

## Related

- versioned changelog snapshot: `2.0.0:CHANGELOG.md`
- `docs/adr/2025-12-06-adopt-a-dual-platform-development-model.md`
- `docs/adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`
- git commit: `6bfc035d`
