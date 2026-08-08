# ADR: adopt standalone esp-idf mqtt firmware as the initial product shape

- Status: Superseded
- Date: 2025-11-01
- Recorded: 2026-07-09 (retrospective)
- Superseded by: 2025-12-06-adopt-esphome-as-the-production-integration-surface.md

## Context

The earliest tagged release, `1.0.0`, presents ESPectre as a standalone ESP32-S3 firmware built on ESP-IDF and exposed primarily through MQTT over Wi-Fi. The initial product shape combined on-device CSI motion detection, runtime configuration via MQTT, and supporting CLI and serial tools.

That architecture made sense for the first stage of the project: it gave the project a working end-to-end sensing product before the later move toward ESPHome and broader frontend separation.

## Decision

Use standalone ESP-IDF firmware with MQTT-over-Wi-Fi as the initial production shape of the project.

Concretely:

- keep motion detection on the device
- publish runtime state and movement data over MQTT
- expose runtime configuration and control through MQTT commands
- support the workflow with CLI and serial monitoring tools

## Alternatives Considered

### Start directly as an ESPHome component

Rejected. At the time, the project first needed a working standalone firmware path before committing to a smart-home-native integration surface.

### Keep the project as an analysis-only or host-side tool

Rejected. The initial goal was to ship a device-side sensing system, not only a research or analysis workflow.

## Consequences

Benefits:

- the project gained a complete standalone sensing product quickly
- MQTT gave an immediate integration and observability surface
- the initial firmware architecture created the base from which later frontends could evolve

Trade-offs:

- ecosystem-specific integration concerns remained coupled to the standalone firmware shape
- this model was later superseded when ESPHome became the primary production surface

## Related

- versioned changelog snapshot: `1.0.0:CHANGELOG.md`
- git commit: `942d3b8e`
