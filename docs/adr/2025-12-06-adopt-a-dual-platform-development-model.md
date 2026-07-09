# ADR: adopt a dual-platform development model

- Status: Superseded in part
- Date: 2025-12-06
- Recorded: 2026-07-09 (retrospective)

## Context

By the `2.0.0` release, the project had reached a clear split in how it wanted
to evolve:

- an ecosystem-facing firmware path for end users and Home Assistant
- a faster experimental path for algorithm exploration, tuning, and data work

The versioned `2.0.0` changelog makes that strategy explicit. It presents
ESPectre as the production C++ path and Micro-ESPectre as the research and
development path, with each side serving a different purpose.

## Decision

Adopt a dual-platform development model:

- ESPectre C++ firmware is the production-facing path for end users
- Micro-ESPectre is the Python-based R&D path for rapid experimentation

Use the production path to ship stable motion-detection behavior, and use the
Python path to explore algorithms, filters, data collection, and ML-oriented
work without the slower firmware iteration loop.

## Alternatives Considered

### Keep only the standalone firmware path

Rejected. That would make rapid experimentation and data-oriented work slower
and more cumbersome.

### Keep only the Python path

Rejected. The project still needed a production-facing firmware surface that fit
the target device ecosystem.

## Consequences

Benefits:

- the project gained a clear separation between shipping behavior and research
- algorithm and tooling work could move faster without destabilizing production
- contributors could understand which platform to use for deployment versus
  experimentation

Trade-offs:

- the project now had to maintain alignment across two implementations
- parts of this decision were later generalized by the multi-frontend platform
  direction captured in newer ADRs

## Related

- versioned changelog snapshot: `2.0.0:CHANGELOG.md`
- `docs/adr/2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`
- `docs/adr/2025-12-06-adopt-esphome-as-the-production-integration-surface.md`
- `docs/adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`
- git commits: `c971f874`, `6bfc035d`
