# ADR: use Classic as the Matter detector default

- Status: Accepted
- Date: 2026-07-28
- Supersedes: 2026-07-15-adopt-persisted-runtime-detector-selection.md

## Context

The shared runtime supports persisted detector selection for frontends with a writable detector control. Matter intentionally exposes no writable detector surface, so its firmware must choose a fixed frontend-owned default.

The persisted-selection ADR originally recorded ML as that default. Before the v3 release candidate, Matter moved to Classic so the read-only occupancy frontend uses the adaptive non-ML detector selected as the platform default. This changes only Matter's fixed default; persisted runtime selection remains active for ESPHome and Native.

## Decision

Use Classic as the fixed detector for the Matter frontend. Keep Matter's detector capability read-only, start Classic calibration during the normal runtime startup flow, and leave persisted runtime detector selection unchanged for ESPHome and Native.

## Alternatives Considered

### Keep ML as the Matter default

Rejected. It would diverge from the platform's default detector without giving Matter users a runtime control for choosing the alternative.

### Add writable Matter detector selection

Rejected. Detector selection is not part of the published Matter occupancy surface, and adding a frontend-specific writable contract is outside the v3 baseline.

## Consequences

Benefits:

- Matter follows the platform's adaptive non-ML default
- ESPHome and Native retain their persisted runtime choice
- the Matter capability surface remains explicit and read-only

Trade-offs:

- Matter inherits Classic's startup calibration and documented long-quiet false-positive limit
- choosing ML on Matter still requires a firmware change

## Related

- `docs/adr/2026-07-15-adopt-persisted-runtime-detector-selection.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- `docs/CHANGELOG.md`
