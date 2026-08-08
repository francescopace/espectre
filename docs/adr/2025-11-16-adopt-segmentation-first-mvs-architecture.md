# ADR: adopt segmentation-first mvs architecture

- Status: Superseded
- Date: 2025-11-16
- Recorded: 2026-07-09 (retrospective)
- Superseded by: 2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md

## Context

The first project versions used a more complex detection and calibration stack, including a multi-state detection engine, dedicated statistics modules, and a broader configuration surface.

The `1.2.0` release changelog records a deliberate simplification: the project removed the older calibration and detection layers and focused the runtime on Moving Variance Segmentation (MVS) as the primary sensing path. That change also reframed feature extraction as an optional analysis path rather than the center of runtime motion detection.

## Decision

Adopt a segmentation-first architecture centered on MVS.

Concretely:

- make motion segmentation the primary runtime path
- remove the older multi-state detection engine and supporting statistics stack
- treat feature extraction as optional and downstream of motion detection
- simplify configuration and MQTT control surfaces around the MVS workflow

## Alternatives Considered

### Keep the broader multi-state detection and calibration system

Rejected. The older structure was harder to maintain and exposed more moving parts than the project could justify at that stage.

### Keep features and classifier-like logic in the center of the runtime path

Rejected. The release direction favored a leaner motion detector first, with features retained for analysis and later experimentation.

## Consequences

Benefits:

- the runtime became simpler and easier to reason about
- MVS became the reference path for subsequent validation and tuning
- feature extraction could evolve more independently from the detection loop

Trade-offs:

- some earlier configurability and richer detection states were removed
- this decision was later superseded as the project converged on newer Classic and ML production paths

## Related

- versioned changelog snapshot: `1.2.0:CHANGELOG.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- git commits: `73b506eb`, `e6d2f53f`
