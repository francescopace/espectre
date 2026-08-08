# ADR: adopt nbvi for runtime subcarrier selection

- Status: Superseded
- Date: 2025-12-03
- Recorded: 2026-07-09 (retrospective)
- Superseded by: 2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md

## Context

The `1.5.0` release changelog presents NBVI as the project's answer to zero-configuration runtime subcarrier selection. At that stage, the goal was to replace manual band tuning with an automatic method that performed nearly as well while adapting to the environment at boot.

The release positioned NBVI as both a Python and C implementation, with automatic calibration at startup and after reset. Later releases strengthened that role further by making NBVI the sole calibrator before the project eventually moved away from the runtime NBVI path.

## Decision

Adopt NBVI as the runtime subcarrier-selection strategy.

Concretely:

- use NBVI to choose the active subcarrier set automatically
- run calibration automatically at startup when no saved configuration exists
- keep Python and C implementations aligned around the same calibration logic
- treat zero-configuration calibration as a key usability goal

## Historical Formulation

The internal NBVI procedure:

1. collected a quiet baseline of about 1,000 packets, approximately 5-10 seconds at the intended cadence;
2. excluded unusable carriers and the lowest 10% of carriers by mean amplitude;
3. applied Hampel outlier filtering;
4. ranked the remaining carriers by baseline variability; and
5. selected 12 carriers while enforcing frequency diversity.

The documented weighted score was:

```text
NBVI = 0.3 * (sigma / mu^2) + 0.7 * (sigma / mu)
```

The `sigma / mu` component is the coefficient of variation and is invariant to a positive common scale factor. The `sigma / mu^2` component is not: multiplying the signal by `a` divides that term by `a`. The weighted NBVI score therefore does not satisfy the exact scale-invariance requirement adopted by the later ML feature work.

## Alternatives Considered

### Keep manual subcarrier selection

Rejected. The project wanted a more deployable baseline that did not require per-environment tuning.

### Stay with simpler variance-only or heuristic-only selection

Rejected. The release evidence presented NBVI as materially more robust and closer to manual tuning quality.

## Consequences

Benefits:

- the project gained a strong zero-configuration calibration story
- runtime calibration behavior stayed aligned across Python and C
- NBVI became the historical baseline that shaped several later releases

Trade-offs:

- runtime calibration gained algorithmic and operational complexity
- this approach was later superseded when the project moved to fixed shared subcarriers and a different bootstrap strategy

## Related

- versioned changelog snapshot: `1.5.0:CHANGELOG.md`
- git commits: `896a5bd9`, `40b2cd2a`, `d10a10e2`
