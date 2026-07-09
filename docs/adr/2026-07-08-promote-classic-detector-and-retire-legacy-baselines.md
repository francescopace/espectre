# ADR: promote classic detector and retire legacy baselines

- Status: Accepted
- Date: 2026-07-08
- Recorded: 2026-07-09 (retrospective)

## Context

The historical moving-variance baseline remained useful for research, but the
later experiments showed that it was not the best production-facing non-ML
path. The decisive motion-feature benchmark showed that L1-delta matched the
historical moving-variance baseline on aggregate quality while giving a much
more stable quiet floor and a stronger S3 result. A later fusion pass then
closed the low-contrast recall residual by keeping L1-delta primary and adding
a guarded MVS recovery vote only in the ambiguous band where MVS still carried
complementary signal.

The changelog shows the same convergence: L1-delta became the primary non-ML
metric, startup calibration became motion-first with an internal quiet-first
fallback, and tooling moved away from historical metadata thresholds toward the
current runtime calibration behavior.

The older MVS startup-threshold and online-adaptation sweep remains part of the
historical context because it established the strongest version of the legacy
variance baseline before that baseline was retired. That work still matters for
offline comparison, but it no longer defines the active runtime direction.

## Decision

Promote `ClassicDetector` as the production non-ML path and retire the legacy
baselines from the active runtime path.

The accepted Classic direction is:

- L1-delta as the primary non-ML metric
- shared startup calibration that prefers a validated
  `quiet -> motion -> quiet` bootstrap
- quiet-first fallback inside the same startup budget when motion-first
  validation does not complete
- gated startup threshold placement as captured in
  `2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`
- supporting recovery logic aligned with the current Classic design rather than
  older standalone baselines

Historical variance-baseline tooling may remain for offline comparison, but it
is no longer the production runtime reference.

## Alternatives Considered

### Keep moving variance as the production default

Rejected. It remained useful as a benchmark, but the later Classic path offered
better quiet-floor stability and a clearer production calibration story.

### Switch to a fixed factory threshold

Rejected. The experiments did not support one fixed threshold as a safe
cross-session production policy.

## Consequences

Benefits:

- the project has one clear, documented non-ML production detector
- runtime calibration and validation workflows can align on the same behavior
- older baselines remain available as historical references without driving the
  active path

Trade-offs:

- offline comparisons must distinguish research baselines from the production
  detector
- legacy metadata and threshold sweeps became less central and required cleanup

## Related

- `docs/adr/2025-11-16-adopt-segmentation-first-mvs-architecture.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/adr/2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`
- `docs/adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`
- git commits: `dc0658ed`, `5b871159`, `dbbe21dd`, `b2e0de00`
