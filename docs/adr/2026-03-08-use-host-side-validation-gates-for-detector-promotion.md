# ADR: use host-side validation gates for detector promotion

- Status: Accepted
- Date: 2026-03-08
- Recorded: 2026-07-09 (retrospective)

## Context

As the project added multiple detectors, multi-chip support, and a larger ML
program, release notes and experiments show a clear shift away from informal
single-metric wins toward stricter host-side promotion criteria.

The `2.6.0` changelog explicitly tightened the quality bar for motion
validation, aligning Python and C++ targets around `Recall >95%` and `FP <5%`.
Historical sweeps then clarified why that stricter rule was necessary. The
2026-05-20 FP-first feature and training campaign showed that grouped CV and
paired-set quality alone were not enough to rank candidates safely for
deployment; the long-run false-positive ceiling had to outrank prettier
cross-validation wins. Later ML and Classic promotions hardened that into the
paired and long-quiet gates, including explicit non-regression rules such as
`MAX_PROMOTION_TOTAL_FP_INCREASE = 0`.

## Decision

Use host-side validation gates as the promotion rule for detector and feature
set changes.

Concretely:

- evaluate candidates with shared host-side validation workflows
- require cross-stack evidence, not only isolated model or detector wins
- treat false-positive regressions on realistic long-quiet recordings as a
  promotion blocker
- use paired and long-quiet evaluation gates to decide whether a detector or ML
  candidate becomes the new baseline

## Alternatives Considered

### Promote candidates based mainly on grouped CV or one benchmark family

Rejected. Later experiments showed that CV-only wins can hide deployment-facing
false-positive regressions.

### Let each runtime or frontend keep its own promotion standard

Rejected. The project benefits from one shared quality bar across Python and
shared C++ paths.

## Consequences

Benefits:

- baseline promotions are tied to deployment-relevant evidence
- Python and C++ validation stay aligned around the same acceptance criteria
- the project can preserve historical experiments without promoting noisy wins

Trade-offs:

- promotion becomes slower and more conservative
- candidates that look strong on one metric can still be rejected by stricter
  regression gates

## Related

- versioned changelog snapshot: `2.6.0:CHANGELOG.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- git commits: `3719e695`, `2217271d`
