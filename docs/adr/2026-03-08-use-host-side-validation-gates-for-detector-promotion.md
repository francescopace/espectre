# ADR: use host-side validation gates for detector promotion

- Status: Accepted
- Date: 2026-03-08
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-07-17

## Context

As the project added multiple detectors, multi-chip support, and a larger ML
program, release notes and experiments show a clear shift away from informal
single-metric wins toward stricter host-side promotion criteria.

The `2.6.0` changelog explicitly tightened the quality bar for motion
validation, aligning Python and C++ targets around `Recall >95%` and `FP <5%`.
Historical sweeps then used paired sets plus long-quiet recordings as a
deploy-like ranking signal. That kept CV-only wins from promoting noisy
baselines, but it also made every train and seed-search pass pay the
long-recording cost and blocked otherwise useful paired improvements when
quiet long FP moved.

On 2026-07-17 the trainer stopped using long recordings as a promotion gate so
iteration stays fast and long-run quality stays in the performance report and
dedicated pytest suites.

## Decision

Use host-side validation gates as the promotion rule for detector and feature
set changes.

Concretely:

- evaluate ML candidates with shared host-side paired validation
- require cross-stack evidence for published detector quality, not only isolated
  model wins
- treat paired pass count, max FP, worst-chip recall, and worst-chip F1 as the
  trainer promotion gate and primary ranking signal
- treat grouped CV as diagnostic evidence and a final tie-breaker, not as an
  early rejection rule for ML candidates
- evaluate curated long recordings in `generate_performance_report` and
  `test_validation_long_recordings.py`, not inside `train_ml_model` promotion
- require explicit artifact promotion after evaluation; experiment campaigns
  remain non-destructive by default and compare finalists across multiple seeds

## Alternatives Considered

### Promote candidates based mainly on grouped CV or one benchmark family

Rejected. Later experiments showed that CV-only wins can hide deployment-facing
false-positive regressions on paired captures.

### Keep long-quiet recordings inside trainer promotion

Rejected for the current workflow. Long gates remain valuable for published
performance evidence, but blocking every training and seed-search trial on
them made training slow and over-constrained relative to the paired gate.

### Let each runtime or frontend keep its own promotion standard

Rejected. The project benefits from one shared quality bar across Python and
shared C++ paths.

## Consequences

Benefits:

- trainer promotion stays tied to paired real-data evidence without paying the
  long-recording cost on every trial
- Python and C++ validation stay aligned around shared acceptance suites
- the project can preserve historical experiments without promoting noisy wins

Trade-offs:

- a promoted ML artifact can still regress on quiet long recordings until the
  performance report or long-recording pytest suite catches it
- long-recording policy metrics are no longer the first-rank signal inside
  seed search and architecture campaigns

## Related

- versioned changelog snapshot: `2.6.0:CHANGELOG.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- `docs/adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md`
- git commits: `3719e695`, `2217271d`
