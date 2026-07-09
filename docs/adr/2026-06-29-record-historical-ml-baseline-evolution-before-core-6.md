# ADR: record historical ml baseline evolution before core-6

- Status: Superseded
- Date: 2026-06-29
- Recorded: 2026-07-09 (retrospective)
- Superseded by: 2026-07-07-use-core-6-as-the-production-ml-feature-set.md

## Context

Before the project reached Core-6, the production ML line had already gone
through several promotions driven by the same product priority: long static-run
false positives mattered more than attractive grouped-CV numbers alone.

Three experiment families shaped that line:

- the 2026-05-20 feature-set reduction sweep cut the earlier raw-feature input
  from 12 to 9 by removing features that hurt long-run robustness more than
  they helped paired validation
- the 2026-05-20 raw-9 topology sweep then promoted the wider `9 -> 32 -> 16 ->
  1` MLP over the older `24 -> 12` line because it improved the FP-first
  ranking without reopening the feature set
- the 2026-06-29 relative-8 topology and `fp_weight` sweep moved the project to
  a gain-stable relative-feature baseline, again choosing the candidate that
  best balanced long-run false positives, paired validation, and gain-stress
  behavior

This progression matters historically because Core-6 did not appear from an
empty slate. It replaced a specific, already hardened MLP line that had been
selected through multiple FP-first promotions.

## Decision

Record the pre-Core-6 ML line as a deliberate historical progression:

- reduce the older raw feature set to the 9-feature baseline
- keep the MLP family and promote the wider raw-9 `32 -> 16` topology over the
  older raw `24 -> 12` baseline
- then move to the gain-stable relative-8 feature set with topology
  `8 -> 32 -> 16 -> 1` and `fp_weight=2.0` as the final pre-Core-6 production
  baseline

Treat those promotions as one historical baseline-evolution thread governed by
FP-first deployment robustness, not as isolated benchmark wins.

## Alternatives Considered

### Keep the earlier raw-9 baseline unchanged

Rejected. The earlier raw line was useful, but repeated sweeps showed that both
the feature set and topology could be improved without abandoning the supported
runtime MLP path.

### Rank candidates mainly by grouped CV or one benchmark family

Rejected. These promotions mattered precisely because the project learned that
paired or CV-only wins can hide the false-positive regressions that show up on
long quiet recordings or under gain shift.

## Consequences

Benefits:

- the historical ML path is now legible without the deleted experiment log
- later Core-6 work has a clear predecessor baseline to supersede
- the repo keeps the record that the MLP line was hardened incrementally rather
  than replaced wholesale

Trade-offs:

- this ADR groups several promotions into one historical thread rather than
  preserving each experiment as its own document
- detailed benchmark tables remain out of line with the ADR format and are not
  reproduced here

## Related

- `docs/adr/2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/adr/2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
