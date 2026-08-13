# ADR: use host-side validation gates for detector promotion

- Status: Accepted
- Date: 2026-03-08
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-12

## Context

Single aggregate metrics and grouped CV can hide deployment-facing failures. ESPectre needs one promotion policy that covers paired motion evidence, empty rooms, weak links, packet-rate variation, long recordings, generated artifacts, and Python/C++ parity without making every exploratory training run execute the complete release suite.

Static-presence captures are also not false-positive ground truth: a present person can produce real micro-motion. Empty-room recordings are the strict no-motion domain for alarm gates.

## Decision

Use layered host-side validation for detector, feature, and model promotion:

1. Use grouped CV and selection-role replays to compare candidates without opening the sealed holdout.
2. Require per-recording paired and quiet safety gates before promotion; do not rely on chip averages alone.
3. Treat empty-room recordings as the strict false-positive and zero-alarm ground truth. Static-presence captures may use an explicit bounded alarm budget.
4. Keep real low-RSSI recordings visible under their documented stress policy rather than hiding them in aggregate metrics.
5. Validate packet-rate behavior, long recordings, generated artifacts, and Python/C++ parity on the selected candidate and for published performance.
6. Keep experiment commands non-destructive by default and require an explicit artifact-promotion step.

Trainer ranking may use the narrow selection gates for iteration speed. Passing trainer selection does not replace the broader parity, performance, and generated-artifact gates required for production.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-03-08 | Align detector promotion around host-side real-data gates | Accepted |
| 2026-07-17 | Run long-recording gates inside every training trial | Moved to final promotion and published-performance validation |
| 2026-07-25 | Treat static-presence baselines as strict false-positive truth | Replaced with empty-room zero-alarm gates and an explicit static-presence budget |

## Alternatives Considered

### Promote from grouped CV or one benchmark family

Rejected. Aggregate wins can hide per-recording false positives, weak-link collapse, or runtime filtering failures.

### Run the complete release suite for every seed

Rejected. It makes exploration unnecessarily slow; the complete suite belongs at finalist promotion and release validation.

### Let each runtime define its own gate

Rejected. Detector behavior is a shared Python/C++ contract.

## Consequences

- Candidate search stays practical while production promotion remains evidence-based.
- Empty-room, weak-link, timing, long-recording, and parity failures remain independently visible.
- A passing exploratory result cannot be described as production-ready until the broader gates run.

## Related

- [`2026-07-23-separate-ml-training-data-from-promotion-replays.md`](2026-07-23-separate-ml-training-data-from-promotion-replays.md)
- [`2026-08-13-use-aggregated-turbulence-iqr-for-classic.md`](2026-08-13-use-aggregated-turbulence-iqr-for-classic.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- versioned changelog snapshot: `2.6.0:CHANGELOG.md`
