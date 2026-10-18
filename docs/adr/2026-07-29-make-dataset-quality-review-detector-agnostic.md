# ADR: make dataset-quality review detector-agnostic

- Status: Accepted
- Date: 2026-07-29
- Updated: 2026-08-12

## Context

`tools/validate_dataset_quality.py` does two jobs: it keeps contaminated or unusable captures out of training, and it summarizes the admitted corpus for people in `data/auto_generated/DATASET_QUALITY_CHECK.md`.

On 2026-07-17 Classic was taken out of dataset admission, but the review tables still replayed `ClassicDetector`, with its startup calibration and threshold-relative wording. That left a circularity: a weak or badly calibrated detector could make a capture look bad even when the production features separated it cleanly. Low-RSSI pairs showed it: the report warned about detector activation while idle and motion were still well separated. That says something about the detector, not about the data.

## Decision

There is still one generated dataset-quality report, but its review metrics no longer depend on any detector.

Admission is unchanged and can still fail the run: integrity, continuity, signal quality, empty and `static_presence` availability and overlap, ML readiness, and long-recording annotation and coverage.

The review tables are built from the current High Accuracy feature set, which is scale-invariant by construction. Feature directions come from what each feature means, not from a detector replay. Pairs are judged on how well motion separates from the idle half; idle captures are judged against their own baseline in feature space. No threshold-relative detector term appears in the report. Lightweight shows up in promotion and performance reports, not here.

The column definitions and marks are documented in the report's own legend, which the validator generates.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-17 | Take Classic out of admission, keep Classic review tables | Detector no longer decides admission |
| 2026-07-29 | Make the review metrics detector-agnostic | Accepted |

## Alternatives considered

- **Keep Classic review and drop only the marked threshold columns.** Even unmarked replay tables keep the reader focused on one detector's calibration.
- **Add a second, agnostic report next to the old one.** Dataset policy belongs in one generated document, and detector performance already has its own report.
- **Score from raw RSSI or absolute energy.** Not robust to gain changes or weak links; it would bring back the same comparability problem.

## Consequences

The report describes the recordings, not a detector's startup calibration. Weak-link and gain-shifted captures stay comparable, and excluded pairs can be shown as diagnostics without detector bias. Detector problems show up where they belong, in performance and promotion gates.

The report now owns a small feature-evidence layer and its scoring rules, which must stay in line with the shared feature definitions. Burst and drift on idle captures describe feature-space behavior, not alarms, and need to be read that way.

## Related

- `tools/validate_dataset_quality.py` and [dataset quality report](../../data/auto_generated/DATASET_QUALITY_CHECK.md)
- [ML data collection](../ML_DATA_COLLECTION.md) and the [performance report](../performance/README.md)
- Commits `a8b0dc4f` (Classic out of admission) and `e3ad4b68` (Classic replay removed from the report)
