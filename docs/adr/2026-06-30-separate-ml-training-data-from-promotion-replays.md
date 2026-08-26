# ADR: separate ML training data from promotion replays

- Status: Accepted
- Date: 2026-06-30
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

The deployed IDLE class includes both a person who remains still and a genuinely empty room. C3 and C6 incidents showed that a detector trained only on `static_presence` versus `motion` could look correct on paired recordings and still raise false positives in an empty room. Retraining with `empty` mapped to IDLE reduced the observed failures.

The broader training workflow also selected early candidates on recordings that participated in training. Synthetic derivatives could cross validation folds independently of their source, one uniform target obscured the different physical limits of normal and low-RSSI links, and one-evaluation margins made seed searches sensitive to incidental training noise.

The project therefore needs one data-role, lineage, ranking, and reproducibility contract that matches the deployed task and keeps promotion evidence out of training.

## Decision

Adopt the following ML data and promotion protocol:

1. Map both `empty` and `static_presence` to IDLE. Keep empty-room captures in training, dataset curation, and strict quiet replay gates.
2. `dataset_info.json` assigns every recording one role: `train`, `selection`, `holdout`, or `exclude`.
3. Training consumes only `train`; candidate selection uses `selection`; `holdout` stays sealed until the winner is fixed; and `exclude` remains indexed without affecting promotion.
4. Grouped CV splits by lineage, so every synthetic derivative stays in the same fold as its real source.
5. Deployment replays are absolute safety gates. Among safe candidates, grouped-CV tail metrics and per-recording comparisons rank candidates.
6. Real low-RSSI captures use the documented stress policy. They remain visible and bounded without treating collapsed physical separation as a normal-link software defect.
7. Per-recording non-regression margins come from measured seed-to-seed dispersion rather than one arbitrary evaluation.
8. Detector-guided sample weighting is not part of the default baseline.
9. Production packet augmentation uses the deterministic constant-size seed mix documented in `ML_TRAINING.md`; exact parameters remain operational documentation rather than ADR content.
10. Artifact export is explicit. A force-promotion escape hatch may reset a demonstrably invalid baseline, but it must remain deliberate and visible.

Interpret a new empty-domain failure as a coverage problem first, not as evidence that the deployed task excludes empty rooms. Holdouts, low-RSSI gates, long recordings, generated-artifact checks, and Python/C++ parity remain necessary because empty-room coverage does not replace other deployment domains.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-06-30 | Treat empty captures as first-class IDLE training and validation data | Accepted after independent C3 and C6 empty-room failures |
| 2026-07-07 | Use detector-guided sample weighting to improve the baseline | Rejected after two campaigns; unweighted training remains the default |
| 2026-07-23 | Separate train, selection, holdout, and excluded data | Accepted |
| 2026-07-27 | Use a one-evaluation non-regression margin | Replaced with margins derived from measured seed noise |
| 2026-08-11 | Train from one packet-augmentation seed view | Replaced with a deterministic constant-size mix of complementary views |

## Validation Policy

- Selection and holdout results retain per-recording provenance.
- Quiet `empty` replays keep the zero-alarm requirement for High Accuracy. Lightweight sequential empty-room tests use the bounded alarm budget in the host-side validation ADR.
- Static-presence replays may use an explicit alarm budget because real micro-motion can occur.
- Weak-link replay changes remain subject to absolute stress targets and the current alarm ratchet.
- Generated artifacts and Python/C++ parity are validated under the shared host-side promotion ADR.

## Alternatives Considered

### Train only on `static_presence` versus `motion`

Rejected. It excludes a normal part of the deployed IDLE distribution and reproduced false positives on more than one chip and room.

### Keep `empty` only as an optional smoke-test domain

Rejected. Empty-room behavior is part of the deployed task and must influence both data curation and promotion.

### Rank on in-sample paired replays

Rejected. Memorization hid generalization failures that appeared once recordings were reserved.

### Let synthetic derivatives cross source folds

Rejected. Deterministic transforms of one recording are not independent validation data.

### Drop weak-link captures

Rejected. They provide the measured view of graceful degradation near the sensitivity floor.

### Double the augmented matrix with two complete seed views

Rejected. A constant-size mix covers complementary stress tails without doubling synthetic weight or memory.

## Consequences

- the training target matches empty-room and static-presence deployment behavior;
- promotion metrics are out-of-sample by construction and keep source lineage intact;
- seed searches are less sensitive to incidental one-event noise;
- data roles and augmentation provenance remain part of dataset and cache identity; and
- new normal-link holdout and empty-room data remain valuable as the corpus evolves.

## Related

- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`../ML_DATA_COLLECTION.md`](../ML_DATA_COLLECTION.md)
- [`../ML_TRAINING.md`](../ML_TRAINING.md)
- [`../FEATURES.md`](../FEATURES.md)
