# ADR: separate ML training data from promotion replays

- Status: Accepted
- Date: 2026-07-23
- Updated: 2026-08-16

## Context

Earlier ML candidates were selected on recordings that also participated in training. Synthetic derivatives could cross CV folds independently of their real source, and one uniform target obscured the different physical limits of normal and low-RSSI links. Later seed searches also showed that one-evaluation margins and a single augmentation view were too sensitive to ordinary training noise.

The training and promotion workflow therefore needs one explicit data-role, lineage, ranking, and reproducibility contract.

## Decision

Adopt the following ML training and promotion protocol:

1. `dataset_info.json` assigns every recording one role: `train`, `selection`, `holdout`, or `exclude`.
2. Training consumes only `train`; candidate selection uses `selection`; `holdout` stays sealed until the winner is fixed; and `exclude` remains indexed without affecting promotion.
3. Grouped CV splits by lineage, so every synthetic derivative stays in the same fold as its real source.
4. Deployment replays are absolute safety gates. Among safe candidates, grouped-CV tail metrics and per-recording comparisons rank candidates.
5. Real low-RSSI captures use the documented stress policy. They remain visible and bounded without pretending that collapsed physical separation is a normal-link software defect.
6. Per-recording non-regression margins are derived from measured seed-to-seed dispersion rather than one arbitrary evaluation.
7. Empty captures remain first-class IDLE data, and detector-guided sample weighting is not part of the default baseline.
8. Production packet augmentation uses the deterministic constant-size mix of seeds `20260807` and `20260808` described in `ML_TRAINING.md`; exact augmentation parameters remain operational documentation rather than separate ADRs.
9. Artifact export is explicit. A force-promotion escape hatch may reset a demonstrably invalid baseline, but it must be deliberate and visible.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-07 | Use detector-guided sample weighting to improve the baseline | Rejected after two campaigns; unweighted training remains the default |
| 2026-07-23 | Separate train, selection, holdout, and excluded data | Accepted |
| 2026-07-27 | Use a one-evaluation non-regression margin | Replaced with margins derived from measured seed noise |
| 2026-08-11 | Train from one packet-augmentation seed view | Replaced with a deterministic constant-size mix of two complementary views |

## Validation Policy

- Selection and holdout results must retain per-recording provenance.
- Quiet `empty` replays keep the zero-alarm requirement for High Accuracy. Lightweight sequential empty-room tests use the one-alarm per-recording budget in the host-side validation ADR.
- Static-presence replays may use the explicit alarm budget because real micro-motion can occur.
- Weak-link replay changes remain subject to absolute stress targets and the current alarm ratchet.
- Generated artifacts and Python/C++ parity are validated under the shared host-side promotion ADR.

## Alternatives Considered

### Rank on in-sample paired replays

Rejected. Memorization hid generalization failures that appeared immediately once the same recordings were reserved.

### Let synthetic derivatives cross source folds

Rejected. Deterministic transforms of one recording do not constitute independent validation data.

### Drop weak-link captures

Rejected. They provide the only measured view of graceful degradation near the sensitivity floor.

### Double the augmented matrix with two complete seed views

Rejected. The constant-size row mix covers complementary stress tails without doubling synthetic weight or memory.

## Consequences

- Promotion metrics are out-of-sample by construction and keep source lineage intact.
- Seed searches are less sensitive to incidental one-event noise.
- Data roles and exact augmentation provenance must be maintained as part of dataset and cache identity.
- New normal-link holdout data remains valuable as the corpus evolves.

## Related

- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md`](2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md)
- [`../ML_TRAINING.md`](../ML_TRAINING.md)
- [`../FEATURES.md`](../FEATURES.md)
- git commits: `51c1357`, `5b914f8`, `d792158`
