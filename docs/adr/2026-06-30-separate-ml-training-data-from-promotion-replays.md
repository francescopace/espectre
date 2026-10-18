# ADR: separate ML training data from promotion replays

- Status: Accepted
- Date: 2026-06-30
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-10-04

## Context

In production, IDLE means both "someone is sitting still" and "the room is empty". The model had only been trained on `static_presence` against `motion`, so it looked fine on paired recordings and still raised false alarms in empty rooms on C3 and C6. Adding `empty` as IDLE fixed most of it.

The same review found other problems in the workflow. Early candidates were picked on recordings that had also been used for training. Synthetic copies of a recording could end up in a different CV fold from their source. One shared target hid the fact that low-RSSI links have different physical limits. And a one-evaluation margin made seed searches react to noise.

## Decision

1. `empty` and `static_presence` both map to IDLE. Empty-room captures are used in training, curation, and the strict quiet gates.
2. Every recording in `dataset_info.json` has one role: `train`, `selection`, `holdout`, or `exclude`. Training sees only `train`, candidates are picked on `selection`, `holdout` stays sealed until the winner is fixed, and `exclude` is indexed but never used for promotion.
3. Grouped CV splits by lineage, so a synthetic derivative always sits in the same fold as its source.
4. Deployment replays are pass/fail safety gates. Among candidates that pass, grouped-CV tail metrics and per-recording comparisons decide the ranking.
5. Real low-RSSI captures follow the stress policy: visible and bounded, but a physical loss of separation is not treated as a software bug.
6. Per-recording non-regression margins come from measured seed-to-seed spread.
7. No detector-guided sample weighting by default.
8. Packet augmentation uses a deterministic, constant-size mix of seeds. The exact parameters live in `ML_TRAINING.md`.
9. Export is an explicit step. A forced promotion can reset a baseline that is clearly broken, but it has to be deliberate and visible.

A new empty-room failure is first treated as missing coverage, not as a sign that empty rooms are out of scope. Empty-room data does not replace holdouts, low-RSSI gates, long recordings, artifact checks, or parity. Alarm budgets for each replay type are defined in the host-side validation ADR.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-06-30 | Empty captures as first-class IDLE data | Accepted after empty-room failures on C3 and C6 |
| 2026-07-07 | Detector-guided sample weighting | Rejected after two campaigns |
| 2026-07-23 | Separate train, selection, holdout, and exclude | Accepted |
| 2026-07-27 | One-evaluation non-regression margin | Replaced by margins from measured seed noise |
| 2026-08-11 | Augmentation from a single seed view | Replaced by a constant-size mix of complementary views |

## Alternatives considered

- **Train on `static_presence` against `motion` only.** Leaves out part of real IDLE and reproduced false alarms on several chips and rooms.
- **Use `empty` only as a smoke test.** Empty rooms are part of the job, so they must shape both data and promotion.
- **Rank on in-sample paired replays.** Memorization hid failures that showed up once recordings were held out.
- **Let synthetic derivatives cross folds.** A transformed copy of a recording is not independent evidence.
- **Drop weak-link captures.** They are the only measured view of how detection degrades near the sensitivity floor.
- **Train on two full seed views.** Doubles synthetic weight and memory; a mix covers the same stress tails.

## Consequences

The training target matches what users see in empty and occupied rooms. Promotion metrics are out-of-sample by construction, and seed searches are less sensitive to one-off noise. Data roles and augmentation settings become part of the dataset and cache identity. New normal-link holdout and empty-room recordings remain worth collecting.

## Related

- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [ML data collection](../ML_DATA_COLLECTION.md), [ML training](../ML_TRAINING.md), and the [feature ledger](../FEATURES.md)
- Commits `52859c06` (empty as IDLE), `b3b8d702` (data roles and lineage CV), `9ea91829` (seed-noise margin), `58d2243a` (augmentation mix)
