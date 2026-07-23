# ADR: separate ML training data from promotion replays

- Status: Accepted
- Date: 2026-07-23

## Context

Until 2026-07-22 the ML trainer selected and promoted candidates using paired
static-presence/motion replays whose recordings were also part of the training
set, and the performance report aggregated the same recordings. This produced
three related distortions:

1. **In-sample promotion metrics.** The exported model at the time (seed
   `1395304627`) reported a near-perfect paired gate (max FP `0.43%`), while
   grouped CV exposed `44.5%` FP on a C5 `empty` session left out of the
   training fold. Once evaluation recordings were reserved, the direct
   measurement was stark: the old model scored `0.00%` FP and `100%` recall on
   replays it had memorized, while 20 freshly trained candidates scored
   `2.9%`–`40%` FP on the same replays out-of-sample.
2. **Synthetic/real CV leakage.** Synthetic low-RSSI derivatives carried their
   own pair identity, so grouped CV could place a synthetic derivative in the
   training fold while its real source recording sat in the validation fold.
3. **A single uniform target for physically different link classes.** The
   2026-07-22 holdout captures were deliberate weak-link recordings
   (−75/−77 dBm) where the motion/static turbulence ratio collapses to
   ~`1.03x`. On the S3 weak pair the ML detector missed the `95%` recall
   target (92.8%), but Classic — which has no training set and cannot overfit
   — produced the identical 7 effective alarms. Identical failure across two
   unrelated detectors indicates a link-physics limit, not a model defect.

## Decision

Adopt a train/evaluation separation for ML promotion, plus an explicit
link-class policy, as one coherent protocol:

1. **Lineage-grouped CV.** Grouped CV splits by `lineage_group`: a synthetic
   derivative always shares the fold of its real source recording
   (`source_dataset` / `generation_group` read from the generated NPZ).
2. **Dataset roles.** `dataset_info.json` entries carry
   `dataset_role: train | selection | holdout` (default `train`). Training
   consumes `train` recordings only. `selection` replays (paired and quiet
   `empty`, replayed at production cadence with runtime hit filtering) gate
   candidate selection. `holdout` recordings stay sealed through the entire
   search and are evaluated exactly once, on the chosen winner.
3. **Safety-first robust ranking.** Deployment replays are absolute safety
   gates; among safe candidates, grouped-CV worst and worst-five-tail session
   metrics lead ranking, with one-changed-evaluation equivalence margins.
   When synthetic derivatives exist, real sessions lead the comparison and
   synthetic session metrics act only as regression guards.
4. **Link-class policy.** Real weak-link pairs (`low_rssi: true`) are stress
   diagnostics, not standard promotion material: the ML detector gates them at
   relaxed stress targets (recall >`90%`, FP <`10%`), the Classic detector is
   report-only there and gains strict per-pair gating on normal-link pairs,
   and the performance report separates normal-link tables (ML further split
   into reserved out-of-sample vs in-sample training recordings) from a
   dedicated Low-RSSI stress section.
5. **Deliberate baseline reset.** `--force-promote --seed <n>` exports a
   candidate even when gates fail, printing the bypass loudly. It exists for
   exactly one scenario: replacing a baseline whose passing gate status is
   itself an in-sample artifact. It was used once to promote seed
   `1921306627` as the first honestly evaluated baseline.

## Validation

- Reserved selection replays reproduce the search metrics exactly
  (deterministic training): max FP `2.90%`, worst recall `98.59%`.
- Sealed holdout, opened once on the winner: C3 `0.14%` FP / `99.7%` recall,
  C5 (weak link, −75 dBm but healthy separation) `0.00%` FP / `100%` recall,
  C6 `1.30%` FP / `95.3%` recall, S3 (weak link, −77 dBm, collapsed
  separation) `4.43%` FP / `92.8%` recall.
- Under the link-class split, Classic passes strict per-pair gates on all
  normal-link pairs, and the full Python suite is green with weak-link pairs
  asserted at stress targets.
- A 10-seed search with train-time augmentation confirmed the reserved
  replays discriminate candidates: max FP spread `1.46%`–`16.5%`
  out-of-sample, versus uniformly perfect in-sample scores before the split.

## Alternatives Considered

### Keep ranking on the in-sample paired gate

Rejected. Memorization masks generalization failures entirely; the C5 quiet
session at `44.5%` OOF FP was invisible to a paired gate the model had
trained on.

### Group CV by session without linking synthetic lineage

Rejected. Synthetic derivatives are deterministic transforms of real
recordings; letting them cross the train/validation boundary leaks the source
session into training and inflates validation scores.

### Uniform strict targets for weak-link replays

Rejected. At ~`1.03x` motion/static separation no threshold-based detector
has headroom left to trade; Classic fails these captures identically to ML.
A permanent red at `95%`/`5%` would encode a physics limit as a code defect
and train reviewers to ignore the suite.

### Drop weak-link captures from the corpus

Rejected. They are the only real measurements of graceful degradation near
the sensitivity floor, and they validated the synthetic low-RSSI pipeline
(the C5 weak pair passes cleanly; only the collapsed-separation S3 pair
degrades).

## Consequences

Benefits:

- Promotion, the validation suite, and the performance report now tell the
  same story from the same split; in-sample numbers are labeled as such.
- CV can no longer leak a synthetic derivative across its source session.
- Weak-link degradation is visible and bounded instead of hidden in chip
  averages.

Trade-offs:

- Training loses 8 real pairs and 8 `empty` recordings to reserved roles
  until new collection lands.
- ESP32 has a single real pair and no `empty` capture, so its gate still uses
  the in-sample legacy fallback.
- No normal-link holdout exists yet: all 2026-07-22 holdout pairs turned out
  to be weak-link captures. New bedroom collection at normal link is required
  for a strict out-of-sample holdout.
- The force-promoted baseline fails its own selection gate by one effective
  alarm, so seed searches run in broken-baseline mode (a candidate must fully
  restore the gate) until better candidates or new training data arrive.

## Related

- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [2026-07-17-separate-dataset-admission-from-classic-diagnostics.md](2026-07-17-separate-dataset-admission-from-classic-diagnostics.md)
- [2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md](2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md)
- [2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md](2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md)
- Commits: `51c1357` (lineage grouping, dataset roles, robust ranking),
  `5b914f8` (report provenance split, `--force-promote`, baseline reset),
  `d792158` (link-class stress policy)
