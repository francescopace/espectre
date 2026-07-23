# ADR: adopt Coherence-6 as the production ML feature set

- Status: Accepted
- Date: 2026-07-23
- Supersedes: 2026-07-07-use-core-6-as-the-production-ml-feature-set.md

## Context

Core-6 shipped two members that measured absolute signal energy:
`turb_skewness` and `l1_delta_waveform_length`. A separability probe over the
real corpus (windowed AUC per session) showed both were weak, and worse: on
real weak-link pairs at `-75/-77 dBm` the absolute L1 features lose or invert
their motion separation (AUC `0.37`-`0.40` on C6/S3), the same failure mode
that broke Classic at low RSSI. Seed searches under the honest train/replay
separation exposed the consequence: Core-6 training was seed-fragile, with
reserved-replay max FP ranging `2.9`-`40%` across seeds.

The physics: the noise floor is white in time, human motion is temporally
coherent. Statistics that measure the coherent fraction of the variance are
invariant to both gain and floor level.

## Decision

Replace the two energy-based members with shift/scale-invariant
temporal-coherence statistics, keeping six inputs and the same runtime cost:

- in: `turb_zcr` (median-crossing rate of the turbulence window) and
  `l1_delta_autocorr` (lag-1 autocorrelation of the L1 displacement series)
- out: `turb_skewness` and `l1_delta_waveform_length` (demoted to
  experiment-only candidates)

`DEFAULT_FEATURES` is now `COHERENCE6_FEATURES`; both new features have C++
extractor ids (`14`, `24`) with Python-parity tests. The promoted production
model (seed `1312857390`, trained with the `--augment` recipe) was selected by
a normal-mode seed search and is the first model promoted end-to-end by the
reserved-replay protocol: selection screening, robust grouped-CV ranking, and
a single sealed-holdout evaluation.

## Validation

- Multi-seed robustness (6 seeds, reserved replays): swap max FP median
  `2.66%`, worst `2.80%`, versus Core-6 median `19.42%`, worst `24.93%`;
  alarms median `2` versus `12`. The coherence swap collapses seed-to-seed
  variance on out-of-sample false positives.
- Same-seed augment ablation on unseen replays favored keeping the
  `--augment` recipe (fewer alarms on every reserved replay class).
- Promoted-model holdout: novel-hardware S3 Waveshare pair `0.29%` FP,
  `100%` recall, zero alarms; all weak-link stress replays improved their
  recall over the previous model (C6 weak `95.3 -> 96.2%`, S3 weak
  `92.8 -> 94.5%`) with fewer alarms.
- Full validation suites: C++ 25/25; Python green except one known in-sample
  dip recorded below.

## Alternatives Considered

### Add the coherence features without removing anything (Core-8)

Rejected. Measured no improvement over Core-6: the weak members act as
unstable inputs that seed-dependent training latches onto.

### Also add `l1_delta_cv`

Rejected. The seven-feature variant measurably worsened reserved-replay
results; the coefficient of variation stays an experiment-only candidate.

### Keep Core-6 and rely on train-time augmentation

Rejected. Augmentation narrowed but did not close the seed fragility
(max FP still up to `16.5%` across seeds); the feature swap addresses the
cause rather than compensating it.

## Consequences

Benefits:

- Reserved-replay FP is low and stable across seeds; promotion no longer
  depends on seed luck.
- Weak-link recall improved without dedicated real weak training data.
- Same model size, architecture, and runtime cost as Core-6.

Trade-offs:

- One known in-sample dip: the promoted model scores `93.9%` recall on the
  C5 living-room training pair (the per-pair suite target is `95%`); the
  reserved replays it was promoted on all pass.
- Synthetic low-RSSI profiles carry reference medians calibrated for Core-6;
  generation fits the intersection until profiles are recalibrated for
  Coherence-6.
- Historical Core-6 remains selectable for experiments via `--features`.

## Related

- [2026-07-07-use-core-6-as-the-production-ml-feature-set.md](2026-07-07-use-core-6-as-the-production-ml-feature-set.md)
- [2026-07-23-separate-ml-training-data-from-promotion-replays.md](2026-07-23-separate-ml-training-data-from-promotion-replays.md)
- Commits: `dc94003` (candidate features, C++ port, `--features`)
