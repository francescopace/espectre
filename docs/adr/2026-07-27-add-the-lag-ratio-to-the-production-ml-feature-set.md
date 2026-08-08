# ADR: add the lag ratio to the production ML feature set

- Status: Superseded
- Date: 2026-07-27
- Supersedes: 2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md
- Superseded by: 2026-07-28-drop-the-absolute-l1-features.md

## Context

Classic replaced its mean L1 displacement with a lag ratio, dividing the displacement at lag `L` by the displacement at lag `1`; see [2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md](2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md). The mean carries the link's noise floor and degrades or inverts on weak links, while the ratio shares a unit with its denominator and drops the floor.

ML kept Coherence-6, which still contains the plain mean and its standard deviation. The same weakness applies there, so the ratio was worth testing as a seventh input rather than as a replacement.

## Decision

The production ML feature set is Coherence-7: Coherence-6 plus `l1_delta_lag_ratio`, exported as feature id `25`.

Nothing was removed. The absolute L1 members stay, because the measurement below is about what the ratio adds, not about what the mean costs.

## Validation

Seed `20260519`, no augmentation, evaluated on reserved replays with `allow_legacy_fallback=False` so every row is data the candidate never trained on. Two of the five chips own no reserved selection pair, and the gate's default fills them with training pairs, which would have made two of five rows in-sample.

Selection only, three reserved pairs:

| | exported Coherence-6 | Coherence-7 seed 20260519 |
| --- | --- | --- |
| pass | 3/3 | 3/3 |
| max FP | 2.63% | **0.73%** |
| worst recall | 100.00% | 100.00% |
| worst F1 | 97.45% | **99.28%** |
| effective alarms | 1 | **0** |

Selection and holdout, ten reserved replays:

| | exported | Coherence-7 |
| --- | --- | --- |
| max FP | 6.43% | **4.43%** |
| worst recall | 97.99% | **99.14%** |
| worst F1 | 92.92% | **95.30%** |
| effective alarms | 8 | **3** |

Nine of the ten replays improve or hold. The C6 selection pair drops from `18` false-positive evaluations and one alarm to `5` and none, and the S3 weak-link holdout goes from `7` alarms to `3`.

## The gate refused it, and why it was overridden

`paired_result_non_regression` ratchets per replay, and one normal-link S3 holdout recording regressed: false positives went from `1` evaluation to `7` out of `685`, against a margin of `100/685`, which is a single evaluation. Two extra false-positive evaluations are enough to block any recording of that length.

None of those seven produced an effective alarm; the consecutive-hit filter absorbed them all. Against that, the candidate removes five real alarms elsewhere and adds none, taking the reserved total from `8` to `3`.

So the export used `--force-promote`, which is the flag's documented purpose: a deliberate, explicit baseline reset at a fixed seed. The rule was left untouched. Relaxing the margin to let this candidate through would have weakened the gate for every future candidate on the strength of one result, which is the failure mode the gate exists to prevent.

**The reserved holdout is now open for this candidate.** It was evaluated to make this decision, so it can no longer serve as blind validation of it. Promoting on this evidence is sound; iterating over seeds until one passes, using holdout feedback, would not be, and was not done.

## Alternatives Considered

### Swap the mean out rather than adding the ratio

Not attempted here. Classic runs on the ratio alone plus turbulence autocorrelation, but ML has six inputs and the interactions are its own question. Removing a member is a separate experiment against a separate baseline.

### Relax the per-replay margin

Rejected. The margin is one evaluation on a `685`-evaluation recording, which is arguably too tight for a quantity that never reached an alarm, and that is worth revisiting on its own evidence. Doing it as part of promoting a candidate it blocks would mean tuning the gate to the answer.

## Consequences

Reserved effective alarms fall from `8` to `3` and the worst reserved F1 rises from `92.92%` to `95.30%`.

The runtime feature declaration moved with the export. `csi_features.py` now sets `DEFAULT_FEATURES = COHERENCE7_FEATURES` and `l1_delta_lag_ratio` left `CANDIDATE_FEATURES`; the report's `EXPORTED_FEATURE_NAMES == RUNTIME_FEATURE_NAMES` assertion caught the gap when only the weights had been regenerated.

**The production set now contains a feature the series extractor cannot derive on its own.** `l1_delta_lag_ratio` comes from the tracker, so every caller passing the default set must supply it, in both runtimes. Python raises when it is missing and C++ has no default argument, so the compiler refuses the call: a forgotten value would otherwise read as `1.0`, which is exactly what a no-motion stream produces.

The model gains one input: `7 -> 32 -> 16 -> 1` against `6 -> 32 -> 16 -> 1`.

## Related

- [FEATURES.md](../FEATURES.md)
- [2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md](2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md)
- [2026-07-28-drop-the-absolute-l1-features.md](2026-07-28-drop-the-absolute-l1-features.md)
