# ADR: drop the absolute L1 features

- Status: Accepted
- Date: 2026-07-28
- Supersedes: 2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md

## Context

The production ML feature set was Coherence-7, which kept two members carrying
absolute magnitude: `l1_delta`, the mean L1 profile displacement, and
`l1_delta_std`, its standard deviation. Classic had already replaced its own use
of the mean with a lag ratio for a known reason, recorded in
[2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md](2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md):
the per-packet CSI scaling factor is never stored, so absolute magnitude carries
the link's noise floor, and on weak links that floor can exceed the motion it is
meant to measure.

The Coherence-7 ADR kept both anyway, and said why:

> Nothing was removed. The absolute L1 members stay, because the measurement
> below is about what the ratio adds, not about what the mean costs.

On 2026-07-27 the cost arrived on its own.

## What happened

A C3 capture recovered from history was added to `train`. On its own it looked
ideal: `1.0000` separation, `100%` recall, and `0.0%` false positives on both
detectors. Under Coherence-7 it took a pre-existing weak-link C3 pair from
`0.0%` to **`100.0%`** false positives, and blocked out-of-fold F1 from `96.2%`
to `90.0%`.

Every window of that weak pair's quiet half was classified as motion. The
feature medians say why:

| feature | added idle (`-42 dBm`) | added motion | weak idle | weak motion |
| --- | --- | --- | --- | --- |
| `l1_delta` | `0.0228` | `0.0587` | **`0.2653`** | **`0.1830`** |
| `l1_delta_std` | `0.0064` | `0.0304` | **`0.0667`** | **`0.0519`** |
| `l1_delta_lag_ratio` | `1.0151` | `1.9399` | `0.9998` | `1.1301` |
| `turb_autocorr` | `0.0169` | `0.8127` | `0.0115` | `0.5817` |

Both absolute members are **inverted** on the weak pair: its idle sits above its
own motion. Meanwhile its idle `l1_delta` of `0.2653` is `4.5` times the added
capture's *motion* at `0.0587`. Adding one strong-link recording sharpened the
association "small absolute L1 means still" until the weak pair's entire quiet
half fell on the far side of it.

The two scale-invariant features ordered both pairs correctly throughout.

## Decision

The production set is the five scale-invariant features, named
`INVARIANT5_FEATURES`:

`turb_mad_over_mean`, `turb_autocorr`, `turb_zcr`, `l1_delta_autocorr`,
`l1_delta_lag_ratio`.

Every member is a ratio, a correlation, or a crossing rate. That is now the
membership rule rather than an observation about the members, and a test asserts
it by scaling both input streams and requiring every feature to hold.

`l1_delta` and `l1_delta_std` are gone from both runtimes, from `MLFeatureId`,
and from `CPP_FEATURE_IDS`. The model is `5 -> 32 -> 16 -> 1`.

## Measurement

Four seeds, on the corpus including the capture that exposed the problem:

| | 7 features | 5 features |
| --- | --- | --- |
| blocked OOF F1 | `90.0%` | `98.0`-`98.3%` |
| worst session FP | `100.0%` | `9.5%` |
| paired max FP | `7.16%` | `4.09%`-`5.90%` |
| paired effective alarms | `8` | `2`-`7` |
| paired worst recall | `99.14%` | `95.76`-`96.26%` |

Out-of-fold F1 sat between `98.0%` and `98.3%` across all four seeds, so the
gain is not one lucky initialisation. It also exceeds the `96.2%` that seven
features reached on the *easier* corpus without that capture.

Seed `1538882188` was exported, reaching `pass=12` on the absolute gate with
`maxFP` `4.09%` and `2` alarms.

## The promotion was forced, and what it cost

`--force-promote` was used. The candidate cleared every absolute gate but failed
per-recording non-regression on two counts: a new effective alarm on the C6
selection pair, and S3 holdout false positives moving from `7` to `17`
evaluations out of `685` against a five-evaluation margin.

The published report shows the trade in full. On reserved replays recall is
`100%` on every chip with `3` effective alarms, but weak-link ML recall on C3
falls from `100%` to `92.5%`. That comparison also spans a corpus change, since
two recovered captures entered at the same time, so the two effects are not
separable from the report alone.

This was a deliberate reset rather than a normal promotion, taken because the
failure mode being removed is structural: under seven features, one added
training capture could move a weak pair by a hundred points, and no gate would
have predicted which capture would do it.

## Alternatives Considered

### Drop the capture that exposed the problem

Rejected, and it was the first thing tried. Removing it restored every number,
which is exactly why it should stay: the fragility did not come from the
capture. Any future capture at an unusual link strength could have triggered it,
and the corpus is meant to grow.

### Keep seven features and rely on the gates

Rejected. The gates measure a candidate against reserved replays; they do not
measure how much a model's behaviour depends on which recordings happen to be in
training. Nothing in the promotion protocol would have caught this before it
landed.

## Consequences

The model is smaller and cheaper: five inputs instead of seven, `5 -> 32 -> 16
-> 1`, and the delta-series path no longer computes a mean or a standard
deviation for the feature vector.

**Worst-replay recall is lower**, `95.98%` against `99.14%`, and that is the
open question this decision leaves. A seed search over the five-feature set is
the next step and may recover it; the exported seed came from four hand-run
trials, not a search.

The low-RSSI generator needed a fitting anchor, since it tunes an impairment
magnitude and no production feature carries magnitude any more. It now measures
the mean displacement itself, under `L1_FIT_ANCHOR`, deliberately outside
`FEATURE_NAMES` so it never reaches the fit score or the exported metadata.

## Related

- [FEATURES.md](../FEATURES.md)
- [2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md](2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md)
- [2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md](2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md)
