# ADR: reduce the feature surface to the production set

- Status: Accepted
- Date: 2026-07-27

## Context

The codebase carried three tiers of feature: the seven in the exported model,
two demoted predecessors kept selectable for experiments, and a handful of
helpers and lists left over from earlier sets. Each tier cost a calc function in
both languages, a C++ id, extractor branches, and tests.

The tests were the part that hid the cost. A helper no production path calls
still looks alive when its own unit tests reference it, so a scan that counts all
references finds nothing dead. Counting only references from `src/` and `tools/`
told a different story.

The detector experiments have settled: Classic runs on the lag ratio and
turbulence autocorrelation, the MLP on Coherence-7, and the two remaining
candidates lost measurements that are already recorded.

## Decision

Exactly seven features exist, in both languages: the Coherence-7 set. There is
no candidate tier, and `ALL_FEATURES == DEFAULT_FEATURES` is asserted by a test.

Classic reads two of the seven directly and needs nothing of its own.

## What was removed, and what rejected it

| feature | C++ id | why it is gone |
| --- | --- | --- |
| `turb_skewness` | `5` | Core-6 member. Swapped out for `turb_zcr` when the coherence set collapsed seed-to-seed variance on out-of-sample false positives; see [2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md](2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md) |
| `l1_delta_waveform_length` | `23` | Core-6 member. Swapped out for `l1_delta_autocorr` in the same promotion |
| `l1_delta_cv` | none | Never shipped, and had no C++ id at all, so it could train but never deploy. The lag ratio normalises the same quantity better; see [2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md](2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md) |
| `band_power_ratio` and the band-power family | none | Removed earlier; the only noise-robust member of its family, but it never beat the L1 and turbulence features on the corpus |
| `turb_kurtosis` | none | Removed earlier, kept here so the list of dead ends is in one place |

Supporting code removed with them: `calc_skewness` and `calc_waveform_length`
in both languages, the `skewness` and `waveform_length` members of
`MLStatNeeds` and `MLSeriesStats`, and their branches in `ml_feature_source`,
`ml_series_needs`, `ml_feature_value_from_stats`, and `compute_ml_series_stats`.

Three more helpers went that no feature ever used:

- `calc_iqr` and its private percentile helper. Python-only, no C++ counterpart,
  referenced by nothing but its own seven assertions.
- `calc_l1_delta`. The plain mean of the displacement series, which is the
  metric Classic replaced with the lag ratio.
- `normalize_features` in `ml_detector.py`. A second normalisation path that
  allocated a list, while `_predict_with_workspace` normalises into a reusable
  buffer. Not in `__all__`, used only by tests, and contrary to the
  allocation-free device design it sat next to.

`CORE6_FEATURES`, `COHERENCE6_FEATURES`, and `L1_DELTA_FEATURES` are gone as
lists. The lineage they encoded lives in this file and in the promotion ADRs.

## Alternatives Considered

### Keep the two demoted members for reproducibility

Rejected. The argument was that removing them prevents re-running the Core-6
against Coherence-6 comparison. But that comparison has an ADR with its
measurements, and the code is in git history, so what is preserved is the
result, which is what a future decision needs. Keeping selectable features to
preserve the ability to re-derive a settled answer trades permanent surface for
a one-off convenience.

### Keep `--features` but restrict it to production members

Kept, in that form. The flag still allows ablation-style subsets of the seven,
which is useful, and the guard that rejects a feature without a C++ id stays as
protection for future additions even though nothing can currently trip it.

## Consequences

`csi_features.py` loses about 150 lines, `csi_features.h` loses two helpers and
four struct members, and the exporter's id map has seven entries.

**A future experiment must read this file before adding a feature back.** The
list above is the point: without it, the next person to reach for skewness or a
coefficient of variation has no way to know it was already measured and
rejected, and the git history that holds the code does not hold the reason.

Re-adding a feature now means a calc function in both languages, an id in
`MLFeatureId` and `CPP_FEATURE_IDS`, a case in `ml_feature_source`, and an
extractor branch. That is deliberate friction, proportionate to a change that
alters what every deployed device computes.
