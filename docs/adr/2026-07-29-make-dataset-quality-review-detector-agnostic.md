# ADR: make dataset-quality review detector-agnostic

- Status: Accepted
- Date: 2026-07-29
- Updated: 2026-08-12

## Context

`tools/validate_dataset_quality.py` still served two distinct needs well:

1. keep contaminated or unusable captures out of ML training
2. summarize the admitted corpus for human review in `data/auto_generated/DATASET_QUALITY_CHECK.md`

The 2026-07-17 split removed Classic from dataset admission, but the review tables still depended on `ClassicDetector` replay, startup calibration, and threshold-relative language. That kept a residual circularity in the operator surface: a weak or miscalibrated detector could still make one capture look bad even when the shared production features separated it cleanly.

This became visible on low-RSSI pairs. The report could warn on detector activation while the intrinsic idle/motion separation stayed strong, which is evidence about detector limits, not about dataset corruption. The report therefore needed to speak directly about the data instead of about one detector's threshold placement.

The same change also had to preserve one source of truth. Dataset admission belongs in the validator, while detector promotion belongs in `docs/performance/README.md`; splitting the report into a second dataset-quality document would have repeated policy and metrics.

## Decision

Keep one generated dataset-quality report, but make its review metrics detector-agnostic.

Dataset admission remains unchanged and may still fail the run:

- integrity, continuity, and signal-quality checks
- empty and `static_presence` availability / overlap checks
- ML readiness
- long-recording annotation and coverage checks

The report-only review tables now derive from the shared scale-invariant feature pipeline already used elsewhere in production and training:

- the core evidence surface is the current production High Accuracy feature set, all gain-invariant by construction: `turb_iqr_over_mean_aggr`, `turb_autocorr`, `turb_zcr`, `l1_delta_lag_ratio`, `chan_shape_spread_subband`, `chan_shape_coherent_innovation_energy`, `chan_shape_excess_path`, and `chan_shape_subband_kendall_lag_excess`
- feature directions are fixed from the feature semantics, not inferred from a detector replay
- pair review compares `static_presence` and `motion` through consensus feature-evidence series
- idle review scores each capture against its own feature-space baseline

Pair rows replace threshold-relative Classic diagnostics with:

- `Cover`: share of motion windows above the idle half's own p95 evidence
- `Sep`: rank-based AUC between idle and motion evidence
- `Tail`: idle-side q95 above the pair's own centered evidence baseline
- `Score`: an indicative 0-100 ranking from separation, coverage, and idle cleanliness

Idle rows replace self-calibrated Classic baseline terms with:

- `Exc`: excursion share above median + 3 MAD on the feature-evidence axis
- `Burst`: sustained burst length measured on block-level idle baselines
- `Tail`: q95 above the capture's own centered evidence baseline
- `Drift`: half-to-half median evidence drift
- `Score`: an indicative 0-100 ranking from tail cleanliness and burst length

Threshold-relative detector terms are removed from the generated report entirely. Lightweight remains visible in detector promotion and performance surfaces, but not in dataset-quality review.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-17 | Separate dataset admission from Classic diagnostics but retain Classic review tables | Removed detector behavior from admission |
| 2026-07-29 | Make the remaining review metrics detector-agnostic | Accepted as the single dataset-quality policy |

## Consequences

Benefits:

- the dataset-quality report now describes the recordings themselves instead of one detector's startup calibration
- weak-link and gain-shifted captures stay comparable because the core metrics come from scale-invariant shared features
- excluded pairs can stay visible as informational diagnostics without reintroducing detector bias
- detector-specific problems remain visible where they belong: performance and promotion gates

Trade-offs:

- the report now owns a small feature-evidence layer and its threshold-free scoring rules, which must stay aligned with the shared feature semantics
- burst and drift review on idle captures require careful interpretation, since they summarize feature-space behavior rather than a detector's binary alarms

## Alternatives Considered

### Keep Classic review but remove only the marked threshold columns

Rejected. Even unmarked replay tables keep the operator anchored on one detector's calibration path and keep detector-specific failure modes in the dataset-quality narrative.

### Create a second agnostic report beside the current one

Rejected. Dataset-quality policy already belongs in one generated document, and performance policy already belongs in `docs/performance/README.md`.

### Score review directly from raw RSSI or absolute-energy features

Rejected. Those quantities are not robust under gain changes or weak links, so they would reintroduce the same comparability problems that motivated the change.

## Related

- `tools/validate_dataset_quality.py`
- `data/auto_generated/DATASET_QUALITY_CHECK.md`
- `docs/ML_DATA_COLLECTION.md`
- `docs/performance/README.md`
