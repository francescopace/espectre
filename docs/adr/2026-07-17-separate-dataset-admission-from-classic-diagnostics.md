# ADR: separate dataset admission from Classic diagnostics

- Status: Superseded
- Date: 2026-07-17
- Superseded by: `2026-07-29-make-dataset-quality-review-detector-agnostic.md`

## Context

`tools/validate_dataset_quality.py` serves two product needs at once:

1. keep contaminated or unusable captures out of ML training
2. reuse the same corpus later to judge `ClassicDetector` quality in host tests and performance reports

If admission is defined as "Classic already performs well on this file", the validator creates a circular chain: the dataset is filtered to what Classic solves, then Classic tests look artificially strong on that filtered set.

Raw L1 / feature-engineering gates for static/motion pair admission were also unconvincing here: if raw L1 separability were enough, Classic would not be needed. The collected corpus is treated as mostly clean unless integrity or empty/static sanity fails.

## Decision

Split the validator into admission checks and Classic indicative diagnostics.

Dataset admission may fail the run. It stays free of Classic's decision boundary:

- integrity, continuity, and signal-quality checks
- empty and `static_presence` availability / overlap checks
- ML readiness

Static/motion pair membership is not filtered by Classic or by raw L1 feature gates.

Classic stays visible but non-blocking:

- production `ClassicDetector` replay on pairs and quiet long recordings
- independent self-calibration of every `empty` and `static_presence` capture
- robust logit-margin, false-positive, drift, and activation-burst diagnostics
- an indicative 0-100 score per pair, quiet file, or idle baseline
- soft PASS/WARN against promotion-style activation targets
- Classic results never veto dataset admission

Indicative pair score:

`0.5 × idle_clean + 0.4 × motion_cover + 0.1 × ratio`

where `idle_clean` is 100 at `static_above=0` and 0 at `>=10%`, `motion_cover` is 100 at `motion_above>=95%`, and `ratio` is 100 at `p95(motion)/threshold >= 4×`. Quiet idle scores are 100 at FP=0 and 0 at FP `>=10%`.

Empty and `static_presence` are not required to be temporally paired. Each idle file is treated as a distinct baseline: Classic calibrates on the capture prefix and evaluates only its remaining tail. Its score weights self-FP cleanliness (50%), robust logit-margin stability (30%), and absence of sustained activation bursts (20%). Soft Presence/Empty review marks follow that baseline alone (`clean`, `unstable`, `motion-like`, or `motion-contaminated`).

The validator CLI defaults keep operators on one entry point: every run refreshes reciprocal pair metadata in `data/dataset_info.json` (write / `updated_at` only on real content change) and writes `data/auto_generated/DATASET_QUALITY_CHECK.md` unless `--no-report` is set. Removed one-shot flags: `--refresh-metadata`, `--report`, and `--strict`.

Detector promotion and production quality gates stay in the performance report and motion-detection tests, not in dataset admission.

## Alternatives Considered

### Keep Classic activation as a hard dataset FAIL

Rejected. It couples corpus membership to the current Classic decision boundary and creates circular validation.

### Admit pairs with raw Hampel-filtered L1 separability

Rejected. Raw L1 admission does not add trust beyond integrity/empty sanity for this corpus, and it still does not equal production Classic behavior.

### Drop Classic replay from the dataset validator

Rejected. Indicative Classic scores on the admitted corpus remain useful for spotting glaring contamination and watching detector trends. They just must not decide which files exist.

### Use Classic both to admit data and to score Classic later

Rejected. That is the circular chain this ADR exists to prevent.

## Consequences

Benefits:

- ML training admission no longer depends on Classic's current boundary
- the same admitted corpus remains a fair later test surface for Classic
- Classic soft misses become review WARNs plus a 0-100 indicative score instead of automatic data rejection

Trade-offs:

- residual label contamination may still require human review of low Classic scores / WARNs
- operators must read admission FAIL and Classic review signals as different things

## Related

- `tools/validate_dataset_quality.py`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/ML_DATA_COLLECTION.md`
