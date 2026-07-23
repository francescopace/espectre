# ADR: use core-6 as the production ml feature set

- Status: Superseded
- Date: 2026-07-07
- Recorded: 2026-07-09 (retrospective)
- Superseded by: 2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md

## Context

The project had already moved through several ML baselines, including raw
feature sets, relative-8 turbulence features, weighting sweeps, and gain-shift
hardening. The experiment history records that progression and the reasons the
earlier baselines were superseded.

The immediate predecessor was a clean-dataset refresh of the relative-8 line.
That pass established two important constraints before any new promotion:

- `waveform_length_over_mean` was the weak relative-8 feature and should be
  removed first
- a one-for-one `l1_delta` substitution did not improve the MLP; the signal had
  to be reconsidered as a richer descriptor, not as a single scalar drop-in

The final promotion decision then came from the Core-6 mixed turbulence and
L1-delta work, which compared the refreshed relative baseline against a larger
mixed descriptor and reduced it to the six features that actually contributed
unique signal.

## Decision

Use the mixed six-feature "Core-6" set as the production ML baseline:

- `turb_mad_over_mean`
- `turb_skewness`
- `turb_autocorr`
- `l1_delta`
- `l1_delta_std`
- `l1_delta_waveform_length`

Keep the exported runtime artifacts and the C++ feature extractor aligned with
that feature set, using exported feature identifiers instead of inferring the
feature path from model input count alone.

## Alternatives Considered

### Keep the relative-8 turbulence baseline

Rejected. Core-6 improved grouped-CV quality, reduced long-quiet false
positives substantially, and stayed slightly lighter.

### Use a pure L1-delta descriptor

Rejected. It reached parity in some aggregate metrics, but it blurred the
static-presence versus motion boundary too much and was not robust enough on its
own.

### Keep larger mixed descriptors

Rejected. Ablation showed that several added L1-delta statistics were
redundant, and some removals improved generalization.

## Consequences

Benefits:

- the ML path becomes more robust to RF interference and quiet-floor drift
- production artifacts are lighter than the previous baseline
- Python and C++ now share a more explicit feature-parity contract

Trade-offs:

- older ML benchmarks must be interpreted as historical baselines
- future ML experiments should compare against Core-6 explicitly, not against
  older relative-8 numbers alone

## Follow-Up Validation

On 2026-07-16, deterministic grouped out-of-fold SHAP for the promoted seed
`1386543369` assigned `turb_skewness` the lowest mean absolute contribution at
`1.9%`. A targeted same-seed ablation therefore compared Core-6 against Core-5
without `turb_skewness`, using grouped blocked CV, paired validation, and all
curated long recordings.

Removing `turb_skewness` improved blocked OOF F1 from `92.4%` to `93.5%`, and
both candidates passed the paired gate with a `0.0%` maximum paired FP rate.
The long-recording gate nevertheless regressed: total false positives increased
from `601` to `979`, and the maximum FP rate increased from `0.9%` to `1.3%`.

Keep `turb_skewness` in Core-6. This follow-up confirms that low aggregate SHAP
importance does not imply safe removal when the feature protects quiet-run
robustness.

## Related

- `docs/adr/2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`
- `docs/adr/2026-06-29-record-historical-ml-baseline-evolution-before-core-6.md`
- `docs/adr/2026-02-15-share-ml-model-artifacts-between-python-and-cpp.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/adr/2026-07-02-use-pytorch-as-the-host-training-stack.md`
- git commit: `2217271d`
