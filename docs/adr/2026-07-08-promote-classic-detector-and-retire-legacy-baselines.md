# ADR: promote classic detector and retire legacy baselines

- Status: Accepted
- Date: 2026-07-08
- Recorded: 2026-07-09 (retrospective)
- Amended: 2026-07-16; 2026-07-17
- Supersedes: `2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`

## Context

The historical moving-variance baseline remained useful for research, but the later experiments showed that it was not the best production-facing non-ML path. The decisive motion-feature benchmark showed that L1-delta matched the historical moving-variance baseline on aggregate quality while giving a much more stable quiet floor and a stronger S3 result. A later fusion pass then closed the low-contrast recall residual by keeping L1-delta primary and adding a guarded MVS recovery vote only in the ambiguous band where MVS still carried complementary signal.

The changelog shows the same convergence: L1-delta became the primary non-ML metric, startup calibration became motion-first with an internal quiet-first fallback, and tooling moved away from historical metadata thresholds toward the current runtime calibration behavior.

The older MVS startup-threshold and online-adaptation sweep remains part of the historical context because it established the strongest version of the legacy variance baseline before that baseline was retired. That work still matters for offline comparison, but it no longer defines the active runtime direction.

The 2026-07-16 amendment replaced the L1-primary plus recovery-vote Classic path with weighted probability fusion. The 2026-07-17 follow-through removed the remaining `auto` / `min` / `manual` threshold modes so the runtime surface matches that probability detector: automatic startup placement, session-only overrides, and Hampel-consistent Classic and ML feature streams.

## Decision

Promote `ClassicDetector` as the production non-ML path and retire the legacy baselines from the active runtime path.

The original L1-primary plus recovery-vote design was superseded after grouped, de-overlapped feature analysis and gain-stressed replay showed that a direct two-feature fusion was both simpler and stronger. The accepted Classic direction is now:

- weighted logistic fusion of `l1_delta` and `turb_autocorr`
- no voting or conditional recovery branch
- Hampel filtering on Classic per-packet feature streams under one master switch, with the same Hampel policy applied to ML feature streams so train, validate, and runtime stay aligned
- a shared `0.0-1.0` probability threshold scale for Classic and ML
- automatic threshold placement at startup: Classic adapts the trained probability boundary from the session startup `q95` in logit space; ML uses its trained default
- no config-time threshold modes (`auto`, `min`, or `manual`); both detectors accept session-only runtime overrides, and recalibration or reboot restores the automatic value
- identical Python and C++ feature, filtering, fusion, and calibration behavior

Historical variance-baseline tooling may remain for offline comparison, but it is no longer the production runtime reference.

## Alternatives Considered

### Keep moving variance as the production default

Rejected. It remained useful as a benchmark, but the later Classic path offered better quiet-floor stability and a clearer production calibration story.

### Switch to a fixed factory threshold

Rejected. The experiments did not support one fixed threshold as a safe cross-session production policy.

### Keep the gated variance recovery vote

Rejected after the amended comparison. It added branching and calibration state, while the weighted autocorrelation fusion produced better paired recall, lower long-recording false positives, and stronger raw-CSI gain stress results.

### Keep config-time threshold modes (`auto`, `min`, `manual`)

Rejected. Once Classic and ML share a probability scale and automatic startup placement, mode selection adds config surface without a durable production benefit. Session overrides already cover temporary sensitivity changes; `min` duplicated maximum-sensitivity tuning that is better expressed as a lower session threshold; compile-time or YAML manual defaults fought the automatic startup path.

### Keep the L1-primary gated `max x factor` calibration as the Classic policy

Rejected for production Classic after the weighted probability amendment. The contamination analysis in `2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md` remains useful historical context, but the active Classic path no longer uses that numeric proposal or its threshold modes.

## Consequences

Benefits:

- the project has one clear, documented non-ML production detector
- runtime calibration, frontend controls, and validation workflows align on the same automatic-plus-session-override behavior
- Classic and ML share one probability threshold scale and Hampel feature-stream policy
- older baselines and gated L1 calibration remain available as historical references without driving the active path

Trade-offs:

- offline comparisons must distinguish research baselines from the production detector
- legacy metadata, threshold sweeps, and mode-bearing configs required cleanup
- temporary sensitivity changes are session-scoped; they are not persisted as a compile-time or YAML threshold mode

## Related

- `docs/adr/2025-11-16-adopt-segmentation-first-mvs-architecture.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/adr/2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`
- `docs/adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`
- git commits: `dc0658ed`, `5b871159`, `dbbe21dd`, `b2e0de00`, `8641425d`, `acec4a2c`, `a593edb1`
