# Historical Experiments

This document records notable host-side experiments that informed production
choices over time, including both rejected candidates and experiments that
eventually promoted a runtime baseline.

The goal is to preserve design history in one place without turning
`ALGORITHMS.md` into a research log.

Historical labels such as `MVS` are preserved in experiment names where they
match the actual benchmark, weighting mode, or training variant that was run.

## Overview

| Date | Experiment | Status | Main Lesson |
|------|------------|--------|-------------|
| 2026-05-20 | Feature-set reduction sweep | Superseded by relative-8 and AGC-active normalization | Dropping unstable raw-window features improved the raw-9 line. |
| 2026-05-20 | FP-first feature and training sweep | Superseded by later ML baselines | Long-recording false positives must outrank CV-only wins. |
| 2026-05-20 | Raw-9 MLP topology sweep | Superseded by relative-8 | `32 -> 16` was the best raw-9 topology among tested candidates. |
| 2026-05-20 | Tiny CNN / TCN sweep | Rejected | Small temporal models did not beat the MLP FP-first ranking. |
| 2026-06-29 | Gain-shift robustness diagnostic | Superseded as a gate by AGC-active normalization | Raw turbulence features were not structurally gain-invariant. |
| 2026-06-29 | Relative-8 topology and FP-weight sweep | Superseded by MVS hard-negative retrain, then by AGC-active clean baseline work | Relative features solved uniform gain-shift sensitivity but not all quiet long-run false positives. |
| 2026-06-30 | C3 empty-room retrain incident | Partly superseded; lesson retained | `empty` captures must be first-class IDLE training data. |
| 2026-07-03 | MVS-guided weighting bias and hard-negative retrain | Superseded by AGC-active normalization and clean data recollection | MVS can help as hard-negative mining, but it can also import MVS quiet-spike bias. |
| 2026-07-04 | Multi-device sync and phase research | Active research note | `stimulus_id` and reference metadata remain useful for future multi-device experiments. |
| 2026-07-05 | MVS startup-threshold and online adaptation sweep | Historical variance-baseline note | `max x 1.3` remained the safest moving-variance baseline; online threshold tracking helped some chips, but not all. |
| 2026-07-05 | Motion-feature benchmark and L1-Delta promotion | Promoted into `ClassicDetector` | L1 profile displacement matched moving-variance quality with a far more stable quiet level; no candidate supported a fixed factory threshold. |
| 2026-07-06 | L1-Delta startup-threshold and online recovery sweep | Superseded by the contaminated-calibration gate sweep | Clean data still favor static `max x 1.1`, but startup-spike recovery is real; the best no-buffer candidate is a conservative decaying-peak tracker. |
| 2026-07-06 | L1-Delta contaminated-calibration gate and extension sweep | Promoted to runtime (Python and C++) | A floor-anchored rolling-chunk consistency gate with calibration extension keeps F1 >= 94.3% from clean startup up to 100% contaminated startup. |
| 2026-07-06 | Clean relative-8 refresh and L1-Delta ML feature check | Active ML baseline candidate | Removing `waveform_length_over_mean` improved the Python ML baseline; `l1_delta` remained useful as a Classic support signal, but not as a winning MLP feature. |

## Current Superseding Events

Date: 2026-07-07

Status: Active ML production baseline.

The production ML feature set is now the mixed turbulence + L1-delta "Core-6"
set (`turb_mad_over_mean`, `turb_skewness`, `turb_autocorr`, `l1_delta`,
`l1_delta_std`, `l1_delta_waveform_length`), replacing the relative-8
turbulence set. It wins every promotion gate (see the Core-6 entry below) and
is slightly lighter than relative-8. `features.DEFAULT_FEATURES` and the
exported artifacts (`ml_weights.h`, `ml_weights.py`, `ml_test_data.npz`) reflect
Core-6, and the C++ feature extractor is now driven by the exported
`ML_FEATURE_IDS` instead of the model input count.

Date: 2026-07-04

Status: Active baseline reset.

The runtime and training path now use AGC-active, coefficient-of-variation
turbulence (`std(amplitudes) / mean(amplitudes)`) as the single production
path. Earlier dataset captures may also have been contaminated by a collection
bug. For that reason, the next clean ML baseline starts from
`--sample-weight-mode none`; moving-variance-guided weighting should be re-evaluated only
after clean data collection and refreshed explicit pair metadata.

## Active Research Notes

### Core-6 Mixed Turbulence + L1-Delta ML Feature Set

Date: 2026-07-07

Status: Promoted to production. Supersedes the "do not add `l1_delta` to the
MLP yet" decision in the "Clean Relative-8 Refresh And L1-Delta ML Feature
Check" entry below.

#### Goal

Decide whether L1-delta profile displacement improves the ML detector as a full
statistical descriptor rather than the single scalar rejected on 2026-07-06,
targeting a model that is more robust to RF interference and has a more stable
quiet floor without growing inference cost or memory.

#### Background

The relative-8 production set is built entirely on spatial turbulence
statistics, so ML inherits the turbulence-path weaknesses (threshold drift, RF
sensitivity, the S3 chip gap). The shared L1-delta axis is a different signal
(normalized amplitude-profile displacement) with a session-stable quiet floor. The
2026-07-06 check fed `l1_delta` to the MLP as a single window-mean scalar and
found no win. Two observations reopened it:

- the turbulence path enters the MLP as eight statistics, while `l1_delta`
  entered as one, so the comparison was not like-for-like
- the training and gate feature paths did not maintain the amplitude-profile
  history, so any `l1_delta*` feature silently resolved to `0.0` at evaluation
  time (a dead constant); this was fixed in the trainer `StreamingEvaluator`
  and the MicroPython `ml_detector.py` guard before any run below

#### Method

`features.py` gained an L1-delta descriptor (`l1_delta` plus `l1_delta_std`,
`l1_delta_max`, `l1_delta_min`, `l1_delta_iqr`, `l1_delta_mad`,
`l1_delta_waveform_length`, `l1_delta_skewness`, `l1_delta_autocorr`), the same
statistics the turbulence path uses, computed on the per-packet L1-delta series.
Sequence of single-variable runs at fixed seeds (`42`, `123`, `7`), production
architecture, scaler, and `fp_weight`:

1. pure L1-delta descriptor (nine features, no turbulence)
2. mixed set (relative-7 turbulence + nine L1-delta) at 16 features
3. leave-one-out ablation on the mixed set
4. the surviving lean candidate, re-validated on grouped CV plus the in-memory
   paired and long-quiet gates (`evaluate_paired_gate` / `evaluate_long_gate`),
   compared head-to-head against the relative-8 baseline

#### Results

Pure L1-delta reached parity with the turbulence baseline on grouped CV
(`OOF F1 79.7%` vs `79.5%`) but with high, session-unstable false positives
(fold FP `8.6%`, C3 static-presence up to `91.6%`): L1-delta alone blurs the
static-presence / motion boundary that turbulence separates well.

The 16-feature mixed set reached `OOF F1 83.0%` at seed 42 and fixed the C3
false-positive spike. Leave-one-out ablation then showed the descriptor was
mostly redundant: `turb_autocorr` is critical (`-8.3%` if removed), and only
three L1-delta features carry unique signal (`l1_delta`,
`l1_delta_waveform_length`, `l1_delta_std`); the remaining spread statistics are
redundant, and removing several of them *improves* generalization.

The resulting six-feature "Core-6" set —
`turb_mad_over_mean`, `turb_skewness`, `turb_autocorr`, `l1_delta`,
`l1_delta_std`, `l1_delta_waveform_length` — wins every gate (median of seeds
`42`, `123`, `7`):

| Metric | relative-8 baseline | Core-6 |
|--------|--------------------:|-------:|
| Grouped-CV OOF F1 | 79.8% | 83.1% |
| Long-quiet total FP | 1846 | 552 (-70%) |
| Long-quiet max FP rate | 1.9% | 0.8% |
| Paired pass count | 3 | 3 (non-regression) |
| Paired max FP rate | 2.6% | 0.6% |
| Model | 8 -> 32 -> 16 -> 1 | 6 -> 32 -> 16 -> 1 |
| Params / weights / inference | 833 / 3.3 KB / ~39 us | 769 / 3.0 KB / ~38 us |

Long-quiet false positives, the promotion-governing gate
(`MAX_PROMOTION_TOTAL_FP_INCREASE = 0`), drop by roughly 70 percent, and Core-6
is slightly lighter than the model it replaces.

#### C++ Parity And Promotion

The C++ extractor previously inferred the feature set from the model input count
(8 = relative, 9 = raw). It is now driven by an exported `ML_FEATURE_IDS` array
(mirroring the `MLFeatureId` enum in `features.h`); `extract_ml_features_by_id`
replaces the count-based extractor, and `MLDetector` maintains an incremental
profile and delta ring, capacity `window_size - lag`, that exactly reproduces
`features.l1_delta_series`. Parity was verified two ways: `compute_ml_series_stats`
matches the Python statistics at float32 precision, and the incremental delta
series is a bit-exact match to the windowed Python reference.

#### Decision

Promote Core-6 to production. `features.DEFAULT_FEATURES` is Core-6, and the
exported `ml_weights.h`, `ml_weights.py`, and `ml_test_data.npz` are regenerated
accordingly.

#### Follow-Up

1. Refresh the ML table in `docs/PERFORMANCE.md` with Core-6 numbers.
2. The per-recording strict gates (`test_motion_detection`,
   `TestPerformanceMetrics`) fail on hard chips (S3, some C3 and C6) on the
   recollected dataset; these are dataset-driven and pre-existing (they also fail
   for the untouched MVS detector), and Core-6 ML is at least as good as the
   relative-8 baseline on the shared cases. Track that dataset/threshold work
   separately.
3. Revisit whether a cheap recursive band-power veto (no FFT window) adds
   interference robustness on top of Core-6 without hurting latency.

---

### Multi-Device Sync and Phase Research

Date: 2026-07-04

Status: Active research note.

#### Goal

Evaluate whether collector-driven `stimulus_id` tagging and optional reference
frames are worth preserving in raw datasets, even when promoted ML baselines are
amplitude-first and usually measurement-only.

#### What The Experiments Show

Historical multi-device experiments showed that `stimulus_id` is the practical
key for cross-device grouping, while frame-level metadata is more useful as
diagnostics than as a guaranteed global identifier. They also showed that:

- packet grouping quality can be very strong even on standard hardware
- raw inter-node phase remains much noisier than packet grouping alone suggests
- reference-assisted paths are still experimental and have not yet displaced the
  best simpler compensated baselines
- amplitude-first single-link inference was more credible than phase-heavy or
  fused multi-link baselines in the tested setup

#### Practical Interpretation

So the presence of `stimulus_id` and optional reference flags in collected
datasets is intentional, not accidental. They are low-cost metadata for future
research tracks, even when a given ML dataset is used today only for the
ordinary measurement-only path.

#### Why The Metadata Stays

Those fields are kept because they remain the practical bridge from ordinary
dataset collection to future host-side experiments that need temporal
association across devices, including:

- `stimulus_id`-anchored multi-device packet grouping
- reference-assisted phase-coherence experiments
- temporally aligned multi-device feature fusion

---

### MVS Startup-Threshold And Online Adaptation Sweep

Date: 2026-07-05

Status: Historical variance-baseline note. The runtime default is now
`ClassicDetector`; this section is retained because the moving-variance
baseline is still useful in offline sweeps.

#### Goal

Test whether the current moving-variance startup threshold and simple online adaptation can
reduce static-room false positives without paying too much recall, before any
runtime C++ changes.

#### Background

The then-current production moving-variance startup path used:

- fixed production subcarriers
- AGC-active coefficient-of-variation turbulence
- startup adaptive threshold from baseline moving variance
- `max x 1.3` as the default startup multiplier

The two main hypotheses tested were:

1. lower startup threshold (`max x 1.1`) might recover weak-motion recall
2. online idle-only threshold tracking might recover some false-positive drift
   after startup

A second candidate path tested slow per-subcarrier EMA normalization:

`amp_norm[i] = amp[i] / ema_amp_baseline[i]`

followed by the usual CV turbulence metric on the normalized amplitudes.

#### Tooling Added

The following tools were realigned to run production-aligned paired sweeps over
all explicit `static_presence` / `motion` pairs from `data/dataset_info.json`:

- `tools/lib/variance_baseline_core.py`: shared pair iterator, startup calibration, and
  continuous baseline -> motion evaluator
- `tools/5_analyze_filter_turbulence.py`: main variance sweep and prototype
  comparison entry point
- `tools/6_optimize_filter_params.py`: paired filter optimizer that now reuses
  the same sweep core instead of latest-file heuristics

#### Global Result

Current all-pairs production baseline:

- recall `96.8%`
- precision `93.6%`
- FP rate `3.3%`
- F1 `95.2%`

Online threshold tracking became active after the sweep harness exposed update
gates and candidate-threshold diagnostics. The best tested global compromise
was approximately:

- startup threshold: `max x 1.1`
- online tracking: idle-only, `p99 x 1.15`, margin `0.95`, transition guard `50`

That variant improved aggregate precision and FP slightly, but it was not a
clear promotion candidate:

- recall `96.3%`
- precision `94.3%`
- FP rate `2.9%`
- F1 `95.3%`

Interpretation: the gain is real but marginal, and it is not uniform across
chips.

#### Chip-Specific Result

The useful diagnostic was to compare startup `max x 1.1` directly against the
current `max x 1.3` baseline on chips that are sensitive to the threshold
choice.

S3 is the clearest negative case:

- `max x 1.3`: recall `91.3%`, FP `12.3%`
- `max x 1.1`: recall `93.2%`, FP `14.6%`

Online threshold tracking on top of `max x 1.1` reduced some of that extra FP,
but not enough to beat the safer startup threshold:

- conservative tracking: recall `92.0%`, FP `14.1%`
- medium tracking: recall `92.5%`, FP `13.8%`

So for S3, online tracking did not rescue the lower startup threshold.

C6 is more encouraging:

- `max x 1.3`: recall `98.9%`, precision `95.9%`, FP `2.3%`
- `max x 1.1`: recall `99.0%`, precision `92.6%`, FP `4.7%`

Online threshold tracking recovered most of that regression:

- conservative tracking over `1.1`: recall `99.0%`, precision `95.1%`, FP `2.8%`
- medium tracking over `1.1`: recall `99.0%`, precision `94.6%`, FP `3.2%`

So for C6, threshold tracking can act as a partial safety net for a more
sensitive startup threshold, although it still did not beat the plain `1.3`
baseline.

#### Per-Subcarrier EMA Normalization Result

The slow per-subcarrier EMA normalization path did not hold up in the all-pairs
sweep.

Representative runs:

- `alpha=0.001`: recall `84.6%`, FP `4.5%`, F1 `87.4%`
- `alpha=0.0005`: recall `85.1%`, FP `4.5%`, F1 `87.7%`

The main issue was not only update speed. The worst C3 paired datasets suffered
severe recall collapse, while some C6 datasets still produced large false
positive spikes. This path remains exploratory and is not close to promotion.

#### Decision

Keep `max x 1.3` as the default global moving-variance startup threshold.

Do not port online threshold tracking or per-subcarrier EMA normalization into
the runtime yet.

The current evidence supports these narrower conclusions:

- online threshold tracking is chip-dependent, not a universal fix
- S3 remains a fragile false-positive case under a lower startup threshold
- C6 may justify further chip-specific host-side tuning if future work needs a
  more aggressive startup threshold
- per-subcarrier EMA normalization is currently too unstable to justify runtime
  work

#### Follow-Up

If this line of research is resumed, the next sensible experiment is not a
global promotion attempt, but a chip-specific host-side study:

1. keep `max x 1.3` as the default control
2. test capped and less frequent threshold tracking only on chips like C5/C6
3. treat S3 as a non-regression chip, not as a candidate for lower startup
   sensitivity

### Motion-Feature Benchmark And L1-Delta Promotion

Date: 2026-07-05

Status: Promoted into `ClassicDetector` as the primary non-ML metric.

#### Goal

Find a per-packet motion metric that discriminates quiet from motion as well
as the MVS moving variance but with a more sustainable threshold over time:
robust to AGC, RF noise, and chip differences, and less dependent on fragile
session-specific startup thresholds.

#### Physical Framing

Motion changes the multipath, so the channel profile `H(f, t)` decorrelates
over time coherently across subcarriers. The AGC applies a scalar per-packet
gain and PLL/STO/CFO randomize absolute phase, so a robust metric must be
invariant to per-packet scale and measure the temporal displacement of the
normalized spectral shape, not absolute energy.

#### Method

Offline benchmark over the repo datasets (paired `static_presence -> motion`
plus `empty` captures; 4 chips x 3 environments), all candidates causal with
windows matched to MVS (~1 s at 100 pps). The MVS baseline reproduction was
verified against `SegmentationContext` to a relative error of `2e-6`.

Protocols:

1. Per-pair AUC (threshold-free separability)
2. Session-calibrated threshold, production semantics
   (`max(calibration) x factor`), with the factor swept per candidate for a
   fair comparison against the MVS-tuned `1.3`
3. Leave-one-chip-out universal threshold (fixed threshold chosen on the other
   chips), in best-F1 and quiet-quantile variants
4. Empty-room FPR, temporal stability (CV, drift), and quiet-median spread
   across sessions
5. Synthetic perturbations applied after clean calibration: AGC ramp/step,
   amplitude spikes, broadband AWGN (5% and 15% RMS), narrowband bursts

One S3 hobby-room pair was found contaminated during the run (static capture
with motion-level turbulence throughout, confirmed by
`tools/11_validate_dataset_quality.py`) and was later removed from the
dataset; headline numbers exclude it.

#### Candidates And Results

Session-calibrated protocol at each candidate's best global factor, valid
pairs only:

| Candidate | AUC mean | Recall | FP rate | F1 | Verdict |
|-----------|----------|--------|---------|----|---------|
| `l1_delta` (lag-10 L1 profile displacement, factor 1.1) | 0.993 | 93.2% | 2.5% | 94.2% | Promoted |
| `mvs` baseline (factor 1.3) | 0.989 | 94.3% | 3.2% | 94.1% | Baseline |
| `turb_cv` (std/mean of turbulence window) | 0.987 | 92.3% | 4.9% | 91.6% | ML feature only |
| `turb_madratio` (MAD/median) | - | 90.0% | 6.4% | 89.1% | ML feature only |
| `band_power_ratio` (0.5-10 Hz energy fraction, window 256) | 0.978 | 82.3% | 6.0% | 85.1% | Reserved as RF-noise gate |
| `eigen_ratio` (top eigenvalue / trace) | 0.887 | 84.5% | 10.3% | 82.9% | Rejected |
| `corr_amp_d10` (1 - Pearson at lag 10) | 0.887 | 78.4% | 7.1% | 81.8% | Rejected |
| `corr_complex_d10` (1 - complex correlation) | 0.830 | 56.2% | 6.8% | 66.5% | Rejected |
| `corr_amp_d1` (1 - Pearson at lag 1) | 0.678 | 25.8% | 1.9% | 39.9% | Rejected |

Key robustness results:

- `l1_delta` per-chip recall is uniform (89-96%, including S3 where MVS drops
  to ~80%), its quiet score has the lowest temporal CV (0.084), and its
  quiet-median spread across sessions is <=1.3x versus up to 14.5x for MVS.
- All per-packet-normalized candidates, MVS included, are exactly invariant to
  scalar AGC perturbations (gain ramp/step produced zero metric change).
- All dispersion-style metrics (MVS, `l1_delta`, `turb_*`, `corr_*`) fail open
  if broadband noise rises after calibration (FPR ~90% under +5% RMS AWGN).
  Only `band_power_ratio` fails closed (FPR drops, recall drops) and it is
  also immune to narrowband bursts.
- No candidate, including fusions, supports a fixed factory threshold:
  leave-one-chip-out thresholds collapse recall or explode FPR on the held-out
  chip. Startup calibration remains necessary; `l1_delta` makes it far less
  fragile rather than unnecessary.

Why the rejected candidates fail:

- Pearson/complex correlations are dominated by the static frequency-selective
  fading shape, which dwarfs the motion perturbation (worst case AUC 0.23 on a
  C5 pair); the complex variant additionally absorbs per-packet STO/CFO phase
  noise.
- Lag-1 differences measure receiver noise: at 10 ms the body has not
  displaced the multipath yet.
- `eigen_ratio` is chip-inconsistent and the most expensive candidate.

#### AND Fusion (Held In Reserve)

`l1_delta (factor 1.1) AND band_power_ratio (factor 1.0)` with session
thresholds: F1 93.4%, FP 1.1%, and FPR under post-calibration AWGN reduced
from ~91% to ~6.5% at a ~4-point recall cost. Deferred because the RF-noise
fail-open scenario has only been demonstrated synthetically, MVS shares the
same vulnerability (no regression), and the gate costs a 256-packet window
(~2.6 s confirmation latency) plus per-hop FFTs. Revisit if long quiet runs
show false positives correlated with RF events (`noise_floor_dbm`, RSSI).

#### Live Validation

Loopback UDP replay of repo captures and a live C3 session with
`./espectre collect --no-save --detector classic,ml` confirmed the
offline picture: independent per-detector startup calibration, stable IDLE in
quiet (metric at ~91% of threshold by construction, see
[ALGORITHMS.md](ALGORITHMS.md)), and clean IDLE -> MOTION transitions with the
metric at ~2x threshold during movement.

#### Computational Cost

Measured on the Python reference implementations (10k real packets):
`l1_delta` is ~20% cheaper than MVS+Hampel per packet in the firmware-like
path (evaluation every 25 packets) and ~50% cheaper when evaluating every
packet, because its running-mean update is O(1) versus the O(window) two-pass
variance and it needs no Hampel sorting; its hot path is allocation-free like
the shared turbulence path. It uses ~100 extra floats of state.
Details in [ALGORITHMS.md](ALGORITHMS.md).

#### Decision

Promote `l1_delta` as the primary metric inside `ClassicDetector` with startup
factor `1.1`, and use the multi-detector live collect for side-by-side
validation against `ml`. Next gates before the full cross-stack promotion:

1. cross-session threshold stability live (expect <=~1.3x spread)
2. long quiet runs for the real false-positive rate at factor 1.1
3. S3 side-by-side, where the offline gap versus MVS is largest

---

### L1-Delta Startup-Threshold And Online Recovery Sweep

Date: 2026-07-06

Status: Active research note.

#### Goal

Test whether `l1_delta` can keep the simple startup threshold
`max(calibration) x 1.1` while recovering automatically from a noisy startup in
real deployments, where the nominal quiet-room calibration may include
micromovements, RF disturbances, or a not-quite-static device.

#### Background

The initial offline promotion used controlled paired datasets, where startup
calibration is relatively clean. That environment answers the question "is
`max x 1.1` a good session threshold in ordinary quiet captures?" but not the
more deployment-oriented question "can the detector recover if startup itself is
contaminated?"

The working assumption for this sweep was:

- the `l1_delta` quiet floor is structurally stable enough that startup
  contamination should be easier to repair than for MVS
- startup contamination matters more than the clean-room optimum, because the
  runtime can ask the user to stay still, but cannot guarantee a perfectly quiet
  environment

#### What Was Evaluated

Host-side sweeps reused the current `l1_delta` production semantics:

- startup threshold from the ready-state calibration metric
- threshold formula anchored to `max(calibration) x 1.1`
- continuous baseline -> motion evaluation on the paired real-data datasets

To stress startup robustness, the startup window was synthetically contaminated
with movement packets before running the normal clean baseline -> motion
evaluation:

- tail contamination at 5%, 10%, 15%, and 20% of startup
- sparse contamination at 10% of startup

The following threshold families were compared:

1. static startup threshold only (`max x 1.1`)
2. alternate startup statistics (`p95`, `p98`, `p99`, `p99.5`, `p99.9`,
   `mean + k*std`)
3. switch/capped hybrids such as "use `p95` when `max/p95` looks suspicious"
4. online threshold tracking without a new buffer (`decaying peak`)
5. exact moving-max tracking with a minimal additional buffer
6. min-based ideas (`moving min`, and min+range variants)

#### Startup-Only Sweep Result

On clean paired datasets, the startup calibration window already shows that
`max` is much less pathological for `l1_delta` than for MVS:

- median `max / p99` during startup: `1.0079x`
- p90 `max / p99`: `1.0196x`
- worst observed `max / p99`: `1.0415x`

So on clean data, `max` is already close to the upper quiet edge rather than an
isolated spike. That explains why the clean all-pairs sweep still favors the
simple static policy:

- `max x 1.10`: recall `94.1%`, FP `2.3%`, F1 `94.7%`
- `max x 1.08`: recall `95.2%`, FP `2.9%`, F1 `94.7%`
- `p95 x 1.10`: recall `96.3%`, FP `3.8%`, F1 `94.5%`

Interpretation: on clean sessions, replacing `max` with a lower quantile buys
recall mostly by spending more quiet-room FP.

#### Contaminated-Startup Result

Under synthetic startup contamination, lower-quantile startup thresholds became
useful. For example, under 10% tail contamination of startup:

- `max x 1.10`: recall `85.8%`, FP `0.95%`, F1 `91.4%`
- `p98 x 1.10`: recall `89.6%`, FP `1.36%`, F1 `93.2%`
- `p95 x 1.10`: recall `93.5%`, FP `2.37%`, F1 `94.3%`

This confirmed the expected trade-off:

- static `max` is still the clean-data winner
- lower startup quantiles are more tolerant of dirty startup
- but they pay with higher quiet FP when startup was already clean

#### Online Recovery Result

The strongest no-new-buffer candidate was an online decaying-peak tracker:

- reference update: `ref = max(metric, ref * 0.998)`
- tracking gate: only when `state == IDLE` and `metric < threshold * 0.70`
- threshold update: `threshold = min(threshold, max(ref * 1.1, startup * 0.85))`

This candidate (`peak9980_safe70_floor85`) outperformed static `max x 1.1`
whenever startup contamination was the dominant problem:

| Policy | Clean F1 | Tail 10% F1 | Tail 20% F1 | Sparse 10% F1 | Quiet FP |
|--------|----------|-------------|-------------|---------------|----------|
| `static max x 1.1` | 94.66% | 91.40% | 59.81% | 95.49% | 2.08% |
| `peak9980_safe70_floor85` | 93.56% | 94.21% | 77.93% | 93.47% | 6.34% |

So the aggressive decaying-peak policy is a good startup-recovery mechanism,
but not a universal default:

- better when startup is genuinely dirty
- worse on clean sessions and on quiet-only FP

#### Conservative Online Recovery Result

A milder family reduced that FP cost while still beating static `max x 1.1` on
the contaminated-startup stress tests. The best practical compromise from the
focused sweep was:

- `ref = max(metric, ref * 0.9995)`
- only update when `state == IDLE` and `metric < threshold * 0.60`
- clamp to at least `0.90 x startup_threshold`

`peak9995_safe60_floor90`:

- clean F1 `94.04%`
- tail 10% F1 `93.25%`
- tail 20% F1 `70.62%`
- sparse 10% F1 `94.62%`
- quiet FP `2.46%`

Compared with the aggressive candidate, this variant gives back some startup
recovery, but it stays much closer to the quiet-room behavior of static
`max x 1.1`.

#### Rejected Directions

Several ideas were explicitly tested and did not beat the decaying-peak line:

- exact moving-max tracking over a small new buffer of recent metrics
- moving-min and min+range heuristics
- quantile-tracking proxies that effectively collapsed toward the floor

Key reasons:

- exact moving-max with a tiny buffer still dropped the threshold too abruptly
  and produced more quiet FP than the best decaying-peak line
- the `min` is indeed more stable across sessions than `max`, but it tracks the
  lower quiet edge, not the upper quiet edge where the threshold must sit
- permanently active downward-only quantile or min tracking tends to ratchet the
  threshold too low unless it is heavily clamped, at which point it stops
  outperforming the simpler peak tracker

#### Decision

Keep static `max x 1.1` as the default startup threshold for now.

For future runtime experimentation, the evidence supports two distinct paths:

1. conservative control: keep static `max x 1.1`
2. recovery candidate: test a guarded decaying-peak line, starting with
   `peak9995_safe60_floor90`

Do not promote quantile-only startup replacement, moving-min tracking, or the
minimal moving-max buffer variants into the runtime yet.

#### Follow-Up

The next live validation should not be another broad offline sweep. The next
useful gates are:

1. long real quiet runs comparing `static max x 1.1` versus
   `peak9995_safe60_floor90`
2. at least one deliberately noisy startup session to verify that the decaying
   peak actually repairs threshold overshoot in a live deployment
3. S3-specific live side-by-side, because that chip remains the strongest
   cross-session false-positive risk

---

### L1-Delta Contaminated-Calibration Gate And Extension Sweep

Date: 2026-07-06

Status: Promoted to the Micro-ESPectre and shared C++ runtimes; supersedes
the startup-threshold and online recovery sweep decision above.

#### Goal

Pick the final `l1_delta` startup-threshold policy for deployments where the
quiet-room calibration cannot be guaranteed, closing the open question from
the same-day startup-threshold and online recovery sweep.

#### What Changed Versus The Previous Sweep

Contamination was made realistic: calibration packets were replaced with real
motion packets from the paired motion capture instead of milder synthetic
perturbations. Under real-motion contamination the static policy collapses
much harder than previously measured, because the calibration max lands at
motion level and recall fails closed:

- `static max x 1.1`: F1 `51.5%` at tail 10%, `24.7%` at tail 20%, `3.4%` at
  tail 100% (motion during the entire calibration)

Scenario set: clean, tail contamination at 5/10/15/20/30/40/60/100% of the
1000-packet calibration window, mid-window blocks at 10/20%, and sparse 10%.
All 11 explicit 64-subcarrier pairs (C3/C5/C6/S3) were evaluated with
production semantics (ready-state metric, `max(calibration) x 1.1`,
continuous baseline -> motion pass). Clean-startup tightness reproduced the
previous note exactly (median `max/p99` `1.0079`, worst `1.0415`).

#### Winner: Rolling-Chunk Consistency Gate With Calibration Extension

Device state cost is `k + 2` floats; no metric buffer:

1. group ready-state calibration metrics into `k = 6` chunks (~150 samples);
   keep a ring of per-chunk maxima and the minimum chunk max ever observed
2. accept calibration only when both hold:
   - spread gate: `max(ring) <= 1.10 x median(ring)`
   - floor anchor: `median(ring) <= 1.5 x min_chunk_ever`
3. on rejection, keep calibrating one chunk at a time (ring slides), up to
   +2000 packets; on budget exhaustion fall back to `median(ring) x 1.1`
4. on acceptance, threshold = `max(ring) x 1.1` — the unchanged production
   formula applied to the accepted window

The floor anchor is what fixes majority-homogeneous contamination (tail 60%),
where a spread-only gate accepts a motion-level ring as "consistent"
(F1 `80-86%` without the anchor).

#### Results

Aggregate over all 11 pairs (`gate` = `k6`, ratio `1.10`, anchor `1.5`):

| Scenario | `static max x 1.1` F1 | Gate F1 | Gate FP | Gate avg extension |
|----------|----------------------|---------|---------|--------------------|
| clean | 94.23% | 94.29% | 2.42% | 256 pkts (2/11 sessions) |
| tail 5% | 81.18% | 94.38% | 2.68% | 915 pkts |
| tail 10% | 51.47% | 94.90% | 2.17% | 1292 pkts |
| tail 20% | 24.68% | 94.90% | 2.17% | 1292 pkts |
| tail 40% | 11.53% | 94.90% | 2.17% | 1292 pkts |
| tail 60% | 10.74% | 94.90% | 2.17% | 1292 pkts |
| tail 100% | 3.39% | 94.90% | 2.17% | 1292 pkts |
| mid 20% | 22.42% | 95.12% | 1.87% | 807 pkts |
| sparse 10% | 94.82% | 94.75% | 2.09% | 215 pkts |

Properties worth keeping in mind:

- on 9/11 clean sessions the gate never fires and the threshold is identical
  to production `static max x 1.1`; the clean cost is ~2.6 s of average extra
  calibration and `+0.10%` quiet FP overall
- recovery is contamination-agnostic: the extension simply slides to the
  first self-consistent window, so tail 10% and tail 100% converge to the
  same threshold
- bounded worst case: contamination homogeneous and mild enough to pass both
  gates can inflate the threshold by at most ~`anchor x factor` (~1.65x the
  quiet max), versus unbounded inflation for static `max` (6x observed)

#### Rejected In This Sweep

- startup quantiles (`p95`, `p98`) and chunk-median-only thresholds: pay
  quiet FP on clean sessions (up to `6.1%`) and still fail from tail 40% up
- decaying-peak online recovery (both `peak9980_safe70_floor85` and
  `peak9995_safe60_floor90`): the safety floors cap repair at 10-15%, so
  recall never recovers under real-motion contamination (F1 `62.6%` at
  tail 10%, `21.9%` at tail 100% for the conservative line); superseded as a
  startup-repair mechanism (unchanged as a possible future drift tracker)
- spread-only gate without the floor anchor: accepts homogeneous majority
  contamination (tail 60%)
- `k = 8` with ratio `1.15` variants: dominated by the `k6` anchored gate

#### Decision

Promote the floor-anchored rolling-chunk gate with calibration extension as
the `l1_delta` startup-threshold policy:

- parameters: `k = 6` chunks, spread ratio `1.10`, floor anchor `1.5`,
  extension cap +2000 packets, fallback `median(ring) x 1.1`
- the threshold formula stays `max x 1.1` on the accepted window; no online
  threshold adaptation is promoted
- MVS keeps `max x 1.3` unchanged: its quiet floor is far less tight (up to
  14.5x cross-session), so this gate is not validated for MVS and would need
  its own sweep

#### Follow-Up

The gate is implemented in `StartupThresholdCalibrator` on both sides
(`src/python/micro_espectre/threshold.py`, `src/cpp/core/threshold.h`), with
an `EXTENDING` status surfaced during extension. Remaining live gates:

1. one deliberately noisy live startup session to confirm on-device recovery
2. long quiet runs on S3, still the strongest cross-session FP risk

---

### L1-Delta Quiet-Tail Rescue For Gate Extension

Date: 2026-07-07

Status: Promoted to the Micro-ESPectre and shared C++ runtimes; amends the
gate-extension decision above.

#### Goal

Fix the C6 long quiet run regression: 11.94% sustained FP on
`test_c6_..._112318` with the gated `l1_delta` startup threshold, versus
0.8% MVS and ~0% ML on the same capture.

#### Root Cause

That session's quiet metric has a heavy upper tail (`p99/p50 = 1.35`, versus
`1.13` on the second C6 long run): recurring short bumps (165 episodes,
median ~18 packets, spread across all 10 minutes) rather than drift or RF
changes (RSSI/noise floor flat, no periodicity). One tail bump inside the
calibration window tripped the spread gate, the extension slid past it, and
the accepted ring locked `max(ring) x 1.1 = 0.0207` — *below* the 0.0248
already observed during calibration. The threshold landed at ~p90 of the
session's own quiet distribution. Ungated `max x 1.1` (0.0272) would have
produced 0.02% FP: the gate turned a passing run into a failing one.

#### What Was Tested

Three candidate policies over: the 10 quiet long runs (self-calibrated FP),
the 11 clean pairs (production semantics), and a re-run of the real-motion
contamination sweep (tail 10/20/100%, mid 20%, sparse 10%):

1. tail rescue: on acceptance after extension, if the max discarded chunk
   stays within the floor-anchor band (`<= 1.5 x median(ring)`), fold it
   back into the threshold metric; discarded chunks above the band keep the
   plain contamination-repair behavior
2. adaptive factor: `factor = clamp(max(all chunks)/median(all chunks),
   1.1, 1.5)` applied to the gated statistic
3. chunk-mean gate: spread/anchor checks on per-chunk means instead of
   maxima (persistence-based contamination detection)

#### Results

Quiet long runs (FP rate, changed rows only):

| Recording | Gate | Tail rescue | Adaptive factor |
|-----------|------|-------------|-----------------|
| C6 112318 | 11.94% | 0.02% | 0.88% |
| S3 181757 | 2.49% | 2.49% | 0.03% |

Contamination sweep (aggregate F1 over the 11 pairs):

| Scenario | Gate | Tail rescue | Adaptive factor |
|----------|------|-------------|-----------------|
| clean | 94.4% | 94.3% | 93.5% |
| tail 10% | 94.6% | 92.4% | 78.0% |
| tail 20% | 95.0% | 92.5% | 75.2% |
| tail 100% | 95.0% | 95.0% | 74.9% |
| mid 20% | 94.4% | 92.4% | 74.0% |
| sparse 10% | 94.4% | 94.4% | 93.7% |

The tail-rescue contamination cost is concentrated in a single weak-motion
pair per scenario (for example C5 hobby tail 10%: the contaminating motion
peaks at 1.43x the quiet floor, inside the anchor band, so it is rescued and
the threshold rises to just above that weak-motion level). The inflation is
bounded by the anchor: `<= 1.5 x median(ring) x 1.1`, the same worst case the
gate decision already accepts for mild contamination that passes the checks.

#### Rejected In This Sweep

- adaptive factor: collapses under contamination (the all-chunk spread is
  motion-inflated, capping at `1.5x` the gated statistic) and pays clean F1
- chunk-mean gate: no separation — the 100-packet metric window smears an
  ~18-packet tail bump across most of a 148-packet chunk, so the bump chunk
  mean (1.21x floor) is identical to a weak-motion chunk mean (1.21x floor)
- amplitude cutoffs tighter than the anchor: the C6 tail bump sits at 1.33x
  the ring median while real weak-motion contamination sits at 1.32-1.43x;
  the ranges overlap, so no fixed ratio separates them

#### Decision

Promote the tail rescue: quiet-tail bumps and weak motion are locally
indistinguishable at startup, and a persistent double-digit FP floor in a
quiet room is a worse failure than reduced sensitivity to sub-anchor motion
after a contaminated startup. Implemented in `StartupThresholdCalibrator` on
both sides (one extra tracked float, `discarded_chunk_max`); threshold
formula and gate parameters unchanged.

---

### L1-Delta Post-Startup Drift-Tracker Re-Test

Date: 2026-07-07

Status: Rejected for the C3/C5 recall residual; confirms the deferral of the
decaying-peak line from the startup-threshold and online recovery sweep above.
Production keeps the static gated startup threshold with no online adaptation.

#### Goal

Decide whether a post-startup online threshold adapter can close the remaining
`l1_delta` paired-recall gap on C3 (`89.7%`) and C5 (`94.9%`) without breaking
the long-quiet false-positive gate. The gap sits on a small number of
low-contrast rooms (C3 hobby room `80.9%`, C3 living room `88.2%`, C5 bedroom
`92.1%`), all with clean startup (`0%` startup-window FP) and a threshold
already placed at the session's true quiet floor.

#### Background

The decaying-peak online tracker (`peak9995_safe60_floor90` and the aggressive
`peak9980_safe70_floor85`) was characterized but deferred in the
startup-threshold and online recovery sweep as a startup-repair mechanism: it
is downward-only and lowers the threshold only when the running quiet floor
falls below the calibration `max`, that is, when startup *overshot*. The C3/C5
residual is the opposite regime — a correctly placed threshold on a genuinely
low motion-to-quiet contrast — so the tracker was expected to have little to
reclaim there. This run measures that directly.

#### Method

Offline harness replaying production semantics: startup gate on the
static-presence (or long-quiet) recording for the initial threshold, then a
continuous baseline -> motion pass with the tracker driving `detector.threshold`
online (decaying peak `ref = max(metric, ref * decay)`, threshold updated only
while `IDLE` and `metric < threshold * safe`, clamped to
`min(threshold, max(ref * 1.1, startup * floor))`). The `off` variant reproduces
the committed paired and long-quiet numbers exactly (C3 long-quiet `0.30/0.42`,
C6 `0.43/0.83`, S3 `3.63/7.90` avg/max; paired recall `89.7/94.9/96.2/96.4`),
so the deltas below are the tracker, not a re-calibration artifact. Variants
swept: conservative `9995/60/90`, aggressive `998/70/85`, and two deeper floors
(`998/70/80`, `998/75/75`).

#### Results

Conservative `peak9995/safe60/floor90` (paired recall / FP, and long-quiet
avg / max FP):

| Chip | Paired off | Paired cons | Long-quiet off | Long-quiet cons |
|------|-----------|-------------|----------------|-----------------|
| C3 | 89.7 / 1.7 | 89.7 / 1.7 | 0.30 / 0.42 | 0.30 / 0.42 |
| C5 | 94.9 / 0.3 | 96.7 / 0.5 | 0.37 / 1.06 | 0.37 / 1.06 |
| C6 | 96.2 / 2.8 | 96.2 / 2.8 | 0.43 / 0.83 | 1.15 / 1.46 |
| S3 | 96.4 / 6.0 | 97.2 / 12.3 | 3.63 / 7.90 | 4.89 / 8.62 |

Three findings settle the question:

- the tracker never engages on the C3 residual: `thr_min / startup = 1.00` on
  every C3 pair (and C6 hobby room), so recall is unchanged `89.7 -> 89.7`.
  Clean startup leaves no overshoot to reclaim, exactly as predicted
- it recovers only overshoot pairs (C5 bedroom `92.1 -> 97.3`, S3 bedroom
  `97.9 -> 99.6`), that is, it reclaims the same C5 bedroom recall the
  quiet-tail rescue had traded away, and on C5 with no long-quiet cost
- but it regresses the promotion gate: on the C6 `112318` long run the
  quiet-tail rescue had just fixed (`11.94 -> 0.02%`), the tracker ratchets
  back to `1.46%` between tail bumps, and S3 long-quiet worsens
  (`3.63 -> 4.89%` avg). The online tracker and the startup tail rescue are in
  direct tension on heavy-quiet-tail sessions: the tracker undoes what the
  rescue protects

Aggressive and deeper floors do move C3/C5 recall (C3 up to `99.8%`, C5 to
`100%`) but detonate the long-quiet gate: C5 `0.37 -> 19-44%`, S3
`3.63 -> 17-43%`, C6 up to `13%` avg. C3 hobby room only clears once the floor
reaches `0.75`, at which point its own quiet baseline fires at `12-19%` FP —
the same "quiet-tail bumps and weak motion are locally indistinguishable" floor
seen at startup, now measured from the online side.

#### Decision

Reject the drift tracker for the C3/C5 recall residual and keep the static
gated startup threshold with no online adaptation. The residual is a
single-feature contrast floor, not a startup overshoot: it is unreachable by
startup calibration (already swept) and by post-startup adaptation (this run).
The only FP-acceptable variant is a net negative on the gate that governs
promotion. The decaying-peak line stays a documented deployment-robustness
candidate for dirty or overshot startup, which is orthogonal to this residual.
For the low-contrast rooms the intended path is the ML detector (C3 ML
`95.9%`), and the residual does not block keeping the new Classic default.

---

### L1-Delta + MVS Gated Fusion For The Low-Contrast Recall Residual

Date: 2026-07-07

Status: Promoted into the current `ClassicDetector` design. This section records
the validation that led to the current fusion: L1-Delta primary metric with a
moving-variance recovery vote.

#### Goal

Close the `l1_delta` paired-recall residual on the low-contrast rooms (C3
`89.7%`, C5 `94.9%`, plus C6 hobby/living) that no calibration policy could
reach.

#### Background

The residual was confirmed as a single-feature contrast floor from four
independent angles: the startup-threshold sweeps, the drift-tracker re-test,
a fixed-threshold study, and a motion-triggered calibrator. In every low-recall
room `motion_p10 / quiet_p99` is `1.05-1.10` (S3 bedroom `0.87`): the motion
barely clears the quiet floor, so no threshold placement separates them without
paying false positives one-for-one. The static baselines are pristine
(`max/p99 ~ 1.03`), so this is genuine low contrast or coarse motion labeling,
not dirty data.

The opening comes from the `PERFORMANCE.md` table: MVS and L1-Delta have
complementary failure modes. MVS wins recall exactly where L1-Delta loses (C3
`93.7` vs `89.7`, C5 `99.7` vs `94.9`, C6 `98.9` vs `96.2`), while L1-Delta wins
S3 (`96.4` vs `79.5`) and the long-quiet floor. A per-packet probe confirmed it:
on the low-contrast rooms MVS recovers `58-100%` of the packets L1-Delta misses,
and the union false-positive rate stays near `0%` — the missed motion has
elevated variance even when its L1 displacement is low.

#### Method

L1-Delta stays primary with its unchanged calibrated threshold `thr`. MVS is
added as a recovery vote with no absolute threshold of its own:

- consulted only in the ambiguous band `alpha*thr < l1_metric <= thr`
- votes MOTION only when `mvs_metric > k * mvs_baseline`, a relative test
  against MVS's own session floor (`mvs_baseline = median` of the MVS metric
  over the reference window). A ratio is scale-invariant, so MVS's `14.5x`
  cross-session floor drift does not matter; L1-Delta's stable floor supplies
  the "we are quiet" signal that maintains the reference
- the vote is enabled only when the session MVS floor is tight
  (`p99/median < cut`), a self-diagnosing gate that keeps MVS out where its
  variance floor is bursty

Parameters: `alpha = 0.6`, `k = 3.0`, `cut = 4.0`. `alpha` is nearly inert (the
band is wide); `k` is the recall/FP knob; `cut` sits in a wide bimodal gap (tight
floors `<= 3.2`, bursty floors `>= 7.7`).

#### Results

Paired (L1-Delta-alone -> gated fusion), per-chip recall/FP:

| Chip | L1-Delta alone | Gated fusion |
|------|----------------|--------------|
| C3 | 89.7 / 1.7 | 95.6 / 1.8 |
| C5 | 94.9 / 0.3 | 96.8 / 0.6 |
| C6 | 96.2 / 2.8 | 99.2 / 2.8 |
| S3 | 96.4 / 6.0 | 97.2 / 6.2 |

The residual closes (C3 `+5.9`, C6 `+3.0`, C5 `+1.9`) at near-zero false-positive
cost (C3 `+0.1`, C6 `+0.0`).

Long-quiet FP is unchanged: the two bursty S3 runs (`p99/median` `13.2` and
`12.9`) gate the vote off and the fusion equals L1-Delta exactly (`7.90%` and
`2.49%`, no change); every other run moves `<= 0.2pp`.

Contamination sweep (1000-packet startup window contaminated with real motion,
then clean baseline -> motion evaluation), aggregate over 11 pairs:

| Scenario | L1-Delta alone | Gated fusion | vote-off pairs |
|----------|----------------|--------------|----------------|
| clean | 94.1 / 2.4 | 98.1 / 2.9 | 1/11 |
| tail 10% | 89.9 / 2.3 | 93.2 / 2.4 | 6/11 |
| tail 20% | 90.5 / 2.2 | 91.4 / 2.2 | 9/11 |
| tail 40% | 95.0 / 2.2 | 95.0 / 2.2 | 11/11 |
| mid 20% | 85.5 / 2.2 | 86.7 / 2.2 | 9/11 |
| sparse 10% | 95.0 / 2.4 | 97.1 / 2.5 | 2/11 |

The dispersion gate does double duty: contamination inflates the MVS dispersion,
so the vote progressively disables (tail 10% `6/11` off, tail 40% `11/11` off)
and the fusion degrades to the already-robust L1-Delta baseline. False positives
never rise (the relative test raises `k*baseline` under contamination, producing
fewer votes, not more), so a dirty startup costs the recovery benefit but never
pays for it.

#### Decision

This was strong enough to settle the non-ML runtime design: the default
`ClassicDetector` is L1-Delta primary plus a threshold-free, dispersion-gated
moving-variance recovery vote, not L1-Delta alone. Moving variance keeps a role
as a complementary second signal rather than as an alternative detector. The
remaining follow-up after promotion is:

- maintain the MVS baseline and dispersion online during L1-Delta IDLE (not from
  a fixed window), and size the device-side state (rolling MVS window or a
  streaming percentile estimator)
- measure the per-packet cost of running both detectors on ESP32
- validate `k` and `cut` on additional sessions and at least one live capture

The motion-triggered calibrator explored en route stays a separate candidate: it
removes the guaranteed-quiet-window requirement and tightens false positives on
high-contrast rooms, but it does not close the recall residual (low-contrast
rooms lack a detectable motion jump to anchor on).

---

### L1-Delta-Guided Sample Weighting And Hard-Negative Retrain

Date: 2026-07-07

Status: Rejected; production keeps `--sample-weight-mode none`. Revisits the
2026-07-03 MVS-guided weighting attempt with the L1-Delta detector as the
training guide.

#### Goal

Decide whether the L1-Delta detector should guide MLP training as a sample
weight, either through `l1_guided` (hard-positive mining on motion plus
hard-negative emphasis on idle) or `l1_hard_negative` (idle-only promotion of
hard negatives), versus the current `none` baseline.

#### Background

`10_train_ml_model.py` can replay the production L1-Delta startup calibration
per source file and weight training windows by the metric-to-threshold ratio:

- `l1_guided`: subtle near-threshold motion is up-weighted (up to 2.4x), easy
  strong motion is down-weighted (0.8x), and idle windows where L1-Delta would
  itself fire (ratio >= 1.0) are promoted to 2.0x; weights are normalized per
  file so no recording dominates
- `l1_hard_negative`: motion stays neutral (1.0x), only the hard idle windows
  are promoted (2.0x)

The 2026-07-03 MVS-guided variant of the same idea was rejected because it read
as a sensitivity shift (long-recording `total_fp` up, only a slight `mean_f1`
gain). This run asks whether the tighter, more session-stable L1-Delta metric
changes that verdict.

#### Method

Single-variable A/B: three modes at three fixed seeds (`42`, `123`, `7`) with
everything else held constant (production feature set, scaler, batch size,
`fp_weight`). Each run trained to a temporary export (production artifacts
backed up and restored, verified by hash), then evaluated on:

- grouped-CV blocked OOF (the Stage-1 robustness prefilter)
- the long-recording ML gate (`TestLongRecordings::test_ml_vs_test_recordings`)
- the paired ML gate (`TestPerformanceMetrics::test_ml_detection_accuracy`)

The long-recording set is quiet-only (no annotated motion segment), so its
recall and F1 are 0 by construction and its only meaningful signal is
`total_fp` / `max_fp_rate` — this gate is a sustained false-positive test.
Recall effects surface in grouped CV and the paired gate instead.

#### Results

Medians over the three seeds:

| Mode | OOF F1 | Long-run total FP | Long-run max FP | S3 paired FP | S3 paired recall |
|------|--------|-------------------|-----------------|--------------|------------------|
| `none` | 79.7% | 1902 | 1.90% | 4.9% | 92% |
| `l1_guided` | 80.6% | 2561 (+35%) | 3.00% | 5.0% | 91% |
| `l1_hard_negative` | 80.3% | 2529 (+33%) | 2.80% | 5.3% | 92% |

Per-seed long-run total FP (none / guided / hard_negative):

| Seed | none | l1_guided | l1_hard_negative |
|------|------|-----------|------------------|
| 42 | 2595 | 2733 | 2529 |
| 123 | 1902 | 1915 | 2610 |
| 7 | 1834 | 2561 | 2148 |

Properties worth keeping in mind:

- both weighting modes raise long-recording total FP at the median (+33-35%),
  the exact metric the FP-first promotion gate minimizes
  (`MAX_PROMOTION_TOTAL_FP_INCREASE = 0` blocks any increase), so both are
  rejected by the repo's own criterion
- `l1_guided` gives a consistent grouped-CV gain (+0.9% OOF F1 on all three
  seeds, with better hard-chip recall, e.g. C6 70 -> 76% at seed 123), but that
  is the expected outcome of its hard-positive mining: more sensitivity buys
  recall and pays deployment FP
- `l1_hard_negative`, which targets FP only, does not reduce it at the median;
  the `none` baseline already reaches low long-run FP at seeds 123/7 (1902,
  1834), and the idle promotion only wins at seed 42 where `none` happened to
  be high (2595 -> 2529)
- the paired gate is a tie across modes (within +/-1pp), so there is no
  standalone paired-FP win either
- seed variance is large (`none` long-run FP spans 1834-2595), so three seeds
  give direction, not a tight estimate; the direction is nonetheless
  consistent with the 2026-07-03 MVS-guided finding

#### Decision

Keep `--sample-weight-mode none`. Neither L1-Delta-guided mode improves the
metric that governs promotion (deployment/long-run FP); both trade a small
cross-validation F1/recall gain for higher FP, matching the earlier
MVS-guided outcome. The support-detector weighting hooks stay in the trainer
as opt-in analysis tooling, not a default. A definitive verdict, if needed,
should use the native `--seed-search-until-improvement` flow (5+ seeds, median
promotion key) rather than this three-seed A/B.

---

### Clean Relative-8 Refresh And L1-Delta ML Feature Check

Date: 2026-07-06

Status: Active ML baseline candidate.

#### Goal

Re-evaluate the AGC-active relative-8 Python ML baseline on the cleaned
dataset, identify weak features under grouped CV and real-data holdouts, and
check whether `l1_delta` should enter the MLP after its strong standalone
detector benchmark.

#### Background

The cleaned July dataset reset changed the meaning of older ML targets. The
current runtime and training path already use AGC-active coefficient-of-
variation turbulence, so the right question was no longer "does the historical
relative-8 export still pass?" but:

1. which existing ML features are actually helping on the refreshed dataset
2. whether the baseline should be simplified before another full export
3. whether `l1_delta` adds complementary information beyond the existing
   relative window statistics

The starting production feature set was:

- `turb_std_over_mean`
- `turb_max_over_mean`
- `turb_min_over_mean`
- `turb_iqr_over_mean`
- `turb_mad_over_mean`
- `waveform_length_over_mean`
- `turb_skewness`
- `turb_autocorr`

#### Tooling Added

`tools/10_train_ml_model.py` gained analysis helpers to make focused feature
studies reproducible:

- `--feature-drop a,b,c`
- `--feature-swap old=new`
- `--feature-sweep feature`

The Python ML runtime was also extended so that exported Python weights can
consume `l1_delta` when the exported `FEATURE_NAMES` require it. C++ parity for
that specific feature is still pending, so modified feature sets remain
analysis-only outside the Python path.

#### What Was Evaluated

- SHAP and ablation-style follow-up on the clean relative-8 line
- direct drop candidates:
  - drop `waveform_length_over_mean`
  - drop `turb_skewness`
  - drop both
  - drop all three of `waveform_length_over_mean`, `turb_skewness`,
    `turb_min_over_mean`
- one-for-one `l1_delta` sweep across every slot of the relative-8 set
- a production-like Python rerun of the most relevant `l1_delta` candidate:
  replace `waveform_length_over_mean` with `l1_delta`

#### Key Result: `waveform_length_over_mean` Was Weak

On grouped blocked CV, the clean relative-8 baseline scored:

- blocked OOF F1: `79.49%`
- fold F1: `81.19%`
- worst-chip recall: `68.8%`

Dropping only `waveform_length_over_mean` improved all three:

- blocked OOF F1: `81.0%`
- fold F1: `83.0%`
- worst-chip recall: `75.1%`

Dropping only `turb_skewness` improved less:

- blocked OOF F1: `80.2%`
- fold F1: `82.5%`

Dropping both `waveform_length_over_mean` and `turb_skewness` matched the CV
headline (`81.0%` blocked OOF F1, `83.0%` fold F1), but did not hold the same
paired-data quality as the simpler 7-feature candidate. Dropping
`turb_min_over_mean` on top of that reduced the global score further, so
`turb_min_over_mean` was kept.

#### Real-Data Python Validation

The two serious post-CV candidates were compared directly on the paired
real-data gate and on the Python long quiet/FP gate:

| Candidate | Features | Paired Mean Recall | Paired Worst-Chip Recall | Paired Mean F1 | Total FP | Long Quiet FP | Long Max FP Rate |
|-----------|----------|-------------------:|-------------------------:|---------------:|---------:|--------------:|-----------------:|
| Relative-8 minus `waveform_length_over_mean` | 7 | 94.0% | 83.6% | 95.7% | 798 | 4057 | 2.67% |
| Relative-8 minus `waveform_length_over_mean`, `turb_skewness` | 6 | 92.8% | 79.3% | 95.0% | 786 | 5781 | 5.87% |

Interpretation:

- removing `waveform_length_over_mean` is the real win
- removing `turb_skewness` as well saves little on paired false positives and
  makes recall materially worse
- the best clean-data Python candidate is the 7-feature set with
  `turb_skewness` still present

#### L1-Delta As An ML Feature

`l1_delta` remained strong as a standalone detector, but it did not become a
winning MLP feature.

The slot-by-slot sweep showed no one-for-one replacement that beat the clean
relative-8 baseline. The most relevant production-like candidate was the exact
replacement:

- drop `waveform_length_over_mean`
- add `l1_delta`

That candidate scored:

- blocked OOF F1: `76.59%`
- fold F1: `78.74%`

And it failed more paired Python datasets than the 7-feature candidate after
the Python runtime was updated to support `l1_delta` extraction.

The qualitative pattern was consistent:

- `l1_delta` lowered quiet-room false positives on some long idle segments
- but it also made the MLP too conservative, hurting recall and aggregate F1
- no tested `l1_delta` substitution beat the simpler 7-feature candidate

#### Decision

For the next ML promotion candidate:

- remove `waveform_length_over_mean`
- keep `turb_skewness`
- do not add `l1_delta` to the MLP feature set yet

Candidate feature set:

- `turb_std_over_mean`
- `turb_max_over_mean`
- `turb_min_over_mean`
- `turb_iqr_over_mean`
- `turb_mad_over_mean`
- `turb_skewness`
- `turb_autocorr`

#### Follow-Up

1. Promote the 7-feature Python baseline into the next full production retrain
   and export flow
2. Align the C++ ML feature path before treating any non-default `l1_delta`
   export as production-like
3. Revisit `l1_delta` for ML only if a later model family wants stronger
   conservative gating, or if a different temporal context makes it less
   redundant with the current relative window statistics

---

## Historical ML Experiments

### Gain-Shift Robustness Diagnostic

Date: 2026-06-29

Status: Superseded as a production gate by AGC-active normalization; retained
as the experiment that motivated relative and normalized turbulence features.

#### Goal

Check whether the then-production ML detector was structurally robust to
device/session gain shifts, or whether it was only empirically robust on the
collected domains.

#### Background

Earlier MVS variants had a split normalization path. The later AGC-active
detector uses coefficient-of-variation turbulence (`std / mean`), which is
invariant to a uniform amplitude scale factor. At the time of the first
gain-stress diagnostic, the ML detector intentionally used raw turbulence std
in both training and runtime inference. Its exported feature scaler was a
global statistical standardization fitted on the training set; it did not
compensate an unseen per-device/session gain shift.

For the previous raw 9-feature production baseline:

- scale-sensitive: `turb_mean`, `turb_std`, `turb_max`, `turb_min`,
  `turb_iqr`, `turb_mad`, `waveform_length`
- scale-invariant under positive uniform gain: `turb_skewness`,
  `turb_autocorr`

#### Tooling Added

`tools/10_train_ml_model.py --gain-stress-gate` evaluated the exported Python
weights without retraining. It extracted the exported feature
set from real `empty`/`static_presence`/`motion` data, applies artificial gain
multipliers only to the scale-sensitive features, and reports overall plus
worst-group metrics.

Example:

```bash
python tools/10_train_ml_model.py --gain-stress-gate
python tools/10_train_ml_model.py --gain-stress-gate --environment bedroom
python tools/10_train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25
```

#### Raw-Feature Result

Exported raw-feature seed: `721498330`.

All environments:

| Scale | Recall | Precision | FP rate | F1 | Worst chip FP |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 93.8% | 97.2% | 2.7% | 95.4% | S3 8.7% |
| 0.75 | 95.3% | 98.5% | 1.5% | 96.8% | S3 5.5% |
| 1.00 | 99.9% | 99.2% | 0.8% | 99.6% | S3 3.5% |
| 1.25 | 99.9% | 92.8% | 7.8% | 96.2% | ESP32 29.3% |
| 1.50 | 99.4% | 86.3% | 15.8% | 92.4% | ESP32 45.6% |
| 2.00 | 99.0% | 77.6% | 28.7% | 87.0% | S3 54.8% |

Bedroom-only:

- `1.00x`: recall `100.0%`, FP `1.3%`, F1 `99.4%`
- `1.25x`: recall `100.0%`, FP `6.5%`, worst-chip FP `ESP32 20.2%`
- `1.50x`: FP `14.2%`, worst-chip FP `ESP32 37.4%`
- `2.00x`: FP `27.6%`, worst-chip FP `S3 61.1%`

#### Decision

The raw-feature ML detector is strong at nominal gain, but it is not
structurally gain-invariant. Global feature standardization is numeric
normalization, not domain compensation. Seed search and long/paired gates can
promote empirically better weights, but they do not by themselves solve
cross-gain robustness.

Follow-up experiments compared:

1. the then-current raw features
2. relative/gain-invariant features such as `std/mean`, `iqr/mean`,
   `mad/mean`, and normalized waveform length
3. a small hybrid set that keeps useful raw features while adding relative
   features

The relative 8-feature set was promoted after a second architecture/training
policy pass. The plain `24-12`, `fp_weight=1.0` relative model was gain-stable
but increased long-recording false positives on C6/S3. A wider `32-16` model
trained with `fp_weight=2.0` recovered the long-recording gate while preserving
flat gain-stress behavior.

Promoted export:

- seed: `1890407301`
- topology: `8 -> 32 -> 16 -> 1`
- training policy: `fp_weight=2.0`, `scaler=standard`
- gain-stress gate: flat at `1.00x`, `1.25x`, `1.50x` with `FP=1.1%`
- long-recording ML gate: `total_fp=654`, worst chip `C6` with `F1=93.5%`

Promoted feature set:

- `turb_std_over_mean`
- `turb_max_over_mean`
- `turb_min_over_mean`
- `turb_iqr_over_mean`
- `turb_mad_over_mean`
- `waveform_length_over_mean`
- `turb_skewness`
- `turb_autocorr`

Rejected near-term additions:

- `range_over_mean`
- `peak_over_mad`
- `robust_relative` p95/p05 replacement for max/min

Those extra relative features produced occasional local wins, but did not beat
the simpler relative set on the combined long-recording and gain-stress
comparison. The later `robust_relative` p95/p05 variant reduced single-spike
feature leverage in isolation, but its grouped-CV result was weaker than the
promoted relative set (`F1=89.8%`, `recall=89.1%`, `FP=4.0%` versus the
promoted retrain's `F1=91.5%`, `recall=91.6%`, `FP=3.5%`). It remains an
analysis-only feature set. At the time, the gain-stress gate was the primary
diagnostic for gain-shift robustness, with the long-recording gate acting as
the false-positive non-regression check. After the AGC-active normalization
refactor, this post-feature gain-stress diagnostic is historical rather than a
production promotion gate.

---

### C3 Empty-Room Retrain Incident

Date: 2026-06-30

Status: Partly superseded by AGC-active normalization and clean data
recollection; retained because it established `empty` as an IDLE training
class.

#### Goal

Diagnose why a C3 ESPHome deployment in a static room produced noisy ML scores
despite good offline validation, then decide whether the failure was caused by
the model, runtime inference, or dataset coverage.

#### Observation

The failing runtime log was a C3 connected to the same AP/BSSID, at the same
distance and packet rate used for collection. ESPHome generated ping traffic at
about `100 pps`; the collector-generated datasets used UDP traffic. The
important offline reproduction was not `static_presence`, but the newer C3
`empty` capture: the previously exported model produced false positives on
that empty-room data.

The new C3 `static_presence` capture did not reproduce the high-score failure
offline. The new C3 `empty` capture did:

- old export on new C3 `empty`: about `4.1%` false positives, with scores up to
  `9.98`
- retrained export including `empty`: about `0.2%` false positives on the same
  C3 `empty` file
- global `empty` false-positive rate after retrain: about `0.3%`

The problematic packets showed frame-scale amplitude jumps while channel, RSSI,
and reported gain metadata stayed stable. Because the then-production ML
pipeline used raw per-packet turbulence as its base signal, those jumps entered
the 100-packet ML window as turbulence spikes. Relative window features reduced
uniform window-level gain shifts, but they did not make the model structurally
immune to arbitrary per-frame amplitude jumps.

#### Decision

The fix at that point was to include `empty` in binary ML training, mapping both
`empty` and `static_presence` to IDLE and `motion` to MOTION. This better
matches the deployed task: the detector must suppress both quiet empty rooms
and static-presence rooms, not only distinguish static presence from motion.

At this point no C++ feature ABI change was required:

- the ML feature set remained the 8 relative features
- the C++ runtime changed only through regenerated exported weights

A live ESPHome C3 smoke test after the retrain produced 37 IDLE publications
and 1 MOTION publication across 38 post-connect samples, with median score
`0.10`, mean score `0.76`, and one score above the fixed threshold `5.0`.

#### Follow-Up

Per-packet normalized turbulence for ML (`std(amplitudes) / mean(amplitudes)`)
was later promoted to the single production path before the same 8 relative ML
features are extracted.

Retraining with seed `1890407301` on the full clean dataset produced blocked
grouped-CV `F1=92.2%` and passed the Python and C++ real-data gates. The C++
ESP32 paired validation is the most direct regression check: ML with CV OFF
had `90.2%` recall, while the gain-aware path reached `100.0%` recall with
`0.0%` FP.

The long-recording gate still exposes noisy C5/C6 idle segments (`C5` ML
`7.7%` FP, `C6` ML `10.1%` FP). This appeared separate from the historical
ESP32 normalization issue because the same long files were also difficult for
MVS (`C5` `11.1%` FP, `C6` `40.2%` FP). The working interpretation was
dataset/environment coverage, not evidence against the ESP32 normalization fix.

#### Second Empty-Domain Capture

A later C6 bedroom `empty` capture (`empty_c6_64sc_20260630_120210.npz`) was a
true empty-room recording but looked motion-like to the previous export:

- old export on new C6 `empty`: about `35.2%` false positives
- retrained export including the C6 `empty`: about `2.1%` false positives on
  that file
- real-data paired C6 validation after gain-aware retrain: `recall=99.2%`,
  `FP=0.0%`

However, grouped OOF validation still identified the held-out C6 `empty` file
as the weakest source (`FP` around `37%`). Raising the global IDLE penalty from
`fp_weight=2.0` to `3.0` did not improve that held-out failure. The full export
therefore handles the known hard negative, but the broader lesson remains
dataset/domain coverage: future empty-room captures should stay in the
regression gate and be evaluated with long-recording holdouts before promotion.

---

### Feature-Set Reduction Sweep

Date: 2026-05-20

Status: Superseded by later relative-8 and AGC-active normalized baselines.

#### Goal

Reduce long-recording false positives without weakening the deployed MLP
architecture or breaking Python/C++ parity.

#### Setup

- Then-production topology kept fixed at `9 -> 24 -> 12 -> 1`
- Candidate feature sets evaluated with grouped CV, paired validation, and
  long-recording holdout
- Ranking favored holdout robustness, not CV alone

#### Decision

The input feature set was reduced from 12 to 9.

Removed features:

- `turb_kurtosis`
- `turb_entropy`
- `turb_slope`

#### Why These Features Were Dropped

- They sometimes improved paired validation slightly, but hurt the
  long-recording holdout where FP robustness mattered more
- They overlapped with more stable signals already captured by
  `turb_autocorr`, `turb_iqr`, `turb_mad`, and `waveform_length`
- They increased deployment complexity without producing a reliable FP-first
  win

#### Outcome

- The 9-feature MLP became the production baseline at that point
- The simpler input set improved holdout robustness while preserving strong
  paired-set quality

---


### FP-First Feature and Training Sweep

Date: 2026-05-20

Status: Superseded by later ML baselines; retained for FP-first ranking
criteria.

#### Goal

Revisit the then-production `mlp-9` from the opposite angle of the temporal sweep:
identify which features amplify long-run false positives, then test whether
feature-set changes or training-policy changes can reduce FP without paired-set
regression.

#### Axes Tested

- per-window profiling on the 4 curated long recordings
- feature diagnostics on `TP/FP/TN/FN` buckets
- FP-first training policies (`fp_weight`, negative emphasis, threshold tuning)
- targeted candidates combining feature-set changes and training policy

#### Result

- Winner: `baseline-9`
- Median long `max_fp_rate`: 7.00%
- Median long `total_fp`: 356.0
- Median long `worst_chip_f1`: 89.00
- Baseline reference (`baseline-9`): `max_fp_rate=7.00%`, `total_fp=356.0`, `worst_chip_f1=89.00`

#### Decision

The campaign only promotes a candidate if the FP-first ranking improves in
median and stays stable in the worst case. See the generated JSON campaign
artifact for the full shortlist and diagnostics.

#### Follow-Up: `drop-turb_min`

The long-run diagnostics flagged `turb_min` as a suspicious feature, so a
focused 5-seed follow-up compared `baseline-9` against a single ablation that
removed only `turb_min`.

Result:

- `baseline-9`: `median_long_max_fp_rate=7.0%`, `median_long_total_fp=356`,
  `median_long_worst_chip_f1=89`, `median_paired_pass=4`
- `drop-turb_min`: `median_long_max_fp_rate=7.0%`, `median_long_total_fp=358`,
  `median_long_worst_chip_f1=70`, `median_paired_pass=3`

Conclusion:

- the ablation did not reduce the long-run FP ceiling
- median total FP was slightly worse
- robustness regressed materially on the weakest seed / chip combinations

So `drop-turb_min` was explicitly rejected, and the production baseline stayed
unchanged at that point.

---

### Raw-9 MLP Topology Sweep

Date: 2026-05-20

Status: Superseded by the relative-8 line.

#### Goal

Check whether the then-current 9-feature MLP could reduce long-run false positives
by changing only the hidden-layer topology, without reopening the feature set
or training-policy axes.

#### Candidates

- `Then-current default (24-12)` -> `9 -> 24 -> 12 -> 1`
- `Legacy (16-8)` -> `9 -> 16 -> 8 -> 1`
- `Shallow (24)` -> `9 -> 24 -> 1`
- `Wider (32-16)` -> `9 -> 32 -> 16 -> 1`
- `Deep (24-12-6)` -> `9 -> 24 -> 12 -> 6 -> 1`

#### Ranking Priority

1. lowest long-run `max_fp_rate`
2. lowest long-run `total_fp`
3. highest long-run `pass_count`
4. highest long-run `worst_chip_f1`
5. paired validation as a non-regression constraint
6. grouped CV only as a final tie-breaker

#### Key Observation During Screening

`Shallow (24)` looked strong on 3-seed median `total_fp`, but `Wider (32-16)`
held a slightly better primary FP ceiling (`max_fp_rate`) and therefore won the
head-to-head slot for the final 5-seed comparison.

#### Final Outcome

| Architecture | Seeds | Median Max FP Rate | Median Total FP | Median Paired Pass Count | Median Worst-Chip F1 |
|--------------|-------|--------------------|-----------------|--------------------------|----------------------|
| Then-current default (24-12) | 5 | 7.89% | 567.0 | 5.0 | 93.46 |
| Wider (32-16) | 5 | 7.86% | 506.0 | 5.0 | 93.96 |

#### Decision

`Wider (32-16)` was promoted for the raw-9 line. The winning export
used seed `20260521`, passed the final paired validation rerun, and kept the
same 9-feature input set while improving the FP-first long-run ranking over the
previous raw `24-12` baseline.

---

### Relative-8 Topology and FP-Weight Sweep

Date: 2026-06-29

Status: Superseded by the MVS hard-negative retrain, then by AGC-active clean
baseline work.

#### Goal

Keep the gain-invariant relative feature set, then recover long-recording
false-positive robustness by changing only the MLP topology and IDLE-class
weighting.

#### Setup

- Feature set fixed to the promoted relative 8-feature view:
  `std/mean`, `max/mean`, `min/mean`, `iqr/mean`, `mad/mean`,
  normalized waveform length, skewness, autocorrelation
- Seed fixed to `1890407301` for the focused screen
- Primary gate: curated 60-second long recordings
- Non-regression checks: paired real-data validation and exported gain-stress
  gate

#### Focused Screen

| Candidate | Params | Long Total FP | Long Max FP Rate | Worst-Chip F1 | Mean Recall | Gain-Stress FP @ 1.5x |
|-----------|-------:|--------------:|-----------------:|--------------:|------------:|----------------------:|
| `24-12`, `fp_weight=1.0` | 529 | 1178 | 18.7% | 91.0% | 100.0% | 1.9% |
| `32-16`, `fp_weight=1.0` | 833 | 829 | 15.1% | 92.5% | 99.9% | 1.4% |
| `48-24`, `fp_weight=1.0` | 1633 | 828 | 12.6% | 93.7% | 99.8% | 0.9% |
| `32-16-8`, `fp_weight=1.0` | 961 | 878 | 11.9% | 94.0% | 100.0% | 1.4% |
| `24-12`, `fp_weight=1.5` | 529 | 1134 | 18.5% | 91.1% | 100.0% | 1.9% |
| `32-16`, `fp_weight=1.5` | 833 | 822 | 13.3% | 93.3% | 99.9% | 1.4% |
| `32-16`, `fp_weight=2.0` | 833 | 712 | 11.3% | 94.2% | 99.7% | 0.7% |

#### Promoted Export

The focused screen selected `32-16` with `fp_weight=2.0`. A full train/export
with seed `1890407301` produced:

- topology: `8 -> 32 -> 16 -> 1`
- grouped blocked OOF F1: `93.0%`
- paired real-data gate: pass
- exported gain-stress gate: flat at `1.00x`, `1.25x`, `1.50x` with
  `FP=1.1%`
- long-recording ML gate: `total_fp=654`; per-chip FP counts `C3=0`,
  `C5=249`, `C6=405`, `S3=0`

#### Decision

Promote `8 -> 32 -> 16 -> 1`, `fp_weight=2.0`, seed `1890407301` as the
relative-feature baseline. C6 remained the weakest long-recording
case, but the candidate kept the gain-shift invariance objective while
substantially reducing long-run false positives versus the initial relative
`24-12` export.

This baseline was later superseded by the hard-negative MVS weighting retrain
described below.

---

### MVS-Guided Weighting Bias and Hard-Negative Retrain

Date: 2026-07-03

Status: Superseded by AGC-active normalization and clean data recollection.

#### Goal

Determine whether MVS-derived metadata should guide ML training after long-run
validation showed that both MVS and ML can produce quiet-room motion spikes.

#### Background

At that stage the training stack experimented with storing a per-file
`optimal_threshold_gridsearch` value in `data/dataset_info.json`. The first use
of this metadata treated MVS as a context-aware training guide: per-file
thresholds affected the moving-variance ratio used for sample weights,
including both hard-positive mining and hard-negative emphasis.

That full MVS-guided approach was not a clear win:

- paired real-data ML validation regressed on `C3`, `C6`, and `S3`
- `ESP32` improved, but the aggregate result was mixed
- long-recording `mean_f1` improved only slightly, while total false positives
  increased
- the change looked like a sensitivity shift rather than a robust generalization
  improvement

The key concern was conceptual, not only numerical: MVS itself is weak on the
same noisy long-run quiet segments. Using MVS as a broad teacher can therefore
import MVS quiet-spike bias into the ML decision boundary.

#### Tooling Added

`tools/10_train_ml_model.py` gained explicit sample-weight policies:

- `none`: uniform sample weights; no MVS involvement
- `mvs_global`: legacy MVS-guided weighting with the global fallback threshold
- `mvs_gridsearch`: full MVS-guided weighting using the then-current per-file
  `optimal_threshold_gridsearch` metadata
- `mvs_hard_negative`: use MVS only to up-weight IDLE windows that look
  motion-like; motion samples remain neutral

Seed-search promotion was also tightened: a candidate was not promoted if it
increased long-recording total false positives or max false-positive rate over
the exported baseline.

#### Seed Search Result

Seed-search with `mvs_hard_negative` found seed `2083554459`.

Compared with the previously documented long-recording ML baseline:

| Metric | Previous ML Baseline | Hard-Negative Retrain |
|--------|----------------------|------------------------|
| Mean F1 | 97.0% | 97.1% |
| Worst F1 | 92.8% | 94.0% |
| Total FP | 578 | 497 |
| Mean recall | 99.0% | 98.6% |

Per-chip long-recording ML result:

| Chip | Recall | Precision | FP Rate | F1 | FP Count |
|------|-------:|----------:|--------:|---:|---------:|
| C3 | 98.5% | 100.0% | 0.0% | 99.2% | 0 |
| C5 | 100.0% | 91.3% | 7.7% | 95.4% | 247 |
| C6 | 96.0% | 92.1% | 7.9% | 94.0% | 250 |
| S3 | 99.9% | 100.0% | 0.0% | 99.9% | 0 |

The retrain deliberately accepts a small recall reduction in exchange for a
material false-positive reduction and a better worst-chip F1. This matches the
product priority for long static runs: avoid buying small recall gains with
large quiet-room FP costs.

#### Fixed-Seed Weighting Comparison

With seed fixed to `2083554459` and `--no-export`, the four weighting modes
were compared using grouped blocked CV:

| Mode | Recall | Precision | FP Rate | Fold F1 | Blocked OOF F1 | Worst C6 FP | Worst Source FP |
|------|-------:|----------:|--------:|--------:|---------------:|------------:|----------------:|
| `none` | 95.6% | 90.1% | 2.9% | 92.5% | 92.2% | 11.2% | 26.7% |
| `mvs_global` | 95.6% | 89.2% | 3.3% | 91.9% | 91.6% | 12.8% | 30.6% |
| `mvs_gridsearch` | 95.2% | 89.5% | 3.0% | 92.0% | 91.8% | 11.0% | 25.6% |
| `mvs_hard_negative` | 95.2% | 90.8% | 2.6% | 92.7% | 92.5% | 9.6% | 23.1% |

`mvs_hard_negative` was the clearest fixed-seed winner: best blocked OOF F1,
best precision, lowest FP rate, lowest C6 FP, and lowest worst-source FP, with
only a small recall trade-off versus `none`.

#### Decision

Promote seed `2083554459` and make `mvs_hard_negative` the default
sample-weight policy for production training at that point.

MVS remained useful as a hard-negative mining signal, but it should not be used
as a general teacher for motion labels unless a future long-recording gate
shows a clear FP-safe improvement.

#### Later Status

This result is superseded by the later AGC-active normalization refactor and by
the discovery that earlier datasets may have been contaminated. Production
training now defaults back to `none` so the first clean retrain establishes an
unbiased baseline.

Compare `mvs_hard_negative` against `none` again only after clean data
collection and refreshed explicit pair metadata; if detector guidance is used,
recompute it from the correct detector-specific startup calibration instead of
relying on stored thresholds.

---

### Tiny CNN / TCN Sweep

Date: 2026-05-20

Status: Rejected; superseded by later MLP baseline work.

#### Goal

Test whether small temporal models could beat the then-production `mlp-9` on an
FP-first ranking over the 4 curated long recordings.

#### Candidates

- `mlp-9`: then-production 9-feature MLP baseline
- `cnn-b`: Tiny 1D CNN using `turbulence + delta_turbulence`
- `tcn-a`: small causal temporal convolution baseline

#### Ranking Priority

1. lowest `max_fp_rate` on the 4 long recordings
2. lowest `total_fp`
3. highest long-run `pass_count`
4. highest `worst_chip_f1`

#### Final Outcome

- `mlp-9` remained the best practical model after the completed 5-seed final
  comparison
- `cnn-b` did not improve the FP-first ranking enough to justify promotion
- `tcn-a` remained non-competitive during screening / initial multi-seed
  evaluation and was not promoted

#### Completed Result Comparison

| Model | Seeds | Median Max FP Rate | Median Total FP | Median Pass Count | Median Worst-Chip F1 | Worst Max FP Rate |
|-------|-------|--------------------|-----------------|-------------------|----------------------|-------------------|
| MLP-9 | 5 | 7.86% | 556.0 | 2.0 | 94.04 | 7.89% |
| CNN-B | 5 | 7.92% | 553.0 | 2.0 | 83.54 | 9.77% |

#### Interpretation

- `cnn-b` occasionally matched the MLP on total FP, but it was less stable and
  substantially worse on the weakest chip/seed combination
- The main failure mode stayed on `C6`, where temporal candidates often traded
  away too much recall to gain only marginal FP improvements
- No tested temporal model showed a clear enough win to justify a deployment
  path beyond the MLP baseline at that time

#### Decision

Keep `mlp-9` as the production baseline at that time and focus follow-up work
on FP-first decision logic or alternative host-side baselines rather than
porting the tested temporal models.

---

## Notes

- `PERFORMANCE.md` is reserved for current product validation metrics
- `ALGORITHMS.md` describes the current promoted pipeline
- This file is for historical experiments, rejected candidates, and lessons
  learned
