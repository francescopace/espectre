# Historical Experiments

This document records notable host-side experiments that informed production
choices over time, including both rejected candidates and experiments that
eventually promoted a runtime baseline.

The goal is to preserve design history in one place without turning
`ALGORITHMS.md` into a research log.

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
| 2026-07-05 | MVS startup-threshold and online adaptation sweep | Active research note | `max x 1.3` remains the safest global baseline; online threshold tracking helps some chips, but not all. |

## Current Superseding Events

Date: 2026-07-04

Status: Active baseline reset.

The runtime and training path now use AGC-active, coefficient-of-variation
turbulence (`std(amplitudes) / mean(amplitudes)`) as the single production
path. Earlier dataset captures may also have been contaminated by a collection
bug. For that reason, the next clean ML baseline starts from
`--sample-weight-mode none`; MVS-guided weighting should be re-evaluated only
after clean data collection and refreshed `optimal_threshold_gridsearch`
metadata.

## Active Research Notes

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

Status: Active research note.

#### Goal

Test whether the current MVS startup threshold and simple online adaptation can
reduce static-room false positives without paying too much recall, before any
runtime C++ changes.

#### Background

The current production MVS startup path uses:

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

- `tools/mvs_sweep_core.py`: shared pair iterator, startup calibration, and
  continuous baseline -> motion evaluator
- `tools/5_analyze_filter_turbulence.py`: main MVS sweep and prototype
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

Keep `max x 1.3` as the default global MVS startup threshold.

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

The training stack was extended to annotate `optimal_threshold_gridsearch` in
`data/dataset_info.json`. The first use of this metadata treated MVS as a
context-aware training guide: per-file thresholds affected the moving-variance
ratio used for sample weights, including both hard-positive mining and hard
negative emphasis.

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
- `mvs_gridsearch`: full MVS-guided weighting using per-file
  `optimal_threshold_gridsearch`
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
collection and refreshed `optimal_threshold_gridsearch` metadata.

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
