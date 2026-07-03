# Historical Experiments

This document records notable host-side experiments that informed the current
production choices, including both rejected candidates and experiments that
eventually promoted a new runtime baseline.

The goal is to preserve design history in one place without turning
`ALGORITHMS.md` into a research log.

---

## Gain-Shift Robustness Diagnostic

### Goal

Check whether the production ML detector is structurally robust to
device/session gain shifts, or whether it is only empirically robust on the
current collected domains.

### Background

MVS has an explicit no-gain-lock path: when gain is not locked, turbulence can
be computed as coefficient of variation (`std / mean`), which is invariant to a
uniform amplitude scale factor. At the time of the first gain-stress
diagnostic, the ML detector intentionally used raw turbulence std in both
training and runtime inference. Its exported feature scaler was a global
statistical standardization fitted on the training set; it did not compensate
an unseen per-device/session gain shift.

For the previous raw 9-feature production baseline:

- scale-sensitive: `turb_mean`, `turb_std`, `turb_max`, `turb_min`,
  `turb_iqr`, `turb_mad`, `waveform_length`
- scale-invariant under positive uniform gain: `turb_skewness`,
  `turb_autocorr`

### Tooling Added

`tools/10_train_ml_model.py --gain-stress-gate` evaluates the currently
exported Python weights without retraining. It extracts the exported feature
set from real `empty`/`static_presence`/`motion` data, applies artificial gain
multipliers only to the scale-sensitive features, and reports overall plus
worst-group metrics.

Example:

```bash
python tools/10_train_ml_model.py --gain-stress-gate
python tools/10_train_ml_model.py --gain-stress-gate --environment bedroom
python tools/10_train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25
```

### Raw-Feature Result

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

### Decision

The raw-feature ML detector is strong at nominal gain, but it is not
structurally gain-invariant. Global feature standardization is numeric
normalization, not domain compensation. Seed search and long/paired gates can
promote empirically better weights, but they do not by themselves solve
cross-gain robustness.

Follow-up experiments compared:

1. current raw features
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
production relative set (`F1=89.8%`, `recall=89.1%`, `FP=4.0%` versus the
promoted retrain's `F1=91.5%`, `recall=91.6%`, `FP=3.5%`). It remains an
analysis-only feature set. The gain-stress gate remains the primary diagnostic
for gain-shift robustness, with the long-recording gate acting as the
false-positive non-regression check.

---

## C3 Empty-Room Retrain Incident

### Goal

Diagnose why a C3 ESPHome deployment in a static room produced noisy ML scores
despite good offline validation, then decide whether the failure was caused by
the model, runtime inference, or dataset coverage.

### Observation

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
and reported gain metadata stayed stable. Because the production ML pipeline
uses raw per-packet turbulence as its base signal, those jumps entered the
100-packet ML window as turbulence spikes. Relative window features reduce
uniform window-level gain shifts, but they do not make the model structurally
immune to arbitrary per-frame amplitude jumps.

### Decision

The production fix was to include `empty` in binary ML training, mapping both
`empty` and `static_presence` to IDLE and `motion` to MOTION. This better
matches the deployed task: the detector must suppress both quiet empty rooms
and static-presence rooms, not only distinguish static presence from motion.

At this point no C++ feature ABI change was required:

- the production ML feature set remains the 8 relative features
- `MLDetector::set_cv_normalization()` remained a no-op in that retrain, so
  generic gain-lock fallback logic could not silently switch ML into a
  different turbulence mode
- the C++ runtime changed only through regenerated exported weights

A live ESPHome C3 smoke test after the retrain produced 37 IDLE publications
and 1 MOTION publication across 38 post-connect samples, with median score
`0.10`, mean score `0.76`, and one score above the fixed threshold `5.0`.

### Follow-Up

Per-packet CV turbulence for ML (`std(amplitudes) / mean(amplitudes)`) was later
promoted for no-gain-lock streams. The production rule is now gain-mode aware:
gain-locked streams use raw turbulence, while streams without gain lock use
CV-normalized turbulence before the same 8 relative ML features are extracted.
`MLDetector::set_cv_normalization()` now follows the runtime request instead of
ignoring it.

Retraining with seed `1890407301` on the full clean dataset produced blocked
grouped-CV `F1=92.2%` and passed the Python and C++ real-data gates. The C++
ESP32 paired validation is the most direct regression check: ML with CV OFF
had `90.2%` recall, while the gain-aware path reached `100.0%` recall with
`0.0%` FP.

The long-recording gate still exposes noisy C5/C6 idle segments (`C5` ML
`7.7%` FP, `C6` ML `10.1%` FP). This appears separate from ESP32 gain-lock
handling because the same long files are also difficult for MVS (`C5` `11.1%`
FP, `C6` `40.2%` FP). Treat it as a dataset/environment coverage issue, not as
evidence against the ESP32 gain-aware ML fix.

### Second Empty-Domain Capture

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

## Feature-Set Reduction Sweep

### Goal

Reduce long-recording false positives without weakening the deployed MLP
architecture or breaking Python/C++ parity.

### Setup

- Production topology kept fixed at `9 -> 24 -> 12 -> 1`
- Candidate feature sets evaluated with grouped CV, paired validation, and
  long-recording holdout
- Ranking favored holdout robustness, not CV alone

### Decision

The input feature set was reduced from 12 to 9.

Removed features:

- `turb_kurtosis`
- `turb_entropy`
- `turb_slope`

### Why These Features Were Dropped

- They sometimes improved paired validation slightly, but hurt the
  long-recording holdout where FP robustness mattered more
- They overlapped with more stable signals already captured by
  `turb_autocorr`, `turb_iqr`, `turb_mad`, and `waveform_length`
- They increased deployment complexity without producing a reliable FP-first
  win

### Outcome

- The current 9-feature MLP became the production baseline
- The simpler input set improved holdout robustness while preserving strong
  paired-set quality

---


## FP-First Feature and Training Sweep

### Goal

Revisit the production `mlp-9` from the opposite angle of the temporal sweep:
identify which features amplify long-run false positives, then test whether
feature-set changes or training-policy changes can reduce FP without paired-set
regression.

### Axes Tested

- per-window profiling on the 4 curated long recordings
- feature diagnostics on `TP/FP/TN/FN` buckets
- FP-first training policies (`fp_weight`, negative emphasis, threshold tuning)
- targeted candidates combining feature-set changes and training policy

### Result

- Winner: `baseline-9`
- Median long `max_fp_rate`: 7.00%
- Median long `total_fp`: 356.0
- Median long `worst_chip_f1`: 89.00
- Baseline reference (`baseline-9`): `max_fp_rate=7.00%`, `total_fp=356.0`, `worst_chip_f1=89.00`

### Decision

The campaign only promotes a candidate if the FP-first ranking improves in
median and stays stable in the worst case. See the generated JSON campaign
artifact for the full shortlist and diagnostics.

### Follow-Up: `drop-turb_min`

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

So `drop-turb_min` was explicitly rejected and the production baseline remains
unchanged.

---

## Raw-9 MLP Topology Sweep

### Goal

Check whether the then-current 9-feature MLP could reduce long-run false positives
by changing only the hidden-layer topology, without reopening the feature set
or training-policy axes.

### Candidates

- `Then-current default (24-12)` -> `9 -> 24 -> 12 -> 1`
- `Legacy (16-8)` -> `9 -> 16 -> 8 -> 1`
- `Shallow (24)` -> `9 -> 24 -> 1`
- `Wider (32-16)` -> `9 -> 32 -> 16 -> 1`
- `Deep (24-12-6)` -> `9 -> 24 -> 12 -> 6 -> 1`

### Ranking Priority

1. lowest long-run `max_fp_rate`
2. lowest long-run `total_fp`
3. highest long-run `pass_count`
4. highest long-run `worst_chip_f1`
5. paired validation as a non-regression constraint
6. grouped CV only as a final tie-breaker

### Key Observation During Screening

`Shallow (24)` looked strong on 3-seed median `total_fp`, but `Wider (32-16)`
held a slightly better primary FP ceiling (`max_fp_rate`) and therefore won the
head-to-head slot for the final 5-seed comparison.

### Final Outcome

| Architecture | Seeds | Median Max FP Rate | Median Total FP | Median Paired Pass Count | Median Worst-Chip F1 |
|--------------|-------|--------------------|-----------------|--------------------------|----------------------|
| Then-current default (24-12) | 5 | 7.89% | 567.0 | 5.0 | 93.46 |
| Wider (32-16) | 5 | 7.86% | 506.0 | 5.0 | 93.96 |

### Decision

`Wider (32-16)` was promoted for the raw-9 production line. The winning export
used seed `20260521`, passed the final paired validation rerun, and kept the
same 9-feature input set while improving the FP-first long-run ranking over the
previous raw `24-12` baseline.

The full campaign payload is stored in
`models/mlp_architecture_experiment.json`.

---

## Relative-8 Topology and FP-Weight Sweep

### Goal

Keep the gain-invariant relative feature set, then recover long-recording
false-positive robustness by changing only the MLP topology and IDLE-class
weighting.

### Setup

- Feature set fixed to the promoted relative 8-feature view:
  `std/mean`, `max/mean`, `min/mean`, `iqr/mean`, `mad/mean`,
  normalized waveform length, skewness, autocorrelation
- Seed fixed to `1890407301` for the focused screen
- Primary gate: curated 60-second long recordings
- Non-regression checks: paired real-data validation and exported gain-stress
  gate

### Focused Screen

| Candidate | Params | Long Total FP | Long Max FP Rate | Worst-Chip F1 | Mean Recall | Gain-Stress FP @ 1.5x |
|-----------|-------:|--------------:|-----------------:|--------------:|------------:|----------------------:|
| `24-12`, `fp_weight=1.0` | 529 | 1178 | 18.7% | 91.0% | 100.0% | 1.9% |
| `32-16`, `fp_weight=1.0` | 833 | 829 | 15.1% | 92.5% | 99.9% | 1.4% |
| `48-24`, `fp_weight=1.0` | 1633 | 828 | 12.6% | 93.7% | 99.8% | 0.9% |
| `32-16-8`, `fp_weight=1.0` | 961 | 878 | 11.9% | 94.0% | 100.0% | 1.4% |
| `24-12`, `fp_weight=1.5` | 529 | 1134 | 18.5% | 91.1% | 100.0% | 1.9% |
| `32-16`, `fp_weight=1.5` | 833 | 822 | 13.3% | 93.3% | 99.9% | 1.4% |
| `32-16`, `fp_weight=2.0` | 833 | 712 | 11.3% | 94.2% | 99.7% | 0.7% |

### Promoted Export

The focused screen selected `32-16` with `fp_weight=2.0`. A full train/export
with seed `1890407301` produced:

- topology: `8 -> 32 -> 16 -> 1`
- grouped blocked OOF F1: `93.0%`
- paired real-data gate: pass
- exported gain-stress gate: flat at `1.00x`, `1.25x`, `1.50x` with
  `FP=1.1%`
- long-recording ML gate: `total_fp=654`; per-chip FP counts `C3=0`,
  `C5=249`, `C6=405`, `S3=0`

### Decision

Promote `8 -> 32 -> 16 -> 1`, `fp_weight=2.0`, seed `1890407301` as the
relative-feature production baseline. C6 remains the weakest long-recording
case, but the candidate keeps the gain-shift invariance objective while
substantially reducing long-run false positives versus the initial relative
`24-12` export.

This baseline was later superseded by the hard-negative MVS weighting retrain
described below.

---

## MVS-Guided Weighting Bias and Hard-Negative Retrain

### Goal

Determine whether MVS-derived metadata should guide ML training after long-run
validation showed that both MVS and ML can produce quiet-room motion spikes.

### Background

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

### Tooling Added

`tools/10_train_ml_model.py` gained explicit sample-weight policies:

- `none`: uniform sample weights; no MVS involvement
- `mvs_global`: legacy MVS-guided weighting with the global fallback threshold
- `mvs_gridsearch`: full MVS-guided weighting using per-file
  `optimal_threshold_gridsearch`
- `mvs_hard_negative`: use MVS only to up-weight IDLE windows that look
  motion-like; motion samples remain neutral

Seed-search promotion was also tightened: a candidate is not promoted if it
increases long-recording total false positives or max false-positive rate over
the current exported baseline.

### Seed Search Result

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

### Fixed-Seed Weighting Comparison

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

### Decision

Promote seed `2083554459` and make `mvs_hard_negative` the default
sample-weight policy for production training.

MVS remains useful as a hard-negative mining signal, but it should not be used
as a general teacher for motion labels unless a future long-recording gate
shows a clear FP-safe improvement.

---

## Tiny CNN / TCN Sweep

### Goal

Test whether small temporal models could beat the production `mlp-9` on an
FP-first ranking over the 4 curated long recordings.

### Candidates

- `mlp-9`: production 9-feature MLP baseline
- `cnn-b`: Tiny 1D CNN using `turbulence + delta_turbulence`
- `tcn-a`: small causal temporal convolution baseline

### Ranking Priority

1. lowest `max_fp_rate` on the 4 long recordings
2. lowest `total_fp`
3. highest long-run `pass_count`
4. highest `worst_chip_f1`

### Final Outcome

- `mlp-9` remained the best practical model after the completed 5-seed final
  comparison
- `cnn-b` did not improve the FP-first ranking enough to justify promotion
- `tcn-a` remained non-competitive during screening / initial multi-seed
  evaluation and was not promoted

### Completed Result Comparison

| Model | Seeds | Median Max FP Rate | Median Total FP | Median Pass Count | Median Worst-Chip F1 | Worst Max FP Rate |
|-------|-------|--------------------|-----------------|-------------------|----------------------|-------------------|
| MLP-9 | 5 | 7.86% | 556.0 | 2.0 | 94.04 | 7.89% |
| CNN-B | 5 | 7.92% | 553.0 | 2.0 | 83.54 | 9.77% |

### Interpretation

- `cnn-b` occasionally matched the MLP on total FP, but it was less stable and
  substantially worse on the weakest chip/seed combination
- The main failure mode stayed on `C6`, where temporal candidates often traded
  away too much recall to gain only marginal FP improvements
- No tested temporal model showed a clear enough win to justify a deployment
  path beyond the current MLP

### Decision

Keep `mlp-9` as the production baseline and focus follow-up work on FP-first
decision logic or alternative host-side baselines rather than porting the
tested temporal models.

---

## Notes

- `PERFORMANCE.md` is reserved for current product validation metrics
- `ALGORITHMS.md` describes the current promoted pipeline
- This file is for historical experiments, rejected candidates, and lessons
  learned
