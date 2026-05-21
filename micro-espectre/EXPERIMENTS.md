# Historical Experiments

This document records notable host-side experiments that informed the current
production choices, including both rejected candidates and experiments that
eventually promoted a new runtime baseline.

The goal is to preserve design history in one place without turning
`ALGORITHMS.md` into a research log.

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

## MLP Topology Sweep

### Goal

Check whether the current 9-feature MLP could reduce long-run false positives
by changing only the hidden-layer topology, without reopening the feature set
or training-policy axes.

### Candidates

- `Current default (24-12)` -> `9 -> 24 -> 12 -> 1`
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
| Current default (24-12) | 5 | 7.89% | 567.0 | 5.0 | 93.46 |
| Wider (32-16) | 5 | 7.86% | 506.0 | 5.0 | 93.96 |

### Decision

`Wider (32-16)` was promoted as the new production topology. The winning export
used seed `20260521`, passed the final paired validation rerun, and kept the
same 9-feature input set while improving the FP-first long-run ranking over the
previous `24-12` baseline.

The full campaign payload is stored in
`micro-espectre/models/mlp_architecture_experiment.json`.

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

---

## License

GPLv3 - See [../LICENSE](../LICENSE)
