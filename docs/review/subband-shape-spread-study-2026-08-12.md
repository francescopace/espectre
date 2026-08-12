# Subband Shape-Spread Model Study

Date: 2026-08-12
Branch: `v3.0`
Experiment base: `d01fae6f`
Scope: host-only economic replacements for `chan_shape_spread`, the selected seven-feature subband model, and the six-feature ablation that removes `chan_shape_excess_path`.

This record preserves the evidence used to choose the current research candidate. `docs/FEATURES.md` remains the compact feature ledger; this document retains the protocol, seed-level result, sealed evaluations, and implementation status needed to resume the study without reconstructing temporary experiment outputs.

## 1. Findings

| ID | Severity | Finding | Precise locations | Status |
| --- | --- | --- | --- | --- |
| `S-1` | Medium | DS2, slow EMA, and subband spread are credible host-only reductions of the production shape-spread cost. DS2 is the lower-risk CPU-plus-RAM option, slow EMA minimizes state, and subband reuses trajectory history with almost no additional persistent state. | `tools/lib/host_feature_trackers.py`, `tools/lib/candidate_features.py`, `tools/train_ml_model.py` | Measured |
| `S-2` | High | The seven-feature subband model improves selection recall and difficult `exclude` recall, but passes only `4/10` seed safety gates and concentrates cross-environment FP in the unseen bedroom. It is not ready for runtime promotion. | `tools/lib/host_feature_trackers.py`, `tools/train_ml_model.py`, `docs/FEATURES.md` | Research candidate |
| `S-3` | High | Removing `chan_shape_excess_path` is CV-safe at one seed but unsafe after sealed evaluation and a fresh ten-seed search: every seed produces a selection quiet alarm. | `tools/train_ml_model.py`, `docs/FEATURES.md` | Rejected |
| `C-1` | High | Feature-subset experiments were recomputing cached supersets. Indexed superset projection now preserves manifests and column order, and semantic pruning removes artifacts whose dependencies no longer resolve. | `tools/lib/npz_cache.py`, `tools/prune_npz_cache.py`, `docs/review/npz-cache-review-2026-07-29.md` | Resolved |

## 2. Protocol

All comparisons used the role-isolated `train`, `selection`, `holdout`, and `exclude` corpus, standard scaling, `base,drift,burst-loss` augmentation, `fp_weight=1.75`, and the production hidden topology. The baseline and subband candidate used `7 -> 24 -> 12 -> 1`; the compact ablation used `6 -> 24 -> 12 -> 1`. Ranking used CV and selection only. Holdout, `exclude`, and leave-one-environment-out results remained sealed until each winner had been selected.

The shared seed set was `1641245296`, `1442571517`, `68673049`, `1089338155`, `680974852`, `1245285065`, `1584727888`, `2070652044`, `543495078`, and `1134469357`. The baseline values come from its existing production-spread campaign; the six-feature model received a fresh search over the same seeds.

The compared inputs were:

| Model | Seed | Inputs |
| --- | ---: | --- |
| Current baseline | `1089338155` | `turb_iqr_over_mean_aggr`, `turb_autocorr`, `turb_zcr`, `l1_delta_lag_ratio`, `chan_shape_spread`, `chan_shape_coherent_innovation_energy`, `chan_shape_excess_path` |
| Subband 7F | `1584727888` | Baseline inputs with `chan_shape_spread_subband` replacing `chan_shape_spread` |
| Subband 6F | `68673049` | Subband 7F without `chan_shape_excess_path` |

The final cached training matrix resolved `49/49` clean captures and `49/49` augmented captures as hits. A previously unseen six-feature subset projected compatible cached supersets rather than regenerating packet features.

## 3. Best-Seed Comparison

All rates are percentages. Each model is shown at its own winner, so the table compares deployable model selections rather than forcing a common random initialization.

| Metric | Current baseline | Subband 7F | Subband 6F |
| --- | ---: | ---: | ---: |
| Selection-safe seeds | `10/10` | `4/10` | `0/10` |
| Blocked OOF F1 | 99.077 | 99.187 | 99.187 |
| CV worst-session recall | 86.517 | 88.764 | 92.135 |
| CV worst-session FP | 2.564 | 1.163 | 2.286 |
| Selection worst recall | 92.550 | 96.264 | 95.977 |
| Selection maximum paired FP | 0.000 | 0.284 | 0.142 |
| Selection mean F1 | 98.543 | 99.272 | 99.329 |
| Selection quiet maximum FP | 0.342 | 0.385 | 0.598 |
| Selection quiet alarms | 0 | 0 | 1 |
| Holdout worst recall | 99.140 | 97.971 | 98.551 |
| Holdout maximum paired FP | 0.000 | 0.000 | 0.143 |
| Holdout mean F1 | 99.856 | 99.811 | 99.812 |
| Holdout quiet maximum FP | 0.299 | 0.725 | 0.981 |
| Holdout quiet alarms | 0 | 0 | 2 |
| `exclude` worst recall | 5.556 | 7.639 | 8.333 |
| `exclude` motion misses | 163 | 146 | 141 |
| `exclude` paired FP | 0.000 | 0.000 | 0.000 |
| `exclude` quiet FP | 0.000 | 0.000 | 0.862 |
| `exclude` alarms | 0 | 0 | 1 |
| Cross-environment macro recall | 98.670 | 98.609 | 98.300 |
| Cross-environment macro FP | 0.083 | 0.213 | 0.296 |
| Cross-environment macro F1 | 99.234 | 99.055 | 98.803 |

The seven-feature subband candidate gains `3.71` selection-recall points and `2.08` `exclude`-recall points over the current baseline, while keeping all sealed alarm counts at zero. Its principal risk is a higher cross-environment FP tail: unseen-bedroom FP is `0.640%`, including `0.539%` on empty data, while the macro FP remains `0.213%`.

The six-feature winner gains CV recall and one more `exclude` detection, but the apparent aggregate benefit is not safe. Its selection, holdout, and `exclude` quiet alarms recur after seed selection, and its cross-environment FP is `3.57x` the baseline value.

## 4. Six-Feature Seed Search

Every six-feature seed produced one selection quiet alarm. This common failure is stronger evidence than the initial single-seed regression because it rules out random initialization as the explanation.

| Seed | OOF F1 | Selection worst recall | Selection max FP | Quiet max FP | Quiet alarms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1641245296` | 99.161 | 94.828 | 0.427 | 0.556 | 1 |
| `1442571517` | 99.187 | 94.828 | 0.284 | 0.556 | 1 |
| `68673049` | 99.187 | 95.977 | 0.142 | 0.598 | 1 |
| `1089338155` | 99.187 | 95.115 | 0.284 | 0.556 | 1 |
| `680974852` | 99.160 | 94.828 | 0.284 | 0.556 | 1 |
| `1245285065` | 99.187 | 95.690 | 0.427 | 0.556 | 1 |
| `1584727888` | 99.268 | 95.115 | 0.427 | 0.556 | 1 |
| `2070652044` | 99.105 | 95.402 | 0.427 | 0.598 | 1 |
| `543495078` | 99.215 | 95.115 | 0.284 | 0.598 | 1 |
| `1134469357` | 99.187 | 95.977 | 0.427 | 0.556 | 1 |

The winner's CV worst-session recall and FP were `92.135%` and `2.286%`. The full search medians were OOF F1 `99.187%`, CV worst-session recall `91.011%`, CV worst-session FP `1.714%`, selection worst recall `95.115%`, selection maximum paired FP `0.356%`, and quiet maximum FP `0.556%`.

## 5. Correlation, SHAP, And Ablation Evidence

The analysis used 200 balanced held-out samples from the seven-feature subband winner. `chan_shape_spread_subband` had low label correlation (`+0.1389`) and the lowest mean absolute SHAP value (`0.01440`), but it was almost orthogonal to the other inputs (`R2=0.0244`). Its low marginal importance therefore did not imply redundancy.

| Feature | Label correlation | Mean absolute SHAP | Reconstruction R2 from other inputs |
| --- | ---: | ---: | ---: |
| `turb_iqr_over_mean_aggr` | +0.821 | 0.19126 | 0.639 |
| `turb_autocorr` | +0.866 | 0.10920 | 0.883 |
| `l1_delta_lag_ratio` | +0.784 | 0.07199 | 0.773 |
| `turb_zcr` | -0.839 | 0.05648 | 0.875 |
| `chan_shape_coherent_innovation_energy` | +0.537 | 0.04270 | 0.731 |
| `chan_shape_excess_path` | +0.617 | 0.02072 | 0.694 |
| `chan_shape_spread_subband` | +0.139 | 0.01440 | 0.024 |

The strongest pairwise relationships were `turb_autocorr` versus `turb_zcr` at `-0.9307`, and coherent innovation versus excess path at `+0.8127`. Despite those correlations, removing autocorrelation, ZCR, innovation, or excess path did not survive the complete evaluation. Removing `chan_shape_excess_path` was the only ablation to pass the robust CV comparison at seed `1584727888`, improving OOF F1 from `99.187%` to `99.268%` and worst-session recall by `3.37` points. Its sealed regressions motivated the fresh seed search in Section 4, which confirmed the rejection.

## 6. Resource Interpretation

The production shape-spread tracker is estimated at approximately `22.1 KiB` of state. DS2 reduces this to approximately `11.2 KiB` and roughly halves its packet work. Either EMA variant reduces state to approximately `2.4 KiB` with similar per-packet arithmetic. Subband spread adds almost no persistent state when both trajectory features remain active because it consumes the eight normalized profiles already retained by the trajectory tracker.

These are design estimates, not measured firmware values. No host-only candidate has been ported to MicroPython or C++, `ml_weights.h` has not been regenerated, and no runtime behavior has changed.

## 7. Decision And Progress

Retain the seven-feature subband model as the selected research candidate, including `chan_shape_excess_path`. Do not promote it until an independent-room gate resolves the unseen-bedroom FP tail and an on-device comparison verifies the expected CPU and RAM benefit. DS2 and slow EMA remain documented fallback candidates; ClassicDetector is outside this study.

- [x] Implement host-only DS2, fast EMA, slow EMA, and subband candidates.
- [x] Run a common single-seed screen.
- [x] Run ten-seed DS2, slow-EMA, and subband comparisons.
- [x] Evaluate correlation, reconstruction, and grouped OOF SHAP.
- [x] Run all seven single-feature ablations.
- [x] Run the fresh ten-seed six-feature search.
- [x] Open sealed holdout, `exclude`, and cross-environment gates only after winner selection.
- [x] Restore cache subset reuse and semantic pruning for repeated experiments.
- [ ] Validate the seven-feature subband candidate on a fresh independent room.
- [ ] Implement exact Python/C++ parity for the selected finalist.
- [ ] Measure CPU time, peak RAM, and persistent tracker state on representative devices.
- [ ] Make the final promotion or rejection decision.
