# Detector Feature Ledger

This document is the source of truth for ESPectre feature status, physical interpretation, retained project evidence, and research backlog. It covers the production ML inputs, features evaluated for ML or Classic, superseded production families, rejected formulations, and physically motivated ideas that remain blocked or unimplemented.

It is a decision ledger, not an experiment log. Durable production decisions and their campaign-level evidence live in [adr/](adr/), current detector behavior lives in [ALGORITHMS.md](ALGORITHMS.md), mutable detector metrics live in [performance/README.md](performance/README.md), collection workflow lives in [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md), and experiment commands live in [ML_TRAINING.md](ML_TRAINING.md) and [tools/README.md](../tools/README.md).

## Evidence Contract

Feature metrics are comparable only when the corpus, split policy, seed, and gate are the same. A higher F1 from an older campaign is not evidence that its feature set would beat the current baseline. The ADR linked from a ledger row owns that campaign context.

Status values mean:

- **Production**: exported in the current model or detector and implemented in both Python and C++.
- **Research**: implemented host-side and worth a targeted follow-up, but not approved for export.
- **Rejected**: measured and not worth continuing in its tested formulation.
- **Historical**: previously shipped or used in a serious baseline, but superseded.
- **Planned**: physically motivated, but not yet implemented.
- **Deferred**: physically plausible, but blocked by an unavailable input, radio capability, or validation corpus.

An em dash means that no trustworthy isolated number was retained. It does not mean zero. Qualitative evidence is retained when it is all that survives.

Every new candidate must:

1. be invariant to the unrecorded per-packet CSI scale factor;
2. add information beyond the production set, starting with pairwise correlation and multivariate linear redundancy;
3. improve at least one material lineage-grouped metric;
4. avoid material per-recording, weak-link, and quiet-replay regressions; and
5. remain host-side until a promotion decision justifies Python/C++ parity.

Host-side candidates live in `tools/lib/candidate_features.py`. Evaluate them with `--no-export` or `--evaluate-gates`; host-side seed searches run their deployment gates in memory and leave runtime artifacts unchanged.

## Current Production ML Set

The current production baseline is the compact phaseless seven-feature set. Only `turb_iqr_over_mean_aggr` reads a dedicated `W=5` adjacent-magnitude average; the other six inputs and Classic keep their normal amplitude paths. The exported topology is `7 -> 24 -> 12 -> 1`, with 505 parameters, the `base,drift,burst-loss` augmentation recipe, and false-positive weight `1.75`.

Current detector results are generated in [performance/README.md](performance/README.md). See [2026-08-07-promote-the-compact-aggregated-iqr-ml-model.md](adr/2026-08-07-promote-the-compact-aggregated-iqr-ml-model.md) for the feature-selection campaign, its reproducibility evidence, and the rejected alternatives. The exported weight files own the metadata of the currently deployed training run.

| Feature | Physical quantity and definition | Invariance | Retained feature evidence | Status |
| --- | --- | --- | --- | --- |
| `turb_iqr_over_mean_aggr` | Robust relative spread `(Q75(x) - Q25(x)) / abs(mean(x))`, where `x` is turbulence from a dedicated `W=5` adjacent-magnitude average | Ratio | Label correlation `+0.8116`; grouped OOF mean absolute SHAP `0.204348` (`39.1%`). It led the robust-dispersion screen and survived the final compact ablation | Production |
| `turb_autocorr` | Lag-1 autocorrelation `C(1) / C(0)` of the normal turbulence series | Correlation | Label correlation `+0.8549`; SHAP `0.089341` (`17.1%`). Weak-pair idle / motion medians retained the correct `0.0115` / `0.5817` ordering | Production |
| `turb_zcr` | Crossing rate of the turbulence series around its window median | Crossing rate | Label correlation `-0.8417`; SHAP `0.061765` (`11.8%`). Its joint Coherence-6 promotion reduced reserved max-FP seed fragility | Production |
| `l1_delta_autocorr` | Lag-1 autocorrelation of the normalized-profile L1-displacement series | Correlation | Label correlation `+0.6877`; SHAP `0.022108` (`4.2%`). No trustworthy isolated promotion metric survives | Production |
| `l1_delta_lag_ratio` | Mean lag-10 normalized-profile displacement divided by mean adjacent displacement | Ratio | Label correlation `+0.7506`; SHAP `0.097502` (`18.6%`). Its original promotion reduced reserved max FP `6.43% -> 4.43%`, raised worst recall `97.99% -> 99.14%`, and reduced alarms `8 -> 3` | Production |
| `chan_shape_spread` | Participation ratio of accumulated lagged normalized-channel-shape motion energy | Ratio | Label correlation `+0.0174`; SHAP `0.021863` (`4.2%`). It is nearly orthogonal to the other production inputs (`abs(r) <= 0.1068`) and survived through conditional tail protection rather than stable marginal correlation | Production |
| `chan_freq_coh_curve_std` | Temporal standard deviation of `(coh_offset2 - coh_offset12) / (coh_offset2 + coh_offset12)` over the full live band | Bounded contrast | Label correlation `+0.4210`; SHAP `0.026193` (`5.0%`). It improved cross-environment bedroom FP and is also the second Classic input | Production |

The five turbulence and L1 signals remain substantially redundant. The largest pairwise relation is `turb_autocorr` versus `turb_zcr` at `r=-0.9274`; aggregated IQR correlates `0.7244` with autocorrelation and `0.7187` with the lag ratio. Frequency coherence is more complementary, with absolute pairwise correlations from `0.2866` to `0.4049` against those five signals. `chan_shape_spread` changes marginal sign across chips, so its non-zero MLP importance must not be interpreted as a stable standalone detector axis.

## Current Production Classic Set

Classic uses weighted logistic fusion of `turb_autocorr` and `chan_freq_coh_curve_std`. The corrected discovery replay for the exported runtime reached `97.90%` weighted recall, `85.59%` worst-session recall, `2.28%` weighted paired FP, and `1.03%` maximum empty FP. Later candidate replays found recall-oriented alternatives, but none matched its quiet-room and packet-stress balance. See [2026-07-30-adopt-frequency-coherence-for-classic.md](adr/2026-07-30-adopt-frequency-coherence-for-classic.md) and [performance/README.md](performance/README.md).

## Active Research And Deferred Axes

| Candidate | Physical quantity and formulation | Redundancy and retained result | Next decision boundary | Status |
| --- | --- | --- | --- | --- |
| `chan_coh_lag_ratio` | Delay-compensated complex coherence dynamics | Nearly the inverse of historical `chan_coh_gap` (`r=-0.9988`); below baseline on the full corpus and fragile on noisy idle replays | Reopen only with new independent data that resolves the quiet tail without post-hoc exclusions | Research |
| `phase_resid_lag_ratio` | CFO/STO-sanitized phase-shape dynamics | Maximum production correlation `0.434`, linear `R2=0.207`; OOF F1 `97.4%`, worst C5 FP `18.5%` | Needs a targeted phase hypothesis and better C5 stationary coverage | Research |
| `phase_closure_var_std` | Temporal standard deviation of circular variance of `angle(H[k-1] H[k+1] conj(H[k])^2)` over guarded adjacent triplets | Maximum production correlation `0.1367`, linear `R2=0.0906`; OOF F1 `97.3%`, worst-lineage recall `92.1%`, and FP `11.8%` | Reopen only if phase curvature has a task-specific role beyond generic motion | Research |
| Classic lag-ratio triplet | `turb_autocorr + chan_freq_coh_curve_std + l1_delta_lag_ratio` | Improved discovery worst recall `85.59% -> 95.68%`, but paired FP `2.28% -> 3.36%`, maximum empty FP `1.03% -> 6.52%`, and alarms `55/1 -> 79/2` | Requires new sealed recordings and an exact shared packet-level comparison path | Research |
| Trusted profile-scale correction | Recover magnitude-domain information using a measured per-packet CSI gain | Unavailable: current callbacks and captures expose acquisition configuration, RSSI, and noise floor, but no selected per-packet scale | Revisit only if a supported SDK and protocol expose the measured value consistently across chips | Deferred |
| Narrowband micro-motion energy | Sub-Hz spectral concentration for breathing-related or occupancy micro-motion | The current one-second motion window and unpaired Presence-versus-Empty sessions cannot distinguish micro-motion from session drift | Collect same-link, same-session paired evidence with longer windows before implementation | Deferred |
| Delay/Doppler or CIR dynamics | Normalized path-delay and velocity structure | HT20 bandwidth, cadence, and exposed CSI do not yet provide validated resolution | Revisit only when the sensing contract can support and validate the map | Deferred |

## Rejected And Historical Feature Formulations

Rows group only formulations that shared one experiment and one failure mode. Different campaigns remain separate even when their feature names overlap.

| Feature or formulation | Definition or axis | Retained evidence and verdict | Status |
| --- | --- | --- | --- |
| `chan_coh_mean` | Mean lagged complex coherence | Low redundancy but weak and unstable | Rejected |
| `chan_coh_gap` | Adjacent-minus-lagged complex coherence | Previously shipped; removed jointly with overlapping coherence summaries after compact ablation | Historical |
| `chan_coh_gap_low_frac`, `chan_coh_gap_q20` | Fraction or lower quantile of small coherence gaps | Neither beat `chan_coh_gap`; no trustworthy isolated metric survives | Rejected |
| `chan_coh_subband_median_gap` | Median robust coherence over four frequency bands | OOF F1 `97.3%`, worst-lineage recall `80.9%`, worst-lineage FP `10.1%`, and mean-of-five-worst FP `5.9%` | Rejected |
| `chan_coh_subband_gap_median` | Median of per-band coherence gaps | Previously shipped; its removal improved blocked OOF F1 and the weakest-session FP | Historical |
| `chan_freq_coh_cv` | Temporal coefficient of variation of offset-4 within-packet coherence | Previously shipped; its independent removal led the compact ablation and its joint removal survived promotion | Historical |
| `chan_shape_lag_ratio` | Lagged L2-normalized profile displacement divided by adjacent displacement | Reconstructed by the production set (`r=0.9831`, linear `R2=0.9686`); stopped before CV | Rejected |
| `chan_rank_gap` | Mean lag-10 Spearman distance minus mean adjacent distance, guarded below `2%` of packet maximum | Repeated the lag-ratio signal (`r=0.7927`); OOF F1 `97.3%`, worst-lineage recall `91.0%`, and FP `7.4%` | Rejected |
| `chan_ratio_gap` | Median bounded change in guarded fixed-bin log ratios at lag 10 minus lag 1 | Repeated lag-ratio and rank-gap information; OOF F1 `97.6%`, worst-lineage recall `85.4%`, and FP `8.4%` | Rejected |
| `chan_coh_gap_spread` | Positive coherence gap multiplied by channel-shape spread | Orthogonal components did not yield robust synergy; OOF F1 `96.7%`, worst C5 FP `26.1%` | Rejected |
| `turb_band_power_ratio` | Low-frequency share of non-DC turbulence power | `r=0.923` with autocorrelation, linear `R2=0.863`; OOF F1 `98.1%`, but two S3 idle replays regressed, and the autocorrelation swap scored `97.2%` | Rejected |
| `chan_shape_scale_curvature` | Log-distance curvature of an eight-subband Hellinger energy profile at physical lags `80/240/720 ms`, after `80 ms` median binning, exact-stutter removal, and missing-bin skipping; evaluated host-side only | Single-feature primary AUC was `0.5805`, worst-pair AUC `0.4271`, and flips `3`; paired with innovation it reached only `0.9366` AUC and `0.8121` worst-pair AUC. The threshold-free `chan_shape_scale_curvature + turb_zcr` result looked stronger at `0.9945` AUC, `0.9735` worst-pair AUC, and zero flips, and C5 q95 stayed bounded under drift (`1.067x`), stutter (`1.123x`), random loss (`1.105x`), and burst loss (`1.087x`). The clean Classic replay nevertheless fell to `81.6%` worst discovery recall and `79.4%` holdout recall, with `28.0%` max empty-room FP; the fitted curvature weight was negligible next to `turb_zcr` | Rejected |
| `chan_shape_coherent_innovation_energy` | Positive low-order DCT energy of the constant-velocity residual of the same time-binned Hellinger profile, with high-order energy subtracted as a noise floor; evaluated host-side only | Threshold-free separation was strong: primary AUC `0.9948`, worst-pair AUC `0.9733`, holdout worst-pair AUC `0.9971`, and flips `0`. C5 q95 remained bounded under drift (`1.218x`), stutter (`1.436x`), random loss (`1.253x`), and burst loss (`1.025x`). The clean Classic replay still reached `19.6%` discovery pair FP, `20.1%` holdout pair FP, and `30.0%` empty-room FP; under `base` stress worst recall fell to `84.6%`, and under combined `base,drift,burst-loss` stress it fell to `66.4%` with `31.6%` holdout max FP. High window AUC did not survive the operational threshold and startup-calibration path | Rejected |
| `turb_iqr_over_mean_aggr_detrended` | Scale-invariant IQR of the residual after a least-squares linear detrend of the `W=5` aggregated turbulence window, normalized by the original mean; evaluated host-side only | On the time-aware real paired corpus, fit on `train` and ranked on `train+selection`, single-feature AUC was `0.9753`, worst weak-pair AUC `0.8062`, holdout AUC `0.9874`, and flips `0`; correlation with production aggregated IQR was `0.9558`. On the limiting C5 static replay with drift seed `20260807`, its q95 still rose `6.01x`, versus `6.14x` for the original IQR, while the original retained higher primary AUC `0.9821` and worst weak-pair AUC `0.8839`. Linear detrending does not remove the per-tone nonlinear drift mechanism and costs weak-link separation | Rejected |
| `turb_iqr_over_mean_aggr_tone_detrended` | Scale-invariant temporal IQR after removing a separate least-squares trend from each of the twelve `W=5` amplitude profiles, preserving each profile mean, and then recomputing spatial `std/mean`; evaluated host-side only | On the same time-aware real paired corpus, single-feature primary AUC was `0.9703`, worst-pair AUC `0.7110`, worst weak-pair AUC `0.8039`, holdout AUC `0.9862`, and flips `0`. On the limiting C5 static replay with drift seed `20260807`, q95 still rose `5.79x` and reached `0.2627`, versus `6.14x` and `0.2651` for production IQR. Per-profile linear detrending removes synthetic ramps but barely changes the nonlinear packet-drift tail, worsens clean separation, and requires a `window x 12` history | Rejected |
| Aggregated MAD and P95 swaps | `turb_mad_over_mean_aggr` or `turb_p95_over_mean_aggr` replacing `turb_zcr` | Redundant with production aggregated IQR (`abs(r)=0.9917` and `0.9257`) and worse on blocked OOF and C3 tails | Rejected |
| Aggregated robust-dispersion screen | W3/W5 variants of IQR, MAD, P95, minimum, P05, CV, range, and maximum relative to mean | IQR, MAD, P95, minimum, P05, and CV improved the limiting pair; range and maximum did not. Only aggregated IQR survived full promotion gates | Historical |
| Coherence-pair combinations | `chan_coh_gap + chan_coh_lag_ratio`, `chan_coh_gap + chan_shape_spread`, and `chan_freq_coh_cv + chan_freq_coh_curve_std` | The first was near-deterministically redundant, the second retained tail regressions, and the third failed paired replay at `12/13`, `8.04%` max FP, and 11 alarms | Rejected |
| Classic robust-dispersion pairs | Aggregated IQR or other robust dispersion paired with autocorrelation, ZCR, lag ratio, or frequency coherence | Some pairs improved clean recall, but quiet tails or packet stress failed; the apparent IQR-plus-autocorrelation Pareto point fell to `67.26%` worst recall under `base` stress | Rejected |
| Classic robust triplets | MAD/autocorrelation/ZCR and MAD/autocorrelation/shape-spread formulations | Strong threshold-free geometry did not survive corrected startup and empty-room replay; the first retained at least `14.84%` discovery-empty FP with hard negatives, and no trustworthy corrected packet result survives for the second | Rejected |
| One-feature Classic formulations | Production inputs evaluated alone | None was competitive; autocorrelation alone had `89.05%` limiting stressed recall and `7.40%` maximum empty FP | Rejected |
| `l1_delta_cv` | Coefficient of variation of the L1-displacement series | Worsened reserved replay results as a Coherence-6 extension | Rejected |
| `turb_p95_over_mean + turb_p05_over_mean` | Paired robust upper- and lower-tail ratios | F1 `89.8%`, recall `89.1%`, and FP `4.0%`, below Relative-8 at `91.5%` / `91.6%` / `3.5%` | Rejected |
| `turb_range_over_mean`, `turb_peak_over_mad` | Relative range or peak-to-MAD extreme statistics | Rejected after combined long-recording and gain-stress comparison; no trustworthy isolated metric survives | Rejected |
| `eigen_ratio` | Dominant-to-residual channel energy ratio | AUC `0.887`, recall `84.5%`, FP `10.3%`, and F1 `82.9%`; inconsistent and expensive | Rejected |
| `corr_amp_d10` | Lag-10 amplitude-profile correlation | AUC `0.887`, recall `78.4%`, FP `7.1%`, and F1 `81.8%`; dominated by static frequency-selective shape | Rejected |
| `corr_complex_d10` | Lag-10 complex-profile correlation | AUC `0.830`, recall `56.2%`, FP `6.8%`, and F1 `66.5%`; absorbed packet phase offsets | Rejected |
| `corr_amp_d1` | Adjacent amplitude-profile correlation | AUC `0.678`, recall `25.8%`, FP `1.9%`, and F1 `39.9%`; mostly receiver noise and did not improve under aggregation | Rejected |

The standalone AUC, recall, FP, and F1 values above used session-calibrated thresholds and an older corpus. They preserve historical behavior but must not be ranked against current grouped-CV campaigns.

## Production Baseline Lineage

| Family | Features | Retained campaign evidence | Why superseded |
| --- | --- | --- | --- |
| Raw-12 | Mean, standard deviation, extrema, IQR, skewness, autocorrelation, MAD, waveform length, kurtosis, entropy, and slope of turbulence | Reduced to Raw-9 in the 2026-05-20 FP-first sweep | Kurtosis, entropy, and slope hurt long-recording robustness and overlapped with more stable descriptors |
| Raw-9 | Raw-12 without kurtosis, entropy, and slope | Promoted with `9 -> 32 -> 16 -> 1`; exact per-feature metrics were not retained | Cross-gain behavior remained structurally fragile |
| Relative-8 | Relative standard deviation, extrema, IQR, MAD, waveform length, skewness, and autocorrelation | Seed `1890407301`; gain-stress FP `1.1%` from `1.00x` to `1.50x`; long-run total FP `654`; worst-chip C6 F1 `93.5%` | Core-6 improved grouped CV and long-quiet behavior with fewer inputs |
| Core-6 | `turb_mad_over_mean`, skewness, autocorrelation, `l1_delta`, `l1_delta_std`, and L1 waveform length | Removing skew raised OOF F1 `92.4% -> 93.5%`, but total long FP worsened `601 -> 979` | Absolute and energy-like members were weak-link and seed fragile |
| Coherence-6 | Core-6 with ZCR and L1 autocorrelation replacing skew and waveform length | Reserved max-FP median / worst `2.66%` / `2.80%` versus Core-6 `19.42%` / `24.93%` | Lag ratio improved reserved replay behavior further |
| Coherence-7 | Coherence-6 plus `l1_delta_lag_ratio` | Ten reserved replays: max FP `4.43%`, worst recall `99.14%`, and 3 alarms | Absolute L1 members let training composition invert weak-link behavior |
| Invariant-5 | MAD ratio, autocorrelation, ZCR, L1 autocorrelation, and L1 lag ratio | Four-seed blocked OOF F1 `98.0-98.3%`; worst-session FP `9.5%` | Broader phaseless features and localized aggregated IQR improved cross-group tails and model economy |
| Phaseless-10 | Invariant-5 plus shape spread and four coherence summaries | Passed deployment gates and removed the phase tracker | Three overlapping coherence summaries were removable together |
| Aggregated-IQR-7 | Current production set | See [2026-08-07-promote-the-compact-aggregated-iqr-ml-model.md](adr/2026-08-07-promote-the-compact-aggregated-iqr-ml-model.md) | Current |

## Individual Historical Signals

| Feature or formulation | Retained evidence | Outcome |
| --- | --- | --- |
| `waveform_length_over_mean` | Relative-8 OOF F1 `79.49% -> 81.0%` and worst-chip recall `68.8% -> 75.1%` when dropped | Historical; weak |
| `turb_skewness` | Dropping it from refreshed Relative-8 gave OOF F1 `80.2%`; later Core-6 SHAP was `1.9%`, but removal worsened long FP | Historical; context-dependent protection, later superseded |
| `l1_delta` as a Relative-8 replacement | OOF F1 `76.59%`; fold F1 `78.74%` | Rejected in that formulation; later shipped inside Core-6, then removed for scale sensitivity |
| `l1_delta` standalone detector | AUC `0.993`, recall `93.2%`, FP `2.5%`, and F1 `94.2%` | Strong historical signal, but absolute magnitude is inadmissible for the current ML set |
| `l1_delta_std` | Weak-link idle / motion medians `0.0667` / `0.0519`, inverted relative to motion | Removed from production |
| `l1_delta_waveform_length` | Historical Core-6 SHAP `2.8%` | Removed in the Coherence-6 swap |
| `turb_cv` | AUC `0.987`, recall `92.3%`, FP `4.9%`, and F1 `91.6%` | Historical precursor to relative dispersion features |
| `turb_madratio` | Recall `90.0%`, FP `6.4%`, and F1 `89.1%` | Historical precursor; superseded |
| Historical `band_power_ratio` | AUC `0.978`, recall `82.3%`, FP `6.0%`, and F1 `85.1%` | Rejected as a primary detector; retained only as a possible real-RF-noise gate |
| `l1_delta AND band_power_ratio` | F1 `93.4%`, FP `1.1%`; synthetic AWGN FPR about `91% -> 6.5%` at roughly four recall points and 2.6 seconds of confirmation latency | Deferred unless real RF-event false positives justify the cost |

## Literature Basis

Source summaries, publication dates, hardware and signal assumptions, reported methods and results, and transfer limits are centralized in [LITERATURE.md](LITERATURE.md). Published results can motivate a planned axis, but they do not validate it on ESPectre data.

## Updating This Ledger

For each seriously evaluated feature or formulation:

1. update one canonical ledger row rather than appending a chronological narrative;
2. retain its exact name and definition, physical interpretation, implementation scope, and invariance claim;
3. retain the campaign corpus, split, seed, baseline, primary metric, worst-group metrics, redundancy evidence, verdict, and dominant failure mode, either in the row or its linked ADR;
4. use an ADR only when the campaign produces a durable production, architectural, validation-policy, or stop-direction decision;
5. keep routine sweeps, intermediate seeds, superseded grids, per-file output, and transient commands out of permanent documentation; and
6. update current behavior, mutable metrics, data-collection priorities, and operator workflow only in their owning documents.

An experiment that produces no durable decision normally needs only its ledger row. When a durable ADR exists, the ledger summarizes the result and links it without copying the campaign narrative. Code remains authoritative for executable formulas; this ledger is authoritative for feature status and retained project evidence.
