# ML Feature Catalog

This document is the source of truth for ESPectre ML feature status, physical
interpretation, experiment results, and research backlog. It covers production
features, host-side candidates, superseded production features, rejected
experiments, and feature ideas that have not yet been implemented.

## Reading The Evidence

Feature metrics are comparable only when the corpus, split policy, seed, and
gate are the same. A higher F1 from an older campaign is not evidence that its
feature set would beat the current baseline. Tables therefore identify their
measurement context and keep historical campaigns separate.

Status values mean:

- **Production**: exported in the current five-input model and implemented in
  both Python and C++.
- **Research**: implemented host-side and still worth a more targeted
  experiment, but not eligible for export.
- **Rejected**: measured and not worth continuing in its tested formulation.
- **Historical**: previously shipped or used in a serious baseline, but
  superseded.
- **Planned**: physically motivated, but not yet implemented.

An em dash means that no trustworthy per-feature number was retained. It does
not mean zero. Qualitative results are recorded when they are all the evidence
that survives.

Every new candidate must:

1. be invariant to the unrecorded per-packet CSI scale factor;
2. add information beyond the production set, starting with pairwise
   correlation and multivariate linear redundancy;
3. improve at least one material lineage-grouped metric;
4. avoid material per-recording, weak-link, and quiet-replay regressions; and
5. remain host-side until a promotion decision justifies Python/C++ parity.

Host-side candidates live in `tools/lib/candidate_features.py` and must be
tested with `--no-export`.

## Current Production Set

The current baseline is the scale-invariant five-feature set introduced by the
[absolute-L1 removal ADR](adr/2026-07-28-drop-the-absolute-l1-features.md).
The exported topology is `5 -> 32 -> 16 -> 1`.

| Feature | Physical quantity | Definition | Scale invariant | Known evidence |
| --- | --- | --- | --- | --- |
| `turb_mad_over_mean` | Robust relative turbulence spread | `MAD(x) / abs(mean(x))` | Yes, ratio | Historical Core-6 label correlation `0.5752`; mean absolute SHAP `0.123743` (`21.4%`) on 460,958 windows, seed `1386543369` |
| `turb_autocorr` | Temporal persistence of turbulence | lag-1 autocorrelation `C(1) / C(0)` | Yes, correlation | Historical Core-6 label correlation `0.7834`; SHAP `0.101985` (`17.6%`). Weak-pair medians retained the correct ordering: idle `0.0115`, motion `0.5817` |
| `turb_zcr` | Temporal coherence versus noise-like crossings | crossing rate around the window median | Yes, crossing rate | In the Coherence-6 swap, helped reduce reserved max-FP median from `19.42%` to `2.66%` together with `l1_delta_autocorr` |
| `l1_delta_autocorr` | Persistence of normalized channel-profile displacement | lag-1 autocorrelation of the L1-delta series | Yes, correlation | Same Coherence-6 joint result; no isolated ablation metric retained |
| `l1_delta_lag_ratio` | Growth of channel-profile displacement with lag | mean lag-10 displacement / mean adjacent displacement | Yes, ratio | Adding it to Coherence-6 reduced reserved max FP `6.43% -> 4.43%`, raised worst recall `97.99% -> 99.14%`, and reduced effective alarms `8 -> 3` across ten reserved replays |

The five-feature replacement was evaluated over four seeds on the corpus that
exposed the absolute-L1 failure:

| Metric | Coherence-7 | Five scale-invariant features |
| --- | ---: | ---: |
| Blocked OOF F1 | `90.0%` | `98.0-98.3%` |
| Worst-session FP | `100.0%` | `9.5%` |
| Paired max FP | `7.16%` | `4.09-5.90%` |
| Paired effective alarms | `8` | `2-7` |
| Paired worst recall | `99.14%` | `95.76-96.26%` |

Recent retrains of the five-feature set have produced blocked OOF F1 around
`97.7-98.2%`, depending on the corpus actually loaded. Always print and retain
the effective corpus counts with a new result.

## Current Host-Side Campaign

The current physical-axis comparison fixes seed `1876849819`. The measured
corpus contained 553,801 windows in 23 lineage groups. The reference run scored
blocked OOF F1 `97.8%`, mean-of-five-worst lineage recall `95.2%`, and
mean-of-five-worst lineage FP `5.0%`.

These numbers are a campaign snapshot, not permanent properties of the
features. No candidate in this campaign is approved for production.

| Feature | Axis | Maximum production correlation / linear R2 | Measured result | Status |
| --- | --- | ---: | --- | --- |
| `chan_coh_lag_ratio` | Delay-compensated complex coherence dynamics | Orthogonal signal; nearly the inverse of `chan_coh_gap` (`r=-0.9988`) | Below baseline on the full corpus and fragile on noisy `static_presence` and `empty` replays; can beat baseline after post-hoc exclusion of suspicious replays | Research |
| `chan_coh_mean` | Mean lagged complex coherence | Low redundancy | Weak and unstable despite its orthogonality | Rejected |
| `chan_coh_gap` | Adjacent-minus-lagged complex coherence | Nearly equivalent to `chan_coh_lag_ratio` | Best tested coherence formulation, but still below the full-corpus baseline | Research |
| `chan_coh_gap_low_frac` | Fraction of small coherence gaps | Not retained | Did not beat `chan_coh_gap` | Rejected |
| `chan_coh_gap_q20` | Lower-tail coherence gap | Not retained | Did not beat `chan_coh_gap` | Rejected |
| `chan_coh_subband_median_gap` | Median robust coherence over four frequency bands | Not retained | OOF F1 `97.3%`; worst-lineage recall `80.9%`; worst-lineage FP `10.1%`; mean-of-five-worst FP `5.9%` | Rejected |
| `chan_coh_subband_gap_median` | Median of per-band coherence gaps | Not retained | OOF F1 `97.8%`; worst-lineage FP `8.4%`; mean-of-five-worst FP `5.6%`; replay gate max FP `6.04%`, worst recall `94.92%`, and 7 alarms | Rejected |
| `turb_band_power_ratio` | Low-frequency share of non-DC turbulence power | `r=0.923` with `turb_autocorr`; `R2=0.863` | OOF F1 `98.1%`, but two S3 static-presence replays regressed; swapping out autocorrelation scored `97.2%` | Rejected |
| `phase_resid_lag_ratio` | CFO/STO-sanitized phase-shape dynamics | max `r=0.434`; `R2=0.207` | OOF F1 `97.4%`; worst C5 FP `18.5%` | Rejected |
| `chan_shape_lag_ratio` | Temporal displacement of the L2-normalized amplitude profile | max `r=0.9831` with `l1_delta_lag_ratio`; `R2=0.9686`; label correlation `0.754` | Stopped before CV because the production set already reconstructs it | Rejected |
| `chan_shape_spread` | Frequency participation of lagged channel-shape motion energy | max `r=0.1094`; `R2=0.0257`; label correlation `0.0314` | OOF F1 `98.4%`; fold recall `98.0%`; precision `98.7%`; worst-lineage recall `86.5%`; worst-lineage FP `3.4%`; mean-of-five-worst recall `93.9%`, FP `2.4%` | Research |
| `chan_freq_coh_cv` | Temporal variability of within-packet frequency coherence at a four-bin offset | max `r=0.2468`; `R2=0.0838`; label correlation `0.2181` | OOF F1 `97.9%`; worst-lineage recall `93.3%`; worst-lineage FP `8.4%`; mean-of-five-worst recall `96.2%`, FP `4.4%` | Rejected |
| `chan_coh_gap_spread` | Product of positive coherence gap and channel-shape spread | Components are almost orthogonal (`r=-0.0058`) | OOF F1 `96.7%`; worst-lineage recall `93.3%`; worst C5 FP `26.1%`; mean-of-five-worst recall `95.9%`, FP `10.9%` | Rejected |

Combination results:

- `chan_coh_gap + chan_coh_lag_ratio` worsened the single-feature result; their
  near-deterministic inverse relation leaves no useful synergy.
- `chan_coh_gap + chan_shape_spread` scored OOF F1 `98.0%`, worst-lineage
  recall `86.5%`, worst C5 FP `14.3%`, and mean-of-five-worst recall/FP
  `94.4%`/`4.5%`. Orthogonality did not resolve the tail trade-off.

The important finding is physical, not merely statistical: channel coherence
and frequency participation contain information that the production
time-domain statistics do not, but noisy stationary replays can look like
broad, coherent pseudo-motion. C5 and C6 recordings dominated several
worst-case false-positive results. This is a robustness problem, not evidence
that coherence contains no motion signal.

The current corpus audit found no undeclared NPZ files: recordings used by the
trainer were represented in `data/dataset_info.json`. When an expected dataset
edit appears to have no effect, verify effective roles, counts, lineage groups,
and worst recordings rather than assuming an undeclared-file path.

## Historical And Rejected Features

### Superseded Production Families

| Family | Features | Scale invariant | Known campaign evidence | Why superseded |
| --- | --- | --- | --- | --- |
| Raw-12 | `turb_mean`, `turb_std`, `turb_max`, `turb_min`, `turb_iqr`, `turb_skewness`, `turb_autocorr`, `turb_mad`, `waveform_length`, `turb_kurtosis`, `turb_entropy`, `turb_slope` | Mixed, mostly no | Reduced to Raw-9 in the 2026-05-20 FP-first sweep | Kurtosis, entropy, and slope hurt long-recording robustness and overlapped with more stable descriptors |
| Raw-9 | Raw-12 without `turb_kurtosis`, `turb_entropy`, and `turb_slope` | Mixed, mostly no | Promoted with `9 -> 32 -> 16 -> 1`; exact per-feature metrics were not retained | Cross-gain behavior remained structurally fragile |
| Relative-8 | `turb_std_over_mean`, `turb_max_over_mean`, `turb_min_over_mean`, `turb_iqr_over_mean`, `turb_mad_over_mean`, `waveform_length_over_mean`, `turb_skewness`, `turb_autocorr` | Yes | Seed `1890407301`; gain-stress FP `1.1%` at `1.00x`, `1.25x`, and `1.50x`; long-run total FP `654`; worst-chip C6 F1 `93.5%` | Core-6 gave better grouped-CV and long-quiet behavior with fewer inputs |
| Core-6 | `turb_mad_over_mean`, `turb_skewness`, `turb_autocorr`, `l1_delta`, `l1_delta_std`, `l1_delta_waveform_length` | Mixed | Removing skew raised OOF F1 `92.4% -> 93.5%`, but total long FP worsened `601 -> 979`; max FP `0.9% -> 1.3%` | Absolute/energy-like members were weak-link and seed fragile |
| Coherence-6 | Core-6 with `turb_zcr` and `l1_delta_autocorr` replacing skew and waveform length | Mixed | Six-seed reserved max-FP median/worst `2.66%`/`2.80%` versus Core-6 `19.42%`/`24.93%`; promoted holdout `0.29%` FP and `100%` recall | Lag ratio improved reserved replay behavior further |
| Coherence-7 | Coherence-6 plus `l1_delta_lag_ratio` | Mixed | Ten reserved replays: max FP `4.43%`, worst recall `99.14%`, worst F1 `95.30%`, and 3 alarms | `l1_delta` and `l1_delta_std` allowed training-corpus composition to invert weak-link behavior |

### Individual Historical Results

| Feature or formulation | Known metric | Outcome |
| --- | --- | --- |
| `waveform_length_over_mean` | Relative-8 OOF F1 `79.49% -> 81.0%` and worst-chip recall `68.8% -> 75.1%` when dropped | Historical; weak |
| `turb_skewness` | Dropping it alone from refreshed Relative-8 gave OOF F1 `80.2%`; later Core-6 SHAP contribution was `1.9%`, but its removal worsened long FP | Historical; context-dependent protection, later superseded |
| `l1_delta` as a Relative-8 slot replacement | OOF F1 `76.59%`; fold F1 `78.74%` | Rejected in that formulation; later shipped inside Core-6, then removed for scale sensitivity |
| `l1_delta` standalone detector | AUC `0.993`, recall `93.2%`, FP `2.5%`, F1 `94.2%` | Strong historical detector signal, but absolute magnitude is not admissible for the current ML set |
| `l1_delta_std` | Weak-link idle/motion medians `0.0667`/`0.0519`, inverted relative to motion | Removed from production |
| `l1_delta_waveform_length` | Historical Core-6 SHAP `2.8%` | Removed in the Coherence-6 swap |
| `l1_delta_cv` | Seven-feature Coherence-6 extension worsened reserved replay results | Rejected |
| `turb_cv` | AUC `0.987`, recall `92.3%`, FP `4.9%`, F1 `91.6%` in the standalone candidate benchmark | Historical precursor to relative dispersion features |
| `turb_madratio` | Recall `90.0%`, FP `6.4%`, F1 `89.1%` | Historical precursor; superseded |
| Historical `band_power_ratio` | AUC `0.978`, recall `82.3%`, FP `6.0%`, F1 `85.1%` | Rejected as a primary detector; reserved RF-noise-gate idea |
| `l1_delta AND band_power_ratio` | F1 `93.4%`, FP `1.1%`; synthetic AWGN FPR about `91% -> 6.5%` at roughly 4 recall points and 2.6 s confirmation latency | Deferred unless real RF-event false positives justify the cost |
| `eigen_ratio` | AUC `0.887`, recall `84.5%`, FP `10.3%`, F1 `82.9%` | Rejected as inconsistent and expensive |
| `corr_amp_d10` | AUC `0.887`, recall `78.4%`, FP `7.1%`, F1 `81.8%` | Rejected; static frequency-selective shape dominated |
| `corr_complex_d10` | AUC `0.830`, recall `56.2%`, FP `6.8%`, F1 `66.5%` | Rejected; also absorbed packet phase offsets |
| `corr_amp_d1` | AUC `0.678`, recall `25.8%`, FP `1.9%`, F1 `39.9%` | Rejected; lag 1 mostly measured receiver noise |
| `turb_p95_over_mean` + `turb_p05_over_mean` | Robust-relative F1 `89.8%`, recall `89.1%`, FP `4.0%`, versus Relative-8 `91.5%`/`91.6%`/`3.5%` | Rejected |
| `turb_range_over_mean` | No retained isolated metric | Rejected after combined long-recording and gain-stress comparison |
| `turb_peak_over_mad` | No retained isolated metric | Rejected after combined long-recording and gain-stress comparison |

The historical standalone benchmark used session-calibrated thresholds and a
different corpus from current trainer CV. Its numbers must not be ranked
against the seed-`1876849819` campaign.

## Planned Physical Axes

Priority reflects expected incremental information, exact scale invariance,
robustness to missing/near-zero bins, and feasibility on HT20 ESP32 CSI.

| Priority | Candidate | Physical aspect | Scale-invariant construction | First useful experiment |
| ---: | --- | --- | --- | --- |
| 1 | Subcarrier rank dynamics | Frequency-selective channel-shape reordering without trusting amplitudes | Spearman correlation or rank-turnover rate between packet-normalized profiles | Compare lag and adjacent rank changes; test null-bin masks and redundancy against `chan_shape_spread` |
| 2 | Multi-offset frequency-coherence curve | Coherence bandwidth and a compact proxy for delay spread | Ratio or normalized slope of coherence at short and long frequency offsets | Measure several offsets, then retain one ratio only if it is more robust than `chan_freq_coh_cv` |
| 3 | Local closure-phase dynamics | Local multipath phase geometry after common phase cancellation | Wrapped phase closure over adjacent subcarrier triplets, summarized by a circular statistic | Start with stability and clipping tests before grouped CV |
| 4 | Trusted gain correction | Recover useful magnitude-domain information without link-floor leakage | Apply a validated packet scale correction before dimensionless normalization | First prove that profile `scale` is consistent across chips and collection paths |
| 5 | Narrowband micro-motion energy | Periodic breathing or occupancy micro-motion | Power fraction or spectral concentration, never absolute power | Evaluate in a separate Presence-versus-Empty task, not the current Motion classifier |
| Deferred | Delay/Doppler or CIR dynamics | Path-delay and velocity structure | Normalize maps or use ratios within a map | Revisit only if bandwidth, packet cadence, and exposed CSI support enough resolution |

## Literature Basis

The source summaries, reported filters and algorithms, publication dates,
metrics, hardware assumptions, and transferability assessments are centralized
in [LITERATURE.md](LITERATURE.md). Published results motivate the planned
physical axes above, but do not constitute validation on ESPectre data.

## Updating This Catalog

For every new experiment:

1. record the exact feature name and formula;
2. state whether scale invariance is proved, conditional, or absent;
3. record corpus window and lineage counts, seed, split policy, and baseline;
4. retain redundancy, grouped-CV, per-recording, and quiet-replay results;
5. assign a status and explain the dominant failure mode; and
6. link any promotion or durable rejection ADR without copying its full
   rationale here.

Code is authoritative for the exact active formula. This catalog is
authoritative for feature status and retained evidence.

If this catalog becomes difficult to scan, move its sections unchanged under
`docs/features/`, keep `FEATURES.md` as the index, and preserve its stable
anchors where practical.
