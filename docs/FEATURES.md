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
- **Deferred**: physically plausible, but blocked by an unavailable input,
  radio capability, or validation corpus.

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

The current production baseline is the compact phaseless ten-feature set chosen
after the refreshed current-catalog sweep. It keeps the five turbulence/L1
invariants and adds the five coherence and channel-shape signals that survived
the promotion gates without needing the phase tracker. The exported topology is
`10 -> 24 -> 12 -> 1`, trained with `--augment`, `--fp-weight 1.75`, and seed
`1876849819`.

| Feature | Physical quantity | Definition | Scale invariant | Known evidence |
| --- | --- | --- | --- | --- |
| `turb_mad_over_mean` | Robust relative turbulence spread | `MAD(x) / abs(mean(x))` | Yes, ratio | Historical Core-6 label correlation `0.5752`; mean absolute SHAP `0.123743` (`21.4%`) on 460,958 windows, seed `1386543369` |
| `turb_autocorr` | Temporal persistence of turbulence | lag-1 autocorrelation `C(1) / C(0)` | Yes, correlation | Historical Core-6 label correlation `0.7834`; SHAP `0.101985` (`17.6%`). Weak-pair medians retained the correct ordering: idle `0.0115`, motion `0.5817` |
| `turb_zcr` | Temporal coherence versus noise-like crossings | crossing rate around the window median | Yes, crossing rate | In the Coherence-6 swap, helped reduce reserved max-FP median from `19.42%` to `2.66%` together with `l1_delta_autocorr` |
| `l1_delta_autocorr` | Persistence of normalized channel-profile displacement | lag-1 autocorrelation of the L1-delta series | Yes, correlation | Same Coherence-6 joint result; no isolated ablation metric retained |
| `l1_delta_lag_ratio` | Growth of channel-profile displacement with lag | mean lag-10 displacement / mean adjacent displacement | Yes, ratio | Adding it to Coherence-6 reduced reserved max FP `6.43% -> 4.43%`, raised worst recall `97.99% -> 99.14%`, and reduced effective alarms `8 -> 3` across ten reserved replays |
| `chan_shape_spread` | Frequency participation of lagged channel-shape motion energy | participation ratio of accumulated lagged normalized-shape energy | Yes, ratio | Promoted in the refreshed compact sweep: paired `14/14`, cross-chip worst FP `1.9%`, and better leave-one-environment bedroom FP than the wider no-phase layout |
| `chan_freq_coh_cv` | Temporal variability of within-packet frequency coherence at four-bin offset | std / mean of fixed-offset within-packet coherence over time | Yes, ratio | Retained in the promoted no-phase family; part of the best `24 -> 12` compact trade-off on the refreshed corpus |
| `chan_freq_coh_curve_std` | Temporal variability of short-versus-long frequency coherence contrast | std of `(coh_offset2 - coh_offset12) / (coh_offset2 + coh_offset12)` | Yes, bounded contrast | Retained in the promoted no-phase family; improved the compact model's cross-environment bedroom tail over the wider no-phase layout |
| `chan_coh_gap` | Adjacent-minus-lagged delay-compensated channel coherence | mean lag-1 coherence minus mean lag-10 coherence | Yes, difference of normalized coherences | Retained in the promoted no-phase family; the best coherence formulation from the current sweep |
| `chan_coh_subband_gap_median` | Median subband coherence-gap across four contiguous HT20 bands | median over subbands of their adjacent-minus-lag mean coherence gaps | Yes, median of normalized coherence gaps | Retained in the promoted no-phase family; helped keep the promoted model robust on weak bedroom and C3 tails |

The previous five-feature replacement was evaluated over four seeds on the corpus that
exposed the absolute-L1 failure:

| Metric | Coherence-7 | Five scale-invariant features |
| --- | ---: | ---: |
| Blocked OOF F1 | `90.0%` | `98.0-98.3%` |
| Worst-session FP | `100.0%` | `9.5%` |
| Paired max FP | `7.16%` | `4.09-5.90%` |
| Paired effective alarms | `8` | `2-7` |
| Paired worst recall | `99.14%` | `95.76-96.26%` |

Recent retrains of that five-feature baseline have produced blocked OOF F1 around
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
| `chan_coh_subband_gap_median` | Median of per-band coherence gaps | Not retained | OOF F1 `97.8%`; worst-lineage FP `8.4%`; mean-of-five-worst FP `5.6%`; replay gate max FP `6.04%`, worst recall `94.92%`, and 7 alarms | Research |
| `turb_band_power_ratio` | Low-frequency share of non-DC turbulence power | `r=0.923` with `turb_autocorr`; `R2=0.863` | OOF F1 `98.1%`, but two S3 static-presence replays regressed; swapping out autocorrelation scored `97.2%` | Rejected |
| `phase_resid_lag_ratio` | CFO/STO-sanitized phase-shape dynamics | max `r=0.434`; `R2=0.207` | OOF F1 `97.4%`; worst C5 FP `18.5%` | Research |
| `phase_closure_var_std` | Temporal variability of local phase curvature across adjacent subcarrier triplets | max `r=0.1367` with `turb_mad_over_mean`; `R2=0.0906`; `r=0.0521` with `phase_resid_lag_ratio`; label correlation `0.0455` | OOF F1 `97.3%`; worst-lineage recall `92.1%`; worst-lineage FP `11.8%`; mean-of-five-worst recall `95.3%`, FP `6.4%` | Research |
| Trusted profile-scale correction | Recover magnitude-domain information after undoing packet gain | Unavailable: the ESP-IDF packet callback exposes no per-packet scale value | All 95 current NPZ recordings include RSSI metadata, but none records scale; current firmware only selects static automatic/manual scaling configuration | Deferred |
| `chan_shape_lag_ratio` | Temporal displacement of the L2-normalized amplitude profile | max `r=0.9831` with `l1_delta_lag_ratio`; `R2=0.9686`; label correlation `0.754` | Stopped before CV because the production set already reconstructs it | Rejected |
| `chan_shape_spread` | Frequency participation of lagged channel-shape motion energy | max `r=0.1094`; `R2=0.0257`; label correlation `0.0314` | OOF F1 `98.4%`; fold recall `98.0%`; precision `98.7%`; worst-lineage recall `86.5%`; worst-lineage FP `3.4%`; mean-of-five-worst recall `93.9%`, FP `2.4%` | Research |
| `chan_freq_coh_cv` | Temporal variability of within-packet frequency coherence at a four-bin offset | max `r=0.2468`; `R2=0.0838`; label correlation `0.2181` | OOF F1 `97.9%`; worst-lineage recall `93.3%`; worst-lineage FP `8.4%`; mean-of-five-worst recall `96.2%`, FP `4.4%` | Research |
| `chan_freq_coh_curve_std` | Temporal variability of short-versus-long frequency coherence | max `r=0.3881` with `turb_mad_over_mean`; `R2=0.1598`; `r=0.7831` with `chan_freq_coh_cv`; label correlation `0.4080` | Isolated OOF F1 `97.8%`; worst-lineage recall `94.4%`; worst-lineage FP `8.4%`; mean-of-five-worst recall `95.7%`, FP `5.1%` | Research |
| `chan_rank_gap` | Lagged reordering of the amplitude profile relative to adjacent-packet reordering | max `r=0.7927` with `l1_delta_lag_ratio`; `R2=0.6750`; `r=-0.0668` with `chan_shape_spread`; label correlation `0.5981` | OOF F1 `97.3%`; worst-lineage recall `91.0%`; worst-lineage FP `7.4%`; mean-of-five-worst recall `93.7%`, FP `5.1%` | Rejected |
| `chan_ratio_gap` | Lagged change in guarded cross-subcarrier amplitude ratios relative to adjacent-packet change | max `r=0.7866` with `l1_delta_lag_ratio`; `R2=0.6303`; `r=0.6914` with `chan_rank_gap`; `r=-0.0282` with `chan_shape_spread`; label correlation `0.5863` | OOF F1 `97.6%`; worst-lineage recall `85.4%`; worst-lineage FP `8.4%`; mean-of-five-worst recall `94.3%`, FP `5.6%` | Rejected |
| `chan_coh_gap_spread` | Product of positive coherence gap and channel-shape spread | Components are almost orthogonal (`r=-0.0058`) | OOF F1 `96.7%`; worst-lineage recall `93.3%`; worst C5 FP `26.1%`; mean-of-five-worst recall `95.9%`, FP `10.9%` | Rejected |

Combination results:

- `chan_coh_gap + chan_coh_lag_ratio` worsened the single-feature result; their
  near-deterministic inverse relation leaves no useful synergy.
- `chan_coh_gap + chan_shape_spread` scored OOF F1 `98.0%`, worst-lineage
  recall `86.5%`, worst C5 FP `14.3%`, and mean-of-five-worst recall/FP
  `94.4%`/`4.5%`. Orthogonality did not resolve the tail trade-off.
- `chan_freq_coh_cv + chan_freq_coh_curve_std` scored OOF F1 `98.1%`,
  worst-lineage recall `94.4%`, worst-lineage FP `6.7%`, and
  mean-of-five-worst recall/FP `96.1%`/`3.8%`. The paired replay gate regressed
  to `12/13` passes, `8.04%` max FP, `95.11%` worst recall, and 11 alarms,
  versus the baseline's `13/13`, `4.82%`, and `96.05%`. The quiet gate passed
  with `0.04%` max FP and no alarms, but paired non-regression rejected the
  combination.

### Adjacent-Subcarrier Aggregation

The August 5, 2026 screen asked whether averaging adjacent bins into each of the
twelve selected subcarriers improves the features that read the amplitude
buffer. It is an input-path transform rather than a candidate feature, so it
changes existing features instead of adding one.

Corpus: the 19 `train` pairs, replayed through the production detectors with the
aggregation injected at the amplitude-buffer fill. Metric: per-pair separation
`max(AUC, 1-AUC)`, which keeps inverted-polarity features comparable. This is a
screen, not a promotion gate: no grouped-CV, replay, or quiet-gate run was
performed, because the Classic result did not warrant one.

| Feature | Worst pair, baseline | Worst pair, W=3 | Same limiting pair | Mean paired delta |
| --- | ---: | ---: | --- | ---: |
| `turb_mad_over_mean` | 0.6190 | 0.8155 | yes | +0.0139 |
| `turb_zcr` | 0.9685 | 0.9457 | yes | -0.0006 |
| `turb_autocorr` | 0.9734 | 0.9502 | yes | -0.0011 |
| `l1_delta_autocorr` | 0.8403 | 0.8687 | no | +0.0042 |
| `l1_delta_lag_ratio` | 0.9824 | 0.9576 | no | -0.0009 |

The five channel-shape and coherence features read the full 56-bin live profile,
never the twelve-tone buffer, and are bit-identical under aggregation.

The two statistics computed on the turbulence series, `turb_autocorr` and
`turb_zcr`, lose on the same limiting pair, while `turb_mad_over_mean` gains
sharply on its own: its quiet floor is the noise being reduced, rather than what
keeps the quiet series structureless. The two `l1_delta` rows are not evidence
either way, because their worst pair is a different recording in the two
configurations and their mean paired deltas are small. Classic is built on
`turb_autocorr`, so aggregation was rejected for the shared path; see
[2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md](adr/2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md)
for the noise measurements, the width sweep, and the mechanism.

`turb_mad_over_mean` is not alone: the robust-dispersion statistics of the
turbulence series all move the same way, including several retired candidates.
The rows below reconstruct those statistics on the unmodified production
turbulence series; the harness reproduces the production `turb_mad_over_mean`
worst pair exactly, so the reconstructions share its basis.

| Candidate | Worst pair, baseline | Worst pair, W=3 | Same limiting pair |
| --- | ---: | ---: | --- |
| `turb_iqr_over_mean` | 0.6317 | 0.8308 | yes |
| `turb_mad_over_mean` (production) | 0.6190 | 0.8155 | yes |
| `turb_p95_over_mean` | 0.6219 | 0.7865 | yes |
| `turb_min_over_mean` | 0.5237 | 0.6864 | yes |
| `turb_p05_over_mean` | 0.5732 | 0.6673 | yes |
| `turb_cv` | 0.5103 | 0.6451 | yes |
| `turb_range_over_mean` | 0.5855 | 0.5432 | yes |
| `turb_max_over_mean` | 0.5819 | 0.5210 | yes |

Robust dispersion statistics gain on the limiting pair, while the two max-based
extremes lose it, so "extreme-order statistics are the most noise-sensitive and
therefore gain most" is not what the corpus shows. `turb_iqr_over_mean`, a
Relative-8 member dropped at the Core-6 transition, edges past the current
production feature. `corr_amp_d1` was the strongest prior candidate, having been
rejected because at lag 1 it mostly measured receiver noise, and it did not
benefit: median separation falls `0.8920` to `0.8793`.

All of this is ML-side and screening-grade. The ADR records what a retrain would
have to settle before any of it could be promoted.

### Classic Linear Candidate Replay

The July 30, 2026 Classic campaign asked whether a linear pair or triplet from
the existing ML feature surface could improve the exported non-ML detector.
The search covered generic pairs and triplets, runtime-ready and host-only
surfaces, threshold-free screening, startup-calibrated logistic replay, and
targeted packet-level confirmation. The discovery stage itself changed no
runtime Python or C++ detector code; the winning pair from this line of work
was promoted later and is the current exported `ClassicDetector`.

#### Valid Method

The corrected workflow has these contracts:

- fit coefficients only on real `train` recordings;
- use `train + selection` for discovery decisions;
- report `holdout` and `exclude` separately, without allowing either to affect
  ranking;
- fit grouped folds on de-overlapped rows, but score every dense runtime
  evaluation tick;
- use `StratifiedGroupKFold` with `random_state=0`;
- limit nominal startup evidence to the 37 ready evaluations available during
  the 1,000-packet production calibration;
- report the exported `ClassicDetector` through exact packet-level replay as
  the baseline; and
- label a refitted current feature pair as a surrogate, never as the exported
  runtime baseline.

Earlier results that fitted or ranked on `holdout`, fitted folds on dense
overlapping rows, used 10 or 64 startup rows, or treated the refitted current
pair as the exported baseline are superseded. In particular, the initial
research scores of about `44.1` for the baseline, `25.0` for
`turb_mad_over_mean + turb_autocorr + turb_zcr`, and `25.6` for
`turb_mad_over_mean + turb_autocorr + chan_shape_spread` are not promotion
evidence.

The historical `holdout` recordings were observed during this campaign.
Their numbers remain useful diagnostics, but they are no longer a sealed
confirmation set and must not be presented as unbiased final validation.
The exact effective file/window counts and the startup-strength/settle-margin
values for the temporary packet-level points were not retained in a repository
artifact; that evidence is unavailable and must not be reconstructed from the
rounded metrics below. All results are tied to the repository corpus as it
existed on July 30, 2026.

#### Retained Results

The exported runtime baseline is
`turb_autocorr + chan_freq_coh_curve_std`.

The current Python and C++ `ClassicDetector` implementations mirror this pair.
The retained runtime policy is the same one validated here: grouped,
de-overlapped coefficient fitting, startup-centering against the training idle
reference, and the later settled-level threshold recovery documented in
`ALGORITHMS.md`.

| Evaluation | Research score | Weighted recall | Worst recall | Weighted paired FP | Maximum empty FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exported runtime, discovery | `28.44` | `97.90%` | `85.59%` | `2.28%` | `1.03%` |
| Exported runtime, historical holdout | — | — | `99.71%` | `2.18%` | `3.45%` |
| Current pair refit, aggressive point | `23.22` | — | `89.91%` | `4.31%` | `7.34%` |
| Current pair refit, conservative point | `28.99` | — | `87.90%` | — | `7.89%` |
| Current pair refit with train-empty hard negatives | `22.38` | `98.76%` | `88.47%` | `3.90%` | `6.25%` |

The hard-negative refit also regressed historical-holdout paired FP to `4.08%`
and maximum empty FP to `5.58%`. Its lower scalar penalty therefore hides
material false-positive regressions. Keep it research-only; it fails
multidimensional non-regression and is not approved for export.

The strongest threshold-free triplet was
`turb_mad_over_mean + turb_autocorr + turb_zcr`, with `0.9966` worst-pair
discovery AUC. Its components were highly redundant: maximum absolute
correlation was `0.919`, and mean absolute correlation was `0.794`. Corrected
surrogate replay retained about `97.99%` worst-session recall, but maximum
discovery-empty FP rose to at least `16.77%`. Adding train-role empty hard
negatives still left it at `14.84%`. Reject this formulation for Classic.

`turb_mad_over_mean + turb_autocorr + chan_shape_spread` was the second initial
shortlist candidate. Its original score was produced by the superseded
methodology, and no trustworthy corrected packet-level result was retained.
Do not promote or cite the original `25.6` score. Reconsider it only if new
independent data supplies a specific reason to reopen the feature hypothesis.

The best host-only triplet did not show a clear advantage over the best
runtime-ready triplet. Do not add new runtime extractors for this Classic
campaign.

An expanded July 31, 2026 replay over all runtime-ready triplets changed the
screening verdict. The strongest surrogate candidate is now
`turb_autocorr + chan_freq_coh_curve_std + l1_delta_lag_ratio`.
After a later same-day dataset edit and model regeneration, the screening order
stayed the same, but the current-pair surrogate became materially cleaner on
discovery empties. With train-role `empty` hard negatives included in the fit,
the lag-ratio triplet still beat the refitted current pair on discovery recall,
while the pair regained ground on quiet tails:

| Evaluation | Research score | Weighted recall | Worst recall | Weighted paired FP | Maximum empty FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current pair refit with train-empty hard negatives | `10.22` | `98.76%` | `91.93%` | `3.50%` | `4.08%` |
| `turb_autocorr + chan_freq_coh_curve_std + l1_delta_lag_ratio` with train-empty hard negatives | `2.81` | `99.31%` | `95.68%` | `3.36%` | `6.52%` |
| `turb_autocorr + chan_freq_coh_curve_std + chan_shape_spread` with train-empty hard negatives | `4.67` | `99.11%` | `96.56%` | `3.20%` | `7.07%` |

The lag-ratio triplet therefore became the only live Classic replacement
hypothesis worth carrying forward on the current corpus. The shape-spread
triplet remains a weaker runner-up: it still beats the refitted pair, but it
trails the lag-ratio triplet on scalar score and on the worst quiet tails.
The `turb_mad_over_mean` triplets remain rejected for deployment despite their
strong threshold-free geometry: once startup calibration and empty-room replay
are restored, their quiet tails stay materially worse than the leading
lag-ratio candidate.

A closer post-refresh comparison against the committed `ClassicDetector`
narrowed the case further. On the updated corpus, the lag-ratio triplet still
improved discovery recall (`97.90% -> 99.31%`) and worst-session recall
(`85.59% -> 95.68%`), but it lost on the quiet metrics that protect the shipped
Classic path: weighted paired FP `2.28% -> 3.36%`, maximum discovery-empty FP
`1.03% -> 6.52%`, and pair/idle effective alarms `55/1 -> 79/2`. The same
pattern held on `exclude`: weighted recall improved `90.84% -> 94.51%`, but
weighted paired FP regressed `3.82% -> 9.00%`, maximum empty FP regressed
`27.69% -> 44.11%`, and idle alarms rose `75 -> 152`. The candidate therefore
remains a recall-oriented research hypothesis, not a drop-in replacement for the
committed detector.

#### Interpretation And Verdict

The screening result is now more specific than a generic triplet rejection.
One runtime-ready triplet clearly dominates the current-pair surrogate:
`turb_autocorr + chan_freq_coh_curve_std + l1_delta_lag_ratio`. However, the
updated direct comparison also shows why it is still not promotable. The
candidate improves recall, but the committed detector remains materially quieter
on the same corpus, and the `exclude` tails still hinge on a small set of C3/C6
pair and quiet-room replays, especially the C6 `empty` session
`empty_c6_64sc_dev00007c2c6742bbac_20260728_134140_988645_0001.npz` and the
C3 pair `static_presence_c3_64sc_dev0000acebe64ae708_20260725_135809_478030_0001.npz`.
Repeated tuning against those same recordings would optimize the corpus rather
than establish generalization.

The current decision is:

1. keep the exported `turb_autocorr + chan_freq_coh_curve_std` Classic detector
   unchanged;
2. keep `turb_autocorr + chan_freq_coh_curve_std + l1_delta_lag_ratio` only as
   the leading recall-oriented research hypothesis, not as a promotion
   candidate on the current corpus;
3. stop the broader pair and triplet grid search on the present corpus; and
4. require new independent recordings and a maintained shared packet-level
   comparison path before any promotion attempt.

The packet-level confirmations that justified the current pair were run through
a temporary local research harness, not a maintained repository entry point.
Before any future Classic replacement or retune, add an exact, shared
packet-level comparison path that evaluates the exported runtime and candidate
with identical calibration, cadence, reset, settling, and runtime-policy
semantics.

#### Restart Point

Collect new same-protocol data before resuming Classic promotion work. See
`data/COLLECTION_PLAN.md` for the exact capture priorities.

- S3 low-RSSI `static_presence` and `motion` pairs across more than one room,
  placement, and collection time;
- C3 paired `static_presence` / `motion` recordings that mirror the current
  `exclude` recall tails;
- C5 and C6 `empty` recordings across rooms, times, and normal/weak links;
- C5 and C6 paired `static_presence` / `motion` recordings around the current
  false-positive tails; and
- enough independent session or lineage groups to reserve a new confirmation
  set before inspecting detector results.

After assigning the new roles, reproduce the narrow research surface with:

```bash
python tools/benchmark_classic_candidate_pairs.py \
  --triple turb_autocorr,chan_freq_coh_curve_std,l1_delta_lag_ratio \
  --triple turb_autocorr,chan_freq_coh_curve_std,chan_shape_spread \
  --triple turb_mad_over_mean,turb_autocorr,turb_zcr

python tools/replay_classic_candidates.py \
  --features turb_autocorr,chan_freq_coh_curve_std \
  --features turb_autocorr,chan_freq_coh_curve_std,l1_delta_lag_ratio \
  --features turb_autocorr,chan_freq_coh_curve_std,chan_shape_spread \
  --include-train-empty
```

Use the replay only for screening. The next candidate must then pass the exact
packet-level comparison and improve recall without exceeding the exported
baseline's discovery paired-FP and empty-FP rates. Final promotion must use the
new sealed confirmation groups, followed by the required Python/C++ parity
gates. If the lag-ratio triplet still fails after that, retain the current
detector and investigate a different model or calibration family instead of
expanding the linear feature grid again.

### Heterogeneous Feature And Model Sweep

The isolated verdicts above did not answer whether individually weak,
physically different measurements become useful jointly. A systematic sweep
therefore extracted one 22-column superset once, then compared 11 feature
families spanning 5 to 20 inputs on the same 553,801 windows, 23 lineage
groups, grouped folds, and selection seed `1876849819`. `P` below denotes the
five production features.

| Family | Added to, or changed from, `P` | Inputs | OOF F1 | Worst-five recall | Worst-five FP |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | None | 5 | `97.77%` | `95.23%` | `4.96%` |
| Lean heterogeneous | `chan_coh_gap`, `chan_shape_spread`, `chan_freq_coh_curve_std` | 8 | `98.45%` | `94.37%` | `2.83%` |
| Compact orthogonal | gap quantile, subband gap, phase closure, shape spread, and frequency-coherence CV | 10 | `98.20%` | `92.57%` | `2.18%` |
| Frequency dynamics | shape spread, frequency CV and curve, rank gap, and ratio gap | 10 | `98.39%` | `94.83%` | `2.78%` |
| Coherence distribution | seven full-band, lower-tail, subband, and composite coherence descriptors | 12 | `95.81%` | `94.57%` | `12.96%` |
| Phase and frequency | phase residual and closure, shape spread, frequency CV and curve, full-band gap, and subband gap | 12 | `98.07%` | `95.95%` | `5.45%` |
| All physical axes | band-power ratio plus coherence, phase, shape, and frequency descriptors | 12 | `97.68%` | `93.48%` | `6.19%` |
| Alternate lag core | Replaced the production lag ratio and added band power, shape lag, rank, ratio, coherence, and frequency curve | 10 | `97.00%` | `95.02%` | `6.97%` |
| Alternate turbulence core | Replaced production autocorrelation and added band power, coherence, shape, frequency, and phase | 10 | `97.98%` | `94.15%` | `4.90%` |
| Broad physics | Thirteen candidate descriptors across all measured axes | 18 | `97.21%` | `91.46%` | `6.30%` |
| Wide non-redundant | All candidates except the two near-deterministic lag-ratio duplicates | 20 | `97.14%` | `93.71%` | `7.98%` |

All rows initially used `32 -> 16` hidden layers and false-positive weight
`1.5`. Adding every available input was harmful; the useful interaction was a
selected mixture of time, phase, frequency, channel-shape, and coherence
measurements. Historical sweeps on the earlier catalog pointed to the
phase-and-frequency family, but the current compact frontier was re-run after
the updated `C6 bedroom normal-link` pair and with the active deployment gate
workflow. The refreshed comparison used 553,690 active training windows, 23
lineage groups, seed `1876849819`, packet augmentation, `32 -> 16` hidden
layers, and false-positive weight `1.75`.

The selected 12-input reference is `P` plus `phase_resid_lag_ratio`,
`phase_closure_var_std`, `chan_shape_spread`, `chan_freq_coh_cv`,
`chan_freq_coh_curve_std`, `chan_coh_gap`, and
`chan_coh_subband_gap_median`. All are dimensionless ratios, correlations,
normalized-profile statistics, or circular phase statistics. An end-to-end
streaming extraction check scaled every raw CSI component by `0.5`, `2`, and
`4`, while avoiding clipping and quantization loss; every feature was bitwise
unchanged. This proves formula and implementation invariance to a common
positive gain. It cannot make clipped samples or information lost to int8
quantization recoverable.

| Model | Features | MLP params | Blocked OOF F1 | Paired gate | Quiet holdout | Cross-chip | Cross-environment | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| Full compact `32 -> 16` | 12 | 961 | `97.9%` | `14/14`, max FP `1.96%`, worst recall `97.96%`, `0` alarms | pass, max FP `0.13%` | macro F1 `98.1%`, worst recall `97.3%`, worst FP `2.5%` | macro F1 `96.0%`, bedroom FP `9.5%` | Current full compact reference |
| No phase `32 -> 16` | 10 | 897 | `97.7%` | `14/14`, max FP `0.29%`, worst recall `98.25%`, `0` alarms | pass, max FP `0.13%` | macro F1 `98.2%`, worst recall `96.8%`, worst FP `2.1%` | macro F1 `97.1%`, bedroom FP `5.7%` | Best tracker-level reduction |
| No phase `24 -> 12` | 10 | 577 | `97.4%` | `14/14`, max FP `0.44%`, worst recall `97.67%`, `0` alarms | pass, max FP `0.13%` | macro F1 `98.2%`, worst recall `96.8%`, worst FP `1.9%` | macro F1 `97.6%`, bedroom FP `3.9%` | Best compact-layout trade-off |
| No phase `20 -> 10` | 10 | 441 | `97.5%` | `14/14`, max FP `0.29%`, worst recall `96.50%`, `0` alarms | pass, max FP `0.13%` | macro F1 `98.1%`, worst recall `96.8%`, worst FP `2.9%` | macro F1 `96.6%`, bedroom FP `7.1%` | Smallest tested layout that still passes |
| No phase, no `l1_delta_autocorr` `32 -> 16` | 9 | 865 | `98.0%` | `14/14`, max FP `0.15%`, worst recall `98.25%`, `0` alarms | pass, max FP `0.08%` | macro F1 `98.2%`, worst recall `97.1%`, worst FP `2.7%` | macro F1 `96.4%`, bedroom FP `7.3%` | Smallest passing feature set so far |

The important refreshed result is physical, not merely statistical. Dropping
both phase features removes the entire phase tracker and improves the paired
gate and leave-one-environment bedroom split, even though blocked OOF F1 dips
slightly. That is a more meaningful firmware simplification than removing a
single first-layer input while keeping the same tracker alive.

Once the phase tracker is gone, layout downsizing is a higher-value lever than
further feature pruning. `24 -> 12` cuts the MLP from 897 to 577 parameters
(`-35.7%`) while improving the worst held-out FP rates on both leave-one-chip
and leave-one-environment checks. `20 -> 10` cuts further to 441 parameters
(`-50.8%` versus the no-phase `32 -> 16` model), but gives back too much on
`bedroom` and the worst C3 false-positive tail to be the default compact
choice.

The failed family removals were equally informative:

- Dropping the shape family and keeping only production, phase, and coherence
  inputs still scored `14/14`, but regressed non-regression alarms on
  `S3:selection:static_presence_s3_64sc_dev000010b41de8ec00_20260728_125456_174006_0001.npz`
  (`4` versus `2`).
- Dropping the coherence family also kept `14/14`, but introduced one
  effective alarm on
  `C3:holdout:static_presence_c3_64sc_dev0000acebe64adb64_20260729_005553_607144_0001.npz`
  and a recall regression on
  `C3:selection:static_presence_c3_64sc_dev0000acebe64ae708_20260728_121127_735859_0001.npz`.

Current feature-economy evidence therefore supports this ordering:

1. remove the full phase family first;
2. if MLP size still matters, drop `l1_delta_autocorr` next; and
3. keep the shape and coherence families intact until new data proves their
   replay protection is replaceable.

The other tested 9-input drop, removing `chan_coh_gap` instead of
`l1_delta_autocorr`, also passed the deployment gates, but its blocked OOF F1
fell to `97.7%` and it offered no tracker-level savings beyond the first-layer
weights, so it is not the preferred compact reduction.

The important finding is physical, not merely statistical: channel coherence
and frequency participation contain information that the production
time-domain statistics do not, but noisy stationary replays can look like
broad, coherent pseudo-motion. C5 and C6 recordings dominated several
worst-case false-positive results. This is a robustness problem, not evidence
that coherence contains no motion signal.

`chan_rank_gap` used the mean lag-10 Spearman distance minus the mean
adjacent-packet distance, where distance is `(1 - rho) / 2`. Bins below `2%`
of either packet's maximum amplitude were excluded from each comparison so
quantization near nulls could not create arbitrary rank changes. The formula
is exactly invariant to positive per-packet gain and almost orthogonal to
`chan_shape_spread`, but it largely repeated the production lag-ratio signal
and regressed both OOF F1 and tail recall. The replay gate was therefore not
run.

`chan_ratio_gap` used fixed bin pairs separated by four subcarriers. Each
packet-pair distance was the median of
`abs(delta log-ratio) / (1 + abs(delta log-ratio))`; a ratio pair was excluded
unless both bins cleared `2%` of their packet maximum in both packets. The
feature was the mean lag-10 distance minus the mean adjacent distance. This
bounded, gain-invariant formulation avoided supervised pair selection and
near-null denominators, but it repeated much of `chan_rank_gap` and the
production lag-ratio signal. Its tail false positives and one C3 motion lineage
regressed, so the replay gate was not run.

`chan_freq_coh_curve_std` computed complex frequency coherence at offsets 2
and 12, formed `(coherence_2 - coherence_12) / (coherence_2 + coherence_12)`
per packet, and reported its standard deviation over the live window.
Normalized complex coherence cancels packet gain, while its magnitude cancels
the common phase and fixed-offset phase ramp. The multi-offset contrast was
more label-correlated than the offset-4 CV, but did not improve the isolated
tail metrics, and their combination failed paired replay non-regression.

`phase_closure_var_std` formed
`angle(H[k-1] H[k+1] conj(H[k])^2)` over contiguous subcarrier triplets,
excluded triplets containing a bin below `2%` of the packet maximum, computed
their circular variance per packet, and reported its temporal standard
deviation. The second phase difference cancels common phase and a linear phase
ramp exactly. The feature was nearly orthogonal to both production and the
earlier phase residual, but noisy C5 empty data produced an `11.8%`
worst-lineage FP rate, so grouped CV rejected it before replay evaluation.

Trusted gain correction cannot be evaluated on the current capture contract.
ESP-IDF exposes `manu_scale` and `shift` on legacy targets and
`val_scale_cfg` on HE-capable targets as acquisition configuration, but
`wifi_csi_info_t` contains no per-packet scale selected by automatic scaling.
ESPectre configures automatic/default scaling, and its saved corpus records
RSSI and noise-floor metadata rather than a CSI scale value. A correction
would therefore reconstruct an unobserved nuisance from the same amplitudes it
is intended to repair. Keep production features exactly scale invariant unless
a future SDK and protocol expose a measured packet scale consistently across
chips.

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
| Deferred | Narrowband micro-motion energy | Periodic breathing or occupancy micro-motion | Power fraction or spectral concentration, never absolute power | First collect same-link, same-session Presence-versus-Empty pairs with windows long enough to resolve sub-Hz motion |
| Deferred | Delay/Doppler or CIR dynamics | Path-delay and velocity structure | Normalize maps or use ratios within a map | Revisit only if bandwidth, packet cadence, and exposed CSI support enough resolution |

The current corpus is not an admissible Presence-versus-Empty benchmark. It
contains 9 active empty recordings (58 minutes) and 31 active static-presence
recordings (93 minutes), with eight environment, chip, and device combinations
shared across labels. However, empty and static-presence recordings have no
pair lineage and come from separate capture sessions. A sub-Hz classifier
could therefore learn session drift rather than micro-motion. The motion
trainer's one-second window also cannot resolve the intended respiratory band.
Collect paired, longer-window evidence before implementing or scoring this
feature.

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
