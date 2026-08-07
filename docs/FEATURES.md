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

- **Production**: exported in the current seven-input model and implemented in
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

Host-side candidates live in `tools/lib/candidate_features.py` and remain
research-only until promotion. Evaluate them with `--no-export` or
`--evaluate-gates`; host-side seed searches also run their deployment gates in
memory and leave the runtime artifacts unchanged.

## Current Production Set

The current production baseline is the compact phaseless seven-feature set
selected by the adjacent-subcarrier follow-up, joint coherence ablation, and
ten-trial seed search. It keeps five turbulence/L1 invariants and the two
channel-shape signals that survived the promotion gates. Only
`turb_iqr_over_mean_aggr` reads a `W=5` adjacent-magnitude average; the other
six inputs and the Classic detector keep their existing amplitude paths. The
exported topology is `7 -> 24 -> 12 -> 1`, trained with
`--augment base,drift,burst-loss`, `--fp-weight 1.75`, and seed `2125739007`.

| Feature | Physical quantity | Definition | Scale invariant | Known evidence |
| --- | --- | --- | --- | --- |
| `turb_iqr_over_mean_aggr` | Robust relative spread of aggregated turbulence | `(Q75(x) - Q25(x)) / abs(mean(x))`, where `x` is computed after averaging adjacent magnitudes with `W=5` | Yes, ratio | Retained in the seven-feature export at seed `2125739007`: blocked OOF F1 `98.7%`; paired `14/14`, `98.57%` worst recall, `0.43%` max FP; quiet max FP `0.30%`; no effective alarms. Pre-ablation grouped OOF SHAP: `0.213933` (`38.6%`) |
| `turb_autocorr` | Temporal persistence of turbulence | lag-1 autocorrelation `C(1) / C(0)` | Yes, correlation | Historical Core-6 label correlation `0.7834`; SHAP `0.101985` (`17.6%`). Weak-pair medians retained the correct ordering: idle `0.0115`, motion `0.5817` |
| `turb_zcr` | Temporal coherence versus noise-like crossings | crossing rate around the window median | Yes, crossing rate | In the Coherence-6 swap, helped reduce reserved max-FP median from `19.42%` to `2.66%` together with `l1_delta_autocorr` |
| `l1_delta_autocorr` | Persistence of normalized channel-profile displacement | lag-1 autocorrelation of the L1-delta series | Yes, correlation | Same Coherence-6 joint result; no isolated ablation metric retained |
| `l1_delta_lag_ratio` | Growth of channel-profile displacement with lag | mean lag-10 displacement / mean adjacent displacement | Yes, ratio | Adding it to Coherence-6 reduced reserved max FP `6.43% -> 4.43%`, raised worst recall `97.99% -> 99.14%`, and reduced effective alarms `8 -> 3` across ten reserved replays |
| `chan_shape_spread` | Frequency participation of lagged channel-shape motion energy | participation ratio of accumulated lagged normalized-shape energy | Yes, ratio | Promoted in the refreshed compact sweep: paired `14/14`, cross-chip worst FP `1.9%`, and better leave-one-environment bedroom FP than the wider no-phase layout |
| `chan_freq_coh_curve_std` | Temporal variability of short-versus-long frequency coherence contrast | std of `(coh_offset2 - coh_offset12) / (coh_offset2 + coh_offset12)` | Yes, bounded contrast | Retained in the promoted no-phase family; improved the compact model's cross-environment bedroom tail over the wider no-phase layout |

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
| `chan_coh_gap` | Adjacent-minus-lagged complex coherence | Nearly equivalent to `chan_coh_lag_ratio` | Previously shipped, then removed jointly with the two overlapping summaries after the seven-feature ablation and seed search | Historical |
| `chan_coh_gap_low_frac` | Fraction of small coherence gaps | Not retained | Did not beat `chan_coh_gap` | Rejected |
| `chan_coh_gap_q20` | Lower-tail coherence gap | Not retained | Did not beat `chan_coh_gap` | Rejected |
| `chan_coh_subband_median_gap` | Median robust coherence over four frequency bands | Not retained | OOF F1 `97.3%`; worst-lineage recall `80.9%`; worst-lineage FP `10.1%`; mean-of-five-worst FP `5.9%` | Rejected |
| `chan_coh_subband_gap_median` | Median of per-band coherence gaps | Not retained | Previously shipped; its independent removal improved blocked OOF F1 and the weakest session FP, and it was removed in the promoted joint ablation | Historical |
| `turb_band_power_ratio` | Low-frequency share of non-DC turbulence power | `r=0.923` with `turb_autocorr`; `R2=0.863` | OOF F1 `98.1%`, but two S3 static-presence replays regressed; swapping out autocorrelation scored `97.2%` | Rejected |
| `phase_resid_lag_ratio` | CFO/STO-sanitized phase-shape dynamics | max `r=0.434`; `R2=0.207` | OOF F1 `97.4%`; worst C5 FP `18.5%` | Research |
| `phase_closure_var_std` | Temporal variability of local phase curvature across adjacent subcarrier triplets | max `r=0.1367` with `turb_mad_over_mean`; `R2=0.0906`; `r=0.0521` with `phase_resid_lag_ratio`; label correlation `0.0455` | OOF F1 `97.3%`; worst-lineage recall `92.1%`; worst-lineage FP `11.8%`; mean-of-five-worst recall `95.3%`, FP `6.4%` | Research |
| Trusted profile-scale correction | Recover magnitude-domain information after undoing packet gain | Unavailable: the ESP-IDF packet callback exposes no per-packet scale value | All 95 current NPZ recordings include RSSI metadata, but none records scale; current firmware only selects static automatic/manual scaling configuration | Deferred |
| `chan_shape_lag_ratio` | Temporal displacement of the L2-normalized amplitude profile | max `r=0.9831` with `l1_delta_lag_ratio`; `R2=0.9686`; label correlation `0.754` | Stopped before CV because the production set already reconstructs it | Rejected |
| `chan_shape_spread` | Frequency participation of lagged channel-shape motion energy | max `r=0.1094`; `R2=0.0257`; label correlation `0.0314` | OOF F1 `98.4%`; fold recall `98.0%`; precision `98.7%`; worst-lineage recall `86.5%`; worst-lineage FP `3.4%`; mean-of-five-worst recall `93.9%`, FP `2.4%` | Research |
| `chan_freq_coh_cv` | Temporal variability of within-packet frequency coherence at a four-bin offset | max `r=0.2468`; `R2=0.0838`; label correlation `0.2181` | Previously shipped; its independent removal led the first ablation screen, and it was removed in the promoted joint ablation | Historical |
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
buffer. The first screen used it as an input-path transform, but the later ML
follow-up kept only explicit host-side feature variants so the rest of the
model could stay on the production path. In that formulation, an `_aggr`
candidate is the named statistic computed on a `W=5` adjacent-bin average,
while the other nine inputs of that ten-feature baseline are unchanged.

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

| Candidate | Worst pair, baseline | Worst pair, W=3 | Worst pair, W=5 | Same limiting pair |
| --- | ---: | ---: | ---: | --- |
| `turb_iqr_over_mean` | 0.6317 | 0.8308 | 0.8838 | yes |
| `turb_mad_over_mean` (production) | 0.6190 | 0.8155 | 0.8683 | yes |
| `turb_p95_over_mean` | 0.6219 | 0.7865 | 0.8474 | yes |
| `turb_min_over_mean` | 0.5237 | 0.6864 | 0.7696 | yes |
| `turb_p05_over_mean` | 0.5732 | 0.6673 | 0.6819 | yes |
| `turb_cv` | 0.5103 | 0.6451 | 0.7071 | yes |
| `turb_range_over_mean` | 0.5855 | 0.5432 | 0.5933 | yes |
| `turb_max_over_mean` | 0.5819 | 0.5210 | 0.5614 | yes |

Robust dispersion statistics gain on the limiting pair, and the same ordering
persists at `W=5`, while the two max-based extremes still fail to justify a
promotion path. "Extreme-order statistics are the most noise-sensitive and
therefore gain most" is not what the corpus shows. `turb_iqr_over_mean`, a
Relative-8 member dropped at the Core-6 transition, edges past the current
production feature at both widths. `corr_amp_d1` was the strongest prior
candidate, having been rejected because at lag 1 it mostly measured receiver
noise, and it did not benefit: median separation falls `0.8920` to `0.8793`.

The comparable ML follow-up used the production augmentation composition
`base,drift,burst-loss` throughout. It first retrained every variant with the
exported baseline seed `636455708`, then ran a five-trial seed search only for
the strongest fixed-seed variant. The earlier `turb_mad_over_mean_aggr` search
used a different training reference, so its numbers are retained only as
historical screening evidence and are not used for promotion.

| Variant, seed `636455708` | Blocked OOF F1 | Worst session recall / FP | Selection + holdout paired gate | Cross-chip macro recall / FP / F1 | Cross-environment macro recall / FP / F1 | Verdict |
| --- | ---: | --- | --- | --- | --- | --- |
| Production baseline | `97.3%` | `77.5%` / `9.5%` | `14/14`, `98.00%` worst recall, `0.00%` max FP; no alarms | `98.2%` / `0.9%` / `98.1%` | `98.4%` / `2.6%` / `96.5%` | Reference |
| `turb_iqr_over_mean_aggr` | `98.1%` | `84.3%` / `4.6%` | `14/14`, `98.29%` worst recall, `0.71%` max FP; no alarms | `98.4%` / `0.5%` / `98.6%` | `98.3%` / `2.3%` / `96.7%` | Seed-search lead |

The IQR replacement improves blocked OOF F1, the weakest session, cross-chip
generalization, cross-environment FP, and both macro F1 scores. Its only broad
regression is `0.1` percentage point of cross-environment macro recall. This is
the strongest evidence among the aggregated robust-dispersion family, so the
campaign did not spend another full seed search on the weaker MAD control.

The five-trial IQR seed search selected seed `1049082371`. On the selection
split it scored `97.7%` blocked OOF F1, `83.1%` / `9.5%` worst-session
recall / FP, `99.71%` worst paired recall, `0.29%` max paired FP, `0.00%` max
quiet FP, and no effective alarms. On the reserved holdout it passed all seven
paired replays with `96.57%` worst recall, `0.57%` max FP, and no alarms; the
three quiet replays reached `0.30%` max FP and no alarms. The production
baseline on that holdout reached `98.29%` worst recall and `0.00%` max FP, but
the candidate's per-recording changes remain inside the measured non-regression
margins. The invariant-feature gain-stress gate was unchanged from `0.5x` to
`2.0x` amplitude.

`turb_iqr_over_mean_aggr` therefore cleared the research selection and reserved
holdout protocol and replaced `turb_mad_over_mean` in production. The final
export uses seed `1049082371` and the full `base,drift,burst-loss` augmentation
recipe. Its exported-artifact replay passed all 14 paired recordings with
`96.57%` worst recall, `0.57%` max FP, and no alarms; the five quiet replays
reached `0.30%` max FP with no alarms. Leave-one-chip-out scored `98.5%` macro
recall, `0.8%` macro FP, and `98.3%` macro F1. Leave-one-environment-out scored
`98.1%` macro recall, `2.1%` macro FP, and `96.8%` macro F1; bedroom remains the
limiting held-out environment at `6.2%` FP. Python and `C++` use published
feature id `45` and matched on the production replay and inference gates.
Aggregation stays local to this ML feature, so the normal turbulence path and
Classic detector remain unchanged. The retired MAD input remains readable only
for migration-time comparison against the immediately preceding artifact.

The first post-promotion screen kept seed `1049082371`, the full
`base,drift,burst-loss` recipe, and the new aggregated-IQR baseline, then asked
whether another robust statistic should replace `turb_zcr`. Both variants were
export-free, and neither justified replay gates:

| Feature replacing `turb_zcr` | Blocked OOF F1 | Worst session recall / FP | Worst chip recall / FP | Redundancy against production | Verdict |
| --- | ---: | --- | --- | --- | --- |
| None, production reference | `97.7%` | `83.1%` / `9.5%` | C3 `96.2%` / `1.9%` | N/A | Reference |
| `turb_mad_over_mean_aggr` | `97.2%` | `77.5%` / `10.1%` | C3 `95.0%` / `2.1%` | closest to aggregated IQR: `|r|=0.9917`, `R2=0.9837` | Rejected |
| `turb_p95_over_mean_aggr` | `97.1%` | `74.2%` / `12.3%` | C3 `94.1%` / `3.1%` | closest to aggregated IQR: `|r|=0.9257`, `R2=0.8710` | Rejected |

MAD carries virtually no information beyond the promoted IQR, while P95 is
still highly redundant and loses more of the C3 tail. Their strong individual
label correlations (`0.8151` and `0.7925`) do not compensate for replacing the
more complementary `turb_zcr` signal (`|r(label)|=0.8417`). Keep both robust
variants retired, keep `turb_autocorr` out of this aggregation campaign, and do
not spend seed-search or cross-generalization runs on either swap.

The next fixed-seed ablation screen used the same seed and full augmentation
recipe. It trained the ten-feature baseline once, removed each low-SHAP channel
feature independently, and evaluated grouped CV plus the seven selection
replays. No runtime artifact was exported.

| Independent removal | Blocked OOF F1 | Worst session recall / FP | Selection paired gate | Verdict |
| --- | ---: | --- | --- | --- |
| None, retrained reference | `97.40%` | `78.65%` / `12.85%` | `7/7`, `99.71%` worst recall, `0.29%` max FP | Reference |
| `chan_coh_subband_gap_median` | `98.07%` | `83.15%` / `5.03%` | unchanged | Finalist |
| `chan_freq_coh_cv` | `98.13%` | `85.39%` / `4.57%` | unchanged | Lead |
| `chan_coh_gap` | `97.31%` | `80.90%` / `14.53%` | `7/7`, `99.71%` worst recall, `0.14%` max FP | Rejected: CV and tail FP regress |

The two successful removals simplify the model and improve the weakest grouped
tail without spending any selection-gate margin. Their joint removal was
stronger than either isolated ablation at the same seed:

| Joint removal | Blocked OOF F1 | Fold recall / FP | Worst session recall / FP | Full paired / quiet gates |
| --- | ---: | --- | --- | --- |
| `chan_freq_coh_cv` + `chan_coh_subband_gap_median` | `98.31%` | `97.62%` / `0.39%` | `82.02%` / `3.35%` | paired `14/14`, `98.29%` worst recall, `0.29%` max FP; quiet `0.47%` max FP; no alarms |

Against the retrained ten-feature reference, the eight-feature candidate gains
`0.91` points of blocked OOF F1, reduces fold FP by `0.85` points, and reduces
the worst-session FP tail by `9.50` points, while fold recall gives back `0.13`
points. Its paired gate also improves over the preceding exported artifact
(`96.57%` worst recall and `0.57%` max FP), while quiet max FP rises from
`0.30%` to `0.47%` without producing an effective alarm.

Leave-one-environment-out favors the candidate: macro recall / FP / F1 become
`98.6%` / `1.0%` / `98.2%`, versus `98.1%` / `2.1%` / `96.8%` for the
production baseline, and worst held-out recall improves from `96.8%` to
`97.3%`. Leave-one-chip-out is the remaining trade-off: macro FP improves from
`0.8%` to `0.3%`, macro recall is effectively flat at `98.4%` versus `98.5%`,
but worst-chip C3 recall falls from `98.3%` to `96.4%`.

A second-level screen removed one more feature from that eight-feature
candidate. Removing `l1_delta_autocorr` or `chan_shape_spread` retained part of
the gain but regressed against the eight-feature result. Removing
`chan_coh_gap`, despite its poor isolated ablation, produced the strongest
fixed-seed model:

| Seven-feature removal set | Blocked OOF F1 | Fold recall / FP | Worst session recall / FP | Selection paired gate | Verdict |
| --- | ---: | --- | --- | --- | --- |
| Prior two + `chan_coh_gap` | `98.75%` | `98.29%` / `0.32%` | `91.01%` / `2.23%` | `7/7`, `99.71%` worst recall, `0.00%` max FP | Lead |
| Prior two + `l1_delta_autocorr` | `98.19%` | `97.81%` / `0.58%` | `83.15%` / `5.14%` | `7/7`, `99.71%` worst recall, `0.14%` max FP | Below eight-feature candidate |
| Prior two + `chan_shape_spread` | `97.95%` | `97.87%` / `0.81%` | `88.37%` / `6.15%` | `7/7`, `99.71%` worst recall, `0.00%` max FP | Below eight-feature candidate |

The lead keeps `chan_shape_spread` and `chan_freq_coh_curve_std`, but removes
the three other channel-coherence summaries: `chan_freq_coh_cv`,
`chan_coh_gap`, and `chan_coh_subband_gap_median`. On the full deployment gate
it passes `14/14` paired replays with `97.43%` worst recall, `0.14%` max FP,
and no alarms; quiet max FP is `0.30%`, also with no alarms. Leave-one-chip-out
scores `98.5%` macro recall, `0.2%` macro FP, and `99.0%` macro F1, with C3
still the worst held-out chip at `97.1%` recall. Leave-one-environment-out
scores `98.6%` macro recall, `0.5%` macro FP, and `98.8%` macro F1, with
`97.3%` worst held-out recall.

This interaction means marginal SHAP rank alone was insufficient: the three
coherence summaries overlap in a way that is only beneficial to remove jointly.

A ten-trial in-memory seed search then compared the seven-feature schema with
the exported ten-feature baseline under the full `base,drift,burst-loss`
recipe. All ten candidates were robustly eligible. Across seeds, blocked OOF F1
stayed within `98.50%` to `98.69%`, worst-session recall within `88.76%` to
`92.13%`, and worst-session FP within `2.23%` to `4.47%`; the retrained baseline
scored `97.40%`, `78.65%`, and `12.85%`, respectively.

Seed `2125739007` won the robust ranking with `98.66%` blocked OOF F1,
`91.01%` worst-session recall, and `2.86%` worst-session FP. Selection passed
`7/7` paired replays with `99.71%` worst recall and `0.14%` max FP, while quiet
max FP remained `0.00%`; the reserved holdout passed `7/7` at `98.57%` worst
recall and `0.43%` max FP, and quiet max FP was `0.30%`. No replay produced an
effective alarm, and no per-recording non-regression check failed.

At the selected seed, leave-one-chip-out scores `98.5%` macro recall, `0.3%`
macro FP, and `98.9%` macro F1. C3 remains the explicit trade-off at `97.3%`
recall versus the production baseline's `98.3%`, but with `0.0%` FP.
Leave-one-environment-out scores `98.6%` macro recall, `0.4%` macro FP, and
`98.9%` macro F1; bedroom FP is `1.1%`. The multi-seed evidence therefore
confirmed the seven-feature schema for promotion. The final export uses seed
`2125739007`, preserves the full `base,drift,burst-loss` recipe, and has a
`7 -> 24 -> 12 -> 1` topology with 505 parameters. Its training run scored
`98.7%` blocked OOF F1, passed all 14 paired replays at `98.57%` worst recall
and `0.43%` max FP, and kept quiet max FP at `0.30%`, with no effective alarms.
The regenerated out-of-sample performance report records `100%` minimum recall
on the reserved C3, C5, C6, and S3 replays, at most `0.4%` FP, and no effective
alarms. The C3 leave-one-chip-out recall trade-off remains explicit even though
the deployment replays clear the promotion gates.

A later Classic-oriented follow-up asked a narrower question: if aggregation is
accepted as a research-only input transform, can the gaining robust-dispersion
family support a different two-feature non-ML detector? The replay used the
current grouped, de-overlapped fit, `train`-role empty hard negatives, and the
production startup and settled-threshold policy over the 98 real paired and
empty replays. Three verdicts were stable:

| Hypothesis | Best retained row | Key replay result | Verdict |
| --- | --- | --- | --- |
| Replace `turb_autocorr` and keep the coherence term | `turb_p95_over_mean + chan_freq_coh_curve_std` at `W=3` | discovery worst recall `43.27%` | Rejected |
| Use only robust-dispersion features | `turb_p95_over_mean + turb_p05_over_mean` at `W=3` | discovery worst recall `34.10%` | Rejected |
| Keep `turb_autocorr` and replace the coherence term | `turb_autocorr + turb_mad_over_mean` at `W=5`; runner-up `turb_autocorr + turb_iqr_over_mean` at `W=5` | discovery worst recall `99.71%`, weighted FP `3.07%`, and idle max `5.38%`, but `exclude` idle max still `37.04%` | Research |

The same replay established that the detector-level width preference is not the
same as the first feature screen width: `W=3` was enough to expose the family,
but the strongest default-policy replay landed at `W=5` for the
`turb_autocorr + turb_mad_over_mean` and `turb_autocorr + turb_iqr_over_mean`
pairs. A calibration sweep could make the `W=3` `mad` and `iqr` pairs look very
clean on discovery only by driving `startup_strength` to `0.0`, which then
raised holdout idle maxima to `27.94%` and `27.73%`. Treat the whole
Classic-with-aggregation line as research-only. It is the only live
non-ML follow-up from this work, but it is not a drop-in replacement for the
committed detector, and it does not weaken the ADR verdict for the shared path.

The August 8 follow-up re-ran the admissible one- and two-feature surface with
the then-current ten runtime features, then applied the ML packet augmentation
components as inference-only stress. Coefficients and the operating point were
fit once on clean `train` data, including train-role empty hard negatives; the
same fitted detector was then replayed on `base`, `drift`, `burst-loss`, and
`base+drift+burst-loss`. Feature-space jitter was excluded because it is a
training transform rather than a packet stream the runtime can observe. All
candidates saw the same deterministic augmented packets. The corpus contained
40 real paired sessions plus 18 empty recordings, or 98 source files. Ranking
used the worst discovery score across clean and augmented streams; `holdout`
and `exclude` remained diagnostics.

The table reports the refitted current pair as a surrogate, not the exported
runtime baseline. Each metric triplet is worst paired recall / weighted paired
FP / maximum empty FP on discovery.

| Candidate | Clean metrics | Limiting augmented metrics | Worst clean/augmented score | Verdict |
| --- | --- | --- | ---: | --- |
| `turb_autocorr + chan_freq_coh_curve_std` | `94.83% / 3.30% / 4.08%` | `base`: `92.22% / 3.56% / 5.75%` | `9.47` | Robust reference |
| `turb_autocorr` | `91.95% / 3.13% / 7.34%` | `base`: `89.05% / 3.12% / 7.40%` | `23.68` | Rejected; historical-holdout worst recall was `79.42%` |
| `turb_autocorr + l1_delta_lag_ratio` | `92.82% / 2.96% / 9.51%` | `base`: `93.66% / 3.47% / 11.23%` | `25.89` | Rejected; quiet tail and holdout recall regress |
| `turb_autocorr + turb_zcr` | `90.23% / 2.06% / 13.04%` | `burst-loss`: `90.80% / 2.11% / 13.22%` | `42.48` | Rejected; shared-buffer resource saving does not recover the quiet and recall tails |
| `turb_iqr_over_mean_aggr + l1_delta_lag_ratio` | `96.54% / 2.29% / 12.47%` | combined: `97.08% / 4.75% / 15.66%` | `42.13` | Rejected; strongest recall-oriented aggregate pair, but not quiet-room safe |
| `turb_iqr_over_mean_aggr + turb_autocorr` | `97.99% / 3.37% / 12.47%` | combined: `98.77% / 4.89% / 18.34%` | `53.16` | Rejected; high recall hides worse empty tails |
| `turb_iqr_over_mean_aggr + turb_zcr` | `94.83% / 2.25% / 11.96%` | `base`: `96.83% / 3.66% / 21.29%` | `62.48` | Rejected; packet noise exposes the largest discovery empty regression |

The aggregated-IQR plus lag-ratio lead and the current pair fail on different
quiet recordings. The candidate's discovery maximum comes from
`empty_s3_64sc_dev000010b41de8ec00_20260712_203314_805494_0001.npz`; the
current-pair surrogate is limited by C6 recordings on discovery, historical
holdout, and `exclude`. The candidate reduces historical-holdout maximum empty
FP from `20.78%` to `2.01%`, but creates the new `12.47%` admitted-discovery S3
tail. This is a chip/session trade-off, not a global robustness gain. Augmented
`exclude` tails remain severe for multiple formulations and are diagnostic,
not a basis for post-hoc selection on already observed recordings.

No one-feature detector is competitive, and the only two-feature formulation
that shares the existing turbulence buffer fails both clean and augmented
tails. Keep `turb_autocorr + chan_freq_coh_curve_std` as the exported Classic
pair. The aggregated IQR evidence remains useful for ML, but it does not
transfer to a calibrated linear boundary without a material quiet-room cost.
No Python or C++ detector change is justified by this campaign.
A follow-up calibration sweep showed that the initial operating point overstated the aggregate pair's disadvantage. Threshold selection was repeated separately for each pair at weighted training FP targets of `3%`, `2%`, and `1%`, with startup strengths from `0.5` to `1.0` and settled margins from `2.8` to `6.0` logits. The best recall/quiet compromise for `turb_iqr_over_mean_aggr + l1_delta_lag_ratio` used a `2%` FP target, `startup_strength=0.75`, and a `4.0`-logit settled margin:

| Calibration | Clean discovery worst recall / weighted FP / maximum empty FP | Packet-stress range | Worst clean/augmented score |
| --- | --- | --- | ---: |
| Current pair, default `3% / 0.5 / 2.8` | `94.83% / 3.30% / 4.08%` | worst recall `92.22%`; empty max up to `5.75%` | `9.47` |
| Aggregate pair, original `3% / 0.5 / 2.8` | `96.54% / 2.29% / 12.47%` | worst recall `96.15%`; empty max up to `15.66%` | `42.13` |
| Aggregate pair, retuned `2% / 0.75 / 4.0` | `96.26% / 1.43% / 9.24%` | worst recall `94.24%` to `96.55%`; empty max `8.17%` to `9.86%` | `17.74` |
| Aggregate pair, quiet-tail attempt `1% / 0.5 / 4.0` | `71.92% / 0.34% / 5.43%` | not replayed under packet stress after the clean recall failure | `121.56` clean |

Changing the settled margin alone had negligible effect on the limiting S3 empty tail. Reaching the existing sub-`6%` discovery empty region required a threshold that reduced worst-session recall to `71.92%`. The retuned candidate is therefore a genuine recall-oriented Pareto point, not the uniformly weak candidate implied by the first operating point, but it still does not dominate the current pair under the quiet-room gate.

A more structural startup experiment replaced the production `q95` prefix statistic with `q99` and the prefix maximum. At `q99`, `startup_strength=0.75` improved clean worst recall to `97.99%` and maximum empty FP to `8.82%`, but the combined packet-stress empty maximum rose to `10.96%` and the robust score to `23.01`. Raising startup strength to `1.0` reduced the clean empty maximum to `6.88%`, while increasing maximum paired FP to `35.69%` and `exclude` empty FP to `25.39%`. The prefix maximum did not improve the limiting `8.82%` empty tail. This indicates that the S3 disturbance develops after the startup prefix; a higher startup quantile cannot isolate it without moving false positives to paired or later recordings.

Adding `chan_freq_coh_curve_std` to the aggregate pair was also evaluated as a diagnostic three-feature formulation. Threshold-free worst-pair AUC improved from `0.9767` for the current pair to `0.9945`, but calibrated replay did not remove the S3 tail. At the aggregate pair's best `2% / 0.75 / 4.0` operating point, the triplet kept `96.26%` discovery worst recall and `9.24%` maximum empty FP, while reducing weighted paired FP from `1.43%` to `1.33%` and maximum paired FP from `8.74%` to `6.02%`. Packet stress improved the aggregate pair's robust score from `17.74` to `13.30`, but the current pair remained better at `9.47`; the triplet's stressed empty maximum ranged from `7.10%` to `9.32%`. It also raised diagnostic `exclude` empty FP from `13.61%` to `19.84%`. Replacing the lag-ratio term instead of adding coherence preserved two inputs, but `turb_iqr_over_mean_aggr + chan_freq_coh_curve_std` collapsed to `34.10%` discovery worst recall. Coherence therefore improves the aggregate pair's paired-FP balance, but it neither repairs the limiting quiet session nor yields a stronger two-feature replacement.

The current-model correlation and SHAP rerun prompted one additional two-input check. Across all 21 pairs from the seven-feature export, threshold-free screening ranked `turb_iqr_over_mean_aggr + turb_zcr` first, followed by `turb_autocorr + l1_delta_lag_ratio`, `turb_iqr_over_mean_aggr + turb_autocorr`, and `turb_iqr_over_mean_aggr + l1_delta_lag_ratio`. Runtime replay rejected the SHAP-orthogonal `l1_delta_lag_ratio + chan_shape_spread` idea at `48.56%` discovery worst recall, confirming that shape spread is not a stable linear detector term.

The only new clean-replay Pareto point was `turb_iqr_over_mean_aggr + turb_autocorr` at a `2%` FP target, `startup_strength=1.0`, and a `4.0`-logit settled margin. It reached `97.99%` discovery worst recall, `2.54%` weighted paired FP, `6.02%` maximum empty FP, `0.61%` holdout empty FP, and `6.18%` diagnostic `exclude` empty FP. Packet stress invalidated that apparent balance: the `base` recipe reduced worst recall to `67.26%` and produced a robust score of `154.20`. Reducing startup strength to `0.75` recovered clean worst recall to `97.99%`, but `base` still fell to `84.66%` and the combined recipe raised maximum empty FP to `11.86%`, for a robust score of `41.06`. The production `0.5` startup strength avoided a material stress-recall collapse (`92.33%` worst under `base`), but raised `base` and combined empty maxima to `10.32%` and `13.42%`, respectively, for a robust score of `30.50`. The pair would remove the full-band coherence tracker from Classic, but its packet-stress failure rules out that resource saving on the current evidence.

A dedicated current-pair calibration sweep then covered FP targets from `2%` to `4%`, startup strengths from `0.25` to `0.7`, settled margins from `2.0` to `6.0` logits, startup quantiles from `q90` through the prefix maximum, calibration budgets from 500 to 2,000 packets, per-session recall constraints, and settled dwells from 30 to 120 seconds. The strongest clean alternative kept the `3%` FP target, raised startup strength to `0.6`, and used a `4.0`-logit margin. It improved discovery worst recall from `94.83%` to `95.14%`, holdout empty FP from `20.78%` to `17.19%`, and `exclude` empty FP from `33.92%` to `22.78%`, at a small discovery-empty cost from `4.08%` to `4.35%`. Packet stress rejected the retune: its worst score was `10.13`, versus `9.47` for the production-policy surrogate, with slightly lower worst recall under `base` and the combined recipe. Keeping startup strength at `0.5` and raising only the settled margin to `4.0` or `6.0` changed the robust score only to `9.42` or `9.41`, without moving recall or empty tails; the difference is not material enough to override the runtime's existing C3-validated `2.8` margin. `q97.5` worsened the robust score to `10.94`, q90 damaged `exclude`, q99 and the prefix maximum raised discovery empty FP to `6.25%`, and both shorter and longer startup budgets regressed at least one recall or quiet tail. Per-session recall floors up to `95%` selected the same threshold, while `97%` made the `3%` FP gate infeasible. Recovery dwell changes were numerically negligible. No scalar parameter retune is justified; further work should test a different calibration family rather than continue the same grid.

Three causal calibration families were then added to the research replay and evaluated on the same current pair, 98 real paired and empty streams, five-fold grouped OOF operating point, 37 ready startup evaluations, `3%` FP target, train-role empty hard negatives, and `2.8`-logit settled margin. Robust logit calibration interpolated both startup median and IQR toward the training-idle reference; the grid covered strengths `0.1`, `0.25`, `0.5`, `0.75`, and `1.0`, with session-IQR floors at `0.1x`, `0.25x`, and `0.5x` the reference IQR. Its best clean point used `0.5 / 0.5x`: it preserved `94.83%` worst recall and reduced weighted paired FP from `3.30%` to `3.14%`, while increasing maximum paired FP from `15.86%` to `18.45%` and maximum discovery-empty FP from `4.08%` to `4.62%`. Holdout empty FP improved from `20.78%` to `6.74%`, but diagnostic `exclude` empty FP worsened from `33.92%` to `67.41%`. Packet stress rejected the apparent Pareto point: the `base` recipe reduced worst recall to `85.25%`, and its robust score rose to `30.71`, versus `9.47` for the production-policy surrogate.

Per-feature location calibration shifted the two raw-feature baselines separately before applying the fixed linear fusion. Strengths `0.25`, `0.5`, `0.75`, and `1.0` were crossed with feature quantiles `q50`, `q75`, and `q95`. The best clean point used `q75 / 0.75`; compared with the production-policy surrogate, worst recall fell from `94.83%` to `92.82%`, weighted paired FP improved from `3.30%` to `3.04%`, maximum paired FP worsened from `15.86%` to `19.89%`, and maximum discovery-empty FP improved from `4.08%` to `2.72%`. Holdout empty FP improved to `16.77%`, while `exclude` empty FP worsened to `49.06%`. Because this point was already dominated on clean recall and paired tails, it did not advance to packet-stress confirmation.

Guarded upward recovery retained the production `q95 / 0.5` startup rule and allowed later quiet blocks to undo settled-level lowering, but never to exceed the initial calibrated threshold. Predominantly positive blocks clear the evidence, preventing the rule from learning sustained motion. The grid covered one, three, and six `20`-evaluation blocks, block quantiles `q75`, `q90`, and `q95`, and positive-fraction guards `0.25` and `0.5`. The best result used one `q95` block; both guards tied. It left all recall and maximum-empty metrics unchanged, reduced clean weighted paired FP only from `3.30%` to `3.29%`, and reduced the worst packet-stress score only from `9.47` to `9.41`. Under `base`, weighted FP moved from `3.56%` to `3.53%`; under burst loss, it moved from `3.28%` to `3.27%`; the combined result was unchanged at the reported precision. This is not a material winner, especially because a single five-second block is the most aggressive recovery setting and adds runtime state and adaptation risk. Keep the exported calibration unchanged; the guarded rule is useful only as a future hypothesis for purpose-built recordings whose noise rises after settling.

No one-feature detector is competitive, and the only two-feature formulation that shares the existing turbulence buffer fails both clean and augmented tails. Keep `turb_autocorr + chan_freq_coh_curve_std` as the exported Classic pair. The aggregated IQR evidence remains useful for ML, but it does not transfer to a calibrated linear boundary without a material quiet-room cost. No Python or C++ detector change is justified by this campaign.

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
