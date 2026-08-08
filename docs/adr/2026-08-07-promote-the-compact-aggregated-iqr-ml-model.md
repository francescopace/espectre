# ADR: promote the compact aggregated-IQR ML model

- Status: Accepted
- Date: 2026-08-07
- Recorded: 2026-08-08 (retrospective)
- Supersedes: 2026-07-28-drop-the-absolute-l1-features.md

## Context

The five-feature invariant model removed the structural weak-link failure caused by absolute L1 inputs. A later physical-axis sweep added phaseless channel-shape and frequency-coherence measurements, removed the phase tracker, and established a ten-feature `24 -> 12` compact baseline. The remaining questions were whether localized adjacent-subcarrier averaging could improve the weakest turbulence statistic and whether overlapping coherence summaries justified their runtime and model cost.

The shared-band aggregation ADR had already rejected averaging for Classic: suppressing receiver noise raises the quiet autocorrelation floor and reduces its separation. That decision did not settle an explicit ML-only feature whose statistic benefits from the same noise reduction while every other input keeps its existing path.

## Decision

Promote a seven-input phaseless ML model with topology `7 -> 24 -> 12 -> 1` and 505 parameters. Train it with `--augment base,drift,burst-loss`, false-positive weight `1.75`, and seed `2125739007`.

The ordered input set is:

1. `turb_iqr_over_mean_aggr`;
2. `turb_autocorr`;
3. `turb_zcr`;
4. `l1_delta_autocorr`;
5. `l1_delta_lag_ratio`;
6. `chan_shape_spread`; and
7. `chan_freq_coh_curve_std`.

Only `turb_iqr_over_mean_aggr` reads a dedicated `W=5` adjacent-magnitude turbulence buffer. The other turbulence inputs, both L1 inputs, and Classic remain on the normal twelve-tone path. The two channel-shape inputs read the 56-bin live profile. Every input is a ratio, correlation, crossing rate, normalized-profile statistic, or bounded coherence contrast.

Remove `chan_freq_coh_cv`, `chan_coh_gap`, and `chan_coh_subband_gap_median` from the production model. Their joint removal improved the tail metrics even when individual marginal importance did not predict the interaction.

## Validation

### Localized aggregated IQR

The first controlled ML comparison used the then-current ten-feature baseline, seed `636455708`, and the production augmentation recipe:

| Variant | Blocked OOF F1 | Worst-session recall / FP | Paired gate | Cross-chip macro F1 | Cross-environment macro F1 |
| --- | ---: | --- | --- | ---: | ---: |
| Production reference | `97.3%` | `77.5% / 9.5%` | `14/14`, `98.00%` worst recall, `0.00%` max FP | `98.1%` | `96.5%` |
| Aggregated-IQR replacement | `98.1%` | `84.3% / 4.6%` | `14/14`, `98.29%` worst recall, `0.71%` max FP | `98.6%` | `96.7%` |

The candidate improved blocked OOF F1, the weakest session, cross-chip generalization, cross-environment FP, and both macro F1 scores. A five-trial search selected seed `1049082371`; it passed the reserved holdout at `96.57%` worst recall, `0.57%` maximum paired FP, `0.30%` maximum quiet FP, and no alarms. Python and C++ matched on the published feature id and inference gates.

Aggregated MAD and P95 were rejected as additional inputs or ZCR replacements. They were highly redundant with aggregated IQR (`abs(r)=0.9917` and `0.9257`) and worsened blocked OOF and C3 tail metrics.

### Heterogeneous and compact feature frontier

The earlier systematic sweep used the same 553,801 windows, 23 lineage groups, and seed `1876849819` to compare feature families before promotion:

| Family | Inputs | OOF F1 | Worst-five recall | Worst-five FP |
| --- | ---: | ---: | ---: | ---: |
| Five-feature baseline | 5 | `97.77%` | `95.23%` | `4.96%` |
| Lean heterogeneous | 8 | `98.45%` | `94.37%` | `2.83%` |
| Compact orthogonal | 10 | `98.20%` | `92.57%` | `2.18%` |
| Frequency dynamics | 10 | `98.39%` | `94.83%` | `2.78%` |
| Coherence distribution | 12 | `95.81%` | `94.57%` | `12.96%` |
| Phase and frequency | 12 | `98.07%` | `95.95%` | `5.45%` |
| All physical axes | 12 | `97.68%` | `93.48%` | `6.19%` |
| Alternate lag core | 10 | `97.00%` | `95.02%` | `6.97%` |
| Alternate turbulence core | 10 | `97.98%` | `94.15%` | `4.90%` |
| Broad physics | 18 | `97.21%` | `91.46%` | `6.30%` |
| Wide non-redundant | 20 | `97.14%` | `93.71%` | `7.98%` |

Adding every available input was harmful. The useful interaction was a selected mixture of time, frequency, channel-shape, and coherence measurements.

The refreshed compact frontier then established that removing both phase inputs improved paired replay and cross-environment behavior while removing the whole phase tracker. Reducing the no-phase layout from `32 -> 16` to `24 -> 12` cut parameters by `35.7%` and improved the worst held-out FP rates. The smaller `20 -> 10` layout gave back too much on bedroom and C3 tails.

### Final joint ablation and seed search

At fixed seed, removing `chan_freq_coh_cv` and `chan_coh_subband_gap_median` jointly improved blocked OOF F1 `97.40% -> 98.31%` and worst-session FP `12.85% -> 3.35%` without losing the paired gate. Removing `chan_coh_gap` as a third member produced the strongest seven-feature schema: `98.75%` blocked OOF F1, `91.01%` worst-session recall, `2.23%` worst-session FP, and `7/7` selection replays with `99.71%` worst recall and `0.00%` maximum FP.

All ten seeds in the final in-memory search were eligible. Blocked OOF F1 stayed within `98.50-98.69%`, worst-session recall within `88.76-92.13%`, and worst-session FP within `2.23-4.47%`. Seed `2125739007` won the robust ranking.

The final exported artifact scored:

| Gate | Result |
| --- | --- |
| Blocked OOF | F1 `98.7%` |
| Paired replay | `14/14`, `98.57%` worst recall, `0.43%` maximum FP, and no alarms |
| Quiet replay | `0.30%` maximum FP and no alarms |
| Leave one chip out | `98.5%` macro recall, `0.3%` macro FP, and `98.9%` macro F1; C3 recall `97.3%` |
| Leave one environment out | `98.6%` macro recall, `0.4%` macro FP, and `98.9%` macro F1; bedroom FP `1.1%` |

The post-promotion grouped OOF diagnostic used 533,400 clean windows and 23 lineage groups. Aggregated IQR contributed `39.1%` of mean absolute SHAP, lag ratio `18.6%`, autocorrelation `17.1%`, ZCR `11.8%`, frequency-coherence curve `5.0%`, L1 autocorrelation `4.2%`, and shape spread `4.2%`. Shape spread stayed nearly orthogonal to the other inputs despite low and chip-dependent marginal label correlation, supporting its role as conditional tail protection rather than a standalone axis.

## Alternatives Considered

### Aggregate the shared twelve-tone path

Rejected. Classic autocorrelation loses monotonically as adjacent-bin noise is removed. Localizing aggregation to one explicit ML feature preserves that decision and avoids silently changing the other inputs.

### Keep the ten-feature no-phase model

Rejected. Three coherence summaries overlapped, and their joint removal improved model size, blocked OOF F1, grouped tails, paired replay, and cross-group generalization.

### Remove one coherence summary at a time

Rejected as the final method. Marginal SHAP and isolated ablation did not expose the full overlap; `chan_coh_gap` looked harmful to remove alone but beneficial after the other two summaries were gone.

### Remove shape or the entire coherence family

Rejected. Both reduced configurations passed the aggregate paired count but introduced per-recording alarm or recall regressions. Shape spread and the frequency-coherence curve remain the two complementary full-band axes.

### Keep phase inputs

Rejected. Dropping both removed an entire tracker and improved the paired and bedroom tails even though blocked OOF F1 moved only slightly.

### Use the smaller `20 -> 10` layout

Rejected. Its parameter saving did not compensate for the bedroom and C3 false-positive regressions.

## Consequences

- Production ML uses seven scale-invariant inputs and 505 parameters.
- The phase tracker and three overlapping coherence summaries are absent from the production inference path.
- Python and C++ keep a dedicated aggregated-IQR buffer only when the exported feature ids request it.
- The shared twelve-tone path and Classic detector remain unaggregated.
- C3 leave-one-chip-out recall remains an explicit trade-off even though deployment replays clear the promotion gates.
- Future feature work must compare against this exact seven-feature baseline and keep localized input transforms explicit in feature names.

## Related

- [FEATURES.md](../FEATURES.md)
- [2026-07-28-drop-the-absolute-l1-features.md](2026-07-28-drop-the-absolute-l1-features.md)
- [2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md](2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md)
- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [ML_TRAINING.md](../ML_TRAINING.md)
- [performance/README.md](../performance/README.md)
