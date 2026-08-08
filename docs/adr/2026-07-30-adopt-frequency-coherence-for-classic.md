# ADR: adopt frequency coherence for Classic

- Status: Accepted
- Date: 2026-07-30
- Recorded: 2026-08-08 (retrospective)
- Supersedes: 2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md

## Context

Classic had converged on weighted fusion of `turb_autocorr` and `l1_delta_lag_ratio`. The ratio fixed the absolute-L1 noise-floor failure, but the broader phaseless feature campaign exposed frequency-domain measurements that were complementary to turbulence and available through a shared allocation-free tracker.

The candidate search covered one-, two-, and three-feature linear formulations. Its corrected method fitted coefficients only on real `train` recordings, used `train + selection` for discovery, reported `holdout` and `exclude` without ranking on either, fitted grouped folds on de-overlapped rows, scored every dense runtime evaluation tick, limited startup evidence to the production calibration horizon, and compared candidates against exact packet-level replay of the exported runtime. A refitted current feature pair was labeled as a surrogate, never as the runtime baseline.

Earlier screens that ranked on holdout, fitted dense overlapping rows, shortened startup evidence, or treated a refit as the exported baseline are superseded. Their scalar scores are not promotion evidence.

## Decision

Classic uses weighted logistic fusion of:

- `turb_autocorr`, the lag-1 autocorrelation of the normal twelve-tone turbulence stream; and
- `chan_freq_coh_curve_std`, the temporal variability of the short-versus-long complex frequency-coherence contrast over the 56-bin live band.

Both inputs are invariant to common positive packet gain. The second input also cancels common packet phase and a fixed-offset phase ramp through normalized coherence magnitude. Python and C++ share the tracker, feature definitions, coefficients, startup-centering policy, settled-threshold recovery, cadence, and reset behavior.

Do not add a third input or retune the scalar calibration policy on the current corpus. The only live replacement hypothesis is the lag-ratio triplet, and it requires new independent data plus a maintained exact packet-level comparison path before another promotion attempt.

## Validation

The exported runtime baseline retained the following corrected replay evidence:

| Evaluation | Weighted recall | Worst recall | Weighted paired FP | Maximum empty FP |
| --- | ---: | ---: | ---: | ---: |
| Discovery | `97.90%` | `85.59%` | `2.28%` | `1.03%` |
| Historical holdout | — | `99.71%` | `2.18%` | `3.45%` |

The historical holdout was observed during the campaign and is no longer a sealed confirmation set. Its values are diagnostic only.

An expanded corrected replay found one recall-oriented triplet that beat the refitted pair on discovery, but it did not beat the exported runtime on the quiet dimensions that protect Classic:

| Formulation | Discovery worst recall | Weighted paired FP | Maximum empty FP | Pair / idle alarms | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Exported runtime pair | `85.59%` | `2.28%` | `1.03%` | `55 / 1` | Production reference |
| Pair plus `l1_delta_lag_ratio` | `95.68%` | `3.36%` | `6.52%` | `79 / 2` | Recall-oriented research only |

The same direction failed more strongly on `exclude`: the triplet improved weighted recall `90.84% -> 94.51%`, but weighted paired FP regressed `3.82% -> 9.00%`, maximum empty FP regressed `27.69% -> 44.11%`, and idle alarms rose `75 -> 152`. Repeated tuning against those observed C3 and C6 tails would optimize the corpus rather than demonstrate generalization.

A later deterministic packet-stress replay refitted every candidate once on clean train data, then replayed identical `base`, `drift`, `burst-loss`, and combined packet transforms. The current pair remained the robust reference. One-feature candidates were not competitive, robust-dispersion pairs exposed quiet tails, and the clean Pareto point from aggregated IQR plus autocorrelation collapsed to `67.26%` worst recall under `base` stress.

Scalar retuning of the current pair covered false-positive targets, startup strengths and quantiles, settled margins and dwells, calibration budgets, and per-session recall constraints. No point materially dominated the production policy under both clean and packet-stress replay. Robust-logit, per-feature location, and guarded-recovery calibration families also failed to improve the multidimensional balance.

## Alternatives Considered

### Keep the lag-ratio pair

Rejected as the production direction. The lag ratio remains a strong ML input, but frequency coherence supplies a more complementary Classic axis and a better quiet-room balance in the promoted pair.

### Use autocorrelation alone

Rejected. The limiting stressed replay reached `89.05%` recall with `7.40%` maximum empty FP; no one-feature formulation was competitive.

### Add the lag ratio as a third term

Deferred. It improves recall, but the current corpus shows material paired-FP, empty-FP, and alarm regressions. New sealed data must decide whether that is a general trade-off or a corpus-specific tail.

### Use robust-dispersion pairs or triplets

Rejected on the current evidence. Strong threshold-free AUC did not survive runtime startup calibration, empty-room replay, and deterministic packet stress.

### Continue tuning the current calibration family

Rejected. The grids moved error among known recordings without producing a material robust winner. Further work should test a different model or calibration family after new data, not extend the same scalar grid.

## Consequences

- Classic reads one twelve-tone turbulence stream and the shared full-band channel-shape tracker.
- The production pair stays two-feature, vote-free, scale-invariant, and allocation-free in its evaluation path.
- `l1_delta_lag_ratio` remains in the ML surface but is no longer a Classic input.
- New Classic feature work must preserve exact runtime cadence, startup evidence, reset, settling, and threshold semantics in a maintained packet-level comparison.
- C3, C5, and C6 quiet tails remain explicit performance limits; current measurements belong in `performance/README.md` rather than this ADR.

## Related

- [FEATURES.md](../FEATURES.md)
- [2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md](2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md)
- [2026-07-25-select-the-classic-band-from-channel-coherence.md](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md](2026-08-05-reject-adjacent-subcarrier-aggregation-on-the-shared-band.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [performance/README.md](../performance/README.md)
