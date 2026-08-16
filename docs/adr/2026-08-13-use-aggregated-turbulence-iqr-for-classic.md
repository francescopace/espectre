# ADR: use aggregated turbulence IQR for Classic

- Status: Accepted
- Date: 2026-08-13
- Updated: 2026-08-16

## Context

Classic is the resource-constrained alternative to `MLDetector`: it should remain a two-feature, linear detector with substantially lower memory, CPU, and implementation complexity, even when ML retains a quality advantage. The previous production pair combined turbulence autocorrelation with the temporal spread of offset-4/12 complex frequency coherence. It was viable, but required a full-band complex tracker whose work and physical feature family were independent from the turbulence primitives already maintained by both detectors.

The replacement campaign kept `turb_autocorr` fixed and compared `turb_zcr`, normal-turbulence `turb_iqr_over_mean`, and `W=5` adjacent-bin aggregated `turb_iqr_over_mean_aggr`. Candidates were fitted on de-overlapped `train` rows, ranked on `train + selection`, reported separately on historical holdout and `exclude`, and replayed under `base`, `drift`, `burst-loss`, and combined packet stress. Vacation-home `exclude` recordings remained diagnostic and did not participate in fitting or ranking.

This record incorporates the useful lineage and operating-point lessons from the former offset-4/12 ADR. That file is removed rather than marked superseded because this ADR is the cumulative current Classic feature decision.

## Decision

Use weighted logistic fusion of lag-adjusted turbulence autocorrelation and robust spread over a dedicated aggregated turbulence stream:

```text
x_aggr[t] = spatial_turbulence(mean_amplitude_over_adjacent_live_bins(W=5))
turb_iqr_over_mean_aggr = (Q75(x_aggr) - Q25(x_aggr)) / max(abs(mean(x_aggr)), 1e-6)
logit = b + w_ac * z(turb_autocorr) + w_iqr * z(turb_iqr_over_mean_aggr)
probability = sigmoid(logit)
```

The normal and aggregated streams share one packet-wide magnitude frame. Each stream retains the same configurable Hampel and low-pass policy. The coefficient fit uses the 22 admitted `train` pairs, grouped de-overlapped folds, and balanced class, chip, and session weights. The promoted operating-point recipe uses a `1%` grouped OOF FP target; sequential report gates remain authoritative because average dense-window FP does not encode per-recording or debounced empty-room alarm contracts.

The reproducible export command is:

```bash
.venv/bin/python tools/fit_lightweight_detector.py --fp-target 1.0 --centered-threshold-logit 1.8 --apply --quiet
```

The offset-4/12 coherence extractor, the normal-IQR alternative, ZCR, and other rejected candidates remain host-only under `tools/`. Production C++ and MicroPython contain only features consumed by Classic or ML.

## Decision History

Detailed feature evidence belongs in [`FEATURES.md`](../FEATURES.md). The cumulative Classic feature lineage is:

| Date | Feature direction | Resolution |
| --- | --- | --- |
| 2026-07-08 | L1-primary Classic with complementary variance behavior | Established the production non-ML direction |
| 2026-07-22 | Add a low-RSSI session-centered L1 blend | Retired when Classic stopped consuming L1 |
| 2026-07-24 | Defer an `l1_delta_std` swap | Closed when the entire L1 family left Classic |
| 2026-07-26 | Replace absolute L1 mean with a lag ratio | Improved scale behavior but was later replaced |
| 2026-07-30 | Fuse turbulence autocorrelation with offset-2/12 frequency coherence | Established the complex coherence family |
| 2026-08-12 | Use offset-4/12 frequency coherence | Reduced pair work and improved its quiet tail |
| 2026-08-13 | Replace complex coherence with aggregated turbulence IQR | Current production definition |
| 2026-08-16 | Refit coefficients on temporal-admission windows, including High Accuracy's idle `fp_weight=1.75` | Rejected; both refits raised one sequential S3 empty-room alarm that the exported coefficients still clear |

## Validation

The clean-fitted replay at a `1%` OOF FP target reported:

| Role or stress | Worst recall | Maximum idle FP |
| --- | ---: | ---: |
| Discovery, clean | `95.69%` | `4.39%` |
| Holdout, clean | `97.97%` | `6.83%` |
| Discovery, combined stress | `94.66%` | `8.99%` |
| Holdout, combined stress | `96.76%` | `9.89%` |
| `exclude`, clean | `83.27%` | `0%` |
| `exclude`, combined stress | `72.01%` | `0%` |

On the 22-pair production fitter, the grouped de-overlapped five-fold point reported F1 `99.320%`, recall `99.587%`, FP `0.947%`, and worst-session recall `95.690%`. The automatic point raised one effective alarm on a short S3 empty-room gate; centered logit `1.8` was the smallest tested sequential point that restored zero short-empty alarms while retaining all recall gates. The final generated report records normal-link per-chip recall `99.7–100.0%`, FP `0–1.4%`, and maximum per-recording FP `0–3.3%`; low-RSSI recall is `98.9–99.7%`, FP is `0.1–1.8%`, and maximum per-recording FP is `0.3–4.7%`. Long-quiet maximum FP is `0.09–4.39%`. The report generation and C++ parity check passed.

A five-process `-O3` host microbenchmark at 100 packets/s and 4 evaluations/s compared the normal-IQR fallback with the promoted aggregated path:

| Resource | Normal IQR | Aggregated IQR |
| --- | ---: | ---: |
| Median packet path | `143.4 ns` | `299.3 ns` |
| Median evaluation path | `759.4 ns` | `835.6 ns` |
| Modeled feature CPU | `17.38 us/s` | `33.34 us/s` |
| Modeled persistent feature state | `800 B` | `1,280 B` |

The aggregated path therefore adds `480 B` and approximately `16 us/s` of host CPU. These are requested C++ state and host timings, not representative-device allocator or cycle measurements.

## Alternatives Considered

### Keep offset-4/12 frequency coherence

Rejected for production. It had already passed the prior promotion gates, but it retained a separate complex full-band extraction family. Aggregated IQR reuses shared phaseless turbulence primitives, improves combined-stress recall and quiet tails in the candidate replay, and removes the coherence implementation from constrained runtimes. The host extractor remains available for research and historical reproduction.

### Use normal-turbulence IQR

Retained host-only as the ultra-low-resource fallback. It adds no dynamic detector state and was slightly stronger on several clean recall measures, but its combined-stress idle maxima were `17.30%` on discovery and `13.58%` on holdout versus `8.99%` and `9.89%` for aggregated IQR. The extra `480 B` is justified by the materially quieter tail.

### Use turbulence ZCR

Rejected. Its threshold-free separation was attractive, but its sequential calibration frontier was unsafe: tightening the OOF FP target reduced false positives only by sacrificing the weak-session recall floor.

### Add nonlinear fusion or a third feature

Rejected. Classic's product role requires a two-term linear model. Nonlinear and three-feature candidates either increased quiet tails, activated channel-shape trajectory state, or narrowed the resource gap to ML without sufficient independent benefit.

## Consequences

- Classic remains linear, vote-free, gain-invariant, and limited to two features.
- C++ and MicroPython maintain a second filtered turbulence ring and share one packet magnitude frame between both streams.
- Production no longer contains the complex frequency-coherence tracker or helpers; those candidates remain host-only.
- The normal-IQR and ZCR alternatives remain research features and do not expand firmware or MicroPython production surfaces.
- Future coefficient or operating-point changes must rerun sequential low-RSSI, empty-room, per-recording, packet-rate, generated-report, and cross-runtime parity gates.

## Related

- [2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [2026-08-11-promote-channel-shape-trajectory-ml-features.md](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [FEATURES.md](../FEATURES.md)
