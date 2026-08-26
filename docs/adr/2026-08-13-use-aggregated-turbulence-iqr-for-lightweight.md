# ADR: use aggregated turbulence IQR for Lightweight

- Status: Accepted
- Date: 2026-08-13
- Updated: 2026-08-26

## Context

Lightweight is the lower-resource alternative to `HighAccuracyDetector`. It must remain a compact, deterministic, two-feature linear detector. The previous production pair combined turbulence autocorrelation with temporal spread from a complex frequency-coherence tracker. That path required a separate full-band complex feature family beside the turbulence primitives used by both detectors.

A host-side campaign kept turbulence autocorrelation fixed and compared zero-crossing rate, normal-turbulence IQR, and IQR from a dedicated adjacent-bin aggregated turbulence stream. Aggregated IQR produced the safer sequential quiet tail under packet stress while preserving the Lightweight recall floor. The normal-IQR fallback was cheaper, but its idle tail was materially worse. Detailed fits, operating points, corpus revisions, metrics, and resource measurements belong in `FEATURES.md` and the generated performance report.

## Decision

Use weighted logistic fusion of lag-adjusted turbulence autocorrelation and robust relative spread over a dedicated aggregated turbulence stream:

```text
x_aggr[t] = spatial_turbulence(mean_amplitude_over_adjacent_live_bins(W=5))
turb_iqr_over_mean_aggr = (Q75(x_aggr) - Q25(x_aggr)) / max(abs(mean(x_aggr)), 1e-6)
probability = sigmoid(b + w_ac * z(turb_autocorr) + w_iqr * z(turb_iqr_over_mean_aggr))
```

The normal and aggregated streams share one packet-wide magnitude frame and the same configurable Hampel and low-pass policy. Generated coefficients and the current operating point remain owned by the fitter artifacts and performance report.

Keep the offset-coherence, normal-IQR, ZCR, and other rejected candidates host-only. Production C++ and MicroPython contain only features consumed by Lightweight or High Accuracy.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-08 to 2026-07-26 | Use L1-primary and lag-ratio Classic variants | Replaced as the non-ML detector moved to a turbulence-only feature family |
| 2026-07-30 to 2026-08-12 | Fuse turbulence autocorrelation with complex frequency coherence | Replaced after aggregated IQR produced a safer quiet tail with simpler shared primitives |
| 2026-08-13 | Use turbulence autocorrelation plus aggregated turbulence IQR | Accepted as the production Lightweight feature pair |
| 2026-08-16 | Refit and rescreen the pair after temporal-admission and occupancy changes | Retained the exported pair and bounded Lightweight empty-room alarm policy |

## Alternatives Considered

### Keep complex frequency coherence

Rejected for production. It retains a separate complex full-band tracker, while aggregated IQR reuses phaseless turbulence primitives and passed the promotion gates.

### Use normal-turbulence IQR

Retained host-only as an ultra-low-resource fallback. It needs less state but produced a worse stressed idle tail.

### Use turbulence ZCR

Rejected. Its threshold-free separation did not survive the sequential calibration frontier without sacrificing weak-session recall.

### Add nonlinear fusion or a third feature

Rejected. The added state or inference would narrow the resource gap to High Accuracy without sufficient independent benefit.

## Consequences

- Lightweight remains linear, vote-free, gain-invariant, and limited to two features;
- C++ and MicroPython maintain a second filtered turbulence ring and share one packet magnitude frame;
- production no longer contains the complex frequency-coherence tracker;
- alternative features remain research-only and do not expand constrained runtimes; and
- coefficient or operating-point changes must rerun sequential low-RSSI, empty-room, per-recording, packet-rate, generated-report, and cross-runtime parity gates.

## Related

- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [`../FEATURES.md`](../FEATURES.md)
