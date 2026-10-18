# ADR: define the Lightweight detector

- Status: Accepted
- Date: 2026-07-08
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

Over time the runtime collected a multi-state detector, moving-variance variants, threshold modes, and several L1-based Classic versions. They were useful while exploring, but each one multiplied configuration, calibration, tests, and frontend behavior. The non-ML detector, first called Classic and now Lightweight Detection (`LightweightDetector`, see the detection profiles ADR), needed one production definition: its role, its features, and how it places its threshold.

Its features changed twice. The previous pair combined turbulence autocorrelation with temporal spread from a complex frequency-coherence tracker, a whole separate full-band feature family next to the turbulence primitives both detectors already use. A host-side campaign then kept autocorrelation fixed and compared three partners: zero-crossing rate, plain turbulence IQR, and IQR over a turbulence stream built from adjacent-bin averages. The aggregated IQR gave the safest quiet tail under packet stress and kept the recall floor; plain IQR was cheaper but had a clearly worse idle tail.

Threshold placement had its own problem. On one ESP32 recording the opening of the session was unusually noisy, so the startup threshold stayed too high once the room went quiet, and recall dropped even though the features still separated motion from idle. Looking only at the opening cannot tell a noisy room from a bad start.

## Decision

**Role.** `LightweightDetector` is the only production non-ML detector, a small deterministic option for deployments that do not pick High Accuracy. It uses the same `0.0-1.0` probability threshold as High Accuracy, has one automatic startup calibration plus overrides that last only for the session, goes back to its automatic threshold after recalibration or restart, and behaves the same in Python and C++ (features, filtering, fusion, reset, calibration). MVS, moving variance, L1-primary, the recovery vote, threshold modes, and the low-RSSI L1 blend are research history, not runtime options.

**Features.** Lightweight is a weighted logistic fusion of lag-adjusted turbulence autocorrelation and the relative IQR of an aggregated turbulence stream:

```text
x_aggr[t] = spatial_turbulence(mean_amplitude_over_adjacent_live_bins(W=5))
turb_iqr_over_mean_aggr = (Q75(x_aggr) - Q25(x_aggr)) / max(abs(mean(x_aggr)), 1e-6)
probability = sigmoid(b + w_ac * z(turb_autocorr) + w_iqr * z(turb_iqr_over_mean_aggr))
```

Both turbulence streams read the same per-packet magnitude frame and go through the same Hampel and low-pass settings. Coefficients and the operating point belong to the fitter output and the performance report. Offset coherence, plain IQR, ZCR, and the other candidates stay in host tools; production C++ and MicroPython only contain features that a production detector uses.

**Settled-session recovery.** Lightweight may lower its live threshold after a long quiet stretch shows the session is clearly quieter than its opening. Evaluations are grouped into fixed blocks; the detector keeps the maximum metric logit of each block in a small ring and takes the median as the settled level. If that level plus a safety margin is below the current threshold, the threshold drops to it. It only goes down, motion keeps it from moving because active blocks have high maxima, and the median stops a single spike or quiet block from deciding. `reset()`, `clear_buffer()`, `on_startup_calibration_begin()`, and `set_adaptive_threshold()` clear the history, so after a restart, a detector switch, or a recalibration the detector must see another full quiet stretch. Block size, ring length, and margin are constants documented in `ALGORITHMS.md`.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-11-16 | Segmentation-first MVS architecture | Superseded by the simpler Classic path |
| 2026-07-07 | Gated startup calibration for an L1-primary detector | Replaced by probability calibration and later threshold adaptation |
| 2026-07-08 | Promote L1-primary Classic, retire legacy baselines | Kept as the product direction |
| 2026-07-16 | Guarded MVS recovery vote | Replaced by direct weighted two-feature fusion |
| 2026-07-22 | Blend toward a session-centered L1 excursion at low RSSI | Dropped when Classic stopped using L1 features |
| 2026-07-26 | Median-of-block-maxima recovery from a bad startup prefix | Accepted after paired, weak-link, and empty-room replay |
| 2026-07-28 | Remove the unused startup variance-floor sampling | Accepted: replay output bit-for-bit identical, calibrator 4,308 bytes smaller (4,636 → 328); it is not the motion-level floor the threshold metric still uses |
| 2026-07-30 to 2026-08-12 | Autocorrelation fused with complex frequency coherence | Replaced: aggregated IQR had a safer quiet tail with simpler shared primitives |
| 2026-08-10 | Revalidate the recovery margin under native-cadence replay with millisecond windows | Value retuned, rule unchanged |
| 2026-08-13 | Autocorrelation plus aggregated turbulence IQR | Accepted |
| 2026-08-15 | Calibration and evaluation fed by the same temporally admitted stream | Recovery rule kept under fixed temporal admission |
| 2026-08-16 | Refit and rescreen after temporal admission and occupancy changes | Pair kept, with the bounded empty-room alarm budget |

## Alternatives considered

- **Keep several production baselines.** Easy comparisons, at the price of duplicated configuration, calibration, reset logic, and frontend support.
- **Keep build-time threshold modes.** Automatic startup placement plus session overrides already cover production.
- **Keep complex frequency coherence.** Needs its own complex full-band tracker; aggregated IQR reuses phaseless turbulence and passed the gates.
- **Plain turbulence IQR.** Kept host-side as an ultra-low-resource fallback. Less state, worse stressed idle tail.
- **Turbulence ZCR.** Its separation did not hold up under sequential calibration without losing recall on weak sessions.
- **Nonlinear fusion or a third feature.** More state or inference would eat into the resource gap with High Accuracy for too little gain.
- **Stronger startup calibration, or a cap from prefix statistics.** Both still rely on the bad prefix, and hit the empty-room limit before they fixed the failing session.
- **Refit to fix the noisy opening.** The failure was threshold placement; the refit lost recall elsewhere without fixing it.
- **Periodic recalibration from scratch.** Would have to assume the sampled window is idle. The settled rule lets motion block the change instead.
- **Track an exact long-window quantile.** Block maxima give the same decision with much smaller, bounded state.

## Consequences

There is one non-ML detector contract, and frontend controls and calibration do not branch on historical baselines. Lightweight stays linear, vote-free, gain-invariant, and limited to two features; C++ and MicroPython keep a second filtered turbulence ring fed by the shared magnitude frame, and the complex coherence tracker is gone from production. A noisy opening no longer fixes a high threshold for the whole session. If the room gets noisier after a recovery, the threshold does not rise on its own; that needs a recalibration.

Historical comparisons need host tools or Git history. Any coefficient, operating-point, or recovery change must rerun the low-RSSI, empty-room, per-recording, packet-rate, generated-report, and parity gates, and C++ and Python must keep the rule, reset points, and constants aligned.

## Related

- [Algorithms](../ALGORITHMS.md), the [feature ledger](../FEATURES.md), and the [performance report](../performance/README.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-07-15-define-the-detection-profiles.md`](2026-07-15-define-the-detection-profiles.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- Commits `e6d2f53f` (MVS segmentation), `71901197` (L1 detector), `09ceb107` (Classic promoted), `911b0c90` (weighted fusion, vote removed), `581e5b80` (low-RSSI blend), `90fccc26` (settled recovery), `63a726f9` (variance floor removed), `b40c0807` (margin retune), `e034feb5` (aggregated IQR)
