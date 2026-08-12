# ADR: use offset-4/12 frequency coherence for Classic

- Status: Accepted
- Date: 2026-08-12
- Updated: 2026-08-12

## Context

Classic must remain a two-feature detector for memory- and CPU-constrained devices. Its production pair combined turbulence autocorrelation with the temporal spread of a normalized short-versus-long frequency-coherence contrast at offsets 2 and 12. A new search therefore kept turbulence autocorrelation fixed and looked for a better physical definition of the same coherence axis instead of adding the channel-profile trajectory tracker used by ML.

The campaign evaluated trajectory replacements, nonlinear fusion, temporal IQR, two-point coherence contrasts, and three-point decay and curvature summaries. Candidates were fitted on de-overlapped `train` rows, ranked on `train + selection`, and reported separately on historical holdout, packet stress, empty rooms, low RSSI, and `exclude`. Vacation-home `exclude` recordings did not participate in fitting or ranking.

## Decision

Define `chan_freq_coh_curve_std` from the per-packet contrast

```text
curve_t = (coh_4 - coh_12) / (coh_4 + coh_12)
chan_freq_coh_curve_std = std_t(curve_t)
```

where each `coh_d` is normalized complex coherence over live HT20 bins separated by `d`, excluding pairs that cross DC. Keep the second Classic input as lag-1 turbulence autocorrelation and retain linear, two-term logistic fusion.

Refit the coefficients on the 22 admitted `train` pairs. Select the exported global threshold with the sequential production replay rather than the dense OOF window sweep. The reproducible export command is:

```bash
.venv/bin/python tools/fit_classic_detector.py --centered-threshold-logit 1.73 --apply --quiet
```

The centered logit is part of the operating-point decision. The automatic OOF point maximized window recall under an average FP ceiling, but it failed the production empty-room alarm and per-recording constraints; it must not replace the sequentially validated point.

## Decision History

Detailed feature evidence belongs in [`FEATURES.md`](../FEATURES.md). The Classic feature lineage is:

| Date | Feature direction | Resolution |
| --- | --- | --- |
| 2026-07-08 | L1-primary Classic with complementary variance behavior | Established the production non-ML direction |
| 2026-07-22 | Add a low-RSSI session-centered L1 blend | Retired when Classic stopped consuming L1 |
| 2026-07-24 | Defer an `l1_delta_std` swap | Closed when the entire L1 family left Classic |
| 2026-07-26 | Replace absolute L1 mean with a lag ratio | Improved scale behavior but was later replaced |
| 2026-07-30 | Fuse turbulence autocorrelation with offset-2/12 frequency coherence | Established the current physical feature family |
| 2026-08-12 | Use offset-4/12 frequency coherence | Current production definition |

## Validation

The corrected candidate replay compared offset 4/12 with a refitted offset-2/12 surrogate:

| Replay | Offset 2/12 | Offset 4/12 | Result |
| --- | ---: | ---: | --- |
| Clean score | `74.826` | `60.042` | Improved |
| Clean worst recall | `97.13%` | `97.13%` | Equal |
| Clean maximum empty FP | `24.21%` | `20.53%` | Improved |
| Base-stress worst recall | `94.59%` | `94.41%` | `-0.18` points |
| Base-stress maximum empty FP | `26.45%` | `23.23%` | Improved |
| Combined-stress score | `108.36` | `94.987` | Improved |
| Combined-stress worst recall | `80.37%` | `80.98%` | Improved |
| Combined-stress holdout worst recall | `83.48%` | `85.25%` | Improved |

The coefficient fit itself reported grouped OOF F1 `98.320%`, recall `99.518%`, FP `2.912%`, and worst-session recall `97.126%`. Exporting that automatic OOF operating point was rejected after it caused six Python real-data failures and failed the C++ low-RSSI, empty-room, and packet-rate suites.

At centered logit `1.73`, sequential replay retained `93.37%` minimum train-session recall, `14.86%` maximum train-session FP, `89.08%` minimum low-RSSI recall, zero effective alarms on the short empty-room gate corpus, and `2.89%` maximum short-empty FP. The final Python performance gate passed all 82 cases. The regenerated performance report, which also covers long quiet recordings, reported these normal-link chip aggregates:

| Chip | Recall | FP |
| --- | ---: | ---: |
| C3 | `95.0%` | `0.1%` |
| C5 | `99.8%` | `1.7%` |
| C6 | `100.0%` | `4.4%` |
| ESP32 | `100.0%` | `0.0%` |
| S3 | `99.1%` | `1.8%` |

The matching C++ core, Classic, ML, motion, long-recording, low-RSSI, empty-room, and packet-rate suites all passed. Python and C++ use the same 4/12 definition, fitted constants, startup reference, and global threshold.

The coherence inner loop now evaluates `48 + 32 = 80` complex bin pairs per packet instead of `52 + 32 = 84`, a `4.8%` reduction. The 90-float curve ring and all other Classic state are unchanged.

## Alternatives Considered

### Keep offset 2/12

Rejected. It remained viable, but offset 4/12 improved clean and combined-stress scores, improved the quiet tail, and reduced complex-pair work without adding memory.

### Add nonlinear fusion

Rejected. An interaction term scored `75.808` versus `74.826` for linear fusion and increased maximum empty FP from `24.21%` to `24.47%`. Adding both squares scored `74.862`; its `0.28`-point worst-recall gain did not justify extra arithmetic or parameters.

### Use three-point coherence decay

Rejected. Offset-2/4/12 decay improved some scores but required 48 additional complex pairs per packet. Offset 4/12 retained the useful scale contrast at lower cost.

### Replace coherence with a trajectory feature

Rejected. Neither coherent innovation nor excess path produced a better two-feature balance, and either would activate the physical-time profile and DCT tracker that Classic intentionally avoids.

### Export the OOF-selected threshold

Rejected. Average dense-window FP is not equivalent to the debounced zero-alarm contract. The failed promotion attempt demonstrated that coefficients and the production operating point require separate evidence.

## Consequences

- Classic remains linear, vote-free, gain-invariant, and limited to two features.
- The frequency tracker performs slightly less work and retains the same memory footprint.
- `chan_freq_coh_curve_std` now means offset 4/12 in both production runtimes; host-only offset-2 candidates remain available only for historical reproduction.
- Future coefficient exports that change the operating point must run sequential low-RSSI, empty-room, per-recording, packet-rate, and cross-runtime gates. The fitter exposes `--centered-threshold-logit` so an accepted replay point can be reproduced exactly.

## Related

- [2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [2026-08-11-promote-channel-shape-trajectory-ml-features.md](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [FEATURES.md](../FEATURES.md)
