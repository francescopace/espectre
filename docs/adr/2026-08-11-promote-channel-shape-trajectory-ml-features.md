# Promote channel-shape trajectory ML features

- Status: Accepted
- Date: 2026-08-11
- Supersedes: 2026-08-07-promote-the-compact-aggregated-iqr-ml-model.md

## Context

Motion collected in a vacation home crossed repeatable WiFi blind spots. The operator moved continuously, but packet loss and weak-link intervals reduced the apparent motion fraction. This is a production-relevant domain-transfer problem: supported commercial chip families are represented in the corpus, while unseen rooms, routers, RSSI regimes, and intermittent packet delivery remain unpredictable.

The Aggregated-IQR-7 model used L1-delta autocorrelation and frequency-coherence curve standard deviation. A host-only screen found that two physical-time channel-shape trajectory features recovered more vacation-home motion and generalized better across environments when they replaced both inputs. Leave-one-chip-out C3 recall fell by `1.36` percentage points, but supported-chip low-RSSI and quiet replay are the deployment gates; chip exclusion remains a conservative diagnostic rather than a veto.

## Decision

Promote this ordered seven-input ML schema:

1. `turb_iqr_over_mean_aggr`;
2. `turb_autocorr`;
3. `turb_zcr`;
4. `l1_delta_lag_ratio`;
5. `chan_shape_spread`;
6. `chan_shape_coherent_innovation_energy`; and
7. `chan_shape_excess_path`.

For the initial promotion export, keep the `7 -> 24 -> 12 -> 1` topology, seed `1161881508`, standard scaling, false-positive weight `1.75`, feature jitter, and the `base,drift,burst-loss` packet augmentation mix from seeds `20260807` and `20260808`. Later corpus refreshes may select a different training seed without changing this feature-schema decision; the exported weight metadata and `FEATURES.md` record the current run.

The trajectory tracker uses a gain-normalized eight-subband energy profile, `80 ms` physical-time median bins, a one-second window, exact duplicate suppression, and missing-bin skipping. Coherent innovation measures positive low-order DCT energy left after a constant-velocity prediction and high-order noise subtraction. Excess path measures positive two-step path length beyond its chord after subtracting high-order DCT path excess.

Remove runtime extractors and tracker state that neither the exported ML model nor Classic consumes. Retain `l1_delta_autocorr`, `turb_mad_over_mean`, `chan_freq_coh_cv`, `chan_coh_gap`, and `chan_coh_subband_gap_median` as host-only candidates. Keep `chan_freq_coh_curve_std` in production only because Classic consumes it; its runtime path computes the two offsets selected by the current Classic decision.

## Validation

The selection-only decision gate used no holdout evidence. The selected candidate passed all three paired selection replays with zero FP and zero alarms: C3 recall `100%`, low-RSSI C5 recall `94.84%`, and low-RSSI S3 recall `95.98%`. It passed all seven selection quiet replays with `0.342%` maximum raw FP and zero alarms. The excluded vacation-home challenge reached `94.828%` motion recall with `0%` static and empty FP, compared with `87.931%`, `0%`, and `1.078%` for the preceding export.

The final augmented training and reserved export gates produced:

| Gate | Result |
| --- | --- |
| Blocked grouped CV | F1 `99.1%`, recall `98.5%`, precision `99.7%`, and FP `0.1%` |
| Paired selection and holdout | `14/14`, `94.84%` worst recall, `0.14%` maximum FP, and zero alarms |
| Quiet selection and holdout | `9/9`, `0.34%` maximum raw FP, and zero alarms |
| Gain stress | Identical results from `0.5x` through `2.0x`; all inputs are gain-invariant |
| Leave one environment out | F1 `99.135%`, recall `98.868%`, FP `0.253%`, `97.674%` worst recall, and `0.391%` worst FP |
| Leave one chip out | F1 `98.889%`, recall `98.329%`, FP `0.230%`, `95.928%` worst recall, and `0.575%` worst FP |

The preceding exported model measured `92.55%` worst paired recall and `0.29%` maximum paired FP on the same final gate.

## Consequences

- The MLP still has 505 parameters; inference multiply-add cost is unchanged.
- ML no longer computes within-packet frequency coherence or historical complex channel-coherence trackers.
- The frequency-coherence path computes only the short-versus-long curve consumed by Classic.
- Production adds one fixed-storage physical-time trajectory tracker. MicroPython avoids a full-payload tuple allocation, and C++ uses bounded arrays.
- Finalized trajectory bins store orthonormal DCT coefficients rather than median profiles. DCT linearity preserves the innovation residual, and Parseval's identity preserves the full-profile L2 distances in excess path, so only the current bin requires a DCT during each extraction and the feature contract remains unchanged. A profile-space reference test and the generated C++/Python replay-parity gate protect this equivalence.
- Historical feature experiments remain reproducible through the host-only registry without device memory or hot-path cost.
- The C3 leave-one-chip-out recall gap remains an explicit secondary risk, while unseen-environment, router, and weak-link behavior remain the primary promotion axes.

## Related

- [FEATURES.md](../FEATURES.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [ML_TRAINING.md](../ML_TRAINING.md)
- [2026-08-11-mix-complementary-training-augmentation-seeds.md](2026-08-11-mix-complementary-training-augmentation-seeds.md)
- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
