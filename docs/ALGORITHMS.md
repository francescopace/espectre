# Algorithms

Current detector and signal-processing reference for ESPectre.

This file documents only algorithms active in the current project surface. Feature experiments and promotion evidence live in [FEATURES.md](FEATURES.md), decision rationale lives in [adr/](adr/), and mutable detector metrics live in the generated [performance report](performance/README.md).

## Overview

ESPectre detects motion from Wi-Fi CSI by extracting a small, fixed slice of subcarriers, deriving gain-robust scalar signals from those amplitudes, and feeding those signals into either:

- `ClassicDetector`, the default non-ML detector
- `MLDetector`, the neural detector with a trained probability threshold

Representative raw CSI amplitude windows for empty room, static presence, and motion:

![CSI amplitude heatmaps for empty, static presence, and motion](web/assets/images/guides/csi-amplitude-heatmap.webp)

The current production detector definition is:

- AGC stays active
- the shared fixed 12-subcarrier set feeds the turbulence and L1-displacement features, while the coherence and channel-shape features read the full 56-bin live band
- the classic path uses weighted `turb_autocorr + chan_freq_coh_curve_std` fusion
- the classic runtime has no voting branch or legacy low-RSSI blend term
- the ML path uses the compact seven-feature scale-invariant production set

## Processing Pipeline

Steady-state detector flow:

```text
CSI packet
  -> fixed 12-subcarrier amplitudes
       -> CV turbulence (std / mean)
       -> optional Hampel / low-pass filtering
  -> 56-bin live complex profile
       -> coherence and channel-shape trackers
  -> detector-specific metric or feature extraction
  -> thresholded motion state
```

At boot:

- `classic` performs startup threshold calibration
- `ml` starts as soon as CSI capture is active from its trained default threshold

With the default `window_size=100`, the `classic` startup budget is `10 x window_size = 1000` packets. This is a maximum, not a mandatory wait.

## Detector Timing

The deployed detector uses a time-relative evaluation cadence and fixed feature geometry:

| quantity | production setting | nominal interpretation at 100 pps |
| --- | --- | --- |
| detector window | 100 packets | `1 s` |
| evaluation interval | `250 ms` | time-based, not packet-count driven |
| channel-shape tracker lag | 10 packets | `100 ms` |
| turbulence autocorrelation lag | 1 packet | `10 ms` |

These offsets define the detector surface. v3 keeps `lag = 10` packets for the shape trackers and `autocorr_lag = 1`, supports `80-133 pps`, and does not derive feature lags from the measured packet rate. Changing the offsets requires a Classic refit plus the normal ML validation workflow. See the [timing ADR](adr/2026-07-28-keep-production-feature-lags-at-nominal-offsets.md) for the decision evidence.

Calibration and steady-state detection share one cadence, so the interceptor that consumes packets during calibration evaluates on the same schedule the detection path does.

The window follows sample count because its features are estimator averages. See the Window Size section in [TUNING.md](TUNING.md) for the evidence behind the 100-sample floor.

Cadence advances on the packet arrival timestamp, never on the loop clock or a packet-count fallback. The loop clock measures how fast packets are processed, which matches arrival on hardware but not on replay, and would let host scheduling reach a detector decision. Wall-clock time is reserved for staleness detection, which arrival time cannot do because a dead stream delivers no timestamps. Live input and supported replay datasets must provide timestamps; a missing or non-advancing timestamp contributes no elapsed coverage.

The `stream_dense` training contract mirrors this cadence and reset behavior; see [ML_TRAINING.md](ML_TRAINING.md).

## AGC-Active Normalization

The shared turbulence signal is:

```text
turbulence = std(amplitudes) / mean(amplitudes)
```

This coefficient-of-variation form is gain-invariant:

```text
CV(kA) = std(kA) / mean(kA) = std(A) / mean(A)
```

If AGC scales all amplitudes by a factor `k`, turbulence stays unchanged. This same AGC-active normalization model is used across:

- runtime detection
- host collection
- dataset schema
- offline ML tooling

## Fixed Subcarrier Set

Both detectors sample the same fixed 12-subcarrier set for their turbulence and L1-displacement features:

```text
[4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]
```

These bins are subcarriers `+/-4, +/-9, +/-14, +/-19, +/-24, +/-28`, and they assume the centered convention where bin `32` is DC. Classic-MAC parts deliver CSI in Espressif's native `0~31, -32~-1` order instead, so the capture path rotates those payloads before band selection; see [`csi_format.h`](../src/cpp/core/csi_format.h) and [`device_utils.py`](../src/python/micro_espectre/device_utils.py).

The active runtime no longer performs per-session runtime subcarrier selection. This set is part of the detector definition for the current project surface. The indices come from measured channel coherence rather than from a detection-metric search: the motion perturbation stays coherent over about 10 subcarriers while quiet noise is nearly per-tone independent, so span is what buys independent looks. For the full rationale behind the band and the count, see [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md).

### Live Band For Frequency-Domain Features

The 12-tone set is a sampling of the spectrum, and it serves the features that build a time series out of it. The channel-shape and coherence features instead measure structure across frequency inside a single packet, so they read the full HT20 live band: bins `4..31` and `33..60`, that is the 56 subcarriers left after the guard bands and the DC null.

| Feature family | Band | Why |
| --- | --- | --- |
| `turb_*`, `l1_delta_*` | 12 selected tones | builds a time series, where span buys independent looks |
| `chan_shape_*`, `chan_freq_coh_*`, `chan_coh_*` | 56 live bins | measures shape across frequency, which decimation would remove |

The split follows from what each family measures, rather than from two independent band choices. Within-packet frequency coherence is evaluated on bin pairs at fixed separations of `2`, `4`, and `12` bins, and the 12-tone set has a minimum spacing of 4, so it contains no pair at all at separations `2` and `12`. Those features are undefined on the sampled band, not merely degraded by it.

Both runtimes define the live band identically, as `HT20_LIVE_BINS` in [`ml_feature_trackers.h`](../src/cpp/core/ml_feature_trackers.h) and [`ml_feature_trackers.py`](../src/python/micro_espectre/ml_feature_trackers.py).

HT20 is the enforced detector input contract on both supported bands, while the current detection corpus validates only 2.4 GHz operation. VHT20, HE20, and wider layouts are not accepted by the production detectors. Band-selection behavior lives in [SETUP.md](SETUP.md), and the PHY rationale lives in the [HT20 ADR](adr/2026-08-05-pin-ht20-on-every-band-instead-of-forcing-2-4-ghz.md).

Non-HT20 payloads are normalized onto the same internal 64-subcarrier HT20 index grid before fixed-subcarrier extraction. Short layouts are centered so the HT20 midpoint remains aligned.

| Input case | Raw layout | Mapping to HT20 | Output |
|------------|------------|-----------------|--------|
| Native HT20 | `128 B = 64 SC` | pass-through | `64 SC / 128 B` |
| Short HT estimate | `114 B = 57 SC` | zero-pad `4` SC left, copy `57` SC, zero-pad `3` SC right | `64 SC / 128 B` |
| Double HT20 payload | `256 B = 2 x 64 SC` | collapse to one `128 B` half | `64 SC / 128 B` |
| Double short HT estimate | `228 B = 2 x 57 SC` | collapse to one `57 SC` half, then pad `4` left and `3` right | `64 SC / 128 B` |

## Signal Conditioning

Optional filters operate on the scalar turbulence stream before detector evaluation.

### Hampel Filter

Default: enabled (`window=7`, `threshold=5.0` MAD)

The Hampel filter removes large outliers using the median absolute deviation:

```text
MAD = median(|x_i - median(x)|)
```

Packets that exceed the configured MAD-scaled deviation are replaced by the current window median.

### Low-Pass Filter

Default: disabled

The low-pass stage is a first-order Butterworth IIR filter applied to the turbulence signal before detector evaluation. Use [`TUNING.md`](TUNING.md) for the operational trade-off between false-positive reduction and responsiveness.

## Classic Detector

`ClassicDetector` is the production non-ML path. It combines:

- lag-1 autocorrelation of the gain-invariant turbulence stream
- temporal standard deviation of the short-versus-long frequency-coherence contrast
- a fixed, weighted logistic fusion with no voting branches

### Turbulence Autocorrelation

Per-packet turbulence is the spatial coefficient of variation:

```text
t_i = std(A_i) / mean(A_i)
```

After Hampel filtering, Classic calculates lag-1 autocorrelation over the turbulence window. This input is invariant under ideal uniform scaling because the coefficient of variation is itself a ratio. The shared `hampel_enabled` setting still controls the turbulence filter in both runtimes, and the same filtered turbulence stream feeds the ML `turb_*` features.

### Channel Frequency-Coherence Curve Spread

Classic's second input comes from the complex CSI profile over the 56-bin HT20 live band, not over the sampled 12-tone set: at separations `2` and `12` the sampled set contains no bin pair at all. For a fixed subcarrier separation `d`, within-packet frequency coherence is:

```text
coh_d = |sum_k conj(H[k]) H[k + d]| /
        (sqrt(sum_k |H[k]|^2) sqrt(sum_k |H[k + d]|^2))
```

Pairs that would cross the DC subcarrier are excluded. Classic evaluates this coherence at offsets `2` and `12`, forms a bounded contrast per packet,

```text
curve_t = (coh_2 - coh_12) / (coh_2 + coh_12)
```

and reports the temporal standard deviation over the live window:

```text
chan_freq_coh_curve_std = std_t(curve_t)
```

Normalized coherence cancels common packet gain, and the short-versus-long contrast keeps the result dimensionless and bounded. The runtime reuses the shared `ChannelShapeTracker` for this input, so Python and C++ evaluate the same per-packet contrast and the same window statistic.

### Weighted Fusion

Classic standardizes `turb_autocorr` and `chan_freq_coh_curve_std` with fixed training statistics, applies a two-term linear model, and converts its logit to a probability:

```text
logit = b + w_ac * z(turb_autocorr) + w_curve * z(chan_freq_coh_curve_std)
probability = 1 / (1 + exp(-logit))
motion = probability > threshold
```

The coefficients come from grouped, de-overlapped out-of-fold training balanced by class, chip, and session. The runtime contains no majority vote or recovery branch in the score itself; all runtime adaptation happens at the threshold.

Startup adaptation thresholds this fitted two-feature logit directly. The older low-RSSI L1 blend path is retired; it is not part of the current detector surface.

### Startup Threshold Calibration

At startup, Classic begins from the validated global probability threshold and shifts its logit using the session's startup `q95` relative to the training idle reference. The shift applies `50%` of the observed session-to-training offset:

```text
adapted_logit = logit(base_threshold) +
                0.5 * (startup_q95 - train_idle_q95)
threshold = sigmoid(adapted_logit)
```

Only the first `64` ready evaluations contribute startup evidence. This keeps the learned two-feature boundary intact while letting the threshold follow a session whose quiet baseline starts above or below the training reference. Runtime adjustments stay on the same `0.0-1.0` probability scale and remain active until recalibration or reboot.

The settled-level rule cannot create a high threshold. It only ever lowers one after a long quiet dwell, so any threshold that lands near `1.0` came from the startup `q95` shift, not from later recovery.

### Known Limits

Classic clears the aggregate normal-link recall target on every chip, but C5 and C6 retain the largest false-positive tails, including on long quiet recordings. Weak-link captures remain report-only stress diagnostics. See the generated [performance report](performance/README.md) for current metrics.

Use `ml` where quiet-room robustness or held-out generalization matters more than zero-training deployment cost. The active Classic feature-selection record now lives in `FEATURES.md`; no additional pair or triplet is approved for export on the current corpus.

### Settled-Level Threshold Recovery

The runtime therefore revisits the threshold once a session proves itself quieter than its own opening. Every `20` evaluations it records the maximum metric logit in that block, keeps the last `12` blocks, and once the ring is full compares the median of those maxima against the live threshold. If that level plus `CLASSIC_SETTLE_MARGIN_LOGITS` sits below the threshold, the threshold drops to it.

Three properties make this safe rather than a drift toward the noise floor:

- **It only ever lowers.** Nothing here can raise a threshold, so it cannot hide motion that the calibrated threshold would have caught.
- **Motion holds it up.** A stretch of real activity puts the block maxima high, the candidate lands above the current threshold, and nothing happens. The rule moves only after a long quiet stretch, which is exactly the evidence that the threshold is too high.
- **A median of block maxima, not a mean or a global maximum.** One spike cannot pull the level down, and one quiet block cannot either.

The current `20`-evaluation blocks, `12`-block ring, and `2.8`-logit margin produce a `60 s` dwell at the nominal cadence. The promotion evidence and margin sweep live in the [settled-level recovery ADR](adr/2026-07-26-recover-the-startup-threshold-once-a-session-settles.md).

Its limit is the mirror of its safety. A room that grows genuinely noisier after the threshold has come down cannot push it back up; only a recalibration does that.

### Implementation Status

Current aligned implementations:

- `src/python/micro_espectre/classic_detector.py`
- `src/cpp/core/classic_detector.*`

## ML Detector

`MLDetector` is the production neural detector. It treats motion detection as a binary classification problem over a sliding window and outputs a probability in the range `0.0-1.0`.

Current threshold:

```text
motion if probability > 0.5
```

Unlike `classic`, the ML path does not need startup threshold calibration.

### Current Runtime Topology

The production export is a compact MLP:

```text
Input (7 features)
  -> Dense(24, ReLU)
  -> Dense(12, ReLU)
  -> Dense(1, Sigmoid)
```

Total parameter count: 505

The runtime accepts exported hidden-layer layouts generated by the training script, but the committed production artifact currently uses the topology above.

### Production Feature Set

The production model consumes these seven scale-invariant inputs, in export order:

1. `turb_iqr_over_mean_aggr`
2. `turb_autocorr`
3. `turb_zcr`
4. `l1_delta_autocorr`
5. `l1_delta_lag_ratio`
6. `chan_shape_spread`
7. `chan_freq_coh_curve_std`

Every member is a ratio, a correlation, or a crossing rate. The exact definitions, physical interpretations, implementation locations, retained metrics, and candidate-admission rules live in [FEATURES.md](FEATURES.md).

The first input uses a dedicated turbulence series computed after averaging adjacent live-bin magnitudes with `W=5`; its statistic is `(Q75 - Q25) / abs(mean)`. This extra buffer exists only when the exported ML feature ids request it. `turb_autocorr` and `turb_zcr` continue to read the normal twelve-subcarrier turbulence series, so the amplitude path is not silently changed for those features or for Classic. `l1_delta_lag_ratio` comes directly from the L1 tracker rather than from a rebuilt series. The final two inputs come from the normalized channel-shape tracker. The production export no longer requests the delay-compensated channel-coherence tracker, so Python and C++ skip that tracker for ML inference. Every caller passing the production set must still supply the selected tracker-derived values explicitly.

### Inference Flow

```text
CSI packet
  -> turbulence path
  -> optional filters
  -> sliding window
  -> scale-invariant feature extraction
  -> MLP inference
  -> probability threshold at 0.5
```

### Runtime Alignment

The same production feature set is used by:

- `src/python/micro_espectre/csi_features.py`
- `src/cpp/core/ml_*`
- `tools/train_ml_model.py` exports

## Calibration Summary

| Detector | Threshold | Startup behavior |
|----------|-----------|------------------|
| `classic` | automatic, session-adjustable | quiet-logit startup adaptation with motion-first completion and quiet-only fallback |
| `ml` | trained default, session-adjustable | immediate detector startup once CSI is active |

Both detectors use the same fixed subcarrier set. Only the detector metric and threshold behavior differ.

## References

See [LITERATURE.md](LITERATURE.md) for the paper index, publication dates, reported preprocessing, algorithms, results, hardware assumptions, and ESPectre transferability notes. This file retains only the active algorithm definition.
