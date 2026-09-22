# Algorithms

This reference describes the detectors and signal processing in the current release. It is written for detector and firmware contributors; operators should start with the [troubleshooting guide](TROUBLESHOOTING.md).

Feature experiments are in the [feature ledger](FEATURES.md), design decisions in the [ADR index](adr/README.md), and measured results in the [performance report](performance/README.md).

Terms used below:

- **CSI:** channel state information, the Wi-Fi channel measurement captured for each packet.
- **Subcarrier:** one narrow frequency bin inside the Wi-Fi channel.
- **HT20:** the 20 MHz 802.11n channel layout the detectors use.
- **AGC:** the radio's automatic gain control. ESPectre leaves it on, so features must not depend on signal scale.
- **CV:** coefficient of variation, the standard deviation divided by the mean.
- **pps:** CSI packets per second. *Accepted* pps is what capture delivers; *admitted* pps is what reaches the detector after temporal admission.

## Overview

ESPectre reads the amplitudes of a fixed set of subcarriers, turns them into scale-invariant signals, and feeds them to one of two detection profiles:

- **Lightweight Detection** (`lightweight`, `LightweightDetector`): the default, a two-feature statistical model.
- **High-Accuracy Detection** (`high_accuracy`, `HighAccuracyDetector`): a small neural network over eight features.

Example raw CSI amplitude windows for an empty room, a person standing still, and motion:

![CSI amplitude heatmaps for empty, static presence, and motion](web/assets/images/guides/csi-amplitude-heatmap.webp)

In short:

- AGC stays on.
- A fixed set of 12 subcarriers feeds turbulence and L1 displacement. Aggregated turbulence averages neighboring bins. Channel-shape features read the full 56-bin band.
- Lightweight combines `turb_autocorr` and `turb_iqr_over_mean_aggr` with fixed weights.
- High Accuracy uses eight scale-invariant features.

## Why two detection profiles

The two profiles trade cost against quality:

- **Lightweight** tracks two features and does less work per packet. It leaves more CPU and memory for the rest of the firmware, but is less accurate and generalizes less well to new rooms.
- **High Accuracy** tracks eight features and runs a small neural network. It costs more CPU and memory, detects better, and needs no quiet-room calibration because its threshold is trained.

In firmware that can switch profiles at runtime, choosing Lightweight reduces working memory and per-packet work, but the ML code and weights may stay in flash.

## Processing pipeline

```text
CSI packet
  -> fixed 12-subcarrier amplitudes
       -> CV turbulence (std / mean)
       -> optional Hampel / low-pass filtering
  -> 56-bin live complex profile
       -> Lightweight aggregated turbulence or ML L1 and trajectory trackers
  -> detector-specific metric or feature extraction
  -> thresholded motion state
```

At startup, Lightweight calibrates its threshold; High Accuracy starts from its trained threshold once its feature window is full. See [calibration summary](#calibration-summary).

## Detector timing

The detector runs on elapsed time, not on packet counts:

| Quantity | Production setting | At 100 pps |
| --- | --- | --- |
| Detector window | `1000 ms` | 100 samples |
| Evaluation interval | `250 ms` | time-based |
| CSI target rate | `100 pps` | one `10 ms` slot |
| Minimum valid occupancy | `70%` | at least 70 valid slots |
| ML L1 profile-displacement lag | `100 ms` | 10 slots |
| Turbulence autocorrelation lag | `10 ms` | 1 slot |

### Temporal admission

The runtime divides time into fixed slots derived from `csi_target_pps`, not from the measured packet rate:

- At most one packet per slot is admitted: the one closest to the slot center. The choice is final only when a packet arrives in a later slot, so a late but better packet is not lost.
- Two selected packets are at least half a slot apart. Other packets in the same slot count as *excess*.
- Duplicate, stale, and out-of-order timestamps are rejected. Wall-clock time is used only to reject packets that waited too long in the processing queue.
- Missing slots stay empty. Window statistics use the valid samples; lagged features need valid samples at the exact slot offsets.
- A gap as long as the window clears detector history at once.
- Detection is ready after one full window with at least 70% valid slots.

Changing the detector or starting calibration clears the window but keeps the slot grid. Only a real break in the CSI session starts a new grid. Changing the target rate or window rebuilds the detector; the measured rate never does.

Live sensing, replay, training, Python validation, and C++ replay all apply the same admission before feature processing. Replay data must carry trustworthy timestamps; packets without advancing timestamps add no evidence. See the [fixed temporal-admission ADR](adr/2026-08-15-use-fixed-temporal-csi-admission.md), [Wi-Fi and CSI lifecycle](ARCHITECTURE.md#shared-wi-fi-and-csi-lifecycle), [CSI collection](API.md#csi-collection), and the [ML training guide](ML_TRAINING.md).

### Window size

The window length and `csi_target_pps` set the number of slots:

```text
window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)
```

The model, replay gates, and published results all use `1000 ms`. Other values change feature timing and response time and are not covered by those results. Do not use the window to tune false positives or latency; use the threshold and motion hits instead. If a product needs another window, revalidate the detector and the C++/Python parity gates at that setting.

### Motion-hit filtering

The detector processes every admitted packet but evaluates only every `evaluation_interval_ms`. Each evaluation gives a raw `IDLE` or `MOTION` reading:

- `motion_on_hits` consecutive `MOTION` readings switch the state to `MOTION`.
- `motion_off_hits` consecutive `IDLE` readings switch it back.
- One reading that agrees with the current state resets the count.

Hits count evaluations, not windows. With the default 250 ms interval:

| Transition | Hits | Confirmation delay |
|------------|------|--------------------|
| `IDLE -> MOTION` | `4` | about `0.75-1.0 s` |
| `MOTION -> IDLE` | `3` | about `0.50-0.75 s` |

The shorter delay applies when the change lines up with an evaluation. The window and missing input can add more delay.

ESPHome, Native, and Matter expose and persist the hit counts. Telemetry is published once `ready_to_publish` is true and a client asks for it; see the [API reference](API.md).

## AGC-active normalization

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

## Fixed subcarrier set

Both detectors sample the same fixed 12-subcarrier set for their turbulence and L1-displacement features:

```text
[4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]
```

These bins are subcarriers `+/-4, +/-9, +/-14, +/-19, +/-24, +/-28`, and they assume the centered convention where bin `32` is DC. Classic-MAC parts deliver CSI in Espressif's native `0~31, -32~-1` order instead, so the capture path rotates those payloads before band selection; see [`csi_format.h`](../src/cpp/core/csi_format.h).

The active runtime no longer selects subcarriers for each session. This set is part of the current detector definition. The indices come from measured channel coherence, not a detection-metric search: motion perturbation stays coherent over about 10 subcarriers while quiet noise is nearly independent per tone, so spreading the selected tones across the band provides independent observations. For the full rationale behind the band and the count, see [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md).

### Bands for frequency-domain features

The 12-tone set is a sampling of the spectrum, and it serves the features that build a time series out of it. Aggregated turbulence averages a five-bin live-band neighborhood around each selected tone. Channel-shape features instead measure structure across frequency inside a single packet, so they read the full HT20 live band: bins `4..31` and `33..60`, the 56 subcarriers left after the guard bands and the DC null.

| Feature family | Band | Why |
| --- | --- | --- |
| Normal `turb_*`, `l1_delta_*` | 12 selected tones | builds a time series, where span buys independent looks |
| `turb_iqr_over_mean_aggr` | five-bin neighborhoods around the 12 selected tones | suppresses per-tone noise before building the turbulence series |
| `chan_shape_*` | 56 live bins | measures shape across frequency, which decimation would remove |

The split follows from what each family measures rather than from independent band choices. Historical frequency-coherence candidates remain host-only because they need live-bin pairs at fixed separations; production no longer pays that complex full-band cost.

Both runtimes use the same guard-band, DC-null, and adjacent-bin aggregation rules in [`csi_format.h`](../src/cpp/core/csi_format.h) and [`segmentation.py`](../tools/lib/segmentation.py). The ML channel-shape live band remains defined identically in [`ml_feature_trackers.h`](../src/cpp/core/ml_feature_trackers.h) and [`ml_feature_trackers.py`](../tools/lib/ml_feature_trackers.py).

Capture layouts and normalization are described in [normalization](CSI.md#normalization). For LLTF, the missing tones ±27 and ±28 stay zero in raw data; the detector copies them from the nearest live ±26 tone before extracting features.

## Signal conditioning

Optional filters operate on the scalar turbulence stream before detector evaluation.

### Hampel filter

Default: enabled (`window=7`, `threshold=5.0` MAD)

Hampel filtering feeds both `lightweight` and `high_accuracy`. It removes large outliers using the median absolute deviation:

```text
MAD = median(|x_i - median(x)|)
```

Packets that exceed the configured MAD-scaled deviation are replaced by the current window median.

### Low-pass filter

Default: disabled

The low-pass stage is a first-order Butterworth IIR filter applied to the turbulence signal before detector evaluation. Use the [troubleshooting guide](TROUBLESHOOTING.md) for the operational trade-off between false-positive reduction and responsiveness.

The current C++ implementations calculate low-pass coefficients against a nominal `100 Hz` sample rate. `lowpass_cutoff` has its nominal frequency meaning when the admitted stream follows that regular cadence. A different target or substantial missing-slot pattern changes the effective time scale, so treat that combination as an experiment and revalidate it.

## Lightweight implementation: LightweightDetector

`LightweightDetector` is the production non-ML path. It combines:

- lag-1 autocorrelation of the gain-invariant turbulence stream
- robust relative IQR of adjacent-bin aggregated turbulence
- a fixed, weighted logistic fusion with no voting branches

### Turbulence autocorrelation

Per-packet turbulence is the spatial coefficient of variation:

```text
t_i = std(A_i) / mean(A_i)
```

After Hampel filtering, Lightweight calculates lag-1 autocorrelation over the turbulence window. This input is invariant under ideal uniform scaling because the coefficient of variation is itself a ratio. The shared `hampel_enabled` setting still controls the turbulence filter in both runtimes, and the same filtered turbulence stream feeds the ML `turb_*` features.

Lightweight does not allocate or update an L1-delta tracker. The tracker remains conditional on the exported feature IDs in ML, where `l1_delta_lag_ratio` consumes it. Current runtime-state ownership and measured feature costs are recorded in the [feature ledger](FEATURES.md).

### Aggregated turbulence IQR

Lightweight's second input reuses the same `W=5` adjacent-magnitude aggregation as ML. Each selected tone is replaced by the mean amplitude of its five-bin live-band neighborhood, with the DC null skipped and edge windows clamped to bins 4–60. Spatial turbulence is then computed as `std/mean` and filtered into a dedicated ring.

```text
turb_iqr_over_mean_aggr = (Q75(x_aggr) - Q25(x_aggr)) / max(abs(mean(x_aggr)), 1e-6)
```

The robust spread is dimensionless and gain-invariant. Lightweight maintains one additional window-sized float ring plus its Hampel and low-pass state, but it no longer extracts complex full-band coherence. The packet magnitude frame is computed once and shared by the normal and aggregated turbulence paths.

### Weighted fusion

Lightweight standardizes `turb_autocorr` and `turb_iqr_over_mean_aggr` with fixed training statistics, applies a two-term linear model, and converts its logit to a probability:

```text
logit = b + w_ac * z(turb_autocorr) + w_iqr * z(turb_iqr_over_mean_aggr)
probability = 1 / (1 + exp(-logit))
motion = probability > threshold
```

The coefficients come from grouped, de-overlapped out-of-fold training balanced by class, chip, and session. The global operating point is then selected on sequential production replay because a dense-window OOF false-positive rate does not encode the empty-room alarm budget. Current results and alarm gates live in the generated [performance report](performance/README.md). The runtime contains no majority vote or recovery branch in the score itself; all runtime adaptation happens at the threshold.

### Startup threshold calibration

At startup, Lightweight begins from the validated global probability threshold and shifts its logit using the session's startup `q95` relative to the training idle reference. The shift applies `50%` of the observed session-to-training offset:

```text
adapted_logit = logit(base_threshold) +
                0.5 * (startup_q95 - train_idle_q95)
threshold = sigmoid(adapted_logit)
```

Only the first `64` ready evaluations contribute startup evidence. This keeps the learned two-feature boundary intact while letting the threshold follow a session whose quiet baseline starts above or below the training reference. Runtime adjustments stay on the same `0.0-1.0` probability scale and remain active until recalibration or reboot.

The settled-level rule cannot create a high threshold. It only ever lowers one after a long quiet dwell, so any threshold that lands near `1.0` came from the startup `q95` shift, not from later recovery.

### Known limits

Lightweight is less robust than High Accuracy in quiet rooms and in rooms it was not trained on. Per-chip results are in the [performance report](performance/README.md). No additional feature pair or triplet is approved for Lightweight on the current corpus; see the [feature ledger](FEATURES.md).

### Settled-level threshold recovery

The detector revisits the threshold once a session proves itself quieter than its own opening. Every `20` evaluations it records the maximum metric logit in that block, keeps the last `12` blocks, and once the ring is full compares the median of those maxima against the live threshold. If that level plus `LIGHTWEIGHT_SETTLE_MARGIN_LOGITS` sits below the threshold, the threshold drops to it. The shared runtime reports that control-plane change through `on_threshold_changed`; frontend and transport propagation are documented in [runtime contract](ARCHITECTURE.md#runtime-contract) and [MQTT topics](API.md#mqtt-topics).

The recovery has these safeguards:

- It only lowers the threshold, so recovery cannot hide motion that the calibrated threshold would have caught.
- Real activity raises the block maxima above the current threshold and prevents a change. A decrease requires a long quiet stretch.
- The candidate is the median of block maxima. One spike or one quiet block cannot move it.

The current `20`-evaluation blocks, `12`-block ring, and `2.7`-logit margin produce a `60 s` dwell at the nominal cadence. The recovery design and current operating point live in the [settled-level recovery ADR](adr/2026-07-26-recover-the-startup-threshold-once-a-session-settles.md). The temporal-admission contract that prompted the `2.7` revalidation is recorded in the [fixed temporal-admission ADR](adr/2026-08-15-use-fixed-temporal-csi-admission.md).

Its limit is the mirror of its safety. A room that grows genuinely noisier after the threshold has come down cannot push it back up; only a recalibration does that.

### Implementation status

Current aligned implementations:

- `tools/lib/lightweight_detector.py`
- `src/cpp/core/lightweight_detector.*`

## High-Accuracy implementation: HighAccuracyDetector

`HighAccuracyDetector` is the production neural detector. It treats motion detection as a binary classification problem over a sliding window and outputs a probability in the range `0.0-1.0`.

Current threshold:

```text
motion if probability > 0.5
```

High-Accuracy Detection skips startup threshold calibration. Detection begins after CSI is ready and the feature window has filled.

### Current runtime topology

The production export is a compact MLP:

```text
Input (8 features)
  -> Dense(24, ReLU)
  -> Dense(12, ReLU)
  -> Dense(1, Sigmoid)
```

Total parameter count: 529

The runtime accepts exported hidden-layer layouts generated by the training script, but the committed production artifact currently uses the topology above.

### Production feature set

The production model consumes these eight scale-invariant inputs, in export order:

1. `turb_iqr_over_mean_aggr`
2. `turb_autocorr`
3. `turb_zcr`
4. `l1_delta_lag_ratio`
5. `chan_shape_spread_subband`
6. `chan_shape_coherent_innovation_energy`
7. `chan_shape_excess_path`
8. `chan_shape_subband_kendall_lag_excess`

Every member is a gain-invariant ratio, correlation, crossing rate, or normalized channel-shape geometry. The exact definitions, physical interpretations, implementation locations, retained metrics, and candidate-admission rules live in the [feature ledger](FEATURES.md).

The first three inputs come from the normal and adjacent-bin aggregated turbulence streams, the fourth comes from normalized profile displacement, and the final four share one physical-time channel-trajectory tracker. Packet timestamps preserve the trajectory scale through rate changes and loss. Consecutive identical CSI payloads contribute no additional profiles, but their timestamps still advance the window and expire old trajectory features. Runtime state remains conditional on the exported feature IDs, so superseded features do not retain inactive trackers. [current production ML set](FEATURES.md#current-production-ml-set) owns the exact formulas, physical interpretations, storage representation, implementation locations, and retained evidence.

### Inference flow

```text
CSI packet
  -> turbulence path
  -> optional filters
  -> sliding window
  -> scale-invariant feature extraction
  -> MLP inference
  -> probability threshold at 0.5
```

### Runtime alignment

The same production feature set is used by:

- `tools/lib/csi_features.py`
- `src/cpp/core/ml_*`
- `tools/train_ml_model.py` exports

## Calibration summary

| Detection profile | Threshold | Startup behavior |
|----------|-----------|------------------|
| `lightweight` | automatic, session-adjustable | motion-first completion with quiet-first fallback inside the valid evidence budget; applies session `q95` logit adaptation |
| `high_accuracy` | trained default, session-adjustable | no threshold calibration; starts once CSI is active and its feature window has filled |

Lightweight calibration uses up to 10 seconds of valid input after the detector becomes ready:

- Stay quiet right after boot. Movement during this first quiet phase lowers calibration quality.
- A clean `quiet -> motion -> quiet` pattern can finish calibration early. This is optional; otherwise calibration falls back to a quiet-only estimate within the same 10 seconds.
- The 10 seconds count valid slots, not wall-clock time. Missing or bursty input makes calibration take longer, and a window-long gap restarts it.

Both profiles use the same fixed subcarrier set and temporal-admission contract. Their feature extraction, working state, readiness gates, motion metric, and threshold-calibration behavior differ.

## References

Published research and how it applies to ESPectre are in the [literature review](LITERATURE.md).
