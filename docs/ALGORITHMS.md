# Algorithms

Current detector and signal-processing reference for ESPectre.

This file documents only algorithms active in the current project surface. Feature experiments and promotion evidence live in [FEATURES.md](FEATURES.md), decision rationale lives in [adr/](adr/), and mutable detector metrics live in the generated [performance report](performance/README.md).

This reference is for detector and firmware contributors. Operators normally need [TROUBLESHOOTING.md](TROUBLESHOOTING.md), which turns these mechanisms into practical settings.

Terms used throughout this document:

- **CSI:** channel state information, the complex Wi-Fi channel measurement captured for each packet.
- **Subcarrier:** one narrow frequency bin inside the Wi-Fi channel.
- **HT20:** the supported 20 MHz 802.11n channel layout.
- **AGC:** automatic gain control in the radio; ESPectre keeps it active and therefore favors scale-invariant features.
- **CV:** coefficient of variation, standard deviation divided by the mean.
- **pps:** CSI packets per second. Raw accepted pps is capture supply; admitted pps is the detector input after temporal slot admission.

## Overview

ESPectre detects motion from Wi-Fi CSI by extracting a small, fixed slice of subcarriers, deriving gain-robust scalar signals from those amplitudes, and passing those signals to one of two production profiles:

- **Lightweight Detection** (`lightweight`), implemented by `LightweightDetector`, the default non-ML detector
- **High-Accuracy Detection** (`high_accuracy`), implemented by `HighAccuracyDetector`, the neural detector with a trained probability threshold

Representative raw CSI amplitude windows for empty room, static presence, and motion:

![CSI amplitude heatmaps for empty, static presence, and motion](web/assets/images/guides/csi-amplitude-heatmap.webp)

The current production detector definition is:

- AGC stays active
- the shared fixed 12-subcarrier set feeds turbulence and L1 displacement, adjacent live bins feed aggregated turbulence, and channel-shape features read the full 56-bin live band
- the Lightweight path uses weighted `turb_autocorr + turb_iqr_over_mean_aggr` fusion
- the Lightweight runtime has no voting branch or legacy low-RSSI blend term
- the High-Accuracy path uses the compact eight-feature scale-invariant production set

## Why Two Detection Profiles

Lightweight and High Accuracy are both production paths because they optimize different constraints.

- **Lightweight Detection minimizes active detector cost.** It uses two scalar feature streams, does not allocate the ML-only L1 and trajectory state, and performs less per-packet work. This leaves more CPU time and working memory for constrained chips or products in which sensing is only one firmware feature. The trade-off is lower accuracy and weaker generalization than High Accuracy on the maintained corpus.
- **High-Accuracy Detection prioritizes detection quality.** Its ML implementation maintains eight production features and runs a compact neural network, increasing memory and computation while improving accuracy and transfer across recorded environments. Its trained threshold also removes Lightweight's initial quiet-room calibration.

Lightweight calibration consumes up to 10 seconds of temporally valid, ready CSI coverage after temporal warmup. It can complete early when the motion-first gate observes a stable quiet anchor, sustained motion, and a return to quiet; otherwise it uses the quiet-first fallback within the same evidence budget. Missing slots can extend the wall-clock duration because an invalid window does not consume the budget. High Accuracy skips threshold calibration but still waits for CSI readiness and enough samples to fill its feature window. In images that support runtime profile switching, choosing Lightweight reduces active working state and per-packet detector work; it does not necessarily remove ML code or weights from flash.

## Processing Pipeline

Steady-state detector flow:

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

At boot:

- `lightweight` performs startup threshold calibration
- `high_accuracy` starts from its trained default threshold once CSI capture is active and its feature window has filled

With the default `1000 ms` detector window and `100 pps` target, the `lightweight` startup budget is ten seconds of valid equivalent slot coverage after the detector first becomes ready. Missing slots do not become synthetic packets, same-slot bursts do not advance calibration, and a contaminating window-sized gap restarts it. Successful motion-first calibration can finish before the budget is exhausted; quiet-first fallback must converge within it. The budget measures admissible evidence, not a wall-clock timeout.

## Detector Timing

The deployed detector uses a time-relative evaluation cadence and fixed feature geometry:

| quantity | production setting | nominal interpretation at 100 pps |
| --- | --- | --- |
| detector window | `1000 ms` | 100 samples |
| evaluation interval | `250 ms` | time-based, not packet-count driven |
| CSI temporal target | `100 pps` | one `10 ms` slot |
| minimum valid occupancy | `70%` | at least 70 valid slots |
| ML L1 profile-displacement lag | derived from `100 ms` | 10 slots |
| turbulence autocorrelation lag | derived from `10 ms` | 1 slot |

The runtime derives fixed slots from `csi_target_pps`, not from measured arrival rate. It admits at most one packet per slot, retaining the candidate nearest the ideal slot center until a later slot is observed. The minimum distance between consecutive selected candidates is half a target slot and is derived from `csi_target_pps`; other same-slot candidates count as excess. Duplicate, stale, and out-of-order timestamps are rejected, and a gap spanning the configured window clears detector history immediately even while the first post-gap candidate stays pending. Detector changes and calibration boundaries clear admitted window data but preserve the active timestamp-grid phase; only a true CSI session discontinuity starts a new temporal epoch. Missing slots remain invalid in feature rings: window statistics consume valid samples, while adjacent and lagged features require valid samples at the exact configured slot offsets. Detection becomes ready after a complete temporal window with at least seven tenths valid occupancy. See the [fixed temporal-admission ADR](adr/2026-08-15-use-fixed-temporal-csi-admission.md).

Calibration and steady-state detection share one cadence. Both paths evaluate admitted packets on the same schedule.

The detector instance, its slot capacity, and startup calibration remain stable under ordinary delivery jitter. A target or window configuration change is an explicit lifecycle boundary; measured receive rate is diagnostic only and never reconstructs a detector. Live sensing, collector-derived sensing, replay, training, Python validation, and C++ integration replay all apply temporal admission before feature processing. Runtime placement and raw-collection behavior are documented in [ARCHITECTURE.md](ARCHITECTURE.md#shared-wi-fi-and-csi-lifecycle) and [API.md](API.md#csi-collection).

Cadence advances on admitted packet timestamps, never on the loop clock or a packet-count fallback. A live slot is closed by observing a packet in a later timestamp slot, not merely because wall-clock time passed, so a delayed but better candidate is not discarded. Wall-clock time is used only to reject processing-backlog staleness. Live input and binding replay datasets must provide trustworthy timestamps and target provenance; missing or non-advancing timestamps contribute no evidence.

The rest of the replay contract mirrors this cadence and reset behavior; see [ML_TRAINING.md](ML_TRAINING.md).

The detector window setting is elapsed time. Together with `csi_target_pps`, it defines the fixed slot count:

```text
window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)
```

The production model, replay gates, training workflow, and published performance evidence use `1000 ms`. Other supported values change feature geometry and response time but are not covered by those published results. Do not use the window as a routine false-positive or latency control; prefer threshold and hit filtering. If a product needs a different window, validate the selected detector profile and the C++/Python parity gates at that setting.

### Motion-hit filtering

The detector processes every admitted CSI packet into its sliding window, but it evaluates and publishes on the coarser `evaluation_interval_ms` cadence. Packet timestamps drive that cadence; there is no packet-count fallback, so live input and supported replay datasets must provide advancing timestamps.

Each evaluation produces a raw `IDLE` or `MOTION` reading. The runtime requires `motion_on_hits` consecutive opposing readings before publishing `MOTION`, and `motion_off_hits` consecutive readings before returning to `IDLE`. One reading in the current published state clears the pending count. These hits are evaluation ticks, not detector windows.

For a sustained threshold crossing, the nominal hit-filter delay includes the wait for the first evaluation plus the remaining confirmation ticks. At regular 250 ms intervals, the default ranges are:

| Transition | Hits | Confirmation latency |
|------------|------|----------------------|
| `IDLE -> MOTION` | `4` | about `0.75-1.0 s` |
| `MOTION -> IDLE` | `3` | about `0.50-0.75 s` |

The lower bound applies when the crossing aligns with an evaluation tick; the upper bound applies when it begins just after one. These ranges describe hit filtering only. Feature-window response and missing valid coverage can add delay between physical movement and the published state.

ESPHome, Native, and Matter expose persisted runtime hit controls through their advertised surfaces. Telemetry is available on detector evaluations once `ready_to_publish` is true and a frontend consumer requests it. [API.md](API.md) describes the published resources and events.

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

These bins are subcarriers `+/-4, +/-9, +/-14, +/-19, +/-24, +/-28`, and they assume the centered convention where bin `32` is DC. Classic-MAC parts deliver CSI in Espressif's native `0~31, -32~-1` order instead, so the capture path rotates those payloads before band selection; see [`csi_format.h`](../src/cpp/core/csi_format.h).

The active runtime no longer selects subcarriers for each session. This set is part of the current detector definition. The indices come from measured channel coherence, not a detection-metric search: motion perturbation stays coherent over about 10 subcarriers while quiet noise is nearly independent per tone, so spreading the selected tones across the band provides independent observations. For the full rationale behind the band and the count, see [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md).

### Bands For Frequency-Domain Features

The 12-tone set is a sampling of the spectrum, and it serves the features that build a time series out of it. Aggregated turbulence averages a five-bin live-band neighborhood around each selected tone. Channel-shape features instead measure structure across frequency inside a single packet, so they read the full HT20 live band: bins `4..31` and `33..60`, the 56 subcarriers left after the guard bands and the DC null.

| Feature family | Band | Why |
| --- | --- | --- |
| Normal `turb_*`, `l1_delta_*` | 12 selected tones | builds a time series, where span buys independent looks |
| `turb_iqr_over_mean_aggr` | five-bin neighborhoods around the 12 selected tones | suppresses per-tone noise before building the turbulence series |
| `chan_shape_*` | 56 live bins | measures shape across frequency, which decimation would remove |

The split follows from what each family measures rather than from independent band choices. Historical frequency-coherence candidates remain host-only because they need live-bin pairs at fixed separations; production no longer pays that complex full-band cost.

Both runtimes use the same guard-band, DC-null, and adjacent-bin aggregation rules in [`csi_format.h`](../src/cpp/core/csi_format.h) and [`segmentation.py`](../tools/lib/segmentation.py). The ML channel-shape live band remains defined identically in [`ml_feature_trackers.h`](../src/cpp/core/ml_feature_trackers.h) and [`ml_feature_trackers.py`](../tools/lib/ml_feature_trackers.py).

Production detectors consume a canonical centered 64-subcarrier, 20 MHz view. [CSI.md](CSI.md#normalization) defines capture layouts and normalization. For LLTF, raw capture keeps the unavailable physical tones ±27 and ±28 at zero; the private detector view copies I/Q from the nearest live ±26 tone before feature extraction. The current detection corpus validates 2.4 GHz HT20 with HT-LTF; 5 GHz VHT20 detection quality remains uncharacterized.

## Signal Conditioning

Optional filters operate on the scalar turbulence stream before detector evaluation.

### Hampel Filter

Default: enabled (`window=7`, `threshold=5.0` MAD)

Hampel filtering feeds both `lightweight` and `high_accuracy`. It removes large outliers using the median absolute deviation:

```text
MAD = median(|x_i - median(x)|)
```

Packets that exceed the configured MAD-scaled deviation are replaced by the current window median.

### Low-Pass Filter

Default: disabled

The low-pass stage is a first-order Butterworth IIR filter applied to the turbulence signal before detector evaluation. Use [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the operational trade-off between false-positive reduction and responsiveness.

The current C++ implementations calculate low-pass coefficients against a nominal `100 Hz` sample rate. `lowpass_cutoff` has its nominal frequency meaning when the admitted stream follows that regular cadence. A different target or substantial missing-slot pattern changes the effective time scale, so treat that combination as an experiment and revalidate it.

## Lightweight Implementation: LightweightDetector

`LightweightDetector` is the production non-ML path. It combines:

- lag-1 autocorrelation of the gain-invariant turbulence stream
- robust relative IQR of adjacent-bin aggregated turbulence
- a fixed, weighted logistic fusion with no voting branches

### Turbulence Autocorrelation

Per-packet turbulence is the spatial coefficient of variation:

```text
t_i = std(A_i) / mean(A_i)
```

After Hampel filtering, Lightweight calculates lag-1 autocorrelation over the turbulence window. This input is invariant under ideal uniform scaling because the coefficient of variation is itself a ratio. The shared `hampel_enabled` setting still controls the turbulence filter in both runtimes, and the same filtered turbulence stream feeds the ML `turb_*` features.

Lightweight does not allocate or update an L1-delta tracker. The tracker remains conditional on the exported feature IDs in ML, where `l1_delta_lag_ratio` consumes it. Current runtime-state ownership and measured feature costs are recorded in [FEATURES.md](FEATURES.md).

### Aggregated Turbulence IQR

Lightweight's second input reuses the same `W=5` adjacent-magnitude aggregation as ML. Each selected tone is replaced by the mean amplitude of its five-bin live-band neighborhood, with the DC null skipped and edge windows clamped to bins 4–60. Spatial turbulence is then computed as `std/mean` and filtered into a dedicated ring.

```text
turb_iqr_over_mean_aggr = (Q75(x_aggr) - Q25(x_aggr)) / max(abs(mean(x_aggr)), 1e-6)
```

The robust spread is dimensionless and gain-invariant. Lightweight maintains one additional window-sized float ring plus its Hampel and low-pass state, but it no longer extracts complex full-band coherence. The packet magnitude frame is computed once and shared by the normal and aggregated turbulence paths.

### Weighted Fusion

Lightweight standardizes `turb_autocorr` and `turb_iqr_over_mean_aggr` with fixed training statistics, applies a two-term linear model, and converts its logit to a probability:

```text
logit = b + w_ac * z(turb_autocorr) + w_iqr * z(turb_iqr_over_mean_aggr)
probability = 1 / (1 + exp(-logit))
motion = probability > threshold
```

The coefficients come from grouped, de-overlapped out-of-fold training balanced by class, chip, and session. The global operating point is then selected on sequential production replay because a dense-window OOF false-positive rate does not encode the empty-room alarm budget. Current results and alarm gates live in the generated [performance report](performance/README.md). The runtime contains no majority vote or recovery branch in the score itself; all runtime adaptation happens at the threshold.

Startup adaptation thresholds this fitted two-feature logit directly. The older low-RSSI L1 blend path is retired; it is not part of the current detector surface.

### Startup Threshold Calibration

At startup, Lightweight begins from the validated global probability threshold and shifts its logit using the session's startup `q95` relative to the training idle reference. The shift applies `50%` of the observed session-to-training offset:

```text
adapted_logit = logit(base_threshold) +
                0.5 * (startup_q95 - train_idle_q95)
threshold = sigmoid(adapted_logit)
```

Only the first `64` ready evaluations contribute startup evidence. This keeps the learned two-feature boundary intact while letting the threshold follow a session whose quiet baseline starts above or below the training reference. Runtime adjustments stay on the same `0.0-1.0` probability scale and remain active until recalibration or reboot.

The settled-level rule cannot create a high threshold. It only ever lowers one after a long quiet dwell, so any threshold that lands near `1.0` came from the startup `q95` shift, not from later recovery.

### Known Limits

Lightweight has weaker quiet-room and held-out generalization than High Accuracy on the maintained corpus. The generated [performance report](performance/README.md) owns the current per-chip, weak-link, occupancy, false-positive, and alarm results.

Use High-Accuracy Detection where accuracy, quiet-room robustness, or held-out generalization matters more than the additional runtime cost. Use Lightweight Detection when CPU and working-memory headroom are the stronger product constraint. The active Lightweight feature-selection record lives in `FEATURES.md`; no additional pair or triplet is approved for export on the current corpus.

### Settled-Level Threshold Recovery

The detector revisits the threshold once a session proves itself quieter than its own opening. Every `20` evaluations it records the maximum metric logit in that block, keeps the last `12` blocks, and once the ring is full compares the median of those maxima against the live threshold. If that level plus `LIGHTWEIGHT_SETTLE_MARGIN_LOGITS` sits below the threshold, the threshold drops to it. The shared runtime reports that control-plane change through `on_threshold_changed`; frontend and transport propagation are documented in [ARCHITECTURE.md](ARCHITECTURE.md#runtime-contract) and [API.md](API.md#mqtt).

The recovery has these safeguards:

- It only lowers the threshold, so recovery cannot hide motion that the calibrated threshold would have caught.
- Real activity raises the block maxima above the current threshold and prevents a change. A decrease requires a long quiet stretch.
- The candidate is the median of block maxima. One spike or one quiet block cannot move it.

The current `20`-evaluation blocks, `12`-block ring, and `2.7`-logit margin produce a `60 s` dwell at the nominal cadence. The recovery design and current operating point live in the [settled-level recovery ADR](adr/2026-07-26-recover-the-startup-threshold-once-a-session-settles.md). The temporal-admission contract that prompted the `2.7` revalidation is recorded in the [fixed temporal-admission ADR](adr/2026-08-15-use-fixed-temporal-csi-admission.md).

Its limit is the mirror of its safety. A room that grows genuinely noisier after the threshold has come down cannot push it back up; only a recalibration does that.

### Implementation Status

Current aligned implementations:

- `tools/lib/lightweight_detector.py`
- `src/cpp/core/lightweight_detector.*`

## High-Accuracy Implementation: HighAccuracyDetector

`HighAccuracyDetector` is the production neural detector. It treats motion detection as a binary classification problem over a sliding window and outputs a probability in the range `0.0-1.0`.

Current threshold:

```text
motion if probability > 0.5
```

High-Accuracy Detection skips startup threshold calibration. Detection begins after CSI is ready and the feature window has filled.

### Current Runtime Topology

The production export is a compact MLP:

```text
Input (8 features)
  -> Dense(24, ReLU)
  -> Dense(12, ReLU)
  -> Dense(1, Sigmoid)
```

Total parameter count: 529

The runtime accepts exported hidden-layer layouts generated by the training script, but the committed production artifact currently uses the topology above.

### Production Feature Set

The production model consumes these eight scale-invariant inputs, in export order:

1. `turb_iqr_over_mean_aggr`
2. `turb_autocorr`
3. `turb_zcr`
4. `l1_delta_lag_ratio`
5. `chan_shape_spread_subband`
6. `chan_shape_coherent_innovation_energy`
7. `chan_shape_excess_path`
8. `chan_shape_subband_kendall_lag_excess`

Every member is a gain-invariant ratio, correlation, crossing rate, or normalized channel-shape geometry. The exact definitions, physical interpretations, implementation locations, retained metrics, and candidate-admission rules live in [FEATURES.md](FEATURES.md).

The first three inputs come from the normal and adjacent-bin aggregated turbulence streams, the fourth comes from normalized profile displacement, and the final four share one physical-time channel-trajectory tracker. Packet timestamps preserve the trajectory scale through rate changes and loss. Consecutive identical CSI payloads contribute no additional profiles, but their timestamps still advance the window and expire old trajectory features. Runtime state remains conditional on the exported feature IDs, so superseded features do not retain inactive trackers. [FEATURES.md](FEATURES.md#current-production-ml-set) owns the exact formulas, physical interpretations, storage representation, implementation locations, and retained evidence.

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

- `tools/lib/csi_features.py`
- `src/cpp/core/ml_*`
- `tools/train_ml_model.py` exports

## Calibration Summary

| Detection profile | Threshold | Startup behavior |
|----------|-----------|------------------|
| `lightweight` | automatic, session-adjustable | motion-first completion with quiet-first fallback inside the valid evidence budget; applies session `q95` logit adaptation |
| `high_accuracy` | trained default, session-adjustable | no threshold calibration; starts once CSI is active and its feature window has filled |

Lightweight startup uses up to 10 seconds of valid, ready coverage after temporal warmup. A clean `quiet -> motion -> quiet` pattern can finish earlier. For Lightweight, stay quiet immediately after boot. After the first quiet phase, one short movement may complete startup early, but it is optional. Repeated movement during the initial quiet phase still reduces calibration quality. Missing or burst-concentrated slots extend the wall-clock duration because they do not count as valid evidence.

Micro-ESPectre bounds the calibration attempt in wall-clock time: it aborts after 15 seconds without admitted CSI or after `max(15000, 2 * CALIBRATION_DURATION_MS + SEGMENTATION_WINDOW_SIZE_MS)` milliseconds overall, even if sparse packets continue arriving. Its deployment settings are in [README.md](../src/python/micro_espectre/README.md#runtime-behavior).

Both profiles use the same fixed subcarrier set and temporal-admission contract. Their feature extraction, working state, readiness gates, motion metric, and threshold-calibration behavior differ.

## References

See [LITERATURE.md](LITERATURE.md) for the paper index, publication dates, reported preprocessing, algorithms, results, hardware assumptions, and ESPectre transferability notes. This file retains only the active algorithm definition.
