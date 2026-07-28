# Algorithms

Current detector and signal-processing reference for ESPectre.

This file documents the algorithms that are active in the current project
surface. Historical promotion rationale, superseded baselines, and longer
decision context now live in ADRs, especially:

- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [`2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`](adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md)
- [`2026-07-07-use-core-6-as-the-production-ml-feature-set.md`](adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](adr/2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`](adr/2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md)
- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [`2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md`](adr/2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md)
- [`2026-07-26-recover-the-startup-threshold-once-a-session-settles.md`](adr/2026-07-26-recover-the-startup-threshold-once-a-session-settles.md)
- [`2026-07-25-gate-classic-false-positives-on-empty-rooms.md`](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md)
- [`2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md`](adr/2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md)

## Overview

ESPectre detects motion from Wi-Fi CSI by extracting a small, fixed slice of
subcarriers, deriving gain-robust scalar signals from those amplitudes, and
feeding those signals into either:

- `ClassicDetector`, the default non-ML detector
- `MLDetector`, the neural detector with a trained probability threshold

Representative raw CSI amplitude windows for empty room, static presence, and
motion:

![CSI amplitude heatmaps for empty, static presence, and motion](web/guides/images/csi-amplitude-heatmap.webp)

The current production detector definition is:

- AGC stays active
- the shared fixed 12-subcarrier set is used
- the classic path uses weighted L1-delta and turbulence-autocorrelation fusion
- the classic runtime has no voting or variance-recovery branch
- the ML path uses the five scale-invariant production features

## Processing Pipeline

Steady-state detector flow:

```text
CSI packet
  -> fixed 12-subcarrier amplitudes
  -> CV turbulence (std / mean)
  -> optional Hampel / low-pass filtering
  -> detector-specific metric or feature extraction
  -> thresholded motion state
```

At boot:

- `classic` performs startup threshold calibration
- `ml` starts as soon as CSI capture is active from its trained default threshold

With the default `window_size=100`, the `classic` startup budget is
`10 x window_size = 1000` packets. This is a maximum, not a mandatory wait.

## Detector Timing

The detector's contract is expressed in microseconds, not packets, and resolved
into packet counts from the cadence the stream actually delivers:

| quantity | duration | packets at 100 pps |
| --- | --- | --- |
| detector window | `1 s` | 100 |
| evaluation interval | `250 ms` | 25 |
| L1 profile-displacement lag | `100 ms` | 10 |
| turbulence autocorrelation lag | `10 ms` | 1 |

A packet count only means what it is supposed to mean at exactly `100 pps`, and
real streams do not run there: the recorded corpus spans `90` to `120`, and
ESP32 tops out near `70-80 pps` delivered in bursts. `derive_detector_timing()`
resolves the table above from the measured cadence, and it is defined once per
language in [runtime_policy.py](../src/python/micro_espectre/runtime_policy.py)
and [detector_timing.h](../src/cpp/core/detector_timing.h).

Two rules govern the resolution, and they are deliberately asymmetric:

- **The lags follow physical time.** They measure how far the channel moved over
  an interval, so they have to track that interval. This dominates at high
  rates: at `1000 pps` a lag-1 autocorrelation spans under a millisecond, where
  consecutive packets are nearly identical, and the feature leaves the range its
  coefficients were fitted over.
- **The window follows sample count.** Its features are estimator averages, so
  what matters is how many samples they average. Holding a one-second span at
  `25 pps` leaves 25 samples, the estimates get noisier, startup calibration
  raises the threshold to hold false positives, and recall collapses instead.

Inside `+/-25%` of the nominal cadence nothing adapts: rounding a duration into
packets flips between neighbouring counts across streams that are all
essentially nominal, which costs feature homogeneity for no gain.

Cadence advances on the packet arrival timestamp, never on the loop clock. The
loop clock measures how fast packets are processed, which matches arrival on
hardware but not on replay, and would let host scheduling reach a detector
decision. Wall-clock time is reserved for staleness detection, which arrival
time cannot do because a dead stream delivers no timestamps. Sources with no
arrival timestamp fall back to counting packets.

See
[2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md](adr/2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md)
for the measurements behind each rule.

## AGC-Active Normalization

The shared turbulence signal is:

```text
turbulence = std(amplitudes) / mean(amplitudes)
```

This coefficient-of-variation form is gain-invariant:

```text
CV(kA) = std(kA) / mean(kA) = std(A) / mean(A)
```

If AGC scales all amplitudes by a factor `k`, turbulence stays unchanged. This
same AGC-active normalization model is used across:

- runtime detection
- host collection
- dataset schema
- offline ML tooling

## Fixed Subcarrier Set

Both detectors use the same fixed 12-subcarrier set:

```text
[4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]
```

These bins are subcarriers `+/-4, +/-9, +/-14, +/-19, +/-24, +/-28`, and they
assume the centered convention where bin `32` is DC. Classic-MAC parts deliver
CSI in Espressif's native `0~31, -32~-1` order instead, so the capture path
rotates those payloads before band selection; see
[`csi_format.h`](../src/cpp/core/csi_format.h) and
[`device_utils.py`](../src/python/micro_espectre/device_utils.py).

The active runtime no longer performs per-session runtime subcarrier selection.
This set is part of the detector definition for the current project surface.
The indices come from measured channel coherence rather than from a
detection-metric search: the motion perturbation stays coherent over about 10
subcarriers while quiet noise is nearly per-tone independent, so span is what
buys independent looks. For the full rationale behind the band and the count,
see
[`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md).

Why HT20 stays the preferred active contract:

- it gives the project one centered, already validated 64-subcarrier sensing
  view shared by runtime detection, offline validation, and ML training
- in normal modern 2.4 GHz deployments, `802.11n` support is effectively
  ubiquitous, so standardizing on `HT20` is usually a practical constraint
  rather than a deployment blocker
- the fixed 12-tone band sits inside the HT-LTF data-bearing region with
  explicit guard-band and DC-null margins
- newer VHT and HE layouts matter for PHY provenance, but they do not by
  themselves justify switching production Classic or ML onto grouped or
  virtual-subcarrier assumptions

![Legacy, HT, VHT, and HE LTF placement compared on the same 20 MHz slice](web/guides/images/ht20-ltf-layout-preferred.png)
*HT20 is the stable, validated sensing view ESPectre standardizes on today: it
matches the current detector contract directly, is broadly available on modern
2.4 GHz networks, and keeps the active sensing surface simple. Wider or newer
PHY layouts are still recorded through per-record provenance, but they need
separate evidence before they become the production baseline.*

Non-HT20 payloads are normalized onto the same internal 64-subcarrier HT20
index grid before fixed-subcarrier extraction. Short layouts are centered so
the HT20 midpoint remains aligned.

| Input case | Raw layout | Mapping to HT20 | Output |
|------------|------------|-----------------|--------|
| Native HT20 | `128 B = 64 SC` | pass-through | `64 SC / 128 B` |
| Short HT estimate | `114 B = 57 SC` | zero-pad `4` SC left, copy `57` SC, zero-pad `3` SC right | `64 SC / 128 B` |
| Double HT20 payload | `256 B = 2 x 64 SC` | collapse to one `128 B` half | `64 SC / 128 B` |
| Double short HT estimate | `228 B = 2 x 57 SC` | collapse to one `57 SC` half, then pad `4` left and `3` right | `64 SC / 128 B` |

## Signal Conditioning

Optional filters operate on the scalar turbulence stream before detector
evaluation.

### Hampel Filter

Default: enabled (`window=7`, `threshold=5.0` MAD)

The Hampel filter removes large outliers using the median absolute deviation:

```text
MAD = median(|x_i - median(x)|)
```

Packets that exceed the configured MAD-scaled deviation are replaced by the
current window median.

### Low-Pass Filter

Default: disabled

The low-pass stage is a first-order Butterworth IIR filter applied to the
turbulence signal before detector evaluation. Use [`TUNING.md`](TUNING.md) for
the operational trade-off between false-positive reduction and responsiveness.

## Classic Detector

`ClassicDetector` is the production non-ML path. It combines:

- lag-ratio of mean L1 displacement between normalized amplitude profiles
- lag-1 autocorrelation of the gain-invariant turbulence stream
- a fixed, weighted logistic fusion with no voting branches

### L1-Delta Primary Metric

Per packet, ESPectre computes a mean-normalized amplitude profile over the fixed
12-subcarrier set:

```text
p_i[k] = A_i[k] / mean(A_i)
```

It then compares the current profile with the one seen `lag` packets earlier:

```text
d_i = mean_k |p_i[k] - p_(i-lag)[k]|
```

Default `lag = 10`, which is roughly 100 ms at 100 packets per second.

The same comparison also runs at `lag = 1`, and Classic's feature is the ratio
of the two window means:

```text
ratio = mean(d_i at lag) / mean(d_i at lag 1)
```

Noise saturates the displacement immediately: adjacent packets already differ by
the full noise amount, so both means carry it and the ratio sits near `1.0`.
Real channel evolution keeps accumulating with the lag and lifts the numerator
alone. Because both terms share a unit, the noise floor divides out.

That matters because the profile normalization above removes per-packet gain but
not the floor, so a plain mean displacement rises whenever the link weakens,
motion or not. Measured across the corpus the mean separates motion from idle at
`1.0000` AUC on normal links and `0.8705` on weak ones, inverting on two of
eight; the ratio holds `1.0000` and `0.9984` and never inverts. Its idle level
also varies `1.82x` across quiet captures against `14.29x` for the mean, which
is what makes startup calibration land closer to where the session actually
sits.

Hampel filtering is applied to each per-packet displacement stream before its
window mean, both alike, since an outlier surviving only in the denominator
would depress the ratio and read as less motion. The lag-1 reference is the ring
slot behind the lagged one, so the pair costs one running sum and no second
normalization. ML still consumes the plain mean as `l1_delta`.

### Turbulence Autocorrelation

Per-packet turbulence is the spatial coefficient of variation:

```text
t_i = std(A_i) / mean(A_i)
```

After Hampel filtering, Classic calculates lag-1 autocorrelation over the
turbulence window. Both inputs are gain invariant under ideal uniform scaling.
The shared `hampel_enabled` setting controls both Hampel filters.
The same rule applies to ML feature extraction: all `turb_*` features use the
filtered turbulence stream, and all `l1_delta*` features use the filtered
per-packet L1-delta stream in training and both runtimes.

### Weighted Fusion

Classic standardizes `l1_delta` and `turb_autocorr` with fixed training
statistics, applies a two-term linear model, and converts its logit to a
probability:

```text
logit = b + w_l1 * z(l1_delta) + w_ac * z(turb_autocorr)
probability = 1 / (1 + exp(-logit))
motion = probability > threshold
```

The coefficients come from grouped, de-overlapped out-of-fold training balanced
by class, chip, and session. The runtime contains no majority vote or recovery
branch.

Startup adaptation now always thresholds the fitted two-feature logit directly.
The older low-RSSI L1 noise-blend safeguard was retired after the lag-ratio
feature replaced the plain mean: on the current corpus it no longer changes any
decision, and keeping it would only preserve dead state around startup
calibration.

### Startup Threshold Calibration

At startup, Classic begins from the validated global probability threshold
and shifts its logit using the session's startup `q95` relative to the training
idle reference. The shift applies `75%` of the observed session-to-training
offset. This preserves the learned two-feature decision boundary while giving
the quiet baseline enough influence to cover weak links whose startup logits
move in either direction. When the weak-link safeguard is fully active, the
session-centered feature uses the validated global threshold instead of the
saturated raw-logit threshold. Runtime adjustments use the same `0.0-1.0`
probability scale and remain active until recalibration or reboot.

The settled-level rule cannot create a high threshold. It only ever lowers one
after a long quiet dwell, so any threshold that lands near `1.0` came from the
startup `q95` shift, not from later recovery. On the current corpus that still
happens on several otherwise healthy captures: the lag ratio fixed the recall
loss that had motivated this work, but it did not eliminate ceiling-hugging
startup thresholds outright. What remains open is headroom on unseen rooms, not
measured recall on the current pairs.

### Known Limits

On the current normal-link corpus Classic clears the project recall target on
every chip. The published aggregates are `98.6%` on C3, `99.2%` on C5, `99.8%`
on C6, `98.8%` on ESP32, and `99.1%` on S3, with per-chip false positives from
`0.0%` to `2.1%`. ESP32 still rests on one bedroom pair only, so that number is
evidence about one recording, not yet a chip trend.

Recall, not spurious motion, is the remaining Classic gap. Alarms on
static-presence baselines were long read as weak-link false positives, but that
diagnosis did not survive measurement: they occur on the strongest links as
readily as on the weakest, and the empty-room recordings raise none at all. They
are the stationary occupant's own micro-motion. False positives are now gated on
the empty-room recordings, which are the only streams in the corpus with nobody
in the room. See [2026-07-25-gate-classic-false-positives-on-empty-rooms.md](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md).

The threshold-pinning symptom is now mostly diagnostic rather than harmful.
After the lag-ratio swap, the two pairs that had motivated the ceiling concern
still calibrate near `1.0` (`0.980` on C5 bedroom `2026-07-24 12:59`, `0.996`
on C6 bedroom `2026-07-22 18:52`), yet they now reach `96.8%` and `99.1%`
recall. A near-ceiling threshold therefore no longer implies missed motion on
the current corpus; it means the startup prefix ran high enough to push the
q95-shifted threshold close to the probability ceiling.

The two excluded C3 bedroom pairs are a different question, and they are kept
out of the generated validation surfaces on purpose because both sides carry
`dataset_role: exclude`. They are detector evidence, not admission material.
They also are not one failure mode:

- `2026-07-22 19:58` is the strongest link in the corpus at `-39/-38 dBm`, yet
  Classic still reaches only `82.5%` recall on `0.9852` idle/motion AUC with
  `0.0%` false positives.
- `2026-07-25 13:58` is moderate RSSI at `-63/-62 dBm`, not one of the
  `-70` to `-80 dBm` weak-link captures, yet Classic reaches `74.2%` recall on
  `0.9872` AUC with `0.0%` false positives.

In both cases the threshold-free separation says the motion is present while the
Classic decision path still misses a material share of it. That is why the
captures stay useful to ML and to detector analysis while still being excluded
from shared selection or holdout duty.

The historical ESP32 failure that had motivated settled-threshold recovery was
also threshold placement, not separability. Its quiet distribution never
exceeded a probability of `0.110` while the calibrated threshold sat at
`0.421`, and the motion it missed sat at `0.32-0.41`: above every idle sample,
below the threshold. Placing the threshold at that session's idle `p99` would
have taken recall from `94.2%` to `99.1%` at `1%` false positives.

No global knob can collect that, and three were measured. Refitting the
coefficients on the current corpus loses on every chip, ESP32 worst at `-2.3`
points. Raising the startup shift strength above `0.75` lifts ESP32 recall but
starts producing alarms in empty rooms at `0.78`, so the shipped value is
already the largest safe one. Capping the calibrated threshold at the session's
own quiet ceiling hits the same wall: by the margin that moves ESP32, empty
rooms alarm and S3 false positives breach the ceiling.

The reason they all fail together is that ESP32 and the marginal empty rooms
respond to the same knob. What distinguishes them is the strength of the motion
response itself: ESP32 motion peaks near `0.4` where other chips reach `1.0`.

The information is not missing. Measured threshold-free, the window features
separate motion from idle almost perfectly on that capture: `0.9999` AUC for
`l1_delta` and `0.9994` for `turb_autocorr`. Widening the band does not help
because there is nothing left to collect; across `12` to `32` tones the mean AUC
moves from `0.9943` to `0.9959`.

What loses the recall is where the threshold lands, and only on that capture.
Comparing the calibrated threshold against the best any threshold could do at
the same false-positive cost, the corpus leaves `+0.34` points on the table on
average, and `16` of `17` pairs leave `0.0-0.3`. ESP32 leaves `+4.7`.

The cause is an unrepresentative calibration prefix. Startup calibration takes
its quantile over the opening of the session, and on that capture the opening is
`4.14x` noisier than the rest of it, so the threshold settles at `3.82x` the
highest idle value the session ever produces. Most pairs have a prefix that is
representative or quieter, which is why they lose nothing.

### Settled-Level Threshold Recovery

The runtime therefore revisits the threshold once a session proves itself
quieter than its own opening. Every `20` evaluations it records the maximum
metric logit in that block, keeps the last `12` blocks, and once the ring is
full compares the median of those maxima against the live threshold. If that
level plus `CLASSIC_SETTLE_MARGIN_LOGITS` sits below the threshold, the
threshold drops to it.

Three properties make this safe rather than a drift toward the noise floor:

- **It only ever lowers.** Nothing here can raise a threshold, so it cannot
  hide motion that the calibrated threshold would have caught.
- **Motion holds it up.** A stretch of real activity puts the block maxima high,
  the candidate lands above the current threshold, and nothing happens. The rule
  moves only after a long quiet stretch, which is exactly the evidence that the
  threshold is too high.
- **A median of block maxima, not a mean or a global maximum.** One spike cannot
  pull the level down, and one quiet block cannot either. Using the maximum
  instead costs `1.8` points of the recovery, because a single spike then
  governs the whole dwell.

The dwell is `60 s` at the nominal cadence. Measured on the corpus, the rule
takes the ESP32 capture from `94.2%` to `98.0%` recall, raises the worst
per-chip recall from `94.2%` to `97.7%`, and leaves every other chip, the
weak-link slice, and the empty-room gate unchanged.

The margin is the safety knob, and the wall is below the shipped value rather
than next to it. Swept on the corpus, `4.0` moves only one pair by `0.9` points
and leaves ESP32 untouched; `2.0` and `1.5` recover more (`98.8%` and `99.1%`)
for `0.05` and `0.15` points of mean false-positive rate; `1.0` is where the
worst pair breaches the weak-link ceiling at `12.3%`; and the empty-room
recordings stay silent all the way down to `1.0`, first alarming at `0.5`.
`3.0` ships as the conservative end of a usable range that extends to about
`1.5`, not as the last value that works.

Its limit is the mirror of its safety. A room that grows genuinely noisier after
the threshold has come down cannot push it back up; only a recalibration does
that.

Use `ml` where either recall or alarm quietness matters more than startup cost.
Classic remains the default because it needs no training set, no exported
weights, and no per-deployment data.

Raising the Classic ceiling is open feature-side research. The most promising
direction is a third feature or a different pair: an offline sweep on
2026-07-23 found a coherence-oriented candidate (`turb_zcr` plus
`l1_delta_autocorr`) that cut replay false positives sharply against the
current set, and its feature ids already exist in `csi_features.h`. That work
needs its own fit and gate run before anything moves into the runtime.

### Implementation Status

Current aligned implementations:

- `src/python/micro_espectre/classic_detector.py`
- `src/cpp/core/classic_detector.*`

## Retired Historical Baseline

The old standalone moving-variance detector is retained only as historical
context in ADRs and the changelog. It is no longer part of the runtime or
host tooling surface.

## ML Detector

`MLDetector` is the production neural detector. It treats motion detection as a
binary classification problem over a sliding window and outputs a probability in
the range `0.0-1.0`.

Current threshold:

```text
motion if probability > 0.5
```

Unlike `classic`, the ML path does not need startup threshold calibration.

### Current Runtime Topology

The production export is a compact MLP:

```text
Input (6 features)
  -> Dense(32, ReLU)
  -> Dense(16, ReLU)
  -> Dense(1, Sigmoid)
```

Total parameter count: 769

The runtime accepts exported hidden-layer layouts generated by the training
script, but the committed production artifact currently uses the topology above.

### Production Feature Set

Five features, and every one of them is scale-invariant: a ratio, a
correlation, or a crossing rate. That is the membership rule, not an
observation about the members.

| # | Feature | Formula | Meaning |
|---|---------|---------|---------|
| 0 | `turb_mad_over_mean` | `median(|x_i - median(x)|) / |mean(x)|` | Relative robust spread |
| 1 | `turb_autocorr` | `C(1) / C(0)` | Lag-1 temporal correlation |
| 2 | `turb_zcr` | crossing rate of `x` around `median(x)` | Turbulence temporal coherence |
| 3 | `l1_delta_autocorr` | `C(1) / C(0)` of `d_i` | Lag-1 L1-delta coherence |
| 4 | `l1_delta_lag_ratio` | `mean(d_i) / mean(d_i at lag 1)` | Displacement growth with lag |

The rule exists because the per-packet CSI scaling factor varies and is never
recorded. A feature carrying absolute magnitude therefore carries the link's
noise floor with it, and on weak links that floor can exceed the motion it is
meant to measure.

The set arrived in three steps. Coherence-6 replaced the two energy-based
Core-6 members (`turb_skewness`, `l1_delta_waveform_length`) with
shift/scale-invariant temporal-coherence statistics; see the
[Coherence-6 ADR](adr/2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md).
Coherence-7 added the lag ratio Classic runs on, which divides the floor out
instead of carrying it, taking reserved effective alarms from `8` to `3`; see
the [Coherence-7 ADR](adr/2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md).
Then `l1_delta` and `l1_delta_std`, the last two absolute members, were dropped
after a single added training capture took a weak-link pair from `0%` to `100%`
false positives through them; see
[2026-07-28-drop-the-absolute-l1-features.md](adr/2026-07-28-drop-the-absolute-l1-features.md).

Unlike the other four, `l1_delta_lag_ratio` is not derived from a series the
extractor holds: it comes from the L1 tracker, so every caller passing the
production set must supply it.

These seven are the whole feature surface: no candidate tier exists in either
runtime, and the members named below as historical no longer have code behind
them. Every removed feature, and the measurement that rejected it, is listed in
[2026-07-27-reduce-the-feature-surface-to-the-production-set.md](adr/2026-07-27-reduce-the-feature-surface-to-the-production-set.md);
read it before adding one back.

### Feature Diagnostics Snapshot

The Core-6 feature diagnostics below were captured on 2026-07-16 from
`460,958` extracted training windows and are retained as historical reference;
they predate the Coherence-6 swap and have not been refreshed for the current
feature set. Correlation is the marginal Pearson correlation with the binary
motion label. SHAP importance comes from `500` balanced, blocked, held-out
windows across three cross-validation folds grouped by session, using the
seed `1386543369`.

| Rank | Feature | Label correlation | Mean absolute SHAP | SHAP contribution |
|------|---------|------------------:|-------------------:|------------------:|
| 1 | `l1_delta` | 0.7358 | 0.297198 | 51.3% |
| 2 | `turb_mad_over_mean` | 0.5752 | 0.123743 | 21.4% |
| 3 | `turb_autocorr` | 0.7834 | 0.101985 | 17.6% |
| 4 | `l1_delta_std` | 0.6909 | 0.028875 | 5.0% |
| 5 | `l1_delta_waveform_length` | 0.5859 | 0.016178 | 2.8% |
| 6 | `turb_skewness` | 0.2904 | 0.010868 | 1.9% |

Correlation measures each feature independently and does not account for the
strong overlap among the L1-delta descriptors. Grouped out-of-fold SHAP
measures contribution to unseen-session predictions, but correlated features
can still divide importance between them. Feature removal decisions therefore
require grouped, multi-seed ablation plus the paired and long-recording gates;
low SHAP importance alone is not a removal criterion.

Reproduce this snapshot without exporting new runtime artifacts:

```bash
python tools/train_ml_model.py --correlation
python tools/train_ml_model.py --shap 500 --seed 1386543369 --no-export
```

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

Both detectors use the same fixed subcarrier set. Only the detector metric and
threshold behavior differ.

## References

1. **Subcarrier selection for efficient CSI-based indoor localization (2018)**  
   Spectral decorrelation and feature diversity.  
   [Read paper](https://www.researchgate.net/publication/326195991)

2. **Indoor Motion Detection Using Wi-Fi Channel State Information in Flat Floor Environments Versus in Staircase Environments (2018)**  
   Moving-variance segmentation background.  
   [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6068568/)

3. **WiFi Motion Detection: A Study into Efficacy and Classification (2019)**  
   Signal-processing background.  
   [Read paper](https://arxiv.org/abs/1908.08476)

4. **CSI-F: A Human Motion Recognition Method Based on Channel-State-Information Signal Feature Fusion (2024)**  
   Hampel filtering and robustness background.  
   [Read paper](https://www.mdpi.com/1424-8220/24/3/862)
