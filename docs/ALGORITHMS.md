# Algorithms

Scientific documentation of the algorithms used in ESPectre for Wi-Fi CSI-based motion detection.

---

## Table of Contents

- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [AGC-Active Normalization](#agc-active-normalization)
- [Fixed Subcarrier Set](#fixed-subcarrier-set)
- [Signal Conditioning](#signal-conditioning)
- [Classic: L1-Delta with Variance Recovery](#classic-l1-delta-with-variance-recovery)
- [Historical Moving-Variance Baseline](#historical-moving-variance-baseline)
- [L1-Delta: Normalized Profile Displacement](#l1-delta-normalized-profile-displacement)
- [ML: Neural Network Detector](#ml-neural-network-detector)
- [References](#references)

---

## Overview

ESPectre uses a combination of signal processing algorithms to detect motion from Wi-Fi Channel State Information (CSI). The production non-ML path is `ClassicDetector`, which uses L1-Delta as the primary metric and a gated moving-variance recovery vote in the ambiguous band below threshold. The standalone historical moving-variance baseline and L1-Delta sections remain here because the repo still uses them in research and comparison tooling.

<details>
<summary>What is CSI? (click to expand)</summary>

**Channel State Information (CSI)** represents the physical characteristics of the wireless communication channel between transmitter and receiver. Unlike simple RSSI (Received Signal Strength Indicator), CSI provides rich, multi-dimensional data about the radio channel.

**What CSI Captures:**

*Per-subcarrier information:*
- **Amplitude**: Signal strength for each OFDM subcarrier (64 for HT20 mode)
- **Phase**: Phase shift of each subcarrier
- **Frequency response**: How the channel affects different frequencies

*Environmental effects:*
- **Multipath propagation**: Reflections from walls, furniture, objects
- **Doppler shifts**: Changes caused by movement
- **Temporal variations**: How the channel evolves over time
- **Spatial patterns**: Signal distribution across antennas/subcarriers

**Why It Works for Movement Detection:**

When a person moves in an environment, they alter multipath reflections, change signal amplitude and phase, create temporal variations in CSI patterns, and modify the electromagnetic field structure. These changes are detectable even through walls, enabling **privacy-preserving presence detection** without cameras, microphones, or wearable devices.

</details>

---

## Processing Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           CSI PROCESSING PIPELINE                                  │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐                              │
│  │ CSI Data │───▶│ Fixed 12 SC  │───▶│ Turbulence  │                              │
│  │ N subcs  │    │ Threshold    │    │ σ/μ         │                              │
│  └──────────┘    └──────────────┘    └──────┬──────┘                              │
│                  (~10s, 10×window)          │                                     │
│                                                             ▼                     │
│  ┌───────────┐    ┌───────────────┐    ┌─────────────────┐  ┌──────────────────┐  │
│  │ IDLE or   │◀───│ Adaptive      │◀───│ Moving Variance │◀─│ Optional Filters │  │
│  │ MOTION    │    │ Threshold     │    │ (window=100)    │  │ LowPass + Hampel │  │
│  └───────────┘    └───────────────┘    └─────────────────┘  └──────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Calibration sequence (at boot):**
1. **Threshold Bootstrap** (ClassicDetector only): keep the fixed 12-subcarrier set, build a quiet anchor, try to observe one short useful motion segment, then derive the L1-Delta threshold and freeze the variance-floor snapshot from validated quiet chunks

With default `window_size=100`, the startup budget is still 1000 packets, but it is now a maximum rather than a fixed wait. If the runtime sees a clean `quiet -> motion -> quiet` pattern, calibration may finish earlier; otherwise it falls back internally to the quiet-only path inside the same budget. If you change `segmentation_window_size`, the budget adjusts automatically.

**Data flow per packet (after calibration):**
1. **CSI Data**: Raw I/Q values for 64 subcarriers (HT20 mode)
   - Espressif format: `[Q₀, I₀, Q₁, I₁, ...]` (Imaginary first, Real second per subcarrier)
2. **Amplitude Extraction**: `|H| = √(I² + Q²)` for selected 12 subcarriers
3. **Spatial Turbulence**: `σ(amplitudes) / μ(amplitudes)` (gain-invariant CV normalization)
4. **Hampel Filter** (optional): Remove outliers using MAD
5. **Low-Pass Filter** (optional): Remove high-frequency noise (Butterworth 1st order)
6. **Moving Variance**: `Var(turbulence)` over sliding window
7. **Classic Fusion**: compare the L1-Delta primary metric to its calibrated threshold, and use moving variance only as a gated recovery vote in the ambiguous band

---

## AGC-Active Normalization

ESPectre no longer forces hardware gain. The receiver AGC stays active on all
supported chips and the shared detector pipeline always computes spatial
turbulence as:

```
turbulence = σ(amplitudes) / μ(amplitudes)
```

This coefficient-of-variation form is gain-invariant:

```
CV(kA) = σ(kA) / μ(kA) = σ(A) / μ(A)
```

If AGC scales all amplitudes by a factor `k`, turbulence remains unchanged. The
project intentionally standardizes on this single path across runtime,
collector, dataset schema, and offline tooling instead of exposing gain-lock
state or compensated raw samples.

Why this design:
- aligns with Espressif's newer public examples, which prefer AGC-active capture
- avoids undocumented PHY coupling and forced-gain edge cases
- removes the dedicated gain-lock startup delay
- keeps datasets consistent across chips, including ESP32 variants that never
  exposed forced-gain control

**Impact on detection**: CV-normalized turbulence values are typically in the
range 0.05-0.25. Adaptive thresholds from calibration are correspondingly
smaller (order of 1e-4 to 1e-3).

---

## Fixed Subcarrier Set

ESPectre uses one shared fixed 12-subcarrier set for both detectors:

`[14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50]`

This set was originally validated offline and is now treated as part of the production detector definition.

![Subcarrier Analysis](images/subcarriers_constellation_diagram.png)
*I/Q constellation diagrams showing the geometric representation of WiFi signal propagation in the complex plane. The baseline (idle) state exhibits a stable, compact pattern, while movement introduces entropic dispersion as multipath reflections change.*

### Why This Set

The fixed set balances three goals:

- avoid guard-band and DC subcarriers
- spread energy across the valid HT20 band instead of clustering tightly
- stay aligned with the ML training pipeline so both detectors consume the same CSI slice

### Adaptive Threshold Calculation

For `classic`, startup calibration keeps this fixed band and derives the adaptive threshold from the L1-Delta primary metric:

```python
def calculate_adaptive_threshold(threshold_metric, factor):
    return threshold_metric * factor
```

`threshold_metric` now depends on which startup path succeeds first:

- successful motion-first bootstrap: midpoint of the validated quiet/motion gap
- internal quiet-first fallback: gated max of the quiet-only ring, with a small conservative margin when no useful motion was confirmed

| Mode | Formula | Effect |
|------|---------|--------|
| Auto (default) | threshold_metric x 1.1 | Lower false positives while preserving recall |
| Min | threshold_metric x 1.0 | Maximum sensitivity (may have FP) |

See [TUNING.md](TUNING.md) for configuration options (`segmentation_threshold`).

### Why Spread-Out Subcarriers?

Using **spread-out** subcarriers provides:
- **Spectral diversity**: Different frequency components respond differently to motion
- **Noise resilience**: Narrowband interference typically affects adjacent subcarriers
- **Environment adaptation**: Works well in complex multipath environments

### Guard Bands and DC Zone

HT20 mode (64 subcarriers) configuration:

| Parameter | Value |
|-----------|-------|
| Total Subcarriers | 64 |
| Guard Band Low | 11 |
| Guard Band High | 52 |
| DC Subcarrier | 32 |
| Valid Subcarriers | 41 |

See [PERFORMANCE.md](PERFORMANCE.md) for detailed fixed-band validation metrics.

---

## Signal Conditioning

Optional filters can be applied to the turbulence stream before detection. Both filters operate on the scalar turbulence value (one per CSI packet) and share the same `SegmentationContext` used by the Classic recovery path, the historical moving-variance baseline, and the ML detector.

---

## Classic: L1-Delta with Variance Recovery

`ClassicDetector` is the production non-ML detector. It combines:

- a primary L1-Delta metric, which owns the startup threshold and calibration gate
- a moving-variance support metric, used only as a gated recovery vote in the ambiguous band below threshold

The startup threshold is calibrated from the L1-Delta primary metric with:

| Mode | Formula | Effect |
|------|---------|--------|
| Auto (default) | threshold_metric x 1.1 | Lower false positives while preserving recall |
| Min | threshold_metric x 1.0 | Maximum sensitivity (may have FP) |

Classic startup now tries to finish from a validated `quiet -> motion -> quiet` pattern first, then falls back internally to a quiet-only bootstrap if no useful motion arrives before the startup budget is exhausted. The moving-variance path does not have its own standalone detector name in the runtime anymore. Instead, Classic freezes a variance floor during startup from validated quiet chunks only, and enables the recovery vote only when that floor is tight enough to trust.

### Hampel Filter

**Enabled by default** (window=7, threshold=5.0 MAD).

The Hampel filter removes statistical outliers using the Median Absolute Deviation (MAD) method, reducing false positives from sudden RF interference.

**How it works:**

1. Maintain a sliding window of recent turbulence values
2. Calculate the median of the window
3. Calculate MAD: `MAD = median(|xᵢ - median|)`
4. If `|x - median| > threshold × 1.4826 × MAD`, replace with median

The constant **1.4826** is the consistency constant that makes MAD a consistent estimator of standard deviation for Gaussian distributions.

```python
# Matches src/python/micro_espectre/filters.py (MicroPython) and the same logic in C++.
# threshold_scaled = threshold * 1.4826  (pre-computed at init)

def insertion_sort(arr, n):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def hampel_filter(value, buffer, sorted_scratch, window_size, index, count,
                  threshold_scaled):
    buffer[index] = value
    index = (index + 1) % window_size
    if count < window_size:
        count += 1
    if count < 3:
        return value

    n = count
    mid = n // 2

    for i in range(n):
        sorted_scratch[i] = buffer[i]
    insertion_sort(sorted_scratch, n)
    median = sorted_scratch[mid]

    for i in range(n):
        sorted_scratch[i] = abs(buffer[i] - median)
    insertion_sort(sorted_scratch, n)
    mad = sorted_scratch[mid]

    if mad > 1e-6:
        deviation = abs(value - median) / mad
        if deviation > threshold_scaled:
            return median
    return value
```

**Embedded optimization**: Circular turbulence buffer, pre-allocated `buffer` and `sorted_scratch` (no per-packet list growth). Insertion sort on the active window (N ≤ 11) on MicroPython; the C++ component uses the same MAD test with `std::sort` on stack copies of the same small window.

**Reference**: [5] CSI-F: Feature Fusion Method (MDPI Sensors)

### Low-Pass Filter

**Disabled by default**. Enable with `lowpass_enabled: true`.

The low-pass filter removes high-frequency noise from turbulence values using a **1st-order Butterworth IIR filter**:

```python
class LowPassFilter:
    def __init__(self, cutoff_hz=11.0, sample_rate_hz=100.0):
        wc = tan(π × cutoff / sample_rate)
        k = 1.0 + wc
        self.b0 = wc / k
        self.a1 = (wc - 1.0) / k
        
        self.x_prev = 0.0
        self.y_prev = 0.0
    
    def filter(self, x):
        y = self.b0 * x + self.b0 * self.x_prev - self.a1 * self.y_prev
        self.x_prev = x
        self.y_prev = y
        return y
```

**Why 11 Hz cutoff?** Human movement generates signal variations typically in the **0.5-10 Hz** range. RF noise and interference are usually **>15 Hz**. The 11 Hz cutoff preserves motion signal while removing high-frequency noise.

See [TUNING.md](TUNING.md) for filter configuration and tuning guidance.

---

## Historical Moving-Variance Baseline

### The Insight

Human movement causes **multipath interference** in Wi-Fi signals, which manifests as:
- **Idle state**: Stable CSI amplitudes → low turbulence variance
- **Motion state**: Fluctuating CSI amplitudes → high turbulence variance

By monitoring the **variance of turbulence** over a sliding window, we can reliably detect when motion occurs.

### Algorithm Steps

1. **Spatial Turbulence**

   Computed per packet from the 12 selected subcarrier amplitudes using gain-invariant CV normalization. ML then extracts relative window features for the neural detector.

2. **Moving Variance (Two-Pass Algorithm)**
   ```
   μ = Σxᵢ / n                    # Mean of turbulence buffer
   Var = Σ(xᵢ - μ)² / n           # Variance (numerically stable)
   ```
   The two-pass algorithm avoids catastrophic cancellation that can occur with running variance on float32.

3. **State Machine**
   ```
   if state == IDLE and variance > threshold:
       state = MOTION
   elif state == MOTION and variance < threshold:
       state = IDLE
   ```

### Performance

For detailed performance metrics, see [PERFORMANCE.md](PERFORMANCE.md).

**Reference**: [2] Moving-variance segmentation: the fused CSI stream and corresponding moving variance sequence

---

## L1-Delta: Normalized Profile Displacement

### The Insight

Human motion changes the multipath geometry, so the **shape** of the amplitude
profile across subcarriers decorrelates over time. Instead of measuring how
much a scalar summary (turbulence) fluctuates, L1-Delta measures the profile
displacement directly:

- **Idle state**: The normalized profile is stable; packet-to-packet differences
  are only receiver noise, which sits on a stable floor.
- **Motion state**: The profile shifts coherently across all subcarriers,
  raising the displacement well above the noise floor.

Because the profile is normalized per packet, the metric is invariant to the
scalar AGC gain, like the CV turbulence path.

### Algorithm Steps

1. **Normalized Amplitude Profile**

   Per packet, amplitudes of the 12 selected subcarriers divided by their mean:
   ```
   A_norm[k] = A[k] / mean(A)
   ```

2. **Lagged L1 Displacement**

   Compare against the profile observed `lag` packets earlier
   (default `lag = 10`, about 100 ms at 100 pps):
   ```
   d[n] = mean_k |A_norm[n][k] - A_norm[n - lag][k]|
   ```
   The lag matters: at 10 ms (lag 1) the body has not displaced the multipath
   yet and the difference is mostly receiver noise; at 100 ms the motion
   signature dominates.

3. **Motion Metric**

   Running mean of `d` over the sliding window (same `window_size = 100` as the
   historical moving-variance baseline), maintained incrementally with a running sum.

4. **State Machine**

   Identical to the historical moving-variance baseline: metric above threshold enters MOTION, below returns IDLE.

### Startup Threshold

L1-Delta uses the shared startup calibration flow with a detector-specific
`auto` factor:

```
threshold = threshold_metric x 1.1
```

The factor is lower than the historical moving-variance baseline `1.3` because the quiet-state metric is much
tighter. In the motion-first path, `threshold_metric` is the midpoint between:

- `motion_ref`: the lower edge of the accepted motion band
- `quiet_ref`: the upper edge of the accepted quiet band

When startup never confirms a useful motion segment, the same calibrator falls
back internally to the quiet-first ring and uses a conservative gated-maximum
threshold metric instead.

### Motion-First Bootstrap And Internal Quiet-First Fallback

Startup no longer assumes that the room will stay perfectly quiet until the end
of calibration. Instead, the shared calibrator runs as a small state machine:

1. build a stable quiet anchor
2. confirm a sustained motion segment rather than a single spike
3. wait for post-motion quiet to return
4. derive the threshold inside the validated quiet/motion gap

This lets Classic finish earlier than the nominal startup budget when a useful
motion example arrives soon after boot, but still converge on quiet-only
startups.

The motion-first path is accepted only when:

- the detector is already ready (`window_size` packets observed)
- quiet chunks are self-consistent enough to define a floor
- motion lasts for multiple chunks, not just one spike
- quiet returns after motion
- the gap between quiet and motion is large enough to trust

If any of these checks fail before the startup budget is exhausted, the
calibrator falls back internally to the previous quiet-first statistic on the
same observed metrics.

### Internal Quiet-First Fallback

The quiet-first fallback still uses a consistency gate to protect the startup
threshold when the startup budget contains unstable or contaminated data.
During fallback, ready-state metrics are grouped into 6 chunks and only the
per-chunk maxima are kept. The ring is accepted when:

- spread: `max(chunk maxima) <= 1.10 x median(chunk maxima)`
- floor anchor: `median(chunk maxima) <= 1.5 x min(chunk maxima ever seen)`

Unlike the previous implementation, the shared startup flow no longer extends
beyond the configured budget. If motion-first did not succeed and the fallback
ring never became fully trustworthy, Classic now ends the startup budget with a
conservative gated-max threshold instead of waiting for extra packets.

This matters most on noisy `S3`-like sessions: the no-motion fallback is high
enough to avoid the old false-positive spikes, while the successful
motion-first path still keeps recall above the promotion target on the paired
datasets.

### Why No Hampel Filter

The historical moving-variance baseline squares deviations inside the moving variance, so a single spiked packet
can dominate the window; the Hampel filter exists to remove those outliers.
L1-Delta averages absolute differences, so one spiked packet contributes at
most `1/window_size` of the metric. The detector therefore accepts the filter
kwargs for interface compatibility but does not apply them.

### Properties (Offline Benchmark)

Measured on the repo datasets (4 chips, 3 environments, paired
static-presence/motion plus empty captures); full protocol and numbers in
[EXPERIMENTS.md](EXPERIMENTS.md):

- Separability (AUC) equal or better than the historical moving-variance baseline on every chip, with more uniform
  per-chip recall (89-96% including S3, where that baseline drops to ~80%).
- Quiet-level stability: the quiet metric median varies <=1.3x across sessions
  on the same chip, against up to 14.5x for the historical moving-variance metric. Startup
  thresholds therefore age much better.
- Same fail-open behavior as the historical moving-variance baseline if the RF noise floor rises after calibration;
  neither detector covers that case today.

### Computational Cost vs Historical Variance Baseline

Per packet, both detectors share the amplitude extraction (12x `sqrt`). After
that:

| Stage | Variance baseline | L1-Delta |
|-------|-----|----------|
| Per-packet metric | CV turbulence (mean + std + `sqrt` + div) | Normalize (12 div) + L1 diff (12 abs/add) |
| Outlier handling | Hampel(7): 2 insertion sorts per packet | Not needed |
| State evaluation | Two-pass variance, O(window) = ~100 ops | Running-sum mean, O(1) |
| Persistent state | ~126 floats | ~232 floats (adds `lag` profiles ring) |

Measured on the Python reference implementations over 10k real packets
(relative numbers are what matter; absolute values are host CPython):

| Path | Variance baseline (Hampel on) | L1-Delta | Delta |
|------|-----------------|----------|-------|
| Evaluation every 25 packets (firmware-like) | 8.7 us/pkt | 7.0 us/pkt | ~20% cheaper |
| Evaluation every packet (host live CLI) | 15.5 us/pkt | 7.3 us/pkt | ~50% cheaper |

The O(1) evaluation means the L1-Delta cost is flat regardless of evaluation
rate, while the historical moving-variance two-pass variance grows with evaluation frequency. Like
the shared turbulence path, the L1-Delta hot path is allocation-free
(pre-allocated profile rings, buffer swap instead of copy), which matters for
MicroPython GC pressure. The extra ~100 floats of state (~0.4 KB in float32)
are negligible on target hardware.

### Status

Implemented as the production non-ML path in both runtimes with aligned
semantics:

- Micro-ESPectre: `src/python/micro_espectre/classic_detector.py`
- Shared C++ core: `src/cpp/core/classic_detector.*`
- Runtime/frontends: `classic` is the default non-ML detector

---

## ML: Neural Network Detector

### The Insight

Motion detection can be framed as a **binary classification problem**:
- **Input**: Statistical features computed from a sliding window of turbulence values
- **Output**: Probability of motion (0.0 to 1.0)

A neural network can learn complex, non-linear patterns that may be missed by simple threshold-based methods. Unlike the historical moving-variance baseline, ML learns decision boundaries from labeled training data and generalizes across environments without per-environment calibration. The production binary model maps both `empty` and `static_presence` captures to IDLE, and maps `motion` captures to MOTION.

### Architecture

The ML detector uses a compact **Multi-Layer Perceptron (MLP)**.
The current production export remains small enough for embedded deployment, while the runtime now accepts any exported hidden-layer layout generated by the training script.
The training script supports `standard`, `robust`, and `clipped_standard` normalization modes. Experimental modes should be validated against the real-data regression suite before replacing the committed production weights.
The trainer currently uses a PyTorch MLP with ReLU hidden layers and exports the learned weights into the shared Python/C++ runtime format used for production artifacts.

Current production topology:

```
Input (6 features)
    ↓
Dense(32, ReLU)      ← 6×32 + 32 = 224 parameters
    ↓
Dense(16, ReLU)      ← 32×16 + 16 = 528 parameters
    ↓
Dense(1, Sigmoid)    ← 16×1 + 1 = 17 parameters
    ↓
Output (probability)
```

**Total**: 769 parameters, ~3.0 KB (constexpr float weights)

The feature set evolved from earlier raw-window and relative-only turbulence
exports to the current mixed turbulence + L1-delta Core-6 set:
`turb_mad_over_mean`, `turb_skewness`, `turb_autocorr`, `l1_delta`,
`l1_delta_std`, and `l1_delta_waveform_length`. The historical
moving-variance baseline is used only to mine difficult IDLE windows during
training, not as a general teacher for motion labels.

### Inference Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│ CSI Packet   │───▶│ Turbulence   │───▶│ Optional Filters  │───▶│ Buffer (100) │
│              │    │ raw base     │    │ Hampel + LowPass  │    │              │
└──────────────┘    └──────────────┘    └───────────────────┘    └──────┬───────┘
                                                                        │
                                                                        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ IDLE/MOTION  │◀───│ Threshold    │◀───│ Motion Prob. │◀───│ 6 Core-6     │
│              │    │ > 0.5        │    │ [0.0-1.0]    │    │ → Neural Net │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Calibration

Both detectors use the same fixed, non-configurable subcarrier set:

| Algorithm |Threshold | Boot Time |
|-----------|---------------------|-----------|
| Classic | Adaptive (`threshold_metric x 1.1` on the L1-Delta primary metric, with motion-first startup and quiet-first fallback) | ~13s max, often less |
| ML | Fixed (0.5 probability threshold) | **~3s** |

The production subcarrier set is `[14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50]`.
Classic uses a startup threshold bootstrap after startup; ML keeps its fixed threshold and therefore starts immediately once CSI capture is active.

### Features

The ML detector extracts **6 production features** from the current sliding
window. Three come from the turbulence series, and three come from the
L1-delta profile-displacement series derived from mean-normalized subcarrier
amplitude profiles.

**Design principles:**
- Keep the exported runtime gain-robust without per-install calibration
- Use a small mixed feature set that survives grouped CV and real-data promotion gates
- Reuse the shared L1-delta profile tracker already promoted into the Classic path
- MicroPython compatible: pure Python implementation without numpy at runtime

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 0 | `turb_mad_over_mean` | median(\|xᵢ - median(x)\|) / \|μ\| | Relative robust spread |
| 1 | `turb_skewness` | E[(X-μ)³]/σ³ | Turbulence asymmetry (3rd moment) |
| 2 | `turb_autocorr` | C(1)/C(0) | Lag-1 autocorrelation |
| 3 | `l1_delta` | mean(dᵢ) | Mean L1 profile displacement |
| 4 | `l1_delta_std` | std(dᵢ) | Spread of the L1-delta series |
| 5 | `l1_delta_waveform_length` | Σ\|dᵢ - dᵢ₋₁\| | Temporal variation of L1-delta |

#### Feature Categories

**Turbulence Statistics (0-2)**: Gain-reduced summary statistics over the
coefficient-of-variation turbulence window.

**Robust Spread (0)**:
- **MAD over mean**: Robust alternative to std, less sensitive to outliers and
  quiet-window spikes.

**Higher-Order Moments (1)**:
- **Skewness**: Asymmetry of turbulence distribution.

**Temporal Structure (2)**:
- **Autocorrelation**: Lag-1 temporal correlation. High during idle (smooth signal), low during motion (turbulent)

**L1-Delta Statistics (3-5)**:
- **L1-delta**: Mean absolute displacement between mean-normalized subcarrier
  profiles separated by a fixed lag.
- **L1-delta std**: Variability of those per-packet profile displacements.
- **L1-delta waveform length**: Short-term jaggedness of the L1-delta stream.

#### Feature Importance

SHAP and correlation can diverge significantly: correlation captures linear association with the label, while SHAP captures non-linear contribution inside the network.

The earlier raw-window and relative-only rankings are now historical; see
[EXPERIMENTS.md](EXPERIMENTS.md) for that log. For the current Core-6 model,
feature candidates should be compared with grouped CV, the paired/long
recording gates, and the exported-artifact gain-stress gate rather than
correlation alone.

#### Feature Definitions

**Skewness** (third standardized moment):
```
γ₁ = E[(X - μ)³] / σ³
```
- γ₁ > 0: Right-skewed (tail on right)
- γ₁ < 0: Left-skewed (tail on left)
- γ₁ = 0: Symmetric

**Lag-1 Autocorrelation**:
```
r₁ = (1/(n-1)) Σ(xᵢ - μ)(xᵢ₊₁ - μ) / σ²
```
Measures temporal correlation between consecutive values. Ranges from -1.0 to 1.0. Smooth signals have high positive autocorrelation; turbulent signals have low autocorrelation.

**Median Absolute Deviation**:
```
MAD = median(|xᵢ - median(x)|)
```
Robust measure of spread. Unlike std, a single outlier cannot dramatically
inflate the MAD.

**L1-Delta**:
```
dᵢ = mean_j |pᵢ[j] - pᵢ₋lag[j]|
```
Where `pᵢ` is the mean-normalized amplitude profile for packet `i`. The
exported ML feature uses the mean of the resulting `d` series over the current
window.

**L1-Delta Waveform Length**:
```
WL_d = Σ |dᵢ - dᵢ₋₁|,  i = 1..n-1
```
Measures short-term temporal variation in the L1-delta stream. It rises when
profile displacements become bursty or irregular even if the mean displacement
alone stays similar.

### Training

Operational training commands, export formats, and validation gates now live in
[ML_TRAINING.md](ML_TRAINING.md). Dataset collection and labeling remain in
[ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md).

### Performance

ML's strength is **generalization without runtime calibration**: it uses fixed subcarriers and pre-trained weights, so it can boot quickly and perform strongly on the paired real-data validation set.

Historical experiment logs that informed the current production choices are collected in [EXPERIMENTS.md](EXPERIMENTS.md). This keeps the algorithm reference focused on the currently promoted pipeline while preserving the rationale behind rejected or superseded approaches.

See [PERFORMANCE.md](PERFORMANCE.md) for detailed per-chip results and [TUNING.md](TUNING.md) for configuration and tuning guidance.

---

## References

1. **Subcarrier selection for efficient CSI-based indoor localization (2018)**  
   Spectral de-correlation and feature diversity.  
   [Read paper](https://www.researchgate.net/publication/326195991)

2. **Indoor Motion Detection Using Wi-Fi Channel State Information in Flat Floor Environments Versus in Staircase Environments (2018)** 
   Moving variance segmentation.  
   [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6068568/)

3. **WiFi Motion Detection: A Study into Efficacy and Classification (2019)**
   Signal processing methods for motion detection.  
   [Read paper](https://arxiv.org/abs/1908.08476)

4. **A Novel Passive Indoor Localization Method by Fusion CSI Amplitude and Phase Information (2019)**
   SNR considerations and noise gate strategies.  
   [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6412876/)

5. **CSI-F: A Human Motion Recognition Method Based on Channel-State-Information Signal Feature Fusion (2024)**
   Hampel filter and statistical robustness.  
   [Read paper](https://www.mdpi.com/1424-8220/24/3/862)

6. **Linear-Complexity Subcarrier Selection Strategy for Fast Preprocessing of CSI in Passive Wi-Fi Sensing Classification Tasks (2025)** 
   Computational efficiency for embedded systems.  
   [Read paper](https://www.researchgate.net/publication/397240630)

7. **CIRSense: Rethinking WiFi Sensing with Channel Impulse Response (2025)**  
   SSNR (Sensing Signal-to-Noise Ratio) optimization.  
   [Read paper](https://arxiv.org/html/2510.11374v1)
