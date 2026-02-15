# Algorithms

Scientific documentation of the algorithms used in ESPectre for Wi-Fi CSI-based motion detection.

---

## Table of Contents

- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [Gain Lock (Hardware Stabilization)](#gain-lock-hardware-stabilization)
- [CV Normalization (Gain-Invariant Turbulence)](#cv-normalization-gain-invariant-turbulence)
- [MVS: Moving Variance Segmentation](#mvs-moving-variance-segmentation)
- [ML: Neural Network Detector](#ml-neural-network-detector)
- [Automatic Subcarrier Selection](#automatic-subcarrier-selection)
- [Low-Pass Filter](#low-pass-filter)
- [Hampel Filter](#hampel-filter)
- [CSI Features](#csi-features-for-ml)
- [References](#references)

---

## Overview

ESPectre uses a combination of signal processing algorithms to detect motion from Wi-Fi Channel State Information (CSI). 

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
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────────┐              │
│  │ CSI Data │───▶│Gain Lock │───▶│ Band Select  │───▶│ Turbulence  │              │
│  │ N subcs  │    │ AGC/FFT  │    │ 12 subcs     │    │ σ/μ (CV)    │              │
│  └──────────┘    └──────────┘    └──────────────┘    └──────┬──────┘              │
│                  (3s, 300 pkt)   (7.5s, 10×window)          │                     │
│                                                             ▼                     │
│  ┌───────────┐    ┌───────────────┐    ┌─────────────────┐  ┌──────────────────┐  │
│  │ IDLE or   │◀───│ Adaptive      │◀───│ Moving Variance │◀─│ Optional Filters │  │
│  │ MOTION    │    │ Threshold     │    │ (window=75)     │  │ LowPass + Hampel │  │
│  └───────────┘    └───────────────┘    └─────────────────┘  └──────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Calibration sequence (at boot):**
1. **Gain Lock** (3s, 300 packets): Collect AGC/FFT, lock values
2. **Band Calibration** (~7.5s, 10 × window_size packets): Select 12 optimal subcarriers, calculate baseline variance

With default `window_size=75`, this means 750 packets. If you change `segmentation_window_size`, the calibration buffer adjusts automatically.

**Data flow per packet (after calibration):**
1. **CSI Data**: Raw I/Q values for 64 subcarriers (HT20 mode)
   - Espressif format: `[Q₀, I₀, Q₁, I₁, ...]` (Imaginary first, Real second per subcarrier)
2. **Amplitude Extraction**: `|H| = √(I² + Q²)` for selected 12 subcarriers
3. **Spatial Turbulence (CV)**: `CV = σ(amplitudes) / μ(amplitudes)` - gain-invariant variability
4. **Hampel Filter** (optional): Remove outliers using MAD
5. **Low-Pass Filter** (optional): Remove high-frequency noise (Butterworth 1st order)
6. **Moving Variance**: `Var(turbulence)` over sliding window
7. **Adaptive Threshold**: Compare variance to `Pxx(baseline_mv)` → IDLE or MOTION

---

## Gain Lock (Hardware Stabilization)

### Overview

**Gain Lock** is a hardware-level optimization that stabilizes CSI amplitude measurements by locking the ESP32's automatic gain control (AGC) and FFT scaling. This technique is based on [Espressif's esp-csi recommendations](https://github.com/espressif/esp-csi).

### The Problem

The ESP32 WiFi hardware includes automatic gain control (AGC) that dynamically adjusts signal amplification based on received signal strength. While this improves data decoding reliability, it creates a problem for CSI sensing:

| Without Gain Lock | With Gain Lock |
|-------------------|----------------|
| AGC varies dynamically | AGC fixed to calibrated value |
| CSI amplitudes oscillate ±20-30% | Amplitudes stable |
| Baseline appears "noisy" | Baseline flat |
| Potential false positives | Cleaner detection |

### How It Works

The gain lock happens in a **dedicated phase BEFORE band calibration** to ensure clean data:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TWO-PHASE CALIBRATION                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: GAIN LOCK (~3 seconds, 300 packets)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Read PHY   │───▶│   Collect   │───▶│  Calculate  │              │
│  │  agc_gain   │    │  agc_samples│    │   Median    │              │
│  │  fft_gain   │    │  fft_samples│    │             │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│                                               │                      │
│  Packet 300:                                  ▼                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  phy_fft_scale_force(true, median_fft)                       │   │
│  │  phy_force_rx_gain(true, median_agc)                         │   │
│  │  → AGC/FFT now LOCKED                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  PHASE 2: BAND CALIBRATION (~7.5 seconds, 10 × window_size packets) │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Now all packets have stable gain!                           │   │
│  │  → Baseline variance calculated on clean data                │   │
│  │  → Subcarrier selection more accurate                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why two phases?** Separating gain lock from band calibration ensures:
- Calibration only sees data with **stable, locked gain**
- Baseline variance is **accurate** (not inflated by AGC variations)
- Adaptive threshold is calculated correctly
- Total time: ~10.5 seconds (3s gain lock + 7.5s calibration)

**Why median instead of mean?** Median is more robust against outliers:
- Occasional packet with extreme gain values doesn't skew the baseline
- Matches Espressif's internal methodology for gain calibration

### Implementation

The gain lock uses undocumented PHY functions available on newer ESP32 variants:

```c
// External PHY functions (from ESP-IDF PHY blob)
extern void phy_fft_scale_force(bool force_en, int8_t force_value);  // fft_gain is signed
extern void phy_force_rx_gain(int force_en, int force_value);

// Calibration logic (300 packets, ~3 seconds)
// Uses median instead of mean for robustness against outliers
if (packet_count < 300) {
    agc_samples[packet_count] = phy_info->agc_gain;   // uint8_t
    fft_samples[packet_count] = phy_info->fft_gain;   // int8_t (signed!)
} else if (packet_count == 300) {
    median_agc = calculate_median(agc_samples, 300);
    median_fft = calculate_median(fft_samples, 300);
    
    phy_fft_scale_force(true, median_fft);
    phy_force_rx_gain(true, median_agc);
    // Gain is now locked, trigger band calibration
    on_gain_locked_callback();
}
```

### CV Normalization (Gain-Invariant Turbulence)

ESPectre uses **CV normalization** for spatial turbulence when gain lock is not active:

```
turbulence = std(amplitudes) / mean(amplitudes)
```

This is the **Coefficient of Variation (CV)**, a dimensionless ratio that is mathematically invariant to linear gain scaling:

```
CV(kA) = std(kA) / mean(kA) = k·std(A) / k·mean(A) = std(A) / mean(A) = CV(A)
```

If the receiver AGC scales all amplitudes by a factor k, the CV remains unchanged.

**When is CV normalization used?**

- **Gain locked**: Raw `std(amplitudes)` is used (better sensitivity when gain is stable)
- **Gain not locked**: CV normalization (`std/mean`) is used (gain-invariant)

CV normalization is automatically enabled when:
1. Gain lock mode is `disabled`
2. Gain lock mode is `auto` and lock was skipped (e.g., signal too strong, AGC < 30)
3. Platform does not support gain lock (ESP32 Base, ESP32-S2)

**Impact on detection**: CV-normalized turbulence values are typically in the range 0.05-0.25 (compared to 2-20 for raw std). Adaptive thresholds from calibration are correspondingly smaller (order of 1e-4 to 1e-3).

**Compatibility with calibrators**: CV normalization works best with **NBVI** (non-consecutive subcarrier selection) because it selects subcarriers with different spectral characteristics, maximizing sensitivity to motion-induced changes.

### Platform Support

| Platform | Gain Lock | CV Normalization |
|----------|-----------|------------------|
| ESP32-S3 | Supported | When lock skipped |
| ESP32-C3 | Supported | When lock skipped |
| ESP32-C5 | Supported | When lock skipped |
| ESP32-C6 | Supported | When lock skipped |
| ESP32 (original) | Not available | Always enabled |
| ESP32-S2 | Not available | Always enabled |

On platforms without gain lock support, CV normalization ensures stable detection despite AGC variations.

**Reference**: [Espressif esp-csi example](https://github.com/espressif/esp-csi/blob/master/examples/get-started/csi_recv_router/main/app_main.c)

---

## MVS: Moving Variance Segmentation

### Overview

**MVS (Moving Variance Segmentation)** is the core motion detection algorithm. It analyzes the variance of spatial turbulence over time to distinguish between idle and motion states.

### The Insight

Human movement causes **multipath interference** in Wi-Fi signals, which manifests as:
- **Idle state**: Stable CSI amplitudes → low turbulence variance
- **Motion state**: Fluctuating CSI amplitudes → high turbulence variance

By monitoring the **variance of turbulence** over a sliding window, we can reliably detect when motion occurs.

### Algorithm Steps

1. **Spatial Turbulence (CV Normalization)**
   ```
   turbulence = σ(amplitudes) / μ(amplitudes)
   ```
   Where `aᵢ` are the amplitudes of the 12 selected subcarriers, σ is the standard deviation, and μ is the mean. This is the **Coefficient of Variation (CV)**, which is gain-invariant: if all amplitudes are scaled by a factor k (e.g., due to AGC), then `σ(kA)/μ(kA) = σ(A)/μ(A)`. This eliminates the need for gain compensation on platforms without gain lock (e.g., ESP32 Base).

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

For detailed performance metrics (confusion matrix, test methodology, benchmarks), see [PERFORMANCE.md](../PERFORMANCE.md).

**Reference**: [1] MVS segmentation: the fused CSI stream and corresponding moving variance sequence (ResearchGate)

---

## ML: Neural Network Detector

### Overview

The **ML Detector** uses a pre-trained neural network to classify motion based on statistical features extracted from CSI turbulence patterns. Unlike MVS which uses hand-crafted thresholds, ML learns decision boundaries from labeled training data.

### The Insight

Motion detection can be framed as a **binary classification problem**:
- **Input**: Statistical features computed from a sliding window of turbulence values
- **Output**: Probability of motion (0.0 to 1.0)

A neural network can learn complex, non-linear patterns that may be missed by simple threshold-based methods.

### Architecture

The ML detector uses a compact **Multi-Layer Perceptron (MLP)**:

```
Input (12 features)
    ↓
Dense(16, ReLU)      ← 12×16 + 16 = 208 parameters
    ↓
Dense(8, ReLU)       ← 16×8 + 8 = 136 parameters
    ↓
Dense(1, Sigmoid)    ← 8×1 + 1 = 9 parameters
    ↓
Output (probability)
```

**Total**: ~350 parameters, ~2 KB (constexpr float weights)

### Feature Extraction

For each sliding window of 75 turbulence values, 12 statistical features are extracted. See [CSI Features](#csi-features-for-ml) for the complete feature list with detailed definitions.

### Inference Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│ CSI Packet   │───▶│ Turbulence   │───▶│ Optional Filters  │───▶│ Buffer (50)  │
│              │    │ σ/μ (CV)     │    │ Hampel + LowPass  │    │              │
└──────────────┘    └──────────────┘    └───────────────────┘    └──────┬───────┘
                                                                        │
                                                                        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ IDLE/MOTION  │◀───│ Threshold    │◀───│ Probability  │◀───│ 12 Features  │
│              │    │ > 0.5        │    │ [0.0-1.0]    │    │ → Neural Net │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Filter support**: The ML detector shares the same `SegmentationContext` as MVS, so it supports optional low-pass and Hampel filters on the turbulence stream before feature extraction. Filters are disabled by default.

### Calibration

ML uses **fixed subcarriers** - no calibration needed:

| Algorithm | Subcarrier Selection | Threshold | Boot Time |
|-----------|---------------------|-----------|-----------|
| MVS | NBVI (~7.5s) | Adaptive (percentile-based) | ~10.5s |
| ML | **Fixed** (12 evenly distributed) | Fixed (0.5 probability) | **~3s** |

ML uses 12 pre-selected subcarriers evenly distributed across the valid range: `[11, 14, 17, 21, 24, 28, 31, 35, 39, 42, 46, 49]`. This eliminates the 7.5-second band calibration phase, reducing boot time to ~3 seconds (gain lock only).

### Training

For the complete training workflow (data collection, training commands, export formats), see [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md).

The training process uses 5-fold stratified cross-validation with early stopping, dropout, and FP-penalized class weights. The FP penalty multiplies the IDLE class weight, making the model more conservative (fewer false positives at the cost of slightly lower recall).

### Architecture Selection

The 12→16→8→1 architecture was validated as optimal through 5-fold CV on 13,711 samples:

| Architecture | F1 (5-fold CV) | Params | Weights |
|---|---|---|---|
| **12→16→8→1** | **98.3% +/- 0.2%** | 353 | 1.4 KB |
| 12→24→12→1 | 98.2% +/- 0.5% | 625 | 2.4 KB |
| 12→24→1 | 98.1% +/- 0.2% | 337 | 1.3 KB |
| 12→12→8→4→1 | 97.8% +/- 0.5% | 301 | 1.2 KB |
| 12→8→1 | 97.7% +/- 0.3% | 113 | 0.4 KB |

The current architecture achieves the highest F1 with the lowest variance and the best FP rate.

### Performance

ML achieves higher recall than MVS with a small tradeoff in precision. ML's strength is **generalization** -- it performs well across different environments without per-environment calibration.

See [PERFORMANCE.md](../PERFORMANCE.md) for detailed per-chip results and [TUNING.md](../TUNING.md) for configuration and tuning guidance.

---

## Automatic Subcarrier Selection

### Overview

ESPectre uses **NBVI** for automatic subcarrier band selection, achieving excellent performance with zero manual configuration:

| Algorithm | Selection | Best For |
|-----------|-----------|----------|
| **NBVI** | 12 non-consecutive subcarriers | Spectral diversity, resilient to interference |

![Subcarrier Analysis](../images/subcarriers_constellation_diagram.png)
*I/Q constellation diagrams showing the geometric representation of WiFi signal propagation in the complex plane. The baseline (idle) state exhibits a stable, compact pattern, while movement introduces entropic dispersion as multipath reflections change.*

### The Problem

WiFi CSI provides 64 subcarriers in HT20 mode, but not all are equally useful for motion detection:
- Some are too weak (low SNR)
- Some are too noisy (high variance even at rest)
- Some are in guard bands or DC zones
- Manual selection works but doesn't scale across environments

**Challenge**: Find an automatic method that selects the optimal band for motion detection.

See [TUNING.md](../TUNING.md) for configuration options.

### NBVICalibrator

The `NBVICalibrator` class handles the complete calibration lifecycle:
- File-based CSI buffer I/O (write during collection, read during calibration)
- Packet counting and buffer-full detection
- Memory-efficient cleanup (buffer file removed after calibration)
- NBVI-based subcarrier selection algorithm

---

### NBVI Algorithm

The **NBVI (Normalized Baseline Variability Index)** algorithm selects 12 non-consecutive subcarriers by analyzing the variability-to-mean ratio of each subcarrier during baseline.

#### Key Insight

NBVI combines two factors for each subcarrier:
1. **σ/μ** (coefficient of variation): Lower = more stable
2. **σ/μ²** (signal strength factor): Favors subcarriers with strong signals

The weighted formula balances stability and signal strength:
```
NBVI = α × (σ/μ²) + (1-α) × (σ/μ)
```

Where α = 0.5 by default (balanced weighting).

#### Algorithm

```python
def nbvi_calibrate(csi_buffer, band_size=12, alpha=0.5):
    # 1. Find quietest baseline window using percentile detection
    windows = find_candidate_windows(csi_buffer, window_size=200)
    
    # 2. For best window, calculate NBVI for each subcarrier
    for window in windows:
        for subcarrier in valid_subcarriers:
            magnitudes = extract_magnitudes(window, subcarrier)
            mean = sum(magnitudes) / len(magnitudes)
            std = standard_deviation(magnitudes)
            
            # NBVI formula
            nbvi[subcarrier] = alpha * (std / mean**2) + (1-alpha) * (std / mean)
        
        # 3. Apply noise gate (exclude weak subcarriers)
        valid = [sc for sc in subcarriers if mean[sc] > percentile(means, 25)]
        
        # 4. Select 12 subcarriers with lowest NBVI and spacing
        selected = select_with_spacing(sorted_by_nbvi(valid), k=12)
        
        # 5. Validate using MVS false positive rate
        fp_rate, mv_values = validate_subcarriers(selected)
        
        if fp_rate < best_fp_rate:
            best_band = selected
            best_mv_values = mv_values
    
    return best_band, best_mv_values
```

#### Why NBVI?

NBVI selects **non-consecutive** subcarriers, which provides:
- **Spectral diversity**: Different frequency components
- **Noise resilience**: Interference typically affects adjacent subcarriers
- **Environment adaptation**: Works well in complex multipath environments

---

### Adaptive Threshold Calculation

After band selection, NBVI returns the **moving variance values** from baseline. The adaptive threshold is then calculated as a percentile with an optional multiplier:

```python
def calculate_adaptive_threshold(mv_values, percentile, factor):
    return calculate_percentile(mv_values, percentile) * factor
```

Two modes are supported:

| Strategy | Formula | Effect |
|----------|---------|--------|
| Auto (default) | P95 × 1.1 | Balanced sensitivity/false positives |
| Min | P100 × 1.0 | Maximum sensitivity (may have FP) |

See [TUNING.md](../TUNING.md) for configuration options (`segmentation_threshold`).

---

### Performance

NBVI is the default calibration algorithm, selecting 12 non-consecutive subcarriers for spectral diversity and resilience to narrowband interference. See [PERFORMANCE.md](../PERFORMANCE.md) for detailed metrics.

---

### Computational Complexity

| Algorithm | Complexity | Calibration Time (Python) | Notes |
|-----------|------------|---------------------------|-------|
| NBVI | O(W × N × P) | ~30-50ms | Single-pass analysis |

Where W = window size, N = subcarriers, P = packets.

**Benchmark Results** (1000 packets, Python on desktop):

| Chip | NBVI Calibration Time |
|------|----------------------|
| C6 | 32ms |
| S3 | 52ms |

NBVI analyzes each subcarrier independently in a single pass, making it efficient for real-time calibration on embedded devices.

### Guard Bands and DC Zone

HT20 mode (64 subcarriers) configuration:

| Parameter | Value |
|-----------|-------|
| Total Subcarriers | 64 |
| Guard Band Low | 11 |
| Guard Band High | 52 |
| DC Subcarrier | 32 |
| Valid Subcarriers | 41 |

### Fallback Behavior

When calibration cannot find valid bands (e.g., poor signal quality):
NBVI falls back to the default band [11-22] when calibration fails (e.g., due to motion during calibration or insufficient data).

---

## Low-Pass Filter

### Overview

The **Low-Pass Filter** removes high-frequency noise from turbulence values. This is particularly useful in noisy RF environments where the selected band may include subcarriers susceptible to interference.

See [TUNING.md](../TUNING.md) for configuration and when to enable.

### How It Works

The filter uses a **1st-order Butterworth IIR filter** implemented for real-time processing:

1. **Bilinear transform** to convert analog filter to digital
2. **Difference equation**: `y[n] = b₀·x[n] + b₀·x[n-1] - a₁·y[n-1]`
3. **Single sample latency** for real-time processing

### Algorithm

```python
class LowPassFilter:
    def __init__(self, cutoff_hz=11.0, sample_rate_hz=100.0):
        # Bilinear transform
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

### Why 11 Hz Cutoff

Human movement generates signal variations typically in the **0.5-10 Hz** range. RF noise and interference are usually **>15 Hz**. The 11 Hz cutoff:
- **Preserves** motion signal (>90% recall)
- **Removes** high-frequency noise
- **Reduces** false positives in noisy environments

---

## Hampel Filter

### Overview

The **Hampel filter** removes statistical outliers using the Median Absolute Deviation (MAD) method. It can be applied to turbulence values before detection to reduce false positives from sudden interference.

See [TUNING.md](../TUNING.md) for configuration and when to enable.

### How It Works

1. **Maintain sliding window** of recent turbulence values
2. **Calculate median** of the window
3. **Calculate MAD**: `MAD = median(|xᵢ - median|)`
4. **Detect outliers**: If `|x - median| > threshold × 1.4826 × MAD`, replace with median

The constant **1.4826** is the consistency constant for Gaussian distributions.

### Algorithm

```python
def hampel_filter(value, buffer, threshold=4.0):
    # Add to circular buffer
    buffer.append(value)
    
    # Calculate median
    sorted_buffer = sorted(buffer)
    median = sorted_buffer[len(buffer) // 2]
    
    # Calculate MAD
    deviations = [abs(x - median) for x in buffer]
    mad = sorted(deviations)[len(deviations) // 2]
    
    # Check if outlier
    scaled_mad = 1.4826 * mad * threshold
    if abs(value - median) > scaled_mad:
        return median  # Replace outlier
    return value       # Keep original
```

### Implementation Optimization

For embedded systems, the implementation uses:
- **Insertion sort** instead of quicksort (faster for N < 15)
- **Pre-allocated buffers** (no dynamic allocation)
- **Circular buffer** for O(1) insertion

**Reference**: [6] CSI-F: Feature Fusion Method (MDPI Sensors)

---

## CSI Features (for ML)

The ML detector extracts **12 non-redundant statistical features** from a sliding window of turbulence values. All features are computed from the 75-sample turbulence buffer (configured via `segmentation_window_size`), ensuring stable statistical estimates.

### Design Principles

- **No redundant features**: Each feature provides unique information (e.g., no variance alongside std, no range alongside max/min)
- **All turbulence-based**: Higher-order moments (skewness, kurtosis) are computed from the 75-sample turbulence buffer rather than from 12-sample packet amplitudes, giving much more stable estimates
- **MicroPython compatible**: Pure Python implementation without numpy at runtime

### Feature List

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 0 | `turb_mean` | μ = Σxᵢ/n | Mean turbulence (central tendency) |
| 1 | `turb_std` | σ = √(Σ(xᵢ-μ)²/n) | Standard deviation (spread) |
| 2 | `turb_max` | max(xᵢ) | Maximum value in window |
| 3 | `turb_min` | min(xᵢ) | Minimum value in window |
| 4 | `turb_zcr` | crossings / (n-1) | Zero-crossing rate around mean |
| 5 | `turb_skewness` | E[(X-μ)³]/σ³ | Turbulence asymmetry (3rd moment) |
| 6 | `turb_kurtosis` | E[(X-μ)⁴]/σ⁴ - 3 | Turbulence tailedness (4th moment) |
| 7 | `turb_entropy` | -Σpᵢ log₂(pᵢ) | Shannon entropy (randomness) |
| 8 | `turb_autocorr` | C(1)/C(0) | Lag-1 autocorrelation |
| 9 | `turb_mad` | median(\|xᵢ - median(x)\|) | Median absolute deviation |
| 10 | `turb_slope` | Linear regression | Temporal trend |
| 11 | `turb_delta` | x[-1] - x[0] | Start-to-end change |

### Feature Categories

**Basic Statistics (0-3)**: Standard statistical measures of the turbulence buffer.

**Signal Dynamics (4)**:
- **Zero-crossing rate**: Fraction of consecutive samples crossing the mean. High ZCR indicates rapid oscillations (motion), low ZCR indicates stable signal (idle). Very fast to compute.

**Higher-Order Moments (5-6)**: Computed from the turbulence buffer (75 samples) for stable estimates.
- **Skewness**: Asymmetry of turbulence distribution. Motion typically increases skewness.
- **Kurtosis**: "Tailedness" of turbulence distribution. Motion produces heavier tails.

**Robust Statistics (7, 9)**:
- **Entropy**: High during motion (unpredictable), low during idle (stable)
- **MAD**: Median Absolute Deviation - robust alternative to std, less sensitive to outliers

**Temporal Structure (8, 10-11)**:
- **Autocorrelation**: Lag-1 temporal correlation. High during idle (smooth signal), low during motion (turbulent)
- **Slope**: Positive = increasing turbulence, negative = decreasing
- **Delta**: Quick indicator of overall change

### Detailed Definitions

**Zero-Crossing Rate**:
```
ZCR = count(sign(x[i] - μ) ≠ sign(x[i-1] - μ)) / (n - 1)
```
Counts how often the signal crosses the mean value. Ranges from 0.0 (monotonic) to 1.0 (alternating every sample).

**Skewness** (third standardized moment):
```
γ₁ = E[(X - μ)³] / σ³
```
- γ₁ > 0: Right-skewed (tail on right)
- γ₁ < 0: Left-skewed (tail on left)
- γ₁ = 0: Symmetric

**Kurtosis** (fourth standardized moment, excess):
```
γ₂ = E[(X - μ)⁴] / σ⁴ - 3
```
- γ₂ > 0: Heavy tails (leptokurtic)
- γ₂ < 0: Light tails (platykurtic)
- γ₂ = 0: Normal distribution (mesokurtic)

**Shannon Entropy**:
```
H = -Σ pᵢ × log₂(pᵢ)
```
Computed by binning turbulence values (10 bins) and calculating the entropy of the histogram. Higher entropy indicates more randomness/unpredictability.

**Lag-1 Autocorrelation**:
```
r₁ = (1/(n-1)) Σ(xᵢ - μ)(xᵢ₊₁ - μ) / σ²
```
Measures temporal correlation between consecutive values. Ranges from -1.0 to 1.0. Smooth signals have high positive autocorrelation; turbulent signals have low autocorrelation.

**Median Absolute Deviation**:
```
MAD = median(|xᵢ - median(x)|)
```
Robust measure of spread. Unlike std, a single outlier cannot dramatically inflate the MAD. Computed using insertion sort (efficient for n=50 on ESP32).

**Linear Regression Slope**:
```
slope = Σ(iᵢ - ī)(xᵢ - x̄) / Σ(iᵢ - ī)²
```
Where i = time index, x = turbulence value. Positive slope indicates increasing motion intensity.

---

## References

### Primary Sources

1. **MVS Segmentation** - ResearchGate  
   The fused CSI stream and corresponding moving variance sequence.  
   [Read paper](https://www.researchgate.net/figure/MVS-segmentation-a-the-fused-CSI-stream-b-corresponding-moving-variance-sequence_fig6_326244454)

2. **Indoor Motion Detection Using Wi-Fi CSI (2018)** - PMC  
   False positive reduction and sensitivity optimization.  
   [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6068568/)

3. **WiFi Motion Detection: Efficacy and Performance (2019)** - arXiv  
   Signal processing methods for motion detection.  
   [Read paper](https://arxiv.org/abs/1908.08476)

### Algorithm-Specific References

4. **Passive Indoor Localization** - PMC  
   SNR considerations and noise gate strategies.  
   [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6412876/)

5. **Subcarrier Selection for Indoor Localization** - ResearchGate  
   Spectral de-correlation and feature diversity.  
   [Read paper](https://www.researchgate.net/publication/326195991)

6. **CSI-F: Feature Fusion Method** - MDPI Sensors  
   Hampel filter and statistical robustness.  
   [Read paper](https://www.mdpi.com/1424-8220/24/3/862)

7. **Linear-Complexity Subcarrier Selection** - ResearchGate  
   Computational efficiency for embedded systems.  
   [Read paper](https://www.researchgate.net/publication/397240630)

8. **CIRSense: Rethinking WiFi Sensing** - arXiv  
   SSNR (Sensing Signal-to-Noise Ratio) optimization.  
   [Read paper](https://arxiv.org/html/2510.11374v1)

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.

