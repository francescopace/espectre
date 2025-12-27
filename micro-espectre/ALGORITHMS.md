# Algorithms

Scientific documentation of the algorithms used in ESPectre for Wi-Fi CSI-based motion detection.

---

## Table of Contents

- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [Gain Lock (Hardware Stabilization)](#gain-lock-hardware-stabilization)
- [MVS: Moving Variance Segmentation](#mvs-moving-variance-segmentation)
- [NBVI: Automatic Subcarrier Selection](#nbvi-automatic-subcarrier-selection)
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
- **Amplitude**: Signal strength for each OFDM subcarrier (up to 64)
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
│  │ CSI Data │───▶│Gain Lock │───▶│ NBVI Select  │───▶│ Turbulence  │              │
│  │ 64 subcs │    │ AGC/FFT  │    │ 12 subcs     │    │ σ(amps)     │              │
│  └──────────┘    └──────────┘    └──────────────┘    └──────┬──────┘              │
│                  (3s, 300 pkt)    (7s, 700 pkt)             │                     │
│                                                             ▼                     │
│  ┌───────────┐    ┌───────────────┐    ┌─────────────────┐  ┌──────────────────┐  │
│  │ IDLE or   │◀───│ Threshold     │◀───│ Moving Variance │◀─│ Normalize+Filter │  │
│  │ MOTION    │    │ Comparison    │    │ (window=50)     │  │ LowPass + Hampel │  │
│  └───────────┘    └───────────────┘    └─────────────────┘  └──────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Calibration sequence (at boot):**
1. **Gain Lock** (3s, 300 packets): Collect AGC/FFT, lock values
2. **NBVI Calibration** (7s, 700 packets): Select 12 optimal subcarriers, calculate baseline variance

**Data flow per packet (after calibration):**
1. **CSI Data**: Raw I/Q values for 64 subcarriers (128 int8 values)
2. **Amplitude Extraction**: `|H| = √(I² + Q²)` for selected 12 subcarriers
3. **Spatial Turbulence**: `σ = std(amplitudes)` - variability across subcarriers
4. **Normalization**: If baseline > 0.25, attenuate by `0.25/baseline_variance`; otherwise no scaling
5. **Hampel Filter**: Remove outliers using MAD (optional, disabled by default)
6. **Low-Pass Filter**: Remove high-frequency noise (Butterworth 1st order, 11 Hz cutoff)
7. **Moving Variance**: `Var(turbulence)` over sliding window
8. **State Machine**: Compare variance to threshold → IDLE or MOTION

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

The gain lock happens in a **dedicated phase BEFORE NBVI calibration** to ensure clean data:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TWO-PHASE CALIBRATION                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: GAIN LOCK (~3 seconds, 300 packets)                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Read PHY   │───▶│  Accumulate │───▶│  Calculate  │              │
│  │  agc_gain   │    │  agc_sum    │    │  Average    │              │
│  │  fft_gain   │    │  fft_sum    │    │             │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│                                               │                      │
│  Packet 300:                                  ▼                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  phy_fft_scale_force(true, avg_fft)                          │   │
│  │  phy_force_rx_gain(true, avg_agc)                            │   │
│  │  → AGC/FFT now LOCKED                                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  PHASE 2: NBVI CALIBRATION (~7 seconds, 700 packets)                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Now all packets have stable gain!                           │   │
│  │  → Baseline variance calculated on clean data                │   │
│  │  → Subcarrier selection more accurate                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why two phases?** Separating gain lock from NBVI calibration ensures:
- NBVI only sees data with **stable, locked gain**
- Baseline variance is **accurate** (not inflated by AGC variations)
- Normalization works correctly (attenuates only when baseline > 0.25)
- Total time: ~10 seconds (3s gain lock + 7s NBVI)

### Implementation

The gain lock uses undocumented PHY functions available on newer ESP32 variants:

```c
// External PHY functions (from ESP-IDF PHY blob)
extern void phy_fft_scale_force(bool force_en, uint8_t force_value);
extern void phy_force_rx_gain(int force_en, int force_value);

// Calibration logic (300 packets, ~3 seconds)
if (packet_count < 300) {
    agc_sum += phy_info->agc_gain;
    fft_sum += phy_info->fft_gain;
} else if (packet_count == 300) {
    phy_fft_scale_force(true, fft_sum / 300);
    phy_force_rx_gain(true, agc_sum / 300);
    // Gain is now locked, trigger NBVI calibration
    on_gain_locked_callback();
}
```

### Platform Support

| Platform | Gain Lock | Notes |
|----------|-----------|-------|
| ESP32-S3 | Supported | Full AGC/FFT control |
| ESP32-C3 | Supported | Full AGC/FFT control |
| ESP32-C5 | Supported | Full AGC/FFT control |
| ESP32-C6 | Supported | Full AGC/FFT control |
| ESP32 (original) | Not available | PHY functions not exposed |
| ESP32-S2 | Not available | PHY functions not exposed |

On unsupported platforms, ESPectre skips the gain lock process without affecting functionality. Motion detection still works, but may have slightly higher baseline variance.

### Configuration

Gain lock is **always enabled** on supported platforms with no configuration required. It operates transparently during the first ~3 seconds after boot (300 packets at 100 Hz), followed by NBVI calibration (~7 seconds, 700 packets).

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

1. **Spatial Turbulence Calculation**
   ```
   turbulence = σ(amplitudes) = √(Σ(aᵢ - μ)² / n)
   ```
   Where `aᵢ` are the amplitudes of the 12 selected subcarriers.

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

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `threshold` | 1.0 | 0.5-10.0 | With normalization: "Nx baseline noise" |
| `window_size` | 50 | 10-200 | Larger = smoother, slower response |

**Note**: Normalization is always enabled. Threshold 1.0 means "4× baseline noise" consistently across all ESP32 variants.

### Performance

📊 **For detailed performance metrics** (confusion matrix, test methodology, benchmarks), see [PERFORMANCE.md](../PERFORMANCE.md).

**Reference**: [1] MVS segmentation: the fused CSI stream and corresponding moving variance sequence (ResearchGate)

---

## NBVI: Normalized Baseline Variability Index

### Overview

**NBVI (Normalized Baseline Variability Index)** is an algorithm for automatic subcarrier selection, achieving **F1=98.2%** with **zero manual configuration**. It was developed as part of ESPectre and represents a key scientific contribution.

![Subcarrier Analysis](../images/subcarriers_constellation_diagram.png)
*I/Q constellation diagrams showing the geometric representation of WiFi signal propagation in the complex plane. The baseline (idle) state exhibits a stable, compact pattern, while movement introduces entropic 
dispersion as multipath reflections change.*

### The Problem

WiFi CSI provides 64 subcarriers, but not all are equally useful for motion detection:
- Some are too weak (low SNR)
- Some are too noisy (high variance even at rest)
- Some are redundant (correlated with neighbors)
- Manual selection works (F1=97.3%) but doesn't scale across environments

**Challenge**: Find an automatic, calibration-free method that adapts to any environment.

### The Solution: NBVI Formula

```
NBVI = α × (σ/μ²) + (1-α) × (σ/μ)
```

**Default: α = 0.5**

**Components**:
- **σ/μ²** (Energy normalization): Penalizes weak subcarriers (small μ)
- **σ/μ** (Coefficient of Variation): Rewards stability (small σ relative to μ)
- **α = 0.5**: Balanced weighting between signal strength and stability

**Interpretation**: Lower NBVI = Better subcarrier (strong + stable signal)

### Geometric Insight

From I/Q constellation analysis:

| State | Radius (μ) | Ring Width (σ) | Pattern |
|-------|------------|----------------|---------|
| **Baseline (Idle)** | Large | Thin | Compact circle - strong, stable |
| **Movement** | Small | Thick | Scattered - weak, dispersed |

Optimal subcarriers show **maximum contrast** between these states.

### Algorithm Components

#### 1. Percentile-Based Baseline Detection

Instead of using fixed thresholds, NBVI uses percentile analysis to find the quietest windows automatically:

```python
# Analyze sliding windows (skip first 300 packets used for gain lock)
for window in sliding_windows(buffer[300:], size=200, step=50):
    variances.append(calculate_variance(window))

# Find quietest windows (adaptive threshold)
p10_threshold = np.percentile(variances, 10)
candidate_windows = [w for w in windows if variance <= p10_threshold]

# Validate each candidate and select with minimum FP rate
best_window = min(candidate_windows, key=lambda w: validate_fp_rate(w))
```

**Advantages**:
- Multi-window validation ensures selected subcarriers produce stable detection
- Adapts to any environment automatically
- Zero configuration required

#### 2. Noise Gate

**Problem**: Weak subcarriers appear stable (low σ) but have low SNR.

**Solution**: Exclude subcarriers below 25th percentile of mean magnitude.

```python
magnitude_threshold = np.percentile(mean_magnitudes, 25)
valid_subcarriers = [i for i in range(64) if mean[i] > magnitude_threshold]
```

**Reference**: [4] Passive Indoor Localization - SNR considerations and noise gate strategies

#### 3. Spectral De-correlation

**Problem**: Adjacent subcarriers are correlated due to OFDM mechanism.

**Solution**: Hybrid spacing strategy with minimum spacing Δf≥1:
- **Top 5**: Always include (absolute priority by NBVI score)
- **Remaining 7**: Select with minimum spacing constraint

**Default: min_spacing = 1** (adjacent subcarriers allowed)

This allows selecting adjacent subcarriers when they have good NBVI scores, maximizing the use of high-quality subcarriers in the stable central band.

**Reference**: [5] Subcarrier Selection for Indoor Localization - Spectral de-correlation and feature diversity

### Complete Algorithm

```python
def nbvi_calibrate(csi_buffer, num_subcarriers=12):
    # 1. Collect baseline data (700 packets, ~7s @ 100Hz)
    #    Skip first 300 packets (used for gain lock)
    magnitudes = calculate_magnitudes(csi_buffer[300:])
    
    # 2. Find candidate windows using percentile threshold
    window_variances = [var(window) for window in sliding_windows(magnitudes, 200, 50)]
    p10 = percentile(window_variances, 10)
    candidates = [w for w in windows if variance(w) <= p10]
    
    # 3. Validate each candidate and select with minimum FP rate
    best_window = min(candidates, key=lambda w: validate_fp_rate(w))
    
    # 4. Calculate NBVI for all 64 subcarriers (α=0.5)
    for i in range(64):
        mean = np.mean(best_window[:, i])
        std = np.std(best_window[:, i])
        nbvi[i] = 0.5 * (std / mean**2) + 0.5 * (std / mean)
    
    # 5. Apply noise gate (exclude weak subcarriers, 25th percentile)
    threshold = percentile(means, 25)
    valid = [i for i in range(64) if means[i] > threshold]
    
    # 6. Select with spacing
    selected = []
    sorted_by_nbvi = sorted(valid, key=lambda i: nbvi[i])
    
    # Top 5 always included
    selected = sorted_by_nbvi[:5]
    
    # Remaining 7 with spacing >= 1 (adjacent allowed)
    for candidate in sorted_by_nbvi[5:]:
        if all(abs(candidate - s) >= 1 for s in selected):
            selected.append(candidate)
        if len(selected) == 12:
            break
    
    return sorted(selected)
```

### Fallback Behavior

When NBVI calibration cannot find valid subcarriers (e.g., poor signal quality, high noise environment), a fallback mechanism ensures motion detection remains functional:

1. **Default subcarriers used**: The system falls back to the default band [11-22]
2. **Normalization still calculated**: Baseline variance is computed using the first candidate window
3. **Motion detection works**: With proper normalization, even default subcarriers provide usable detection

This fallback ensures ESPectre remains operational even in challenging WiFi environments where optimal subcarrier selection fails.

### Configuration

```python
# Python (Micro-ESPectre)
NBVICalibrator(
    buffer_size=700,             # 7s @ 100Hz (after 3s gain lock)
    window_size=200,             # 2s sliding window for baseline detection
    percentile=10,               # 10th percentile for quietest window selection
    noise_gate_percentile=25,    # 25th percentile for noise gate
    alpha=0.5,                   # NBVI weighting factor (balanced)
    min_spacing=1                # Minimum subcarrier spacing (adjacent allowed)
)
```

### NBVI Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `buffer_size` | 700 | 500-1000 | Packets for calibration (~7s at 100Hz) |
| `window_size` | 200 | 100-300 | Sliding window for baseline detection (2s @ 100Hz) |
| `percentile` | 10 | 5-20 | Percentile for quietest window selection |
| `noise_gate_percentile` | 25 | 10-50 | Percentile for excluding weak subcarriers |
| `alpha` | 0.5 | 0.0-1.0 | Higher = more weight on signal strength |
| `min_spacing` | 1 | 1-3 | Minimum gap between subcarriers (1=adjacent OK) |

**Tuning notes:**
- **alpha = 0.5**: Balanced between signal strength (σ/μ²) and stability (σ/μ)
- **noise_gate_percentile = 25**: Excludes the weakest 25% of subcarriers by signal strength
- **min_spacing = 1**: Allows adjacent subcarriers, maximizing quality selection
- These defaults work well for most environments; rarely need adjustment

---

## Low-Pass Filter

### Overview

The **Low-Pass Filter** removes high-frequency noise from turbulence values. This is particularly useful in noisy RF environments where NBVI may select subcarriers susceptible to interference.

> ℹ️ **Default: Disabled** - The low-pass filter is disabled by default for simplicity. Enable it (11 Hz cutoff recommended) if you experience false positives in noisy RF environments.

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

### Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lowpass_enabled` | false | - | Enable/disable filter |
| `lowpass_cutoff` | 11.0 | 5-20 Hz | Lower = more smoothing, slower response |

### Why 11 Hz Cutoff

Human movement generates signal variations typically in the **0.5-10 Hz** range. RF noise and interference are usually **>15 Hz**. The 11 Hz cutoff:
- **Preserves** motion signal (>90% recall)
- **Removes** high-frequency noise
- **Reduces** false positives in noisy environments

### Performance (60s noisy baseline)

| Configuration | Recall | FP Rate | F1 Score |
|---------------|--------|---------|----------|
| No filter | 98.3% | 51.2% | N/A |
| Low-pass 11 Hz | **92.4%** | **2.34%** | **88.9%** |
| Low-pass 11 Hz + Hampel | **92.1%** | **0.84%** | **93.2%** |

---

## Hampel Filter

### Overview

The **Hampel filter** removes statistical outliers using the Median Absolute Deviation (MAD) method. It can be applied to turbulence values before MVS calculation to reduce false positives from sudden interference.

> ⚠️ **Default: Disabled** - The Hampel filter is disabled by default because MVS already provides robust motion detection with 0% false positives in typical environments. Enabling it reduces Recall from 98.1% to 96.3%. Only enable in environments with high electromagnetic interference causing sudden spikes (e.g., industrial settings, proximity to microwave ovens or multiple WiFi access points).

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

### Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `hampel_enabled` | false | - | Enable/disable filter |
| `hampel_window` | 7 | 3-11 | Larger = more context, slower |
| `hampel_threshold` | 4.0 | 1.0-10.0 | Lower = more aggressive filtering |

### Why Disabled by Default

Testing showed that in clean environments:
- **Without Hampel**: 98.1% Recall, 0% FP
- **With Hampel**: 96.3% Recall, 0% FP

The filter reduces recall because it treats the first packets of real movement as "outliers" and replaces them with the baseline median, delaying detection.

**Reference**: [6] CSI-F: Feature Fusion Method (MDPI Sensors)

---

## CSI Features (for ML)

ESPectre extracts statistical features from CSI data for future machine learning applications (planned for v3.x).

### Available Features

| Feature | Fisher J | Source | Description |
|---------|----------|--------|-------------|
| `iqr_turb` | 3.56 | Turbulence buffer | Interquartile range approximation |
| `skewness` | 2.54 | Current packet | Distribution asymmetry |
| `kurtosis` | 2.24 | Current packet | Distribution tailedness |
| `entropy_turb` | 2.08 | Turbulence buffer | Shannon entropy |
| `variance_turb` | 1.21 | Turbulence buffer | Moving variance (from MVS) |

**Fisher's Criterion (J)**: Measures class separability. Higher J = better feature for distinguishing idle vs motion.

### Feature Definitions

**Skewness** (third standardized moment):
```
γ₁ = E[(X - μ)³] / σ³
```
- γ₁ > 0: Right-skewed (tail on right)
- γ₁ < 0: Left-skewed (tail on left)
- γ₁ = 0: Symmetric

**Kurtosis** (fourth standardized moment):
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
Measures uncertainty/randomness in the turbulence distribution.

---

## References

### Primary Sources

1. **MVS Segmentation** - ResearchGate  
   The fused CSI stream and corresponding moving variance sequence.  
   📄 [Read paper](https://www.researchgate.net/figure/MVS-segmentation-a-the-fused-CSI-stream-b-corresponding-moving-variance-sequence_fig6_326244454)

2. **Indoor Motion Detection Using Wi-Fi CSI (2018)** - PMC  
   False positive reduction and sensitivity optimization.  
   📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6068568/)

3. **WiFi Motion Detection: Efficacy and Performance (2019)** - arXiv  
   Signal processing methods for motion detection.  
   📄 [Read paper](https://arxiv.org/abs/1908.08476)

### Algorithm-Specific References

4. **Passive Indoor Localization** - PMC  
   SNR considerations and noise gate strategies.  
   📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6412876/)

5. **Subcarrier Selection for Indoor Localization** - ResearchGate  
   Spectral de-correlation and feature diversity.  
   📄 [Read paper](https://www.researchgate.net/publication/326195991)

6. **CSI-F: Feature Fusion Method** - MDPI Sensors  
   Hampel filter and statistical robustness.  
   📄 [Read paper](https://www.mdpi.com/1424-8220/24/3/862)

7. **Linear-Complexity Subcarrier Selection** - ResearchGate  
   Computational efficiency for embedded systems.  
   📄 [Read paper](https://www.researchgate.net/publication/397240630)

8. **CIRSense: Rethinking WiFi Sensing** - arXiv  
   SSNR (Sensing Signal-to-Noise Ratio) optimization.  
   📄 [Read paper](https://arxiv.org/html/2510.11374v1)

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.

