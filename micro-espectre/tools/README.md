# Analysis Tools

**Python scripts for CSI data analysis, algorithm optimization, and validation**

This directory contains analysis tools for developing and validating ESPectre's motion detection algorithms. These scripts are essential for parameter tuning, algorithm validation, and scientific analysis.

For algorithm documentation (MVS, NBVI, Hampel filter), see [ALGORITHMS.md](../ALGORITHMS.md).

For production performance metrics, see [PERFORMANCE.md](../../PERFORMANCE.md).

For data collection and ML datasets, see [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Analysis Scripts](#analysis-scripts)
- [Usage Examples](#usage-examples)
- [Key Results](#key-results)

---

## Quick Start

```bash
# Activate virtual environment
source ../venv/bin/activate

# Collect CSI data samples (requires ESP32 streaming)
cd ..
./me stream --ip <PC_IP>     # On one terminal
./me collect --label baseline_noisy --duration 60 --chip c6  # On another
./me collect --label movement --duration 10 --chip c6
cd tools

# Optimize filter parameters
python 6_optimize_filter_params.py c6           # Low-pass optimization
python 6_optimize_filter_params.py c6 --hampel  # Hampel optimization

# Run analysis
python 2_analyze_system_tuning.py --quick
python 3_analyze_moving_variance_segmentation.py --plot
```

---

## Analysis Scripts

### 1. Raw Data Analysis (`1_analyze_raw_data.py`)

**Purpose**: Visualize raw CSI amplitude data and identify patterns

- Analyzes subcarrier patterns and noise characteristics
- Helps identify most informative subcarriers
- Visualizes signal strength distribution

```bash
python 1_analyze_raw_data.py
```

---

### 2. System Tuning (`2_analyze_system_tuning.py`)

**Purpose**: Grid search for optimal MVS parameters

- Tests subcarrier clusters, thresholds, and window sizes
- Shows confusion matrix for best configuration
- Finds optimal parameter combinations

```bash
python 2_analyze_system_tuning.py          # Full grid search
python 2_analyze_system_tuning.py --quick  # Reduced parameter space
```

---

### 3. MVS Visualization (`3_analyze_moving_variance_segmentation.py`)

**Purpose**: Visualize MVS algorithm behavior

- Shows moving variance, threshold, and detection states
- Displays confusion matrix and performance metrics
- Validates current configuration

```bash
python 3_analyze_moving_variance_segmentation.py
python 3_analyze_moving_variance_segmentation.py --plot  # Show graphs
```

---

### 4. Filter Location Analysis (`4_analyze_filter_location.py`)

**Purpose**: Compare filter placement in processing pipeline

- Tests pre-filtering vs post-filtering approaches
- Evaluates impact on motion detection accuracy
- Determines optimal filter location

```bash
python 4_analyze_filter_location.py
python 4_analyze_filter_location.py --plot  # Show visualizations
```

---

### 5. Filter Turbulence Analysis (`5_analyze_filter_turbulence.py`)

**Purpose**: Compare how different filters affect turbulence and motion detection

- **Hampel vs Lowpass comparison**: Shows the fundamental difference between outlier removal and frequency smoothing
- Tests multiple filter configurations (EMA, SMA, Butterworth, Chebyshev, Bessel, Hampel, Savitzky-Golay, Wavelet)
- Visualizes raw vs filtered turbulence signal and resulting moving variance

**Key insight**: Hampel and Lowpass are NOT the same type of filter!
- **Hampel**: Removes spikes/outliers (preserves signal shape)
- **Lowpass**: Smooths high-frequency noise (introduces lag)
- **Combined**: Best of both - spike removal + noise smoothing

```bash
python 5_analyze_filter_turbulence.py              # Run all filter comparisons
python 5_analyze_filter_turbulence.py --plot       # Show 4-panel visualization:
                                                   #   No Filter | Hampel | Lowpass | Combined
python 5_analyze_filter_turbulence.py --optimize-filters  # Optimize parameters
```

---

### 6. Filter Parameters Optimization (`6_optimize_filter_params.py`)

**Purpose**: Optimize low-pass and Hampel filter parameters

- Optimizes normalization target and low-pass cutoff frequency
- Grid search for Hampel filter parameters (window, threshold)
- Supports chip-specific data filtering (c6, s3)
- Finds optimal configuration for noisy environments

```bash
python 6_optimize_filter_params.py              # Low-pass optimization
python 6_optimize_filter_params.py c6           # Use only C6 data
python 6_optimize_filter_params.py --hampel     # Hampel optimization
python 6_optimize_filter_params.py c6 --hampel  # C6 + Hampel
```

**Current optimal configuration (60s noisy baseline):**
- Low-pass: Cutoff=11 Hz, Target=28 → Recall 92.4%, FP 2.3%
- With Hampel: Window=9, Threshold=4.0 → **Recall 92.1%, FP 0.84%, F1 93.2%**

---

### 7. Detection Methods Comparison (`7_compare_detection_methods.py`)

**Purpose**: Compare different motion detection methods

- Compares RSSI, Mean Amplitude, Turbulence, and MVS
- Demonstrates MVS superiority
- Shows separation between baseline and movement

```bash
python 7_compare_detection_methods.py
python 7_compare_detection_methods.py --plot  # Show 4×2 comparison
```

![Detection Methods Comparison](../../images/detection_method_comparison.png)

---

### 8. I/Q Constellation Plotter (`8_plot_constellation.py`)

**Purpose**: Visualize I/Q constellation diagrams

- Compares baseline (stable) vs movement (dispersed) patterns
- Shows all 64 subcarriers + selected subcarriers
- Reveals geometric signal characteristics

```bash
python 8_plot_constellation.py
python 8_plot_constellation.py --packets 1000
python 8_plot_constellation.py --subcarriers 47,48,49,50
python 8_plot_constellation.py --grid  # One subplot per subcarrier
```

---

### 9. ESP32 Variant Comparison (`9_compare_s3_vs_c6.py`)

**Purpose**: Compare CSI characteristics between ESP32 variants

- Compares signal quality between S3 and C6 chips
- Analyzes SNR differences and detection performance
- Helps choose optimal hardware for specific environments

```bash
python 9_compare_s3_vs_c6.py
python 9_compare_s3_vs_c6.py --plot
```

---

### 10. NBVI Parameters Optimization (`10_optimize_nbvi_params.py`)

**Purpose**: Grid search for optimal NBVI calibration parameters

- Optimizes alpha (NBVI weighting factor), min_spacing, and percentile
- Compares NBVI-selected band vs fixed band [11-22]
- Generates optimal configuration for config.py

```bash
python 10_optimize_nbvi_params.py              # Full grid search
python 10_optimize_nbvi_params.py c6           # Use only C6 data
python 10_optimize_nbvi_params.py --quick      # Quick search (fewer combinations)
```

**Current optimal configuration:**
- `NBVI_ALPHA = 0.5`, `NBVI_MIN_SPACING = 1`, `NBVI_PERCENTILE = 10`, `NOISE_GATE_PERCENTILE = 25`
- Achieves **Recall 96.4%, F1 98.2%** with zero configuration

---

## Usage Examples

### Basic Analysis Workflow

```bash
cd tools

# 1. Collect data (files saved in data/)
cd ..
./me run --collect-baseline
./me run --collect-movement
cd tools

# 2. Analyze raw data
python 1_analyze_raw_data.py

# 3. Optimize parameters
python 2_analyze_system_tuning.py --quick

# 4. Visualize MVS
python 3_analyze_moving_variance_segmentation.py --plot

# 5. Run unit tests
cd ..
pytest tests/ -v
```

### Advanced Analysis

```bash
# Compare detection methods
python 7_compare_detection_methods.py --plot

# Plot I/Q constellations
python 8_plot_constellation.py --packets 1000 --grid

# Compare ESP32 variants
python 9_compare_s3_vs_c6.py --plot
```

---

## Key Results

### Filter Optimization (Noisy Environment)

Tested on 60-second noisy baseline with C6 chip:

| Configuration | Recall | FP Rate | F1 Score |
|---------------|--------|---------|----------|
| Low-pass 11Hz only | 92.4% | 2.34% | 88.9% |
| **Low-pass 11Hz + Hampel (W=9, T=4)** | **92.1%** | **0.84%** | **93.2%** |

### NBVI Automatic Subcarrier Selection

**NBVI with `alpha=0.5`, `min_spacing=1`** achieves excellent results with zero configuration.

For complete NBVI algorithm documentation, see [ALGORITHMS.md](../ALGORITHMS.md#nbvi-automatic-subcarrier-selection).

For detailed performance metrics, see [PERFORMANCE.md](../../PERFORMANCE.md).

---

## Additional Resources

- [ALGORITHMS.md](../ALGORITHMS.md) - Algorithm documentation (MVS, NBVI, Hampel)
- [Micro-ESPectre](../README.md) - R&D platform documentation
- [ESPectre](../../README.md) - Main project with Home Assistant integration

---

## License

GPLv3 - See [LICENSE](../../LICENSE) for details.
