# Analysis Tools

**Python scripts for CSI data analysis, algorithm optimization, and validation**

This directory contains analysis tools for developing and validating ESPectre's motion detection algorithms. These scripts are essential for parameter tuning, algorithm validation, and scientific analysis.

For algorithm documentation (MVS, P95 Band Selection, Hampel filter), see [ALGORITHMS.md](../ALGORITHMS.md).

For production performance metrics, see [PERFORMANCE.md](../../PERFORMANCE.md).

For data collection and ML datasets, see [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md).

---

## Table of Contents

- [Analysis Scripts](#analysis-scripts)
- [Usage Examples](#usage-examples)
- [Key Results](#key-results)

---

## Analysis Scripts

### 1. Raw Data Analysis (`1_analyze_raw_data.py`)

**Purpose**: Analyze data quality and verify dataset integrity

- Automatically discovers and analyzes all available chip datasets
- Verifies labels are correct (baseline vs movement)
- Compares turbulence variance between states
- Shows summary table with status for each chip

```bash
python 1_analyze_raw_data.py           # Analyze all datasets (C6, S3, etc.)
python 1_analyze_raw_data.py --chip C6 # Analyze only C6 dataset
python 1_analyze_raw_data.py --chip S3 # Analyze only S3 dataset
```

---

### 2. System Tuning (`2_analyze_system_tuning.py`)

**Purpose**: Grid search for optimal MVS parameters

- Tests subcarrier clusters, thresholds, and window sizes
- Shows confusion matrix for best configuration
- Finds optimal parameter combinations

```bash
python 2_analyze_system_tuning.py              # Full grid search (default: C6)
python 2_analyze_system_tuning.py --chip S3    # Use S3 dataset
python 2_analyze_system_tuning.py --quick      # Reduced parameter space
```

---

### 3. MVS Visualization (`3_analyze_moving_variance_segmentation.py`)

**Purpose**: Visualize MVS algorithm behavior

- Shows moving variance, threshold, and detection states
- Displays confusion matrix and performance metrics
- Validates current configuration

```bash
python 3_analyze_moving_variance_segmentation.py              # Use C6 dataset
python 3_analyze_moving_variance_segmentation.py --chip S3    # Use S3 dataset
python 3_analyze_moving_variance_segmentation.py --plot       # Show graphs
```

---

### 4. Filter Location Analysis (`4_analyze_filter_location.py`)

**Purpose**: Compare filter placement in processing pipeline

- Tests pre-filtering vs post-filtering approaches
- Evaluates impact on motion detection accuracy
- Determines optimal filter location

```bash
python 4_analyze_filter_location.py              # Use C6 dataset
python 4_analyze_filter_location.py --chip S3    # Use S3 dataset
python 4_analyze_filter_location.py --plot       # Show visualizations
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
python 5_analyze_filter_turbulence.py              # Use C6 dataset
python 5_analyze_filter_turbulence.py --chip S3    # Use S3 dataset
python 5_analyze_filter_turbulence.py --plot       # Show 4-panel visualization
python 5_analyze_filter_turbulence.py --optimize-filters  # Optimize parameters
```

---

### 6. Filter Parameters Optimization (`6_optimize_filter_params.py`)

**Purpose**: Optimize low-pass and Hampel filter parameters

- Optimizes low-pass cutoff frequency and threshold parameters
- Grid search for Hampel filter parameters (window, threshold)
- Auto-detects chip from baseline file metadata (ensures matching movement data)
- Automatically selects optimal subcarrier band based on subcarrier count
- Finds optimal configuration for noisy environments

```bash
python 6_optimize_filter_params.py              # Low-pass optimization
python 6_optimize_filter_params.py c6           # Use only C6 data
python 6_optimize_filter_params.py --hampel     # Hampel optimization
python 6_optimize_filter_params.py c6 --hampel  # C6 + Hampel
python 6_optimize_filter_params.py --all        # Combined optimization (low-pass + Hampel)
```

**Current optimal configuration (60s noisy baseline):**
- Low-pass: Cutoff=11 Hz, Target=28 → Recall 92.4%, FP 2.3%
- With Hampel: Window=9, Threshold=4.0 → **Recall 92.1%, FP 0.84%, F1 93.2%**

---

### 7. Detection Methods Comparison (`7_compare_detection_methods.py`)

**Purpose**: Compare different motion detection algorithms

- Compares RSSI, Mean Amplitude, Turbulence, PCA, and MVS detection methods
- Uses production `PCADetector` from `src/pca_detector.py`
- Demonstrates MVS superiority with simpler approach and lower CPU
- Shows separation between baseline and movement

```bash
python 7_compare_detection_methods.py              # Use C6 dataset
python 7_compare_detection_methods.py --chip S3    # Use S3 dataset
python 7_compare_detection_methods.py --plot       # Show 5×2 comparison
```

![Detection Methods Comparison](../../images/detection_method_comparison.png)

---

### 8. I/Q Constellation Plotter (`8_plot_constellation.py`)

**Purpose**: Visualize I/Q constellation diagrams

- Compares baseline (stable) vs movement (dispersed) patterns
- Shows all 64 subcarriers (HT20) + selected subcarriers
- Reveals geometric signal characteristics

```bash
python 8_plot_constellation.py              # Use C6 dataset
python 8_plot_constellation.py --chip S3    # Use S3 dataset
python 8_plot_constellation.py --packets 1000
python 8_plot_constellation.py --packets 200 --offset 50  # Start from packet 50
python 8_plot_constellation.py --subcarriers 47,48,49,50
python 8_plot_constellation.py --grid       # One subplot per subcarrier
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

## Usage Examples

### Basic Analysis Workflow

```bash
cd tools

# 0. Collect data (files saved in data/)
# Requires two terminals:
#   Terminal 1: ./me stream --ip <PC_IP>
#   Terminal 2: ./me collect --label baseline --duration 60
#               ./me collect --label movement --duration 30
# see ../ML_DATA_COLLECTION.md for details

# 1. Analyze raw data
python 1_analyze_raw_data.py

# 2. Optimize parameters
python 2_analyze_system_tuning.py --quick

# 3. Visualize MVS
python 3_analyze_moving_variance_segmentation.py --plot

# 4. Run unit tests
cd ..
pytest tests/ -v
```

### Advanced Analysis

```bash
# Compare detection methods
python 7_compare_detection_methods.py --plot

# Plot I/Q constellations (auto-finds most recent dataset)
python 8_plot_constellation.py --chip S3 --packets 1000 --grid

# Compare ESP32 variants (auto-finds most recent C6 and S3 datasets)
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

### P95 Automatic Band Selection

**P95 Band Selection** achieves excellent results with zero configuration, automatically selecting the optimal 12-subcarrier band for each environment.

For complete algorithm documentation, see [ALGORITHMS.md](../ALGORITHMS.md#automatic-subcarrier-selection).

For detailed performance metrics, see [PERFORMANCE.md](../../PERFORMANCE.md).

---

## Additional Resources

- [ALGORITHMS.md](../ALGORITHMS.md) - Algorithm documentation (MVS, P95 Band Selection, Hampel)
- [Micro-ESPectre](../README.md) - R&D platform documentation
- [ESPectre](../../README.md) - Main project with Home Assistant integration

---

## License

GPLv3 - See [LICENSE](../../LICENSE) for details.
