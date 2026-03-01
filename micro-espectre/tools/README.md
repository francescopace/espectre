# Analysis Tools

**Python scripts for CSI data analysis, algorithm optimization, and validation**

This directory contains analysis tools for developing and validating ESPectre's motion detection algorithms. These scripts are essential for parameter tuning, algorithm validation, and scientific analysis.

## Supported Chips

All analysis tools support any ESP32 variant with CSI capability:
- **ESP32** (original)
- **ESP32-C3**
- **ESP32-S3**
- **ESP32-C6**

Use `--chip <name>` to specify the chip (e.g., `--chip c3`, `--chip s3`). Most tools default to C6 if not specified.

For algorithm documentation (MVS, NBVI calibration, Hampel filter), see [ALGORITHMS.md](../ALGORITHMS.md).

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

- Default mode reads `dataset_info.json` and analyzes all explicit historical pairs
- Verifies labels are correct (baseline vs movement)
- Compares turbulence variance between states
- Prints a compact table with per-pair metrics (`Baseline Var`, `Movement Var`, `Ratio`, `Gap end->start`, status)
- Supports per-chip detailed mode on the most recent dataset for that chip

```bash
python 1_analyze_raw_data.py           # Historical table from dataset_info.json
python 1_analyze_raw_data.py --chip C6 # Detailed analysis on latest C6 dataset
python 1_analyze_raw_data.py --chip C3 # Detailed analysis on latest C3 dataset
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

- Compares RSSI, Mean Amplitude, Turbulence, and MVS detection methods
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

### 9. ESP32 Variant Comparison (`9_compare_chips.py`)

**Purpose**: Compare CSI characteristics between ESP32 variants

- Compares signal quality between S3 and C6 chips
- Analyzes SNR differences and detection performance
- Helps choose optimal hardware for specific environments

```bash
python 9_compare_chips.py
python 9_compare_chips.py --plot
```

---

### 10. ML Model Training (`10_train_ml_model.py`)

**Purpose**: Train, evaluate, and export the production ML model

- Trains the MLP detector with weighted binary cross-entropy
- Default training uses `--fp-weight 2.0` and context-aware MVS-guided sample weights
- Supports architecture experiments and feature-importance analysis
- Exports weights for both platforms:
  - `micro-espectre/src/ml_weights.py`
  - `components/espectre/ml_weights.h`

```bash
python 10_train_ml_model.py                # Train with default settings
python 10_train_ml_model.py --info         # Show dataset and split info
python 10_train_ml_model.py --experiment   # Compare model architectures
python 10_train_ml_model.py --fp-weight 2.0  # Penalize false positives 2x
python 10_train_ml_model.py --seed-search-until-improvement 20  # Stop at first better seed
python 10_train_ml_model.py --shap         # Show SHAP feature importance
```

For full training workflow and dataset preparation, see [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md#5-train-model).

---

### 11. Grid-Search Metadata Refresh (`11_refresh_gridsearch_metadata.py`)

**Purpose**: Refresh `dataset_info.json` with context-aware grid-search metadata

- Runs full MVS grid search on valid baseline/movement temporal pairs
- Applies pairing policy with configurable max delta (default: 30 minutes)
- Uses single-dataset fallback when a valid pair is not available
- Writes per-file metadata used by tests/training:
  - `optimal_subcarriers_gridsearch`
  - `optimal_threshold_gridsearch`
  - `optimal_pair_movement_file` / `optimal_pair_baseline_file` (paired mode only)

```bash
python 11_refresh_gridsearch_metadata.py
python 11_refresh_gridsearch_metadata.py --max-pair-minutes 30
python 11_refresh_gridsearch_metadata.py --max-pair-minutes 15
```

Use this tool after:
- collecting new datasets
- changing grid-search objective/logic
- changing temporal pairing policy

---

### 12. Gesture Model Training (`12_train_gesture_model.py`)

**Purpose**: Train, calibrate, and export the multi-class gesture model (`wave`, `circle_cw`, `no_gesture`, ...)

- Trains an OVR SVM-RBF classifier from `data/<label>/*.npz`
- Builds synthetic `no_gesture` from `baseline` + `movement`
- Uses fixed 2.0 s negative windows aligned with runtime behavior
- Auto-calibrates reject thresholds (`confidence`, `margin`) on hold-out data
- Optimizes reject thresholds with macro-F1 / balanced accuracy and recall constraints:
  - `no_gesture` recall target >= 50%
  - gesture class recall target >= 65%
- Exports weights for both platforms:
  - `micro-espectre/src/gesture_weights.py`
  - `components/espectre/gesture_weights.h`

```bash
python 12_train_gesture_model.py --info
python 12_train_gesture_model.py --seed 13 --window-seconds 2.0 --window-overlap 0.5 --window-labels wave,circle_cw --no-gesture-max-per-source 5
```

Notes:
- `packet-rate` is fixed to 100 pps (CLI option removed).
- Exported thresholds are consumed at runtime by both Python and C++ gesture detectors.

---

### 13. Gesture Streaming Benchmark (`13_test_gesture_stream.py`)

**Purpose**: Evaluate runtime gesture detection on a synthetic continuous stream (production-like)

- Always runs in **continuous** mode
- Always includes synthetic `no_gesture` from `baseline` + `movement`
- Always uses fixed runtime subcarriers (`[12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]`)
- Reports:
  - overall accuracy
  - per-class accuracy
  - confusion matrix
  - macro-F1 (3-class)
  - balanced accuracy (3-class)
  - constraint check (`no_gesture>=50%`, gesture classes `>=65%`)

```bash
python 13_test_gesture_stream.py
python 13_test_gesture_stream.py --seed 42 --segments 80 --chunk-seconds 2.0 --labels wave,circle_cw
```

Notes:
- Legacy options were intentionally removed to keep the benchmark deterministic:
  - `--mode`
  - `--model`
  - `--runtime-subcarriers`
  - `--packet-rate`
  - `--no-gesture`
  - `--no-gesture-sources`

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

# 2b. Refresh dataset metadata for context-aware tests/training
python 11_refresh_gridsearch_metadata.py

# 3. Train gesture model (includes threshold calibration)
python 12_train_gesture_model.py --seed 13 --window-seconds 2.0 --window-overlap 0.5 --window-labels wave,circle_cw --no-gesture-max-per-source 5

# 4. Run continuous gesture benchmark
python 13_test_gesture_stream.py --seed 42 --segments 80 --chunk-seconds 2.0 --labels wave,circle_cw

# 5. Visualize MVS
python 3_analyze_moving_variance_segmentation.py --plot

# 6. Run unit tests
cd ..
pytest tests/ -v
```

### Advanced Analysis

```bash
# Compare detection methods
python 7_compare_detection_methods.py --plot

# Plot I/Q constellations (auto-finds most recent dataset)
python 8_plot_constellation.py --chip S3 --packets 1000 --grid

# Compare ESP32 variants (auto-finds most recent datasets for available chips)
python 9_compare_chips.py --plot
```

---

## Key Results

### Filter Optimization (Noisy Environment)

Tested on 60-second noisy baseline with C6 chip:

| Configuration | Recall | FP Rate | F1 Score |
|---------------|--------|---------|----------|
| Low-pass 11Hz only | 92.4% | 2.34% | 88.9% |
| **Low-pass 11Hz + Hampel (W=9, T=4)** | **92.1%** | **0.84%** | **93.2%** |

### Automatic Band Selection

**NBVI** achieves excellent results with zero configuration, automatically selecting the optimal 12 subcarriers for each environment.

For complete algorithm documentation, see [ALGORITHMS.md](../ALGORITHMS.md#automatic-subcarrier-selection).

For detailed performance metrics, see [PERFORMANCE.md](../../PERFORMANCE.md).

---

## Additional Resources

- [ALGORITHMS.md](../ALGORITHMS.md) - Algorithm documentation (MVS, NBVI calibration, Hampel)
- [Micro-ESPectre](../README.md) - R&D platform documentation
- [ESPectre](../../README.md) - Main project with Home Assistant integration

---

## License

GPLv3 - See [LICENSE](../../LICENSE) for details.
