# 🛜 Micro-ESPectre 👻 - Analysis Tools

**Python scripts for CSI data analysis, algorithm optimization, and validation**

This directory contains analysis tools for developing and validating ESPectre's motion detection algorithms. These scripts are essential for parameter tuning, algorithm validation, and scientific analysis.

📚 **For algorithm documentation** (MVS, NBVI, Hampel filter), see [ALGORITHMS.md](../ALGORITHMS.md).

📊 **For production performance metrics**, see [PERFORMANCE.md](../../PERFORMANCE.md).

📊 **For data collection and ML datasets**, see [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md).

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Analysis Scripts](#analysis-scripts)
- [Usage Examples](#usage-examples)
- [Key Results](#key-results)

---

## 🚀 Quick Start

```bash
# Activate virtual environment
source ../venv/bin/activate

# Collect CSI data samples
cd ..
./me run --collect-baseline
./me run --collect-movement
cd tools

# Run analysis
python 2_analyze_system_tuning.py --quick
python 3_analyze_moving_variance_segmentation.py --plot
```

---

## 🔧 Analysis Scripts

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

**Purpose**: Analyze turbulence calculation with different filters

- Compares filtered vs unfiltered turbulence
- Validates Hampel filter effectiveness
- Optimizes filter parameters

```bash
python 5_analyze_filter_turbulence.py
python 5_analyze_filter_turbulence.py --plot             # Show plots
python 5_analyze_filter_turbulence.py --optimize-filters # Optimize
```

---

### 6. Hampel Parameter Optimization (`6_optimize_hampel_parameters.py`)

**Purpose**: Find optimal Hampel filter parameters

- Grid search over window sizes (3-9) and thresholds (2.0-4.0)
- Tests outlier detection configurations
- Validates filter effectiveness

```bash
python 6_optimize_hampel_parameters.py
```

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

## 📊 Usage Examples

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

## 🎯 Key Results

### NBVI Automatic Subcarrier Selection

**NBVI Weighted α=0.3 with Percentile p10** achieves **F1=97.1%** with zero configuration.

📚 **For complete NBVI algorithm documentation**, see [ALGORITHMS.md](../ALGORITHMS.md#nbvi-automatic-subcarrier-selection).

📊 **For detailed performance metrics**, see [PERFORMANCE.md](../../PERFORMANCE.md).

---

## 📚 Additional Resources

- [ALGORITHMS.md](../ALGORITHMS.md) - Algorithm documentation (MVS, NBVI, Hampel)
- [Micro-ESPectre](../README.md) - R&D platform documentation
- [ESPectre](../../README.md) - Main project with Home Assistant integration

---

## 📄 License

GPLv3 - See [LICENSE](../../LICENSE) for details.
