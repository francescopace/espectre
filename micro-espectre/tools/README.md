# 🛜 Micro-ESPectre 👻 - Analysis Tools

**Comprehensive suite of Python tools for CSI data analysis, algorithm optimization, and subcarrier selection**

This section presents the advanced analysis tools developed to optimize the ESPectre motion-detection system. These tools support data-driven parameter tuning, algorithm validation, and scientific analysis of CSI-based motion detection.

📊 **For ESPectre production performance metrics and benchmarks**, see [PERFORMANCE.md](../../PERFORMANCE.md) in the main repository.

> 🧪 **Unit Tests**: For automated unit tests of the core modules, see `micro-espectre/tests/`. Run with `pytest tests/ -v` from the `micro-espectre` directory.

## 📋 Table of Contents

- [Analysis Scripts](#analysis-scripts)
- [NBVI Algorithm](#nbvi-automatic-subcarrier-selection)
- [Usage Examples](#usage-examples)

> 📊 **Data Collection**: For CSI streaming, data collection, and ML dataset creation, see [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md).

## 🔧 Analysis Scripts

### 1. Raw Data Analysis (`1_analyze_raw_data.py`)
**Purpose**: Visualize raw CSI amplitude data and identify patterns

- Analyzes subcarrier patterns and noise characteristics
- Helps identify most informative subcarriers
- Visualizes signal strength distribution

**Usage**:
```bash
python 1_analyze_raw_data.py
```

---

### 2. System Tuning (`2_analyze_system_tuning.py`)
**Purpose**: Comprehensive grid search for optimal MVS parameters

- Tests subcarrier clusters, thresholds, and window sizes
- Shows confusion matrix for best configuration
- Finds optimal parameter combinations

**Usage**:
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

**Usage**:
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

**Usage**:
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

**Usage**:
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

**Usage**:
```bash
python 6_optimize_hampel_parameters.py
```

---

### 7. Detection Methods Comparison (`7_compare_detection_methods.py`)
**Purpose**: Compare different motion detection methods

- Compares RSSI, Mean Amplitude, Turbulence, and MVS
- Demonstrates MVS superiority
- Shows separation between baseline and movement

**Usage**:
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

**Usage**:
```bash
python 8_plot_constellation.py
python 8_plot_constellation.py --packets 1000
python 8_plot_constellation.py --subcarriers 47,48,49,50
python 8_plot_constellation.py --grid  # One subplot per subcarrier
```

---

### 9. ESP32-S3 vs ESP32-C6 Comparison (`9_compare_s3_vs_c6.py`)
**Purpose**: Compare CSI characteristics between different ESP32 variants

- Compares signal quality between S3 and C6 chips
- Analyzes SNR differences and detection performance
- Helps choose optimal hardware for specific environments

**Usage**:
```bash
python 9_compare_s3_vs_c6.py
python 9_compare_s3_vs_c6.py --plot
```

---

## 🧬 NBVI: Automatic Subcarrier Selection Algorithm

### Overview

**NBVI (Normalized Baseline Variability Index)** is a breakthrough algorithm for automatic subcarrier selection, achieving near-optimal performance (97.1% F1-score) with **zero manual configuration**.

### The Problem

WiFi CSI provides 64 subcarriers, but:
- ❌ Some are too weak (low SNR)
- ❌ Some are too noisy (high variance)
- ❌ Some are redundant (correlated)
- ✅ Manual selection works (97.3%) but doesn't scale

### The Solution

**NBVI Weighted Formula**:
```
NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
```

**Components**:
- **σ/μ²**: Energy normalization (penalizes weak subcarriers)
- **σ/μ**: Coefficient of Variation (rewards stability)
- **0.3/0.7**: Optimal weighting (validated empirically)

**Lower NBVI = Better subcarrier** (strong + stable)

### Algorithm Steps

1. **Collect Data**: 500-1000 packets (5-10s @ 100Hz)
2. **Percentile Analysis**: Find quietest windows using p10
3. **Calculate NBVI**: For all 64 subcarriers
4. **Noise Gate**: Exclude weak subcarriers (10th percentile)
5. **Select with Spacing**: Top 5 + 7 with Δf≥3
6. **Save to NVS**: Persist selected band

### Percentile-Based Detection ⭐

**NO threshold configuration needed!**

```python
# Analyzes sliding windows
for window in sliding_windows(buffer, size=100, step=50):
    variances.append(calculate_variance(window))

# Finds quietest windows using percentile
p10_threshold = np.percentile(variances, 10)  # Adaptive!
baseline_windows = [w for w in windows if variance <= p10_threshold]

# Uses best window for NBVI calibration
best_window = min(baseline_windows, key=lambda x: x.variance)
```

**Advantages**:
- ✅ Adapts to any environment automatically
- ✅ Finds better baseline windows (variance 0.26 vs 0.58)
- ✅ +3.0% average improvement vs threshold-based
- ✅ 100% success rate (4/4 vs 3/3)
- ✅ Zero configuration required

**Trade-off**: Requires larger buffer (500-1000 packets) for reliable percentile analysis

### Performance Comparison

**vs All 23+ Strategies Tested**:

| Rank | Strategy | Pure | Mixed | Config | Overall |
|------|----------|------|-------|--------|---------|
| 🥇 | **NBVI Percentile p10** | **97.1%** | **91.2%** | **Zero** | **WINNER** |
| 🥈 | NBVI Threshold | 97.1% | 86.9% | Adaptive | Good |
| 🥉 | Default (manual) | 97.3% | N/A | Manual | Baseline |
| 4 | Variance-only | 92.4% | FAIL | Adaptive | Fails |
| 5-23 | Others | 63-92% | N/A | Various | Inferior |

### Geometric Insight

From I/Q constellation analysis:
- **Baseline**: Large radius (μ) + thin ring (σ) = stable, strong
- **Movement**: Small radius + thick ring = dispersed, weak

**NBVI captures both patterns** in a single metric!

### Implementation Status

- ✅ **Validated**: 1000+ packets real CSI data
- ✅ **Robust**: Works on mixed data (80/20)
- ✅ **Threshold-free**: Percentile-based (zero config)
- ✅ **Production-ready**: Implemented in Python (micro-espectre)
- ✅ **Python Implementation**: `src/nbvi_calibrator.py` (fully functional)

### Python Implementation

The NBVI algorithm is fully implemented in `micro-espectre/src/nbvi_calibrator.py`:

```python
from src.nbvi_calibrator import NBVICalibrator

# Initialize calibrator
calibrator = NBVICalibrator(
    buffer_size=1000,      # Packets to collect (500-1000 recommended)
    percentile=10,         # 10th percentile for baseline detection
    alpha=0.3,             # NBVI weighting factor
    min_spacing=3          # Minimum spacing between subcarriers
)

# Collect CSI packets
for packet in csi_stream:
    progress = calibrator.add_packet(packet['csi_data'])
    if progress >= calibrator.buffer_size:
        break

# Calibrate and get optimal band
selected_band = calibrator.calibrate(current_band=[11,12,13,14,15,16,17,18,19,20,21,22])

if selected_band:
    print(f"Optimal band: {selected_band}")
    # Save to NVS and use for detection
```

**Key Features**:
- Two-pass variance algorithm for numerical stability
- Memory-efficient buffer (stores magnitudes as int16)
- Configurable window size and step for sliding window analysis
- Automatic noise gate and spectral spacing

### Configuration Parameters

**Recommended settings**:

```python
# Percentile-based (recommended)
NBVICalibrator(
    buffer_size=1000,          # 10s @ 100Hz (ideal)
    percentile=10,             # 10th percentile
    alpha=0.3,                 # NBVI weighting
    min_spacing=3              # Spectral de-correlation
)

# Memory-constrained (alternative)
NBVICalibrator(
    buffer_size=500,           # 5s @ 100Hz (minimum)
    percentile=10,
    alpha=0.3,
    min_spacing=3
)
```

**Window Analysis Parameters** (configurable in `config.py`):
- `NBVI_WINDOW_SIZE = 100`: Window size for baseline detection
- `NBVI_WINDOW_STEP = 50`: Step size for sliding window analysis

**Note**: 
- The Python implementation (`src/nbvi_calibrator.py`) uses **only percentile-based detection**
- Unit tests in `tests/test_nbvi_calibrator.py` validate the algorithm correctness
- README recommends `buffer_size=1000` for optimal performance when memory allows

### C++ Implementation (ESPHome)

The NBVI algorithm is now fully implemented in the ESPHome component (`components/espectre/calibration_manager.cpp`). The calibration runs automatically at first boot and can be triggered via factory reset.

See the main [ESPectre documentation](https://github.com/francescopace/espectre) for configuration details.

## 📊 Usage Examples

### Basic Analysis Workflow

```bash
cd tools

# 1. Collect data (files will be saved in data/)
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

## 🎯 Key Results Summary

### Best Automatic Subcarrier Selection Method

**NBVI Weighted α=0.3 with Percentile p10**:
- Pure data: **F1=97.1%** (gap -0.2% vs manual)
- Mixed data: **F1=91.2%** (4/4 calibrated)
- **Zero configuration** (no threshold needed)
- **Production-ready** for ESP32

### Comparison with Other Methods

| Method | Pure | Mixed | Threshold? | Status |
|--------|------|-------|------------|--------|
| NBVI Percentile | 97.1% | 91.2% | ❌ NO | 🏆 Best |
| NBVI Threshold | 97.1% | 86.9% | ⚠️ Yes | Good |
| Variance-only | 92.4% | FAIL | ⚠️ Yes | Fails |
| Ring Geometry | 92.2% | N/A | N/A | Insights |
| Hybrid | 92.2% | N/A | N/A | No improvement |

### Ring Geometry Analysis Results

Tested 17 strategies (12 geometric + 4 hybrid + 1 variance):
- Best geometric: movement_instability (F1=92.2%)
- Best hybrid: hybrid_variance_instability (F1=92.2%)
- **Conclusion**: Hybrid strategies add NO value (0.0% improvement)

## 📚 Scientific Contributions

### Validated Findings

1. **NBVI Algorithm**: Near-optimal automatic selection (97.1%)
2. **Percentile-Based Detection**: Superior to threshold (+3.0%)
3. **Ring Geometry**: Provides insights but limited practical value
4. **Hybrid Strategies**: Redundant - no improvement over pure methods
5. **Variance Dominance**: Simple variance outperforms complex features

### Key Insights

- **Simplicity wins**: Variance-only beats complex geometric features
- **Weighting essential**: α=0.3 critical for NBVI performance
- **Spacing matters**: Δf≥3 improves diversity and performance
- **Percentile adapts**: Finds better baselines than fixed threshold
- **Manual still best**: +0.2% gap shows room for improvement

## 🔬 Technical Details

### The Problem: Subcarrier Selection at Scale

WiFi CSI provides 52-64 usable subcarriers (IEEE 802.11n/ax 20MHz), but not all are equally informative for motion detection. Each subcarrier's sensitivity depends on its specific multipath components, which can be dominated by:
- Line-of-Sight (LoS) paths
- Static reflections (walls, furniture)
- Environmental noise

**Challenge**: Manual selection works (F1=97.3%) but doesn't scale. Optimal subcarriers vary significantly with environment (layout, materials, object placement).

**Solution Needed**: Unsupervised, calibration-free selection that adapts to any environment in real-time.

### The Geometric Insight: I/Q Constellation Analysis

Visual analysis of I/Q constellation diagrams reveals the key discriminator:

**Baseline (Idle State)**:
- Large radius (high magnitude μ) → Strong signal
- Thin ring (low std σ) → Stable, coherent signal
- Geometric pattern: Compact, well-defined circle

**Movement (Active State)**:
- Small radius (low magnitude) → Weaker signal
- Thick ring (high std) → Dispersed, entropic pattern
- Geometric pattern: Scattered, irregular distribution

**Optimal subcarriers** exhibit maximum contrast between these states, maximizing the Sensing Signal-to-Noise Ratio (SSNR).

### NBVI Formula: Capturing Both Patterns

**Original Formula** (from technical report):
```
NBVI = σ/μ²
```

**Problem**: Energy normalization (1/μ²) too aggressive
- Over-penalizes strong subcarriers
- Example: μ=35, σ=2 → NBVI=0.00163 vs μ=25, σ=1 → NBVI=0.00160
- Second subcarrier preferred despite worse SNR (25/1=25 vs 35/2=17.5)

**Optimized Formula** (validated empirically):
```
NBVI_weighted = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
              = 0.3 × Energy_Component + 0.7 × CV
```

**Why α=0.3 is Optimal**:
- **30% Energy normalization** (σ/μ²): Penalizes weak subcarriers (small μ)
- **70% Coefficient of Variation** (σ/μ): Rewards relative stability (thin ring)
- **Balance**: Prevents over-penalization while maintaining energy bias

**Lower NBVI = Better subcarrier** (strong + stable)

### Algorithm Components

#### 1. Noise Gate (Quality Control)

**Problem**: Weak subcarriers appear stable (low σ) but have low SNR
- Example: μ=1, σ=0.1 → CV=0.1 (looks stable!)
- But signal too weak for reliable detection

**Solution**: Exclude subcarriers below 10th percentile of mean magnitude
- Ensures only subcarriers with sufficient energy are considered
- Prevents selection of "dead" or extremely weak subcarriers

**Reference**: [13] A Novel Passive Indoor Localization Method by Fusion CSI Amplitude and Phase Information

#### 2. Spectral De-correlation (Feature Diversity)

**Problem**: Adjacent subcarriers are correlated (OFDM mechanism)
- Selecting all 12 from same region → redundant information
- Can lead to overfitting or poor multipath diversity

**Solution**: Hybrid spacing strategy
- **Top 5**: Always include (absolute priority by NBVI)
- **Remaining 7**: Select with minimum spacing Δf≥3
- Balances quality (NBVI score) with diversity (spectral separation)

**Reference**: [16] Subcarrier selection for efficient CSI-based indoor localization

#### 3. Percentile-Based Detection (Threshold-Free)

**Problem**: Absolute thresholds fail across environments
- Noisy environment: baseline variance = 1.0-2.0
- Quiet environment: baseline variance = 0.1-0.3
- Fixed threshold doesn't adapt

**Solution**: Percentile-based adaptive detection

**How it works**:
1. Collect buffer (500-1000 packets = 5-10s @ 100Hz)
2. Calculate variance for sliding windows (100 packets, step 50)
3. Compute p10 threshold from window variances (adaptive!)
4. Select window with minimum variance below p10
5. Calibrate NBVI on that window

**Implementation Details**:
- **Buffer size**: 500-1000 packets (configurable based on memory)
- **Window size**: 100 packets (1s @ 100Hz)
- **Step size**: 50 packets (50% overlap for better coverage)
- **Percentile**: 10th percentile (finds quietest 10% of windows)

**Example**:
```python
# Stream with mixed activity
variances = [6.1, 5.8, 0.3, 0.2, 0.3, 5.9, 0.4, ...]

# Adaptive threshold (no configuration!)
p10 = np.percentile(variances, 10)  # = 0.35

# Baseline windows (variance ≤ 0.35)
baseline_windows = [0.3, 0.2, 0.3]  # Automatically found!

# Use best window (variance=0.2) for calibration
```

**Why it's better**:
- ✅ Finds variance=0.26 vs 0.58 (threshold-based) → Better baseline
- ✅ +3.0% average improvement, +4.3% best case
- ✅ Zero configuration required
- ✅ Adapts to any environment automatically

**Trade-offs**:
- Requires larger buffer (500-1000 vs 200 packets)
- Longer calibration time (5-10s vs 2s)
- Worth it for +4.3% performance improvement

**Reference**: [7] Using Wi-Fi Channel State Information (CSI) for Human Activity Recognition

#### 4. Threshold-Based Detection (Alternative)

**When to use**: Memory-constrained environments or when faster calibration is needed

**How it works**:
1. Collect initial sample (100 packets)
2. Calculate adaptive threshold: `threshold = baseline_variance × 2.0`
3. Monitor sliding window (100 packets)
4. When variance drops below threshold → calibrate

**Advantages**:
- Faster calibration (2s vs 5-10s)
- Lower memory footprint (200 packets vs 500-1000)
- Still performs well (F1=86.9%)

**Trade-offs**:
- Requires threshold tuning for different environments
- Less robust than percentile-based (-4.3% performance)

**Note**: The Python implementation (`src/nbvi_calibrator.py`) implements **only percentile-based detection**. Unit tests in `tests/test_nbvi_calibrator.py` validate the algorithm correctness.

### Computational Efficiency

**Complexity**: O(N·L) - Linear in subcarriers and packets
- Mean/std calculation: O(L) per subcarrier
- Total: O(N·L) = O(64·1000) = ~64k operations
- Suitable for all ESP32 variants

**Avoids**:
- ❌ PCA (Principal Component Analysis): O(N³) - too expensive
- ❌ SVD (Singular Value Decomposition): O(N³) - prohibitive
- ❌ Complex ML models: Require training data

**Reference**: [11] Linear-Complexity Subcarrier Selection Strategy for Fast Preprocessing

### Integration with MVS Detection

**Pipeline (Micro-ESPectre)**:
```
CSI Data → NBVI Selection → MVS Detection → MQTT Publishing
```

**Pipeline (ESPHome)**:
```
CSI Data → NBVI Selection → MVS Detection → ESPHome Native API → Home Assistant
```

1. **NBVI Selection** (one-time, at boot):
   - Analyzes 1000 packets baseline
   - Selects optimal 12 subcarriers
   - Saves to NVS (persistent)

2. **MVS Detection** (continuous, runtime):
   - Uses selected subcarriers only
   - Calculates spatial turbulence
   - Applies moving variance segmentation
   - Detects motion with high SSNR

**Benefit**: NBVI optimization ensures movement causes high variance change, improving sensitivity and reducing false positives.

**Reference**: [8] MVS segmentation: the fused CSI stream and corresponding moving variance sequence

## 📖 References

For the complete list of scientific references and academic papers that informed the development of these algorithms, see the **[References section in the main README](../../README.md#-references)**.

## 📚 Additional Resources

- [Micro-ESPectre](../README.md) - Python implementation for MicroPython
- [ESPectre ](../../README.md) - Main project with native Home Assistant integration

## 📄 License

GPLv3 - See [LICENSE](../../LICENSE) for details.
