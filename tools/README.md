# Analysis And Benchmark Tools

**Python scripts for CSI data analysis, algorithm optimization, firmware benchmarking, and validation**

This directory contains analysis tools for developing and validating ESPectre's motion detection algorithms. These scripts are essential for parameter tuning, algorithm validation, and scientific analysis.

## Supported Chips

All analysis tools support any ESP32 variant with CSI capability:
- **ESP32** (original)
- **ESP32-C3**
- **ESP32-S3**
- **ESP32-C6**

Use `--chip <name>` to specify the chip (e.g., `--chip c3`, `--chip s3`). Most tools default to C6 if not specified.

For algorithm documentation (Classic, historical baselines, fixed subcarriers, Hampel filter), see [ALGORITHMS.md](../docs/ALGORITHMS.md).

For production performance metrics, see [docs/performance](../docs/performance/README.md).

For data collection and ML datasets, see [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md).

---

## Table of Contents

- [Analysis Scripts](#analysis-scripts)
- [Usage Examples](#usage-examples)
- [Key Results](#key-results)

---

## Analysis Scripts

### 1. Raw Data Analysis (`analyze_raw_data.py`)

**Purpose**: Analyze data quality and verify dataset integrity

- Default mode reads `dataset_info.json` and analyzes all explicit historical pairs
- Verifies labels are correct (`static_presence` vs `motion`)
- Compares turbulence variance between states
- Prints a compact table with per-pair metrics (`Static Presence Var`, `Motion Var`, `Ratio`, `Gap end->start`, status)
- Supports per-chip detailed mode on the most recent dataset for that chip

```bash
python analyze_raw_data.py           # Historical table from dataset_info.json
python analyze_raw_data.py --chip C6 # Detailed analysis on latest C6 dataset
python analyze_raw_data.py --chip C3 # Detailed analysis on latest C3 dataset
```

---

### 2. System Tuning (`analyze_system_tuning.py`)

**Purpose**: Grid search for optimal Classic detector parameters on the fixed production band

- Tests threshold and window-size combinations using the fixed production subcarriers
- Does not search subcarrier combinations anymore; the band is shared from `config.py`
- Shows confusion matrix for best configuration
- Finds optimal parameter combinations

```bash
python analyze_system_tuning.py              # Full grid search (default: C6)
python analyze_system_tuning.py --chip S3    # Use S3 dataset
python analyze_system_tuning.py --quick      # Reduced parameter space
```

### 3. Filter Location Analysis (`analyze_filter_location.py`)

**Purpose**: Compare filter placement in processing pipeline

- Tests pre-filtering vs post-filtering approaches
- Evaluates impact on motion detection accuracy
- Determines optimal filter location

```bash
python analyze_filter_location.py              # Use C6 dataset
python analyze_filter_location.py --chip S3    # Use S3 dataset
python analyze_filter_location.py --plot       # Show visualizations
```

---

### 4. Filter Turbulence Analysis (`analyze_filter_turbulence.py`)

**Purpose**: Run the production-aligned paired variance-baseline sweep and compare candidate detector variants

- Sweeps all explicit `static_presence` / `motion` pairs from `data/dataset_info.json` by default
- Mirrors the current startup/runtime path: fixed production subcarriers, startup adaptive threshold, and continuous baseline -> motion evaluation
- Compares detector variants such as `baseline`, `baseline_tracking`, and `subcarrier_ema_norm`
- Supports optional filter-profile comparison mode (`production`, `no_filter`, `hampel_only`, `lowpass_only`, `hampel_lowpass`)
- Reports aggregate metrics, per-chip breakdown, worst-pair regressions, and tracking diagnostics
- Supports `--plot` for a single selected pair to visualize moving variance and threshold evolution

**Current lesson**: the plain production baseline remains the safest global default. Online threshold tracking is chip-dependent, and per-subcarrier EMA normalization is still experimental.

```bash
python analyze_filter_turbulence.py
python analyze_filter_turbulence.py --variant baseline_tracking
python analyze_filter_turbulence.py --chip S3 --variant baseline_tracking
python analyze_filter_turbulence.py --compare-filters --filter-profile all
python analyze_filter_turbulence.py --dataset-id <pair_id> --plot
```

---

### 5. Filter Parameters Optimization (`optimize_filter_params.py`)

**Purpose**: Run paired filter-parameter sweeps on top of the same production-aligned variance evaluator

- Reuses the shared paired sweep core instead of selecting the latest files by modification time
- Evaluates explicit `dataset_info.json` pairs, optionally filtered by chip
- Low-pass sweep mode compares runtime-relevant low-pass settings over the paired datasets
- Hampel sweep mode compares `(window, threshold)` combinations over the paired datasets
- `--all` runs low-pass first, then a Hampel sweep using the best low-pass setting found in that run

```bash
python optimize_filter_params.py
python optimize_filter_params.py c6
python optimize_filter_params.py --hampel
python optimize_filter_params.py c6 --hampel
python optimize_filter_params.py --all
```

---

### 6. Detection Methods Comparison (`compare_detection_methods.py`)

**Purpose**: Compare different motion detection algorithms

- Compares the RSSI baseline against the production Classic and ML detectors
- Classic calibrates its threshold from the selected static capture using its production startup logic
- Shows separation between static presence and motion

```bash
python compare_detection_methods.py              # Use C6 dataset
python compare_detection_methods.py --chip S3    # Use S3 dataset
python compare_detection_methods.py --plot       # Show per-method comparison
```

---

### 7. I/Q Constellation Plotter (`plot_constellation.py`)

**Purpose**: Visualize I/Q constellation diagrams

- Compares static presence (stable) vs motion (dispersed) patterns
- Shows all 64 subcarriers (HT20) plus the fixed production subcarriers
- Reveals geometric signal characteristics

```bash
python plot_constellation.py              # Use C6 dataset
python plot_constellation.py --chip S3    # Use S3 dataset
python plot_constellation.py --packets 1000
python plot_constellation.py --packets 200 --offset 50  # Start from packet 50
python plot_constellation.py --grid       # One subplot per subcarrier
```

---

### 8. ESP32 Variant Comparison (`compare_chips.py`)

**Purpose**: Compare CSI characteristics between ESP32 variants

- Compares signal quality between S3 and C6 chips
- Analyzes SNR differences and detection performance
- Helps choose optimal hardware for specific environments

```bash
python compare_chips.py
python compare_chips.py --plot
```

---

### 9. ML Model Training (`train_ml_model.py`)

**Purpose**: Train, evaluate, and export the production ML model

Install the ML requirements before using this script:

```bash
pip install -r requirements-ml.txt
```

The main repository workflow and this training stack target Python `3.14`.

- Trains the MLP detector with weighted binary cross-entropy
- Default training uses `--fp-weight 2.0`, `--scaler standard`, `--batch-size 1024`, `--device cpu`, and grouped session-level CV with uniform sample weights
- Caches the derived feature matrix for repeated local runs; use `--no-cache` to rebuild
- Reports blocked out-of-fold metrics plus worst session/chip/source-file groups
- Uses a PyTorch MLP trainer and exports runtime-compatible weights for both platforms only after explicit promotion
- Supports FP-first architecture and FP-weight campaigns, gain-shift diagnostics, and feature-importance analysis
- Ranks candidates with the paired gate first, then grouped CV; long-recording
  checks stay in the performance report and dedicated pytest suites
- Exports weights for both platforms:
  - `src/python/micro_espectre/ml_weights.py`
  - `src/cpp/core/ml_weights.h`

```bash
python train_ml_model.py                # Train and export if the paired gate passes
python train_ml_model.py --no-export    # Evaluate without replacing runtime artifacts
python train_ml_model.py --info         # Show dataset and split info
python train_ml_model.py --experiment   # Run the FP-first MLP topology campaign
python train_ml_model.py --experiment --experiment-promote  # Promote the winner if it beats the baseline
python train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"  # Custom shortlist
python train_ml_model.py --experiment-fp-weights "1,1.5,2,2.5,3"  # Gated multi-seed FP-weight campaign
python train_ml_model.py --fp-weight 2.0  # Penalize false positives 2x
python train_ml_model.py --scaler clipped_standard  # Robust clipping + z-score
python train_ml_model.py --batch-size 32  # Smaller-batch comparison
python train_ml_model.py --device cuda    # Force CUDA when available
python train_ml_model.py --device mps     # Force Apple GPU when available
python train_ml_model.py --no-cache       # Rebuild cached training matrix
python train_ml_model.py --exclude-chip ESP32  # Run a chip-exclusion experiment
python train_ml_model.py --seed-search-until-improvement 20  # Stop at first better seed
python train_ml_model.py --gain-stress-gate  # Stress exported model with artificial feature gain shifts
python train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25  # Custom stress multipliers
python train_ml_model.py --shap         # Grouped OOF SHAP (200 samples)
python train_ml_model.py --shap 500     # Grouped OOF SHAP (500 samples)
python train_ml_model.py --ablation-feature turb_skewness --seed 1386543369  # Targeted CV and real-data ablation
```

For the complete ML training workflow, promotion guidance, gain-stress
diagnostics, and post-training regressions, see
[ML_TRAINING.md](../docs/ML_TRAINING.md). For dataset preparation and labeling,
see [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md).

### 10. Dataset Quality Validation (`validate_dataset_quality.py`)

Validates the shared Classic and ML datasets for metadata completeness, file
integrity, signal quality, pair diagnostics, training readiness, and long-recording
coverage. Admission FAILs stop the run; Classic replay scores stay review-only.
See
[2026-07-17-separate-dataset-admission-from-classic-diagnostics.md](../docs/adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md).

Defaults on every run:

- refresh derived `static_presence` / `motion` pair fields in
  `data/dataset_info.json`, writing the file and bumping `updated_at` only when
  those fields actually change
- write `data/auto_generated/DATASET_QUALITY_CHECK.md` unless `--no-report` is
  set

Report layout: Quality Check Summary and Validation Domains first, then the
score tables, then Validation rule / computed-metric notes. `Breath` folds
segment coverage and quiet-adult human-rate Hz weight into one score; Presence
and Empty share that ladder, and Empty only inverts the soft marks (high
`Breath` means presence-like contamination).

**Checks performed:**
- Metadata completeness — required dataset metadata exists, disk captures are
  registered, pair links are reciprocal, and paired chip, subcarrier, device,
  and environment metadata agree
- File integrity — NPZ loads, CSI I/Q shape and declared subcarrier count agree,
  per-packet arrays align, and the embedded label matches the dataset directory
- Signal quality — amplitude range, zero-packet detection, packet cadence, and stream continuity
- Pair validation — production-aligned threshold replay on explicit `static_presence` / `motion` pairs
- Empty sanity — each `empty` capture is evaluated independently;
  self-calibrated motion activation and segmented
  respiration evidence flag motion-like, presence-like, or unstable empty files
- Presence evidence — 30-second `static_presence` segments combine respiration-band
  spectral prominence, temporal autocorrelation, and subcarrier support, then keep
  only frequency-consistent peaks near the median candidate, without requiring a
  paired `empty` capture
- Quiet-test sanity — idle-only `test` recordings stay quiet under Classic replay
- ML readiness — binary balance with `empty + static_presence` mapped to IDLE,
  usable windows after per-file warm-up, chip/environment coverage, and grouped
  session coverage for three-fold CV
- Long-recording coverage — quiet recordings are distinguished from mixed
  recordings with `motion starts at packet N`; mixed annotations must leave
  usable IDLE and MOTION segments after warm-up

Turbulence mode follows runtime conventions: CV-normalized turbulence for every
file. ML uses the same normalized base turbulence and exports the production
Core-6 neural-detector features.

```bash
python validate_dataset_quality.py                  # Full validation (auto report + metadata refresh)
python validate_dataset_quality.py --chip C6        # Validate C6 only
python validate_dataset_quality.py --no-report      # Skip markdown report
```

---

### 11. Performance Report Generation (`generate_performance_report.py`)

**Purpose**: Regenerate `docs/performance/README.md` from the current validation
datasets

- Reuses the shared performance replay helpers that also back the Python
  paired real-data and long-recording validation suites
- Recomputes the published Classic and ML aggregate tables directly from the
  current `data/` captures
- Builds and runs the host-side C++ integration suites before publishing so
  Python and C++ drift is caught immediately
- Keeps the checked regression behavior and the published documentation aligned
  without copying metric logic into the Markdown file or trusting only one
  implementation

```bash
python generate_performance_report.py
python generate_performance_report.py --stdout
python generate_performance_report.py --output /tmp/PERFORMANCE.md
```

---

### 12. Firmware Benchmark (`benchmark_firmware.py`)

**Purpose**: Run the live ESPHome and Native firmware benchmark for one
connected chip and write its generated report under `docs/performance/`

The benchmark auto-detects the serial port and flashes and monitors these
variants in order:

1. ESPHome Dev Classic
2. ESPHome Dev ML
3. Native Classic
4. Native ML

Each frontend starts with a clean Classic build. Its ML variant reuses the same
build directory for an incremental build. To reduce total runtime, the ML build
runs while the already-flashed Classic firmware is being monitored. No two
builds run at the same time. Each variant is monitored for three minutes, and
the tool evaluates firmware size, application-partition space, packet rate,
motion-state logging, heap, runtime load, loop timing, and detector timing.
Motion transitions are recorded for context but do not affect the result
because the environment may be occupied.

Detector timing covers the shared runtime state-evaluation step, not the
per-packet ingestion step. The runtime measures every evaluation tick and
reports accumulated duration, sample count, minimum, and maximum. The tool
computes the overall average from total duration divided by total samples and
ignores telemetry windows without detector samples for timing statistics.
Keep local ESPHome and Native Wi-Fi credentials configured. Native falls back
to the local Streamer `sdkconfig.wifi` when it has no frontend-local Wi-Fi
defaults.

```bash
python benchmark_firmware.py --chip c3
```

The command exits successfully only when all four variants pass. It still
writes a partial report if a build, flash, monitor, or runtime check fails.

---

## Usage Examples

### Basic Analysis Workflow

```bash
cd tools

# 0. Collect data (files saved in data/)
# Requires two terminals:
#   Terminal 1: ESPectre streamer firmware running with collector IP/port set to this host
#   Terminal 2: ./espectre collect --label static_presence --duration 60
#               ./espectre collect --label motion --duration 30
# Optional live-inspection terminal:
#               ./espectre collect --target 192.168.1.50
# see ../docs/ML_DATA_COLLECTION.md for details

# 1. Analyze raw data
python analyze_raw_data.py

# 2. Optimize parameters
python analyze_system_tuning.py --quick

# 3. Compare filter placement
python analyze_filter_location.py --plot

# 4. Run unit tests
cd ..
pytest test/python -v
```

### Advanced Analysis

```bash
# Compare detection methods
python compare_detection_methods.py --plot

# Plot I/Q constellations (auto-finds most recent dataset)
python plot_constellation.py --chip S3 --packets 1000 --grid

# Compare ESP32 variants (auto-finds most recent datasets for available chips)
python compare_chips.py --plot
```

---

## Key Results

### Filter Optimization (Noisy Environment)

Tested on 60-second noisy static-presence capture with C6 chip:

| Configuration | Recall | FP Rate | F1 Score |
|---------------|--------|---------|----------|
| Low-pass 11Hz only | 92.4% | 2.34% | 88.9% |
| **Low-pass 11Hz + Hampel (W=9, T=4)** | **92.1%** | **0.84%** | **93.2%** |

### Fixed Subcarriers

ESPectre now uses one shared fixed 12-subcarrier set across `classic` and `ml`. The startup-calibrated runtime paths tune detector-specific thresholds from baseline data, and user-facing tooling now treats `classic` as the only non-ML runtime detector name.

For detailed performance metrics, see [README.md](../docs/performance/README.md).

---

## Additional Resources

- [ALGORITHMS.md](../docs/ALGORITHMS.md) - Algorithm documentation (Classic, ML, fixed subcarriers, Hampel)
- [Micro-ESPectre](../src/python/micro_espectre/README.md) - R&D platform documentation
- [ESPectre](../README.md) - Main project with Home Assistant integration
