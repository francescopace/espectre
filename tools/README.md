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

### 3. Detection Methods Comparison (`compare_detection_methods.py`)

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

### 4. I/Q Constellation Plotter (`plot_constellation.py`)

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

### 5. CSI Amplitude Heatmaps (`plot_heatmap.py`)

**Purpose**: Plot paper-style CSI amplitude heatmaps (time × subcarrier)

- Selects representative `empty` / `static_presence` / `motion` samples from `dataset_info.json`
- Renders amplitude as a viridis heatmap with packet index on x and subcarrier index on y
- Supports chip/environment filters, explicit NPZ paths, optional per-subcarrier detrending, and PNG export

```bash
python plot_heatmap.py
python plot_heatmap.py --chip S3 --environment bedroom
python plot_heatmap.py --labels empty,static_presence,motion,test
python plot_heatmap.py --files data/empty/foo.npz data/motion/bar.npz
python plot_heatmap.py --packets 400 --offset 100 --detrend
python plot_heatmap.py --output /tmp/csi_heatmaps.png --no-show
python plot_heatmap.py --chip C5 --environment living_room --detrend --shared-scale \
    --publication --output docs/web/guides/images/csi-amplitude-heatmap.webp --no-show
```

---

### 6. ESP32 Variant Comparison (`compare_chips.py`)

**Purpose**: Compare CSI characteristics between ESP32 variants

- Compares signal quality between S3 and C6 chips
- Analyzes amplitude and turbulence differences
- Helps choose optimal hardware for specific environments

```bash
python compare_chips.py
python compare_chips.py --plot
```

---

### 7. ML Model Training (`train_ml_model.py`)

**Purpose**: Train, evaluate, and export the production ML model

Install the ML requirements before using this script:

```bash
pip install -r requirements-ml.txt
```

The main repository workflow and this training stack target Python `3.14`.

- Trains the MLP detector with weighted binary cross-entropy
- Default training uses `--fp-weight 1.5`, `--scaler standard`, `--batch-size 1024`, `--device cpu`, and lineage-grouped CV with uniform sample weights
- Caches the derived feature matrix for repeated local runs; use `--no-cache` to rebuild
- Reuses the seed embedded in the current exported weights when `--seed` is omitted
  (`--seed-search-until-improvement` still samples fresh seeds)
- Optional `--augment` applies the current validated train-time augmentation recipe
  (feature jitter + moderate packet augmentation; inference stays clean)
- Reports blocked out-of-fold metrics plus worst and worst-five-tail session,
  lineage, chip, and source-file groups, splitting session metrics by real and
  synthetic provenance when synthetic derivatives are present
- Uses a PyTorch MLP trainer and exports runtime-compatible weights for both platforms only after explicit promotion
- Supports FP-first architecture and FP-weight campaigns, gain-shift diagnostics, and feature-importance analysis
- Uses paired and reserved quiet replays as deployment safety gates, then ranks
  safe candidates by robust grouped-CV worst and tail metrics
- Exports weights for both platforms:
  - `src/python/micro_espectre/ml_weights.py`
  - `src/cpp/core/ml_weights.h`

```bash
python train_ml_model.py                # Train and export if the paired gate passes
python train_ml_model.py --no-export    # Evaluate without replacing runtime artifacts
python train_ml_model.py --info         # Show dataset and split info
python train_ml_model.py --experiment   # Run the FP-first MLP topology campaign
python train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"  # Custom shortlist
python train_ml_model.py --experiment-fp-weights "1,1.5,2,2.5,3"  # Gated multi-seed FP-weight campaign
python train_ml_model.py --fp-weight 1.5  # Penalize false positives 1.5x
python train_ml_model.py --scaler clipped_standard  # Robust clipping + z-score
python train_ml_model.py --batch-size 32  # Smaller-batch comparison
python train_ml_model.py --device cuda    # Force CUDA when available
python train_ml_model.py --device mps     # Force Apple GPU when available
python train_ml_model.py --no-cache       # Rebuild cached training matrix
python train_ml_model.py --exclude-chip ESP32  # Run a chip-exclusion experiment
python train_ml_model.py --seed-search-until-improvement 20  # Evaluate all seeds and keep the best robust improvement
python train_ml_model.py --seed 12345 --force-promote  # Deliberate baseline reset: export even if the gates fail
python train_ml_model.py --features turb_mad_over_mean,turb_autocorr,turb_zcr,l1_delta,l1_delta_std,l1_delta_autocorr --no-export  # Evaluate a candidate feature set
python train_ml_model.py --features turb_mad_over_mean,turb_autocorr,turb_zcr,l1_delta,l1_delta_lag_ratio,l1_delta_autocorr --experiment  # Multi-seed lag-ratio candidate sweep
python train_ml_model.py --augment            # Robustness-winner train-time augmentation
python train_ml_model.py --augment --seed-search-until-improvement 10
python train_ml_model.py --cross-environment  # LOEO using the exported model seed by default
python train_ml_model.py --cross-chip         # LOCO using the exported model seed by default
python train_ml_model.py --gain-stress-gate  # Stress exported model with artificial feature gain shifts
python train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25  # Custom stress multipliers
python train_ml_model.py --shap         # Grouped OOF SHAP (200 samples)
python train_ml_model.py --shap 500     # Grouped OOF SHAP (500 samples)
python train_ml_model.py --ablation-feature turb_skewness --seed 1386543369  # Targeted CV and real-data ablation
```

For the complete ML training workflow, promotion guidance, gain-stress
diagnostics, and post-training regressions, see
[ML_TRAINING.md](../docs/ML_TRAINING.md). For dataset preparation and labeling,
see [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md). For the host CLI
entry points that drive collection and related workflows, see
[CLI.md](../docs/CLI.md).

### 8. Synthetic Low-RSSI Dataset Generation (`generate_low_rssi_dataset.py`)

Generates a reproducible weak-link derivative from one registered real NPZ
capture. The C3, C5, C6, and S3 profiles are anchored to retained real low-RSSI
pairs and report the resulting Core-6 feature medians against their
reference values. Calibration jointly fits all six production ML features,
using sensing-band profile deformation and controlled spatial turbulence rather
than matching only the L1 floor.

`reference_match` independently matches the observed static and motion phases
for Classic detector testing. `shared_session` calibrates the quiet L1 floor
and reuses the same impairment strength for motion, avoiding label-conditioned
generation in datasets that may later be considered for training.

```bash
python generate_low_rssi_dataset.py \
  static_presence_c3_64sc_dev0000acebe64adb64_20260716_003306_439990_0001.npz \
  --profile c3_weak_link --mode reference_match --seed 42
python generate_low_rssi_dataset.py \
  motion_c3_64sc_dev0000acebe64adb64_20260716_003708_700607_0001.npz \
  --profile c3_weak_link --mode reference_match --seed 42

python generate_low_rssi_dataset.py \
  static_presence_c5_64sc_dev000030eda0e46278_20260716_005225_855502_0001.npz \
  --profile c5_moderate_link --mode shared_session --seed 43
python generate_low_rssi_dataset.py \
  motion_c5_64sc_dev000030eda0e46278_20260716_010414_615480_0001.npz \
  --profile c5_moderate_link --mode shared_session --seed 43

python generate_low_rssi_dataset.py \
  empty_s3_64sc_dev000010b41de8ec00_20260712_203314_805494_0001.npz \
  --profile s3_weak_link --mode shared_session --seed 44

python generate_low_rssi_dataset.py \
  empty_c6_64sc_dev00007c2c6742bbac_20260712_215645_774938_0001.npz \
  --profile c6_moderate_link --mode shared_session --seed 45
```

Generate `static_presence` before `motion` in `shared_session` mode. The second
command finds the quiet calibration in the paired synthetic NPZ,
registers reciprocal pair metadata, and prints a production Classic replay.

Outputs live in the standard `data/<label>/` directories. Their
`data/dataset_info.json` entries use the compact `low_rssi: true` and
`synthetic: true` markers alongside the normal dataset and reciprocal-pair
fields. Detailed source provenance, fitted parameters, feature targets, and
errors remain inside the NPZ. Use `--no-register` with an explicit `--output`
for disposable experiments.

### 9. Dataset Quality Validation (`validate_dataset_quality.py`)

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
score tables, then Validation rule / computed-metric notes. Presence and Empty
tables show self-calibrated Classic idle FP plus the indicative 0-100 baseline
score.

How to read the review tables:

- `dataset_role` stays manual. The validator refreshes pair metadata, but it
  does not assign `train`, `selection`, `holdout`, or `exclude`.
- Motion `Ratio` marks still use empirical review thresholds from passing pairs,
  per chip when enough references exist and otherwise with a global fallback.
- Idle-table `Burst` marks now use same-chip clean-idle references only; when a
  chip does not have enough clean references, the cells fall back to fixed
  review thresholds instead of cross-chip empirical marks.
- Idle-table `PPS` shows the observed packet rate from dataset metadata.
- Idle-table `Q95` and `Drift` are exploratory diagnostics. They are shown to
  compare tail proximity and half-to-half stability alongside `FP` / `Burst`,
  and they receive soft marks only when enough same-chip references exist.
- The `Basis` column shows which threshold source was applied in that row:
  `chip`, `global`, or `fixed`.

**Checks performed:**
- Metadata completeness — required dataset metadata exists, disk captures are
  registered, pair links are reciprocal, and paired chip, subcarrier, device,
  and environment metadata agree
- File integrity — NPZ loads, CSI I/Q shape and declared subcarrier count agree,
  per-packet arrays align, and the embedded label matches the dataset directory
- Signal quality — amplitude range, zero-packet detection, packet cadence, and stream continuity
- Pair validation — production-aligned threshold replay on explicit `static_presence` / `motion` pairs
- Empty sanity — each `empty` capture is evaluated independently; self-calibrated
  Classic idle activation flags motion-like or unstable empty files
- Presence quality — each `static_presence` capture uses the same self-calibrated
  Classic idle baseline to flag motion-contaminated or unstable files, without
  requiring a paired `empty` capture
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

### 10. Performance Report Generation (`generate_performance_report.py`)

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
python generate_performance_report.py --skip-cpp-parity-check
```

---

### 11. Firmware Benchmark (`benchmark_firmware.py`)

**Purpose**: Run the live Native, ESPHome, Matter, and Streamer firmware
benchmark for one connected chip and write its generated report under
`docs/performance/`

The benchmark auto-detects the serial port and flashes and monitors these
variants in order:

1. Native Classic
2. Native ML
3. ESPHome Dev Classic
4. Matter Default
5. Streamer Collect

Native starts with a clean Classic build, then reuses that same Native firmware
and switches the detector at runtime with `set_detector` over MQTT before the
ML pass. MQTT is a benchmark prerequisite, not an optional optimization.
ESPHome now benchmarks only the Classic detector. Matter uses a smoke
benchmark with the frontend's default detector, and Streamer runs the host
collect workflow. Each runtime variant is monitored for three minutes, and the
tool evaluates firmware size, application-partition space, packet rate,
motion-state logging, heap, runtime load, loop timing, and detector timing.
Motion transitions are recorded for context but do not affect the result
because the environment may be occupied.

Detector timing covers the shared runtime state-evaluation step, not the
per-packet ingestion step. The runtime measures every evaluation tick and
reports accumulated duration, sample count, minimum, and maximum. The tool
computes the overall average from total duration divided by total samples and
ignores telemetry windows without detector samples for timing statistics.
Keep local ESPHome and Native Wi-Fi credentials configured. The benchmark looks
for `tools/benchmark_firmware.local.env` first and uses it for benchmark-local
Wi-Fi and MQTT settings. Copy `tools/benchmark_firmware.local.env.example` to
`tools/benchmark_firmware.local.env` and fill in the values for your lab
before running the benchmark. The benchmark derives the MQTT device id from the
MAC address reported by the Native flash step.

```bash
python benchmark_firmware.py --chip c3
```

The command exits successfully only when all five cases pass. It still
writes a partial report if a build, flash, monitor, or runtime check fails.

---

### 12. Seed Dispersion Analysis (`analyze_seed_dispersion.py`)

**Purpose**: Measure how far a paired-gate metric moves between training seeds
of the same model on the same recordings

A non-regression margin is only defensible above that dispersion. Below it, the
gate rejects candidates over the noise of weight initialization rather than
over behavior a user would notice, which has happened.

Reads the JSON written by `train_ml_model.py --seed-search-until-improvement`,
and the older `--experiment` reports. Reports per replay the range of the
metric in evaluations rather than percent, because the gate margin is one
evaluation and percentages hide that scale, and reports alarm movement
separately: rate jitter and a new effective alarm are different findings.

Chip-level fallback rows are labeled `n/a` and excluded from the verdict. They
appear when a chip owns no reserved pair and the gate falls back to an
aggregate over training data, which cannot answer a reserved question. Weak and
normal links are counted apart, since the `low_rssi` exemption is its own
question.

```bash
python analyze_seed_dispersion.py ../data/auto_generated/mlp_seed_search.json
python analyze_seed_dispersion.py report.json --metric recall
```

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

# 2. Optimize Classic parameters
python analyze_system_tuning.py --quick

# 3. Compare detection methods
python compare_detection_methods.py --plot

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

# Paper-style CSI amplitude heatmaps (time × subcarrier)
python plot_heatmap.py --chip S3 --environment bedroom --detrend

# Compare ESP32 variants (auto-finds most recent datasets for available chips)
python compare_chips.py --plot
```

---

## Key Results

### Fixed Subcarriers

ESPectre now uses one shared fixed 12-subcarrier set across `classic` and `ml`. The startup-calibrated runtime paths tune detector-specific thresholds from baseline data, and user-facing tooling now treats `classic` as the only non-ML runtime detector name.

For detailed performance metrics, see [README.md](../docs/performance/README.md).

---

## Additional Resources

- [ALGORITHMS.md](../docs/ALGORITHMS.md) - Algorithm documentation (Classic, ML, fixed subcarriers, Hampel)
- [Micro-ESPectre](../src/python/micro_espectre/README.md) - R&D platform documentation
- [ESPectre](../README.md) - Main project with Home Assistant integration
