# Analysis And Benchmark Tools

This directory contains host-side Python tools for CSI inspection, dataset validation, detector research, model training, and firmware benchmarking. It is written for contributors who already understand the basic ESPectre workflow; start with [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md) if you need to collect data, or [ALGORITHMS.md](../docs/ALGORITHMS.md) if you need the detector concepts first.

Run commands from the repository root through the project virtual environment. Use `python tools/<tool>.py --help` for the complete and current option reference; this README explains which tool to choose and the safe mainline workflows.

## Common Terms

- **CSI:** channel state information, the per-packet radio-channel measurement used by the detectors.
- **Replay:** running recorded CSI through production-aligned feature or detector logic.
- **Candidate:** a research-only feature, model, or detector configuration that has not been promoted to production.
- **Gate:** a validation requirement that must pass before generated runtime artifacts can change.
- **OOF:** out-of-fold metrics, computed on data excluded from the fold that fitted the model.

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install `requirements-ml.txt` only for training and ML-specific analysis. The main repository workflow targets Python `3.14`.

The tools support the original ESP32, ESP32-C3, ESP32-C5, ESP32-C6, and ESP32-S3 when the required datasets or connected hardware are available. A missing dataset or hardware report does not mean that a chip is unsupported.

## Tool Index

| Tool | Use it when you need to |
|---|---|
| `analyze_raw_data.py` | inspect registered CSI pairs and basic signal quality |
| `analyze_system_tuning.py` | grid-search Lightweight parameters on the fixed production band |
| `compare_detection_methods.py` | compare RSSI, Lightweight, and ML behavior on recorded data |
| `compare_chips.py` | compare CSI characteristics across available chip datasets |
| `plot_constellation.py` | visualize I/Q samples by subcarrier |
| `plot_heatmap.py` | render time-by-subcarrier CSI amplitude heatmaps |
| `validate_dataset_quality.py` | validate metadata, files, signal quality, pair consistency, and training readiness |
| `train_ml_model.py` | train, evaluate, and conditionally export the production ML model |
| `generate_performance_report.py` | regenerate the aggregate detector performance report and run its parity checks |
| `benchmark_firmware.py` | build, flash, monitor, and report representative live firmware cases |
| `analyze_seed_dispersion.py` | measure replay-metric variation across training seeds |
| `compare_reserved_selection.py` | compare one candidate on reserved selection roles with an explicit seed |
| `benchmark_subcarrier_aggregation.py` | evaluate adjacent-subcarrier aggregation as a host-side experiment |
| `sweep_occupancy_floor.py` | replay reserved pairs while thinning admitted CSI and relaxing the readiness occupancy floor |
| `benchmark_lightweight_candidate_pairs.py` | screen Lightweight features and combinations without threshold coupling |
| `test/cpp/support/benchmark_lightweight_iqr_resources.cpp` | compare host C++ RAM and hot-path cost for the normal- and aggregated-IQR Lightweight finalists |
| `test/cpp/support/benchmark_detector_resources.cpp` | measure current production Lightweight and High Accuracy host memory, packet cost, inference latency, and nominal CPU load |
| `replay_lightweight_candidates.py` | fit and replay research-only Lightweight candidates end to end |
| `fit_lightweight_detector.py` | fit production Lightweight coefficients and optionally apply an approved result |
| `prune_npz_cache.py` | remove cached analysis artifacts whose sources or implementation dependencies are no longer current |
| `espectre_traffic_generator.py` | run the standalone laboratory traffic-generator service |

## Dataset Inspection And Validation

Use lightweight inspection before changing a detector:

```bash
python tools/analyze_raw_data.py
python tools/analyze_raw_data.py --chip C6
python tools/compare_detection_methods.py --chip C6 --plot
```

Run the quality validator before training or publishing dataset conclusions:

```bash
python tools/validate_dataset_quality.py
python tools/validate_dataset_quality.py --chip C6
python tools/validate_dataset_quality.py --no-report
python tools/validate_dataset_quality.py --check-current
python tools/validate_dataset_quality.py --data-dir data/untracked/example --preserve-pairs
python tools/validate_dataset_quality.py --data-dir data/untracked/example --diagnostic-all-phy
```

The validator checks catalog metadata, NPZ integrity, CSI shape, packet timing, stream continuity, explicit pair consistency, quiet recordings, and ML readiness. It refreshes derived pair metadata and normally regenerates `data/auto_generated/DATASET_QUALITY_CHECK.md`. Use `--data-dir` for a standalone ESPectre-format corpus; its report defaults to `<data-dir>/auto_generated/DATASET_QUALITY_CHECK.md`. Add `--preserve-pairs` when the external catalog already contains deliberate reciprocal pairs and its timestamps must not drive automatic re-pairing. `--report-output` overrides the generated report path when needed. `--diagnostic-all-phy` evaluates all explicitly tagged PHY rows while retaining the supported HT20/HT-LTF contract failure in the report; it is for external holdouts and never admits those rows into production training.

`dataset_role` remains a manual curation decision. The validator never promotes a recording to `train`, `selection`, or `holdout`; entries without an explicit role remain excluded. Review scores are diagnostic and do not replace admission gates. Excluded idle captures that produce no usable feature rows after temporal admission are listed first in the generated report with `n/a ⚠️`. The generated report owns the detailed tables and definitions.

## ML Training

Read [ML_TRAINING.md](../docs/ML_TRAINING.md) before running a promotion workflow. The safe progression is:

```bash
python tools/train_ml_model.py --info
python tools/train_ml_model.py --augment --no-export
python tools/train_ml_model.py --augment
```

`--no-export` evaluates without replacing runtime weights. A normal production run exports only after its promotion gates pass. Host-only candidate features remain research artifacts even when their metrics improve; use [FEATURES.md](../docs/FEATURES.md) to record the evidence and promotion status.

Use an explicit seed and the same corpus, roles, preprocessing, features, and augmentation when comparing two changes. Seed searches, cross-environment checks, cross-chip checks, gain stress, ablations, and feature-importance modes are advanced workflows documented by `--help` and [ML_TRAINING.md](../docs/ML_TRAINING.md).

## Generated Performance Report

`generate_performance_report.py` publishes Lightweight and High Accuracy replay tables only for the combined `selection + holdout` corpus, executes the current production C++ resource microbenchmark, and runs the host-side C++/Python parity checks before writing the report. Training-role recordings remain covered by the validation suites but are neither replayed nor summarized by the report generator. Detector replay summaries, augmented rows, and training matrices use the shared `.npz_cache`; a warm generation reuses them instead of replaying the corpus. Its augmentation diagnostic applies the production two-seed packet recipe to the same reserved pairs and compares Lightweight and High Accuracy on matching alternating replay positions; it never reads augmented training rows.

```bash
python tools/generate_performance_report.py
python tools/generate_performance_report.py --check-current
python tools/generate_performance_report.py --stdout
python tools/generate_performance_report.py --data-dir data/untracked/example
python tools/generate_performance_report.py --data-dir data/untracked/example --diagnostic-all-phy
```

Do not edit `docs/performance/README.md` manually. `--check-current` is a lightweight input-revision check; a normal warm regeneration measures resources again, loads the cached replay summary and robustness artifacts, runs parity, and renders the report. A replay-summary miss rebuilds only from the lower-level row cache and never starts ML training. `--data-dir` writes an external holdout report to `<data-dir>/auto_generated/PERFORMANCE_REPORT.md` by default and skips primary-corpus resource, augmentation, and C++ parity sections. Add `--diagnostic-all-phy` only for explicitly tagged external views such as LLTF or HT40; the report records that non-production evaluation view.

## Firmware Benchmark

`benchmark_firmware.py` operates on one connected chip and writes its generated report under `docs/performance/`. The representative matrix is:

1. Native Lightweight
2. Native High Accuracy by runtime switching of the same Native firmware
3. ESPHome Lightweight
4. Matter with its build-time default detector
5. Streamer collection

This matrix is not a capability table. ESPHome, Native, and Matter support Lightweight and High Accuracy; ESPHome and Native can switch at runtime, while Matter selects the detector at build time.

The benchmark requires local ESPHome and Native Wi-Fi credentials plus MQTT for the Native runtime switch. Copy `tools/benchmark_firmware.local.env.example` to `tools/benchmark_firmware.local.env`, fill in the laboratory values, connect the target board, and run:

```bash
python tools/benchmark_firmware.py --chip c3
```

The command writes a partial report when a case fails and returns success only when every selected case passes. Each report is a snapshot of the Git revision, hardware, environment, and run time recorded in its header; it does not certify later source revisions. Do not edit or reformat generated chip reports separately from a hardware benchmark run.

## Research-Only Detector Experiments

The candidate tools answer different questions:

| Question | Tool |
|---|---|
| Does one feature or combination separate paired states before threshold tuning? | `benchmark_lightweight_candidate_pairs.py` |
| Does a candidate survive causal calibration, clean replay, and packet stress? | `replay_lightweight_candidates.py` |
| Does adjacent-bin aggregation change channel statistics or detector behavior? | `benchmark_subcarrier_aggregation.py` |
| How much does a metric move across training seeds? | `analyze_seed_dispersion.py` |
| Does a selected ML candidate survive the reserved selection roles? | `compare_reserved_selection.py` |

These tools are diagnostic by default and must not write production runtime artifacts. Their durable conclusions belong in [FEATURES.md](../docs/FEATURES.md); detector formulas belong in [ALGORITHMS.md](../docs/ALGORITHMS.md). Use an ADR only for a durable architectural or project-level decision.

Replay independent catalogs only after fitting and primary-corpus ranking. Repeat `--external-data-dir` for multiple sealed holdouts; add `--external-diagnostic-all-phy` only for a matching external catalog whose explicit non-production PHY rows are intentionally being evaluated as diagnostics:

```bash
python tools/replay_lightweight_candidates.py \
  --features turb_autocorr,turb_iqr_over_mean_aggr,chan_shape_excess_path \
  --stress-augment \
  --external-data-dir data/untracked/csi_sense_zero \
  --external-data-dir data/untracked/wisdom_lab \
  --external-diagnostic-all-phy data/untracked/wisdom_lab
```

`fit_lightweight_detector.py --apply` and ML export are deliberate promotion actions. Run the required real-data, long-recording, packet-rate, and C++/Python parity gates before applying their output.

## Visual Analysis

```bash
python tools/plot_constellation.py --chip S3 --packets 1000 --grid
python tools/plot_heatmap.py --chip S3 --environment bedroom --detrend
python tools/compare_chips.py --plot
```

Plots help diagnose signal structure; they do not establish detector quality by themselves. Use grouped replay metrics and the maintained gates for production conclusions.

## Cache Maintenance

Training and replay tools share a persistent NPZ cache. Normal runs validate cache provenance automatically. Runtime-supported features use complete replay matrices; host-only experiments use one row-spine artifact plus one column artifact per feature. Adding a variant to an existing provider family leaves sibling columns valid, so later model comparisons compute only columns that are actually missing. Reordering or selecting a subset reads the same columns without rebuilding packet rows. Cold producers serialize on a per-key process lock and recheck the cache after acquiring it. Long fills emit periodic `[npz-cache]` lines on stderr for hits, misses, in-progress builds, and writes when stderr is a TTY; `ESPECTRE_NPZ_CACHE_PROGRESS=0` disables that output, `=1` forces it, and `ESPECTRE_NPZ_CACHE_PROGRESS_INTERVAL_S` overrides the default `10` second heartbeat.

Pruning removes only artifacts that can no longer be used because their capture, implementation dependencies, layout, or artifact version changed. Historical but still reachable feature columns remain until an explicit age or size policy is requested:

```bash
python tools/prune_npz_cache.py
python tools/prune_npz_cache.py --artifact ml_replay_rows
```

## Related Documentation

- [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md): collection labels, metadata, and dataset roles
- [ML_TRAINING.md](../docs/ML_TRAINING.md): training, model selection, promotion, and export
- [ALGORITHMS.md](../docs/ALGORITHMS.md): production detector behavior
- [FEATURES.md](../docs/FEATURES.md): feature evidence and verdicts
- [performance/README.md](../docs/performance/README.md): generated detector metrics
- [CLI.md](../docs/CLI.md): supported repository entry points for collection and device workflows
