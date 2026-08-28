# Analysis And Benchmark Tools

This directory contains host-side Python tools for CSI inspection, dataset validation, detector research, model training, and firmware benchmarking. It is written for contributors who already understand the basic ESPectre workflow; start with [ML_DATA_COLLECTION.md](../docs/ML_DATA_COLLECTION.md) if you need to collect data, or [ALGORITHMS.md](../docs/ALGORITHMS.md) if you need the detector concepts first.

Run commands from the repository root through the project virtual environment. Use `python tools/<tool>.py --help` for the complete and current option reference; this README explains which tool to choose and the recommended mainline workflows.

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

The tools support the original ESP32, ESP32-C3, ESP32-C5, ESP32-C6, ESP32-S2, and ESP32-S3 when the required datasets or connected hardware are available. A missing dataset or hardware report does not mean that a chip is unsupported.

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
| `sweep_occupancy_floor.py` | replay reserved pairs while thinning admitted CSI; `--always-evaluate` keeps occupancy holes scored |
| `benchmark_lightweight_candidate_pairs.py` | screen Lightweight features and combinations without threshold coupling |
| `test/cpp/support/benchmark_lightweight_iqr_resources.cpp` | compare host C++ RAM and hot-path cost for the normal- and aggregated-IQR Lightweight finalists |
| `test/cpp/support/benchmark_detector_resources.cpp` | measure current production Lightweight and High Accuracy host memory, packet cost, inference latency, and nominal CPU load |
| `replay_lightweight_candidates.py` | fit and replay research-only Lightweight candidates end to end |
| `fit_lightweight_detector.py` | fit production Lightweight coefficients and optionally apply an approved result |
| `prune_npz_cache.py` | remove cached analysis artifacts whose sources or implementation dependencies are no longer current |
| `espectre_traffic_generator.py` | send phase-paced, DSCP 46 unicast or local-link multicast UDP traffic to devices in `csi_traffic_mode: external` |

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

`dataset_role` remains a manual curation decision. The trainer treats a missing role as excluded, but the validator requires every catalog entry to declare its role explicitly, including `exclude`. It never promotes a recording to `train`, `selection`, or `holdout`. Review scores are diagnostic and do not replace admission gates: mean valid-slot occupancy warns below 85%, fails admission below 70%, and caps every affected score. Excluded idle captures that produce no usable feature rows after temporal admission are listed first in the generated report with `n/a ⚠️`. The generated report owns the detailed tables and definitions.

## ML Training

Read [ML_TRAINING.md](../docs/ML_TRAINING.md) before running a promotion workflow. Use this promotion sequence:

```bash
python tools/train_ml_model.py --info
python tools/train_ml_model.py --augment --no-export
python tools/train_ml_model.py --augment --seed SEED --evaluate-selection
python tools/train_ml_model.py --augment --seed SEED
```

`--no-export` runs CV without replacing runtime weights or opening deployment replays. `--evaluate-selection` evaluates a candidate against selection while keeping holdout sealed. A normal production run opens selection and holdout, then exports only after its promotion gates pass. Host-only candidate features remain research artifacts even when their metrics improve; use [FEATURES.md](../docs/FEATURES.md) to record the evidence and promotion status.

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
3. Micro-ESPectre Lightweight
4. ESPHome Lightweight
5. ESPHome High Accuracy by runtime switching of the same ESPHome firmware
6. Matter build-and-flash smoke with its initial default detector

This matrix is not a capability table. ESPHome, Native, and Matter support persisted runtime switching between Lightweight and High Accuracy. Micro-ESPectre deploys Lightweight only, and the Matter smoke case does not commission the device or exercise runtime switching.

The benchmark reads laboratory settings from `tools/benchmark_firmware.local.env`, with exported `ESPECTRE_BENCHMARK_*` variables taking precedence.

Native compiles with empty Wi-Fi, device-label, and MQTT defaults, erases NVS, provisions the SSID and password at runtime through standard Improv Serial, and applies an optional BSSID pin through Direct after the first connection. ESPHome injects only the laboratory SSID and password into the generated YAML, then applies the optional BSSID through Direct so the benchmark exercises the ESPectre-owned pin instead of a YAML pin. When `ESPECTRE_BENCHMARK_WIFI_CHANNEL` is set alongside the BSSID, the benchmark also verifies that the connected access point uses that channel; a channel without a BSSID is rejected before hardware access.

Native and ESPHome reuse one flashed Lightweight image and select both scored detectors through Direct. Matter is a build-and-flash smoke case: it stops after a successful flash and requires neither commissioning nor benchmark Wi-Fi settings. Micro-ESPectre copies the laboratory Wi-Fi settings into an isolated temporary `config_local.py`, explicitly enables the native ICMP generator, and connects Direct through the Wi-Fi address reported by its serial launcher. Copy `tools/benchmark_firmware.local.env.example` to `tools/benchmark_firmware.local.env`, fill in the laboratory values required by the selected frontends, connect the target board, and run:

```bash
python tools/benchmark_firmware.py --chip c3 --port /dev/cu.usbmodem01
```

The benchmark always passes `--chip` to `./espectre` and delegates serial selection, chip verification, Native NVS erasure, provisioning, and monitoring to the repository CLI. Omit `--port` to auto-select a single compatible device or choose interactively among multiple compatible candidates; pass `--port` to require that exact compatible device. Matter is omitted automatically for ESP32-S2 because the supported commissioning flow requires Bluetooth.

Use `--duration SECONDS` for a longer scored window, such as a five-minute Micro-ESPectre heap soak:

```bash
python tools/benchmark_firmware.py --chip c3 --frontend micro --duration 300 --update
```

Use `--resume` to keep passing results from the chip report and rerun only failed or missing cases. Optional frontend and detector filters limit which failed or missing cases are retried:

```bash
python tools/benchmark_firmware.py --chip c3 --resume
```

The command writes a partial report when a case fails and returns success only when every case in the resulting report passes. Native and ESPHome refresh that report and their structured artifacts after each detector completes, while retaining their shared build, flash, and serial-monitor session. It stores normalized Direct samples and events, transport outcomes, firmware hashes, structured analysis, and a run manifest under `data/untracked/firmware_benchmarks/<run-id>/`. Runtime artifacts exclude raw serial output, raw Direct payloads, credentials, device identity, and local addresses. Matter artifacts contain only build-and-flash evidence.

Every sensing case waits for five consecutive ready, non-zero Direct diagnostics samples before scoring. Native, ESPHome, and Micro confirm their fixed or requested detector and traffic profile through production responses. The Native and ESPHome runtime cases use `ping` traffic and each frontend's configured CSI rate by default. Set `ESPECTRE_BENCHMARK_TRAFFIC_GENERATOR_MODE=dns` when the laboratory gateway rate-limits sustained ICMP replies. `ESPECTRE_BENCHMARK_CSI_TARGET_PPS` overrides the compiled Native and ESPHome rate; in external mode, it also sets the host traffic rate.

Heap-decline scoring begins 10 seconds into the scored window, device uptime must remain monotonic, Direct transport failure counters must not increase when available, detector timing must be present, and telemetry events are collected from Direct SSE. Native, ESPHome, and Matter bootstrap builds reuse the existing per-chip directory when its image and configuration stamp still match; they pass `--clean` only after a target, detector, or defaults change. Micro-ESPectre keeps generated `sdkconfig` when the board Kconfig inputs are unchanged.

The manifest and generated report record the starting and ending Git revisions, worktree states, and firmware-source fingerprints. A revision change during the run invalidates every executed case; a source-fingerprint change on the same revision is reported as a warning without invalidating results. Each report therefore identifies its source state, hardware, environment, and run time; it does not certify later source revisions. Do not edit or reformat generated chip reports separately from a hardware benchmark run.

Expected sample counts tolerate one sample at the scored-window boundary. Direct cadence uses host-monotonic receive times and device uptime for every sensing frontend; any real gap over the cadence tolerance fails the case.

When `--update` or `--resume` preserves cases from an existing report, the report header identifies the updating run, not the provenance of every preserved case. Use the matching per-run artifact directory for the exact revision, duration, and structured evidence of each executed case.

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

Training and replay tools share a persistent NPZ cache. Normal runs validate cache provenance automatically. Runtime-supported features use complete replay matrices; host-only experiments use one row-spine artifact plus one column artifact per feature.

Adding a variant to an existing provider family leaves sibling columns valid, so later model comparisons compute only columns that are actually missing. Reordering or selecting a subset reads the same columns without rebuilding packet rows.

Cold producers serialize on a per-key process lock and recheck the cache after acquiring it. Long fills emit periodic `[npz-cache]` lines on stderr for hits, misses, in-progress builds, and writes when stderr is a TTY; `ESPECTRE_NPZ_CACHE_PROGRESS=0` disables that output, `=1` forces it, and `ESPECTRE_NPZ_CACHE_PROGRESS_INTERVAL_S` overrides the default `10` second heartbeat.

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
