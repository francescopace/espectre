# Analysis and benchmark tools

This directory holds the Python tools that run on your computer: CSI inspection, dataset validation, detector research, model training, and firmware benchmarks. It is for contributors. To collect data, start with the [data collection guide](../docs/ML_DATA_COLLECTION.md); for the detector concepts, see the [algorithms reference](../docs/ALGORITHMS.md).

Run the tools from the repository root, inside the virtual environment. `python tools/<tool>.py --help` lists every option; this page explains which tool to use and the main workflows.

## Common terms

- **CSI:** channel state information, the per-packet Wi-Fi measurement the detectors use.
- **Replay:** running recorded CSI through the same feature and detector code as the device.
- **Candidate:** a feature, model, or detector setting under research, not in production.
- **Gate:** a check that must pass before model or detector files can change.
- **OOF:** out-of-fold metrics, computed on data the model was not trained on.

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python `3.14`. For training and ML analysis, also run `python -m pip install -r requirements-ml.txt` (use the same command to upgrade, so NumPy, Numba, SHAP, and PyTorch stay compatible).

The tools work with every supported chip. If a chip has no dataset or hardware report yet, that does not mean it is unsupported.

## Tool index

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
| `ha_traffic_generator_addon/espectre_traffic_generator.py` | send phase-paced unicast or multicast UDP traffic to devices in `traffic_generator_mode: external`; configurable DSCP and multicast TTL default to 46 and 8; shared by the CLI and Home Assistant add-on |
| `ha_traffic_generator_addon/` | Home Assistant OS add-on that runs the external traffic generator continuously |

## Home Assistant Traffic Generator add-on

The add-on runs the external traffic generator continuously on 64-bit Home Assistant OS. See the [add-on documentation](ha_traffic_generator_addon/DOCS.md) for installation and configuration.

## Dataset inspection and validation

Take a quick look before changing a detector:

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

The validator checks the catalog, file integrity, CSI shape, timing, continuity, pairs, quiet recordings, and ML readiness. It updates the pair fields and writes `data/auto_generated/DATASET_QUALITY_CHECK.md`, which explains every table.

- Occupancy below 85% gives a warning, below 70% a failure. Review scores are informative; failures are what block training.
- Every catalog entry needs a `dataset_role`, set by hand. The validator never assigns one.
- Excluded quiet recordings with no usable rows are listed first in the report, as `n/a ⚠️`.
- `--chip` limits the check to one chip. `--check-current` must use the same `--chip` as the run that wrote the report. A filtered run still overwrites the default report; use `--report-output` to write elsewhere.
- `--data-dir` checks a separate corpus in ESPectre format, with its report in `<data-dir>/auto_generated/`. Add `--preserve-pairs` if that catalog already has its own pairs.
- `--diagnostic-all-phy` also evaluates non-HT20 rows of an external corpus. They are reported, never used for training.

## ML training

Read the [ML training guide](../docs/ML_TRAINING.md) first. The usual sequence:

```bash
python tools/train_ml_model.py --info
python tools/train_ml_model.py --augment --no-export
python tools/train_ml_model.py --augment --seed SEED --evaluate-selection
python tools/train_ml_model.py --augment --seed SEED
```

The guide explains each step and the advanced modes (seed search, cross-room and cross-chip checks, gain stress, ablations, feature importance). Record results in the [feature ledger](../docs/FEATURES.md).

The `clipped_standard` scaler works only for CV with `--no-export`; export needs `standard`, `robust`, or `session_balanced_robust`, because the device model has no clipping bounds.

## Generated performance report

`generate_performance_report.py` writes `docs/performance/README.md`. It replays Lightweight and High Accuracy on the `selection + holdout` recordings (not `train`), measures C++ resource use, and runs the C++/Python parity checks. It also includes a robustness check that applies the training augmentation to the same recordings. Results are cached, so a second run is fast.

```bash
python tools/generate_performance_report.py
python tools/generate_performance_report.py --check-current
python tools/generate_performance_report.py --stdout
python tools/generate_performance_report.py --data-dir data/untracked/example
python tools/generate_performance_report.py --data-dir data/untracked/example --diagnostic-all-phy
```

- Never edit the report by hand.
- `--check-current` only checks whether the inputs changed; a normal run measures resources again and reruns parity. It never trains a model.
- `--data-dir` reports on an external corpus, in `<data-dir>/auto_generated/PERFORMANCE_REPORT.md`, without the resource, augmentation, and parity sections. Add `--diagnostic-all-phy` for external data in other formats, such as LLTF or HT40.

## Firmware benchmark

`benchmark_firmware.py` flashes one connected board with each firmware, runs it on a real network, and writes a report to `docs/performance/<chip>.md`. It tests:

1. Native Lightweight
2. Native High Accuracy (same firmware, switched at runtime)
3. ESPHome Lightweight
4. ESPHome High Accuracy (same firmware, switched at runtime)
5. Matter Lightweight
6. Matter High Accuracy (same firmware, switched at runtime)
7. Micro-ESPectre Lightweight

Matter is skipped on ESP32-S2, which has no Bluetooth for commissioning.

### Run it

Copy `tools/benchmark_firmware.local.env.example` to `tools/benchmark_firmware.local.env` and fill in your lab settings (exported `ESPECTRE_BENCHMARK_*` variables override the file). Connect the board and run:

```bash
python tools/benchmark_firmware.py --chip c3 --port /dev/cu.usbmodem01
```

- `--port` is required; the benchmark never searches for a port or resets the board on its own.
- `--duration SECONDS` makes the scored window longer, for example a five-minute heap test: `--frontend micro --duration 300 --update`.
- `--resume` keeps passing results and reruns only failed or missing cases. Frontend and detector filters limit what is rerun.
- On ESP32-S2 (USB CDC), the benchmark pauses before each flash so you can put the board in download mode. In non-interactive runs, prepare it beforehand.

Optional settings:

- `ESPECTRE_BENCHMARK_WIFI_BSSID` tests reconnecting to a specific access point; `ESPECTRE_BENCHMARK_WIFI_CHANNEL` also checks its channel (a channel without a BSSID is rejected).
- Matter needs CHIP Tool built from the same `connectedhomeip` revision as the firmware's `esp-matter`. Set `ESPECTRE_BENCHMARK_CHIP_TOOL` if it is not on `PATH` or in `~/.local/bin/`. BLE commissioning is retried twice; change this with `ESPECTRE_BENCHMARK_MATTER_COMMISSIONING_ATTEMPTS`. On macOS, install Apple's Bluetooth Central Matter Client Developer Mode profile and restart first.

### What it does

- **Setup.** Each firmware is built with its normal configuration and flashed with a full erase through the repository CLI. Native and ESPHome get Wi-Fi through Improv Serial; Matter is commissioned with CHIP Tool and a temporary controller store; Micro-ESPectre gets only the lab Wi-Fi settings in a temporary `config_local.py`. The same Lightweight image is reused and High Accuracy is selected through Direct.
- **Access point test.** With a BSSID set, the pin is applied once with `force=true`. The benchmark waits for the switch to finish and checks the new association; a lost SSE stream during the switch is expected. A missing acknowledgement, failed check, rollback, or device restart fails the run.
- **Readiness.** The device must first report the production defaults (Lightweight, the frontend's traffic mode, 100 pps; published firmware uses `ping`). Scoring starts after five consecutive ready samples.
- **Scoring.** The heap trend is measured over two consecutive 10-second windows, starting 10 seconds in; a window too short for that fails. Uptime must never go back, Direct error counters must not grow, detector timing must be present, and the sampling cadence must hold (500 ms tolerance on C++ frontends; Micro samples every 4.5 seconds with 1 second of tolerance). Sample counts may be off by one at the window edges.
- **Readiness details.** If SSE reports "not ready" for five samples, the benchmark rereads the sensing resource over HTTP in case it missed a change. Readiness samples (state, admitted rate, occupancy) are kept as evidence but not scored.
- **CLI delegation.** Build, flash (a single esptool operation, erase included), provisioning, Matter onboarding, and Micro readiness all go through `./espectre`, using its final JSON records rather than its human-readable output. A non-zero exit status is final. Micro-ESPectre is reached through the Wi-Fi address its serial launcher reports.
- **Error reports** name the failed phase, and HTTP errors name the method and resource, without request bodies or addresses.
- **Serial output** is watched only for crashes, resets, or a dead monitor, never for metrics.

### Results

- The run stops at the first failed case, writes a partial report, and succeeds only if every case passes. It never retries a flash or power-cycles the board.
- Detailed, anonymized data (Direct samples and events, firmware hashes, analysis, and a manifest) goes to `data/untracked/firmware_benchmarks/<run-id>/`. No serial logs, credentials, onboarding data, device IDs, or local addresses are stored.
- The report records the Git revision and source fingerprint at start and end. A revision change during the run invalidates it; a fingerprint change alone is a warning.
- With `--update` or `--resume`, the report header shows the latest run; check each run's artifact folder for the exact source of kept cases.
- Never edit a chip report by hand.

### Firmware benchmark contract

The following rules are normative for `benchmark_firmware.py` and its owners under `tools/lib/firmware_benchmark/`:

- Build the canonical frontend configuration in its ordinary build directory, retain production defaults, and allow the normal incremental build system to reuse valid artifacts. Do not generate benchmark-specific YAML, sdkconfig overlays, or dedicated build directories, and do not force a clean build.
- Clear all device data as part of the frontend's normal `flash --erase` operation. Do not add partial-erasure exceptions.
- Provision Wi-Fi through standard Improv Serial on Native and ESPHome. Matter's read-only Improv surface exposes firmware identity and onboarding data but does not provision Wi-Fi; commission Matter through a revision-compatible CHIP Tool controller over BLE and Wi-Fi. Micro-ESPectre may inject only connectivity settings because it does not support Improv Serial.
- Apply and verify the optional target BSSID through Direct where supported. Send `force=true` so the setup exercises the Wi-Fi transition even when the device is already associated with that access point. Require the successful Direct acknowledgement, including its pre-apply `current_bssid`, before reconnecting. Before configuring a detector, verify the target association and restoration of the pre-apply sensing state; the old association may remain visible while the frontend temporarily suspends sensing. A timeout, reset, dropped response, failed association check, or observed device restart fails the benchmark.
- Use Direct responses, diagnostics, and events for runtime configuration, validation, and metrics. Scored samples request the time-series fields in `BENCHMARK_DIAGNOSTIC_FIELDS`. The first successful scored response also supplies detector-timing support and the initial Direct failure counters; the last scheduled sample adds the final counters, minimum free heap, and largest free block. These selections share the ordinary diagnostics requests. Derive the detector average from its duration sum and sample count using integer division. Do not request task-stack headroom or fetch the field catalog before a run. C++ frontend handshakes and Micro's timestamp baseline request only uptime; Micro's handshake additionally validates the Direct transport group. Readiness requests admitted CSI rate, and SSE teardown requests only `direct_http.event_clients`. Use serial output only to detect fatal firmware errors, unexpected resets, or an unexpectedly terminated monitor.
- Reuse one keep-alive Direct control connection during readiness and scored sampling so the harness does not perturb lwIP socket and packet-buffer pressure. Non-persistent TCP timing is an explicit diagnostic opt-in, not the canonical matrix workload.
- Retry only GET requests after a persistent-connection transport failure. Never resend a mutation automatically after losing its response.
- Use one SSE connection per detector from readiness through scored sampling. Close it after the detector window, wait until the device releases the connection, and open a distinct connection for the next detector.
- Reuse capabilities and device identity within the same frontend session. Read fresh sensing, Wi-Fi, and diagnostics before each detector, apply only the required sensing changes in one PATCH, and confirm only the changed sensing resource. Diagnostics provide uptime and request-success evidence without a separate health request. Reconnects and new firmware sessions require a fresh device identity check.
- During readiness, read sensing once and consume its complete SSE snapshots when the frontend advertises sensing events and the stream is active; otherwise, keep polling sensing. Always retain the one-second diagnostics cadence, consecutive-ready-sample requirement, and reboot checks. Scored windows send one diagnostics request per sampling interval. Poll for SSE teardown every 250 ms after scoring.
- Verify production runtime defaults before applying case-specific mutations. Reuse one canonical Lightweight image and select another supported detector through Direct instead of rebuilding it.

Behavioral tests under `test/python/host/benchmark/test_benchmark_*.py` are the executable enforcement of this contract. Generated performance reports describe individual runs and are not contract owners.

## Research-only detector experiments

The candidate tools answer different questions:

| Question | Tool |
|---|---|
| Does one feature or combination separate paired states before threshold tuning? | `benchmark_lightweight_candidate_pairs.py` |
| Does a candidate survive causal calibration, clean replay, and packet stress? | `replay_lightweight_candidates.py` |
| Does adjacent-bin aggregation change channel statistics or detector behavior? | `benchmark_subcarrier_aggregation.py` |
| How much does a metric move across training seeds? | `analyze_seed_dispersion.py` |
| Does a selected ML candidate survive the reserved selection roles? | `compare_reserved_selection.py` |

These tools never change production files. Record their conclusions in the [feature ledger](../docs/FEATURES.md), formulas in the [algorithms reference](../docs/ALGORITHMS.md), and use an ADR only for a lasting project decision.

Candidate replay follows the dataset roles: a recording without a role is excluded, and `--include-train-empty` never uses it for training.

Test on external catalogs only after fitting and ranking on the main corpus. Repeat `--external-data-dir` for each one, and add `--external-diagnostic-all-phy` for an external catalog in another capture format:

```bash
python tools/replay_lightweight_candidates.py \
  --features turb_autocorr,turb_iqr_over_mean_aggr,chan_shape_excess_path \
  --stress-augment \
  --external-data-dir data/untracked/csi_sense_zero \
  --external-data-dir data/untracked/wisdom_lab \
  --external-diagnostic-all-phy data/untracked/wisdom_lab
```

`fit_lightweight_detector.py --apply` and ML export change production. Run the real-data, long-recording, packet-rate, and C++/Python parity gates first.

## Visual analysis

```bash
python tools/plot_constellation.py --chip S3 --packets 1000 --grid
python tools/plot_heatmap.py --chip S3 --environment bedroom --detrend
python tools/compare_chips.py --plot
```

Plots help understand the signal, but do not prove detector quality; use replay metrics and gates for that.

## Cache maintenance

Training and replay tools share a cache under `.cache/npz/` and check automatically that it is still valid.

- Production features are cached as complete matrices. Research features are cached one column per feature, so a new variant computes only its own column, and reordering or picking a subset rebuilds nothing.
- Parallel runs wait for each other instead of computing the same entry twice.
- Long fills print `[npz-cache]` progress on stderr when it is a terminal. `ESPECTRE_NPZ_CACHE_PROGRESS=0` turns it off, `=1` forces it on, and `ESPECTRE_NPZ_CACHE_PROGRESS_INTERVAL_S` changes the default 10-second interval.

Pruning removes entries whose inputs or code have changed. Older entries that can still be used stay until you set an age or size limit:

```bash
python tools/prune_npz_cache.py
python tools/prune_npz_cache.py --artifact ml_replay_rows
```

## Related documentation

- [data collection guide](../docs/ML_DATA_COLLECTION.md): collection labels, metadata, and dataset roles
- [ML training guide](../docs/ML_TRAINING.md): training, model selection, promotion, and export
- [algorithms reference](../docs/ALGORITHMS.md): production detector behavior
- [feature ledger](../docs/FEATURES.md): feature evidence and verdicts
- [Performance report](../docs/performance/README.md): generated detector metrics
- [CLI reference](../docs/CLI.md): supported repository entry points for collection and device workflows
