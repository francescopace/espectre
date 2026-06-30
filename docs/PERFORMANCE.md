# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

---

## Performance Targets

| Scope | Metric | Target | Rationale |
|-------|--------|--------|-----------|
| MVS | Recall | >95% | Minimize missed detections |
| MVS | FP Rate | <5% | Avoid false alarms |
| ML | Recall | >95% | Maintain high sensitivity |
| ML | FP Rate | <5% | Avoid false alarms |

--
### Test Configuration

Configuration used for all test results (unified across chips):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 100 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | Fixed subcarriers + threshold bootstrap | Shared 12-subcarrier set, adaptive threshold for MVS |
| Hampel Filter | ON | Enabled for both MVS and ML (window=7, threshold=5.0 MAD) |
| Adaptive Threshold | Percentile-based | P95 × 1.1 (`DEFAULT_ADAPTIVE_FACTOR`) |
| CV Normalization | Gain-mode aware | Based on `gain_locked` metadata (`false` => apply CV norm) |

CV normalization is applied per-file based on whether data was collected with AGC gain lock enabled. Gain-locked streams use raw turbulence; streams without gain lock use CV-normalized turbulence (`std/mean`). ML then exports relative neural-detector features such as `std/mean`, `iqr/mean`, `mad/mean`, and normalized waveform length.

---

## Training Dataset

`data/dataset_info.json` contains canonical `empty` / `static_presence` /
`motion` labels across multiple collection sessions and environments. The
counts below are aggregated packet totals across all currently available
training captures, including the dedicated empty-room recordings.

| Chip | Empty | Static Presence | Motion | Total | Gain Lock |
|------|-------|-----------------|--------|-------|-----------|
| ESP32-C3 | 21113 | 23204 | 11100 | 55417 | Yes |
| ESP32-C5 | 21144 | 23359 | 11380 | 55883 | Yes |
| ESP32-C6 | 21003 | 23770 | 11891 | 56664 | Yes |
| ESP32-S3 | 21007 | 23364 | 11376 | 55747 | Yes |
| ESP32 | 18495 | 20535 | 9513 | 48543 | No |
| Total | 102762 | 114232 | 55260 | 272254 | Mixed |

Data location: `data/`

---

## Running Tests

```bash
source .venv/bin/activate

# C++
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
ctest --test-dir test/cpp/build -R test_long_recordings --output-on-failure

# Python (real-data validation)
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v

# Python (60-second long recordings, prints summary tables)
pytest test/python/test_validation_long_recordings.py -v -s
```

---

## Current Results

**Last verified:** 2026-06-30 (`test_motion_detection`, C++ `test_long_recordings`, Python `TestPerformanceMetrics`, Python `test_validation_long_recordings.py`)

### Python + C++ real-data validation

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS Default | 98.5% | 100.0% | 0.0% | 99.3% |
| ESP32-C3 | MVS Runtime | 98.5% | 100.0% | 0.0% | 99.3% |
| ESP32-C3 | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| ESP32-C5 | MVS Default | 99.4% | 100.0% | 0.0% | 99.7% |
| ESP32-C5 | MVS Runtime | 99.4% | 100.0% | 0.0% | 99.7% |
| ESP32-C5 | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| ESP32-C6 | MVS Default | 99.7% | 100.0% | 0.0% | 99.9% |
| ESP32-C6 | MVS Runtime | 99.7% | 100.0% | 0.0% | 99.9% |
| ESP32-C6 | ML | 99.2% | 100.0% | 0.0% | 99.6% |
| ESP32-S3 | MVS Default | 99.7% | 100.0% | 0.0% | 99.9% |
| ESP32-S3 | MVS Runtime | 99.7% | 100.0% | 0.0% | 99.9% |
| ESP32-S3 | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| ESP32 | MVS Default | 99.4% | 100.0% | 0.0% | 99.7% |
| ESP32 | MVS Runtime | 99.4% | 100.0% | 0.0% | 99.7% |
| ESP32 | ML | 100.0% | 100.0% | 0.0% | 100.0% |

**MVS Default**: Uses fixed default subcarriers with adaptive threshold from baseline.
**MVS Runtime**: Current production startup path; matches `MVS Default`.
**ML**: Neural network with grouped session-level blocked CV for model selection, context-aware MVS-guided weights, Hampel filtering, gain-mode-aware turbulence normalization, and exported relative turbulence-window features. Binary training uses `empty`, `static_presence`, and `motion`; `empty` and `static_presence` are both IDLE targets.

---

## System Resources

Resource usage benchmarks for ESPectre with full ESPHome stack (WiFi, API, OTA, debug sensors).

Development YAML files (`-dev.yaml`) include ESPHome debug sensors for runtime monitoring of free heap, max block size, and loop time. 
These sensors are available in Home Assistant for continuous monitoring.

Additional performance logs are available at DEBUG level (`logger.level: DEBUG`):
- `[resources]` - Free heap at startup and post-calibration
- `[perf]` - Detection time per packet (logged every ~10 seconds)

---

### Flash Usage

| Chip | Firmware Size | Flash Used | Free App Slot |
|------|---------------|------------|---------------|
| ESP32-C3 | 1370 KB | 73.8% | 486 KB |
| ESP32-C5 | 1587 KB | 85.5% | 269 KB |
| ESP32-C6 | 1539 KB | 82.9% | 317 KB |
| ESP32-S3 | 1246 KB | 67.1% | 610 KB |

Partition layout uses two app slots (`app0`/`app1`, 1.81 MB each) plus a small `otadata` partition for OTA metadata.
 `Free App Slot` is the remaining space in one app slot after placing the firmware image.

---

### RAM Usage

| Chip | Phase | Free Heap | Notes |
|------|-------|-----------|-------|
| ESP32-C3 | Post-setup | 179 KB | After ESPectre init |
| ESP32-C3 | Post-calibration | 83 KB | After startup calibration completes |
| ESP32-C5 | Post-setup | 162 KB | After ESPectre init |
| ESP32-C5 | Post-calibration | 71 KB | After startup calibration completes |
| ESP32-C6 | Post-setup | 272 KB | After ESPectre init |
| ESP32-C6 | Post-calibration | 180 KB | After startup calibration completes |
| ESP32-S3 | Post-setup | 8425 KB | After ESPectre init (includes PSRAM heap) |
| ESP32-S3 | Post-calibration | 8331 KB | After startup calibration completes (includes PSRAM heap) |

---

### Detection Timing

Time to process one CSI packet (feature extraction + detection, measured on hardware).
At 100 pps, each packet has a 10 ms budget. 

| Chip | Algorithm | Detection Time | CPU @ 100 pps |
|------|-----------|----------------|---------------|
| ESP32-C3 | MVS | ~440 µs | ~4.4% |
| ESP32-C3 | ML | ~3400 µs | ~34% |
| ESP32-C5 | MVS | ~220 µs | ~2.2% |
| ESP32-C5 | ML | ~1500 µs | ~15% |
| ESP32-C6 | MVS | ~250 µs | ~2.5% |
| ESP32-C6 | ML | ~1900 µs | ~19% |
| ESP32-S3 | MVS | ~150 µs | ~1.5% |
| ESP32-S3 | ML | ~430 µs | ~4.3% |

The worst-case path is ML on ESP32-C3 (~3.5 ms peak, ~35% CPU), which still leaves substantial budget for WiFi, ESPHome, and Home Assistant communication.

**MVS**: Extracts a single feature (spatial turbulence) and its moving variance.

**ML**: Extracts 8 relative statistical features from the sliding window, then runs MLP inference (8 -> 32 -> 16 -> 1 = 784 MACs).
The MLP itself is lightweight; most time is spent on feature extraction. 
For ML architecture details, see [ALGORITHMS.md](ALGORITHMS.md#architecture).

---

## 60-Second Test Recordings

Continuous recordings (~30s idle + ~30s motion) provide a realistic production-style scenario. These files are not used during training.

Test data: `data/test/`
Source of truth: `test/python/test_validation_long_recordings.py`

The Python and C++ long-recording suites currently produce matching packet-level metrics on all available long-test datasets (`C3`, `C5`, `C6`, `S3`).

Methodology:
- `MVS Fixed`: keep the shared fixed subcarrier set, run baseline threshold bootstrap on the idle segment, then evaluate the full recording with adaptive threshold and Hampel enabled
- `ML`: use exported production weights with threshold `5.0` and Hampel enabled
- Both paths skip the first `100` packets of each segment as warmup when scoring packet-level metrics

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score | FP Count |
|------|-----------|--------|-----------|---------|----------|----------|
| C3 | MVS Fixed | 100.0% | 99.8% | 0.2% | 99.9% | 5 |
| C3 | ML | 100.0% | 100.0% | 0.0% | 100.0% | 0 |
| C5 | MVS Fixed | 100.0% | 87.9% | 11.1% | 93.5% | 357 |
| C5 | ML | 100.0% | 91.3% | 7.7% | 95.4% | 248 |
| C6 | MVS Fixed | 100.0% | 70.4% | 40.2% | 82.6% | 1270 |
| C6 | ML | 97.3% | 90.2% | 10.1% | 93.6% | 319 |
| S3 | MVS Fixed | 100.0% | 94.8% | 4.9% | 97.3% | 151 |
| S3 | ML | 99.5% | 100.0% | 0.0% | 99.7% | 1 |

---

## Result History (ESP32-C6)

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-06-30 | v3.0.0 | C6 |  -   | ML + Hampel | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-06-30 | v3.0.0 | C6 |  -   | MVS + Hampel | 99.7% | 100.0% | 0.0% | 99.9% |
| 2026-05-21 | v2.8.0 | C6 |  -   | ML + Hampel | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-05-21 | v2.8.0 | C6 | NBVI | MVS + Hampel| 99.6% | 100.0% | 0.0% | 99.8% |
| 2026-03-11 | v2.6.1 | C6 |  -   | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-03-11 | v2.6.1 | C6 | NBVI | MVS | 99.3% | 100.0% | 0.0% | 99.7% |
| 2026-03-08 | v2.6.0 | C6 |  -   | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-03-08 | v2.6.0 | C6 | NBVI | MVS | 99.9% | 98.4% | 2.3% | 99.2% |
| 2026-02-15 | v2.5.0 | C6 |   -  | ML  | 99.9% | 100.0% | 0.0% | 99.9% |
| 2026-02-15 | v2.5.0 | C6 | NBVI | MVS | 99.9% | 99.9% | 0.1% | 99.9% |
| 2026-01-23 | v2.4.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | MVS | 96.4% | 100.0% | 0.0% | 98.2% |

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.
