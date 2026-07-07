# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

---

## Performance Targets

| Metric | Target | 
|-------|--------|
| Recall | >95% |
| FP Rate | <5% |

---
### Test Configuration

Shared validation configuration across chips and detectors:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 100 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | Fixed subcarriers + detector-specific threshold bootstrap | Shared 12-subcarrier set across MVS, L1-Delta, and ML validation |
| MVS Threshold | `max x 1.3` | Shared startup bootstrap (`DEFAULT_ADAPTIVE_FACTOR`) |
| L1-Delta Threshold | Startup gate + detector factor | `StartupThresholdCalibrator` with factor `1.1`, rolling-chunk consistency gate, and calibration extension |
| Hampel Filter | MVS on, L1-Delta off | MVS still uses the runtime Hampel path (`window=7`, `threshold=5.0 MAD`); L1-Delta does not require it |
| ML Threshold | Fixed | `5.0` exported runtime threshold |
| CV Normalization | Always on | Shared AGC-active turbulence path (`std/mean`) |

CV normalization is applied uniformly across the production and validation
pipeline. Both detectors use CV-normalized turbulence (`std/mean`), while ML
still exports relative neural-detector features such as `std/mean`, `iqr/mean`,
`mad/mean`, and normalized waveform length.

---

## Training Dataset

`data/dataset_info.json` contains canonical `empty` / `static_presence` /
`motion` labels across multiple collection sessions and environments.
The counts below are aggregated packet totals across all currently available training captures, including the dedicated empty-room recordings.

| Chip | Empty | Static Presence | Motion | Total |
|------|-------|-----------------|--------|-------|
| ESP32-C3 | 23899 | 35587 | 17803 | 77289 |
| ESP32-C5 | 11962 | 17444 | 8727 | 38133 |
| ESP32-C6 | 23956 | 53014 | 26429 | 103399 |
| ESP32-S3 | 23930 | 25594 | 17104 | 66628 |
| Total | 83747 | 131639 | 70063 | 285449 |

Data location: `data/`

---

## Current Results

- C++ `test_motion_detection`
- C++ `test_long_recordings`
- Python `TestPerformanceMetrics`
- Python `test_validation_long_recordings.py`

### Paired Real-Data Validation

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS | 93.7% | 95.4% | 2.7% | 94.2% |
| ESP32-C3 | L1-Delta | 89.7% | 96.9% | 1.7% | 92.7% |
| ESP32-C3 | ML | 94.9% | 99.9% | 0.1% | 97.3% |
| ESP32-C5 | MVS | 99.7% | 96.5% | 1.8% | 98.1% |
| ESP32-C5 | L1-Delta | 95.8% | 99.3% | 0.3% | 97.5% |
| ESP32-C5 | ML | 98.4% | 91.9% | 5.1% | 94.6% |
| ESP32-C6 | MVS | 98.9% | 95.9% | 2.3% | 97.3% |
| ESP32-C6 | L1-Delta | 96.2% | 94.8% | 2.8% | 95.3% |
| ESP32-C6 | ML | 95.1% | 99.6% | 0.2% | 97.3% |
| ESP32-S3 | MVS | 79.5% | 88.5% | 6.4% | 82.4% |
| ESP32-S3 | L1-Delta | 96.4% | 89.9% | 6.0% | 92.7% |
| ESP32-S3 | ML | 88.4% | 91.6% | 4.2% | 89.7% |

---

## Long Quiet Recordings

The current long-run validation set contains quiet-only captures with no
annotated motion segment (`motion_start_packet` is absent, so the full stream is
treated as baseline). In practice this section is a sustained false-positive
gate, not a recall benchmark.

| Chip | Recordings | MVS Avg FP Rate | MVS Max FP Rate | L1D Avg FP Rate | L1D Max FP Rate | ML Avg FP Rate | ML Max FP Rate |
|------|------------|-----------------|-----------------|-----------------|-----------------|----------------|----------------|
| C3 | 2 | 0.72% | 1.02% | 0.30% | 0.42% | 0.00% | 0.00% |
| C5 | 3 | 0.53% | 1.33% | 0.37% | 1.06% | 1.42% | 2.90% |
| C6 | 2 | 0.75% | 0.82% | 6.38% | 11.94% | 0.02% | 0.03% |
| S3 | 3 | 6.68% | 11.84% | 3.63% | 7.90% | 4.60% | 7.71% |

---

## Result History (ESP32-C6)

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-07-07 | v3.0.0 | C6 | legacy| L1D | 96.2% | 94.8% | 2.8% | 95.3% |
| 2026-07-07 | v3.0.0 | C6 |  -   |  ML  | 95.1% | 99.6% | 0.2% | 97.3% |
| 2026-07-07 | v3.0.0 | C6 | `max x 1.3` | MVS + Hampel | 98.9% | 95.9% | 2.3% | 97.3% |
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
