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

Shared validation configuration across chips and detectors:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 100 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | Fixed subcarriers + threshold bootstrap | Shared 12-subcarrier set; threshold bootstrap applies to MVS, while ML keeps a fixed threshold |
| Hampel Filter | ON | Enabled for both MVS and ML (window=7, threshold=5.0 MAD) |
| Adaptive Threshold | Percentile-based | P100 x 1.3 (`DEFAULT_ADAPTIVE_FACTOR`) |
| CV Normalization | Always on | Shared AGC-active turbulence path (`std/mean`) |

CV normalization is applied uniformly across the production and validation
pipeline. Both detectors use CV-normalized turbulence (`std/mean`), while ML
still exports relative neural-detector features such as `std/mean`, `iqr/mean`,
`mad/mean`, and normalized waveform length.

---

## Training Dataset

`data/dataset_info.json` contains canonical `empty` / `static_presence` /`motion` labels across multiple collection sessions and environments.  
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

### Real-data validation

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS  | 99.3% | 84.0% | 9.6% | 91.0% |
| ESP32-C3 | ML | 94.1% | 100.0% | 0.0% | 97.0% |
| ESP32-C5 | MVS  | 100.0% | 86.8% | 7.7% | 92.9% |
| ESP32-C5 | ML | 100.0% | 87.4% | 7.3% | 93.3% |
| ESP32-C6 | MVS  | 58.8% | 99.6% | 0.1% | 73.9% |
| ESP32-C6 | ML | 89.9% | 99.5% | 0.2% | 94.4% |
| ESP32-S3 | MVS | 75.3% | 96.7% | 1.3% | 84.7% |
| ESP32-S3 | ML | 92.6% | 89.8% | 5.2% | 91.2% |

---

## Long Test Recordings

| Chip | MVS Recall | MVS FP Rate | ML Recall | ML FP Rate |
|------|-------|-----|------|----|
| C3 | 2.7% | 4.4% | 0.0% | 0.0% |
| C5 | 2.0% | 1.5% | 1.7% | 1.6% |
| C6 | 8.0% | 6.9% | 0.2% | 0.2% |
| S3 | 14.5% | 11.9% | 7.4% | 4.4% |

---

## Result History (ESP32-C6)

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-07-04 | v3.0.0 | C6 |  -   | ML + Hampel | 89.9% | 99.5% | 0.2% | 94.4% |
| 2026-07-04 | v3.0.0 | C6 |  -   | MVS + Hampel | 58.8% | 99.6% | 0.1% | 73.9% |
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
