# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

- **Classic Detector**: Uses L1-Delta as the primary metric, with a gated moving-variance recovery vote.
- **ML Detector**: Uses a pretrained neural network model based on turbulence and spectral features.

See [ALGORITHMS.md](ALGORITHMS.md) for the full detector design.

---

## Performance Targets

| Metric | Target | 
|-------|--------|
| Recall | >95% |
| FP Rate | <5% |

---

## Test Scripts

- C++ `test_motion_detection`
- C++ `test_long_recordings`
- Python `TestPerformanceMetrics`
- Python `test_validation_long_recordings.py`

---

## Paired Real-Data Validation (empty+static_presence / motion)

### Classic Detector

| Metric | ESP32-C3 | ESP32-C5 | ESP32-C6 | ESP32-S3 |
|--------|----------|----------|----------|----------|
| Recall | 96.0% | 99.4% | 98.5% | 95.8% |
| Precision | 95.1% | 98.4% | 94.4% | 89.8% |
| FP Rate | 2.8% | 0.8% | 3.0% | 5.9% |
| F1-Score | 95.3% | 98.9% | 96.4% | 92.5% |

### ML Detector

| Metric | ESP32-C3 | ESP32-C5 | ESP32-C6 | ESP32-S3 |
|--------|----------|----------|----------|----------|
| Recall | 97.9% | 98.9% | 94.6% | 92.2% |
| Precision | 99.8% | 94.8% | 100.0% | 98.2% |
| FP Rate | 0.1% | 3.0% | 0.0% | 0.9% |
| F1-Score | 98.8% | 96.7% | 97.2% | 95.0% |

---

## Long Quiet Real-Data Validation

### Classic Detector

| Metric | C3 | C5 | C6 | S3 |
|--------|----|----|----|----|
| Avg FP Rate | 0.30% | 0.43% | 0.60% | 4.83% |
| Max FP Rate | 0.40% | 1.10% | 1.00% | 7.90% |

### ML Detector

| Metric | C3 | C5 | C6 | S3 |
|--------|----|----|----|----|
| Avg FP Rate | 0.00% | 1.03% | 0.05% | 4.90% |
| Max FP Rate | 0.00% | 2.00% | 0.10% | 11.50% |
