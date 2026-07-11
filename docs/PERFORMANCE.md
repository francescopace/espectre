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
| Recall | 96.0% | N/A | 98.6% | 99.4% |
| Precision | 99.7% | N/A | 96.8% | 91.4% |
| FP Rate | 0.2% | N/A | 1.7% | 4.7% |
| F1-Score | 97.8% | N/A | 97.6% | 95.3% |

### ML Detector

| Metric | ESP32-C3 | ESP32-C5 | ESP32-C6 | ESP32-S3 |
|--------|----------|----------|----------|----------|
| Recall | 99.8% | N/A | 98.8% | 100.0% |
| Precision | 100.0% | N/A | 99.9% | 100.0% |
| FP Rate | 0.0% | N/A | 0.1% | 0.0% |
| F1-Score | 99.9% | N/A | 99.3% | 100.0% |

---

## Long Quiet Real-Data Validation

### Classic Detector

| Metric | C3 | C5 | C6 | S3 |
|--------|----|----|----|----|
| Avg FP Rate | 0.30% | 0.43% | 0.56% | N/A |
| Max FP Rate | 0.42% | 1.06% | 0.96% | N/A |

### ML Detector

| Metric | C3 | C5 | C6 | S3 |
|--------|----|----|----|----|
| Avg FP Rate | 0.00% | 0.08% | 0.07% | N/A |
| Max FP Rate | 0.00% | 0.24% | 0.14% | N/A |
