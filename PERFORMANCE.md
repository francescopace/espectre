# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

## Test Data

| Chip | Baseline | Movement | Total | Gain Lock | CV Norm |
|------|----------|----------|-------|-----------|---------|
| ESP32-C3 | 2656 | 3154 | 5810 | Mixed | Mixed |
| ESP32-C6 | 7379 | 1000 | 8379 | Yes | No |
| ESP32-S3 | 1353 | 1366 | 2719 | Yes | No |
| ESP32 | 961 | 1103 | 2064 | No | Yes |

Data location: `micro-espectre/data/`

### Gain Lock and CV Normalization

**AGC Gain Lock** stabilizes CSI amplitudes by locking the receiver's automatic gain control. Without it, amplitudes vary with signal strength, making raw values unreliable for detection.

- **ESP32-C6, ESP32-S3**: Support gain lock via `esp_wifi_set_csi_rx_ctrl()`
- **ESP32-C3**: Supports gain lock, but ESPectre skips it when AGC gain < 30 (weak signal). Some C3 datasets were collected with gain 28, so gain lock was skipped.
- **ESP32 (original)**: Does not support gain lock in the CSI driver

**CV Normalization** (`std/mean`) makes detection gain-invariant by normalizing spatial turbulence. It's applied during feature extraction for files marked with `use_cv_normalization: true` in `dataset_info.json`.

## Performance Targets

### MVS Detector (NBVI Calibration)

| Metric | C3 Target | C6 Target | S3 Target | ESP32 Target | Rationale |
|--------|-----------|-----------|-----------|--------------|-----------|
| Recall | >90% | >95% | >90% | >90% | Minimize missed detections |
| FP Rate | <20% | <5% | <15% | <20% | Avoid false alarms |

NBVI's non-consecutive subcarrier selection provides spectral diversity for robust detection.

### ML Detector

| Metric | C3 Target | C6 Target | S3 Target | ESP32 Target | Rationale |
|--------|-----------|-----------|-----------|--------------|-----------|
| Recall | >93% | >93% | >93% | >93% | Minimize missed detections |
| FP Rate | <10% | <10% | <10% | <10% | Avoid false alarms |

ML uses fixed sparse subcarriers and pre-trained weights (no calibration needed). CV normalization is applied during training for datasets without gain lock.

See [TUNING.md](TUNING.md) for environment-specific adjustments.

### Test Coverage Matrix

| Test | C3 | C6 | S3 | ESP32 (control) |
|------|-----|-----|-----|-----------------|
| MVS + NBVI auto-calibration | ✅ | ✅ | ✅ | ✅ |
| MVS + Optimal subcarriers | ✅ | ✅ | ✅ | ✅ |
| ML detection | ✅ | ✅ | ✅ | ✅ |
| Threshold sensitivity (C++) | ✅ | ✅ | ✅ | ✅ |
| Window size sensitivity (C++) | ✅ | ✅ | ✅ | ✅ |

**MVS + Optimal subcarriers**: Uses offline-tuned subcarriers (best case reference).
**MVS + NBVI auto-calibration**: Uses NBVI for runtime subcarrier selection (production case).

Tests run on both C++ (PlatformIO) and Python (pytest) platforms with identical methodology and identical results.

## Running Tests

```bash
source venv/bin/activate

# C++
cd test && pio test -f test_motion_detection -v

# Python
cd micro-espectre && pytest tests/test_validation_real_data.py -v
```

Both platforms produce identical results with the same methodology.

---

## Current Results

Results from C++ and Python tests are **identical** (same algorithms, same data, same methodology).

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS Optimal | 99.6% | 98.2% | 3.4% | 98.9% |
| ESP32-C3 | MVS + NBVI | 99.6% | 98.2% | 3.4% | 98.9% |
| ESP32-C3 | ML | 99.8% | 99.9% | 0.1% | 99.9% |
| ESP32-C6 | MVS Optimal | 99.3% | 96.9% | 4.6% | 98.1% |
| ESP32-C6 | MVS + NBVI | 99.9% | 96.0% | 6.0% | 97.9% |
| ESP32-C6 | ML | 99.8% | 100.0% | 0.0% | 99.9% |
| ESP32-S3 | MVS Optimal | 99.7% | 97.6% | 3.2% | 98.6% |
| ESP32-S3 | MVS + NBVI | 96.6% | 95.6% | 5.8% | 96.1% |
| ESP32-S3 | ML | 98.5% | 100.0% | 0.0% | 99.3% |
| ESP32 (control) | MVS Optimal | 100.0% | 97.4% | 4.5% | 98.7% |
| ESP32 (control) | MVS + NBVI | 100.0% | 97.4% | 4.5% | 98.7% |
| ESP32 (control) | ML | 100.0% | 100.0% | 0.0% | 100.0% |

**MVS Optimal**: Uses offline-tuned subcarriers (best case reference).
**MVS + NBVI**: Uses NBVI auto-calibration (production case).

ESP32 (Base) is excluded from ML training data and used as a control set. Despite this, it achieves 100.0% recall and 100.0% F1-score, demonstrating strong cross-chip generalization.

---

## Chip Comparison

Real-time NBVI calibration results from the same environment.

| Chip              | RSSI | Baseline MV | Band    | Stability |
|-------------------|------|-------------|---------|-----------|
| ESP32-C6          | -46 dB | 0.73 | [11-22] | ✅        |
| ESP32-S3          | -64 dB | 0.80 | [15-26] | ✅        |
| ESP32             | -49 dB | 0.82 | [19-30] | ⚠️        |
| ESP32-C3          | -73 dB | 0.83 | [37-48] | ⚠️        |
| ESP32-S3 (clone)  | -80 dB | 1.71 | [37-48] | ❌        |

**Recommendation**: ESP32-C6 for best CSI stability. Avoid clone chips with non-Espressif silicon.

---

## ML Detector Performance

### Resource Usage

| Resource | Value |
|----------|-------|
| Weights memory | ~1.4 KB (353 params, constexpr) |
| Buffer memory | ~800 B (shared with MVS) |
| Dependencies | None |

### Inference Time

| Platform | Time | Throughput |
|----------|------|------------|
| Python (Mac M-series) | ~14 µs | 70,000 inf/sec |
| C++ ESP32-C6 (est.) | <200 µs | >5,000 inf/sec |
| C++ ESP32-S3 (est.) | <150 µs | >6,600 inf/sec |

The manual MLP inference (no TFLite dependency) is extremely lightweight: 12 → 16 → 8 → 1 = **328 multiply-accumulate operations (MACs)** per inference.
CSI packets arrive every ~10 ms, so inference time is never a bottleneck.

For architecture comparison and training pipeline details, see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md#architecture-selection).

---

## Result History

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-02-15 | v2.5.0 | All | - | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-02-14 | v2.5.0 | C6 | - | ML | 99.6% | 100.0% | 0.0% | 99.8% |
| 2026-02-14 | v2.5.0 | C6 | NBVI | MVS | 99.9% | 96.1% | 5.9% | 97.9% |
| 2026-01-23 | v2.4.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | MVS | 96.4% | 100.0% | 0.0% | 98.2% |

### Test Configuration

Configuration used for all test results (unified across chips):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 75 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | NBVI | Auto-selects 12 non-consecutive subcarriers |
| Hampel Filter | OFF | Can be enabled for noisy environments |
| Adaptive Threshold | Percentile-based | `DEFAULT_ADAPTIVE_PERCENTILE` (P95) |
| CV Normalization | Per-file | Based on `use_cv_normalization` in `dataset_info.json` |

CV normalization is applied per-file based on whether data was collected with AGC gain lock enabled. See Test Data section for details.

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
