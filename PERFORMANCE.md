# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

## Test Data

| Dataset | Baseline | Movement | Total | Source |
|---------|----------|----------|-------|--------|
| ESP32-C6 | 1000 | 1000 | 2000 | `baseline_c6_64sc_20251212_142443.npz` |
| ESP32-S3 | 1353 | 1366 | 2719 | `baseline_s3_64sc_20260117_222606.npz` |

Data location: `micro-espectre/data/`

## Configuration

| Parameter | ESP32-C6 | ESP32-S3 |
|-----------|----------|----------|
| Window Size | 50 | 100 |
| Subcarriers | [11-22] | [18-29] |
| Hampel Filter | OFF | ON |
| Threshold | P95 × 1.4 | P95 × 1.4 |

---

## Results

### ESP32-C6 (HT20)

```
CONFUSION MATRIX (1000 baseline + 1000 movement packets):
                    Predicted
                IDLE        MOTION
Actual IDLE     998 (TN)    2 (FP)
Actual MOTION   8 (FN)      992 (TP)
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Recall | 99.2% | >90% | ✅ |
| Precision | 99.8% | - | ✅ |
| FP Rate | 0.2% | <10% | ✅ |
| F1-Score | 99.5% | - | ✅ |

### ESP32-S3 (HT20)

S3 uses larger window (100) and Hampel filter to handle higher baseline noise.

```
CONFUSION MATRIX (1353 baseline + 1366 movement packets):
                    Predicted
                IDLE        MOTION
Actual IDLE     1159 (TN)   194 (FP)
Actual MOTION   12 (FN)     1354 (TP)
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Recall | 99.1% | >90% | ✅ |
| Precision | 87.5% | - | ✅ |
| FP Rate | 14.3% | <15% | ✅ |
| F1-Score | 92.9% | - | ✅ |

---

## Chip Comparison

Real-time calibration results from the same environment.

| Chip | RSSI | P95 MV | FP Rate | Band | Stability |
|------|------|--------|---------|------|-----------|
| ESP32-C6 | -46 dB | 0.73 | 0.0% | [11-22] | ✅ |
| ESP32-S3 | -64 dB | 0.80 | 0.0% | [15-26] | ✅ |
| ESP32 | -49 dB | 0.82 | 0.8% | [19-30] | ⚠️ |
| ESP32-C3 | -73 dB | 0.83 | 0.0% | [37-48] | ⚠️ |
| ESP32-S3 (clone) | -80 dB | 1.71 | 65.2% | [37-48] | ❌ |

**Recommendation**: ESP32-C6 for best CSI stability. Avoid clone chips with non-Espressif silicon.

---

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

## Performance Targets

| Metric | C6 Target | S3 Target | Rationale |
|--------|-----------|-----------|-----------|
| Recall | >90% | >90% | Minimize missed detections |
| FP Rate | <10% | <15% | Avoid false alarms |
| Precision | >90% | >85% | Ensure reported motion is real |

S3 has relaxed FP rate target due to higher baseline noise.

See [TUNING.md](TUNING.md) for environment-specific adjustments.

---

## Version History

| Date | Version | Dataset | Algorithm | Mode | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-----------|------|--------|-----------|---------|----------|
| 2026-01-23 | v2.4.0 | C6 | MVS | P95 | 99.2% | 99.8% | 0.2% | 99.5% |
| 2026-01-23 | v2.4.0 | C6 | MVS | NBVI | 99.8% | 96.5% | 3.6% | 98.1% |
| 2026-01-23 | v2.4.0 | C6 | PCA | auto | 19.5% | 100.0% | 0.0% | 32.6% |
| 2026-01-23 | v2.4.0 | S3 | MVS | P95 + Hampel | 99.1% | 87.5% | 14.3% | 92.9% |
| 2026-01-23 | v2.4.0 | S3 | PCA | auto | 17.6% | 100.0% | 0.0% | 29.9% |
| 2025-12-27 | v2.3.0 | C6 | MVS | NBVI | 96.4% | 100.0% | 0.0% | 98.2% |
| 2025-12-27 | v2.3.0 | C6 | MVS | Fixed | 98.1% | 100.0% | 0.0% | 99.0% |

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.