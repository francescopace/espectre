# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

## Test Data

| Dataset | Baseline | Movement | Total | Source |
|---------|----------|----------|-------|--------|
| ESP32-C6 | 1000 | 1000 | 2000 | `baseline_c6_64sc_20251212_142443.npz` |
| ESP32-S3 | 1353 | 1366 | 2719 | `baseline_s3_64sc_20260117_222606.npz` |

Data location: `micro-espectre/data/`

## Performance Targets

| Metric | C6 Target | S3 Target | Rationale |
|--------|-----------|-----------|-----------|
| Recall | >90% | >90% | Minimize missed detections |
| FP Rate | <10% | <15% | Avoid false alarms |
| Precision | >90% | >85% | Ensure reported motion is real |

S3 has relaxed FP rate target due to higher baseline noise.

See [TUNING.md](TUNING.md) for environment-specific adjustments.

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

## Chip Comparison

Real-time nbvi calibration results from the same environment.

| Chip              | RSSI | P95 MV | Band    | Stability |
|-------------------|------|--------|---------|-----------|
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
| Weights memory | ~2 KB (constexpr) |
| Buffer memory | ~800 B (shared with MVS) |
| Dependencies | None |

### Inference Time

| Platform | Time | Throughput |
|----------|------|------------|
| Python (Mac M-series) | ~14 µs | 70,000 inf/sec |
| C++ ESP32-C6 (est.) | <200 µs | >5,000 inf/sec |
| C++ ESP32-S3 (est.) | <150 µs | >6,600 inf/sec |

The manual MLP inference (no TFLite dependency) is extremely lightweight: 12 → 16 → 8 → 1 = **337 float operations** per inference.
CSI packets arrive every ~10 ms, so inference time is never a bottleneck.

---

## Result History

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-02-04 | v2.5.0 | C3 | - | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-02-04 | v2.5.0 | C6 | - | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-02-04 | v2.5.0 | S3 | - | ML | 98.0% | 99.2% | 0.8% | 98.6% |
| 2026-02-04 | v2.5.0 | C3 | NBVI | MVS | 92.2% | 84.3% | 16.9% | 88.0% |
| 2026-02-04 | v2.5.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2026-02-04 | v2.5.0 | C3 | P95 | MVS | 92.2% | 84.3% | 16.9% | 88.0% |
| 2026-02-04 | v2.5.0 | C6 | P95 | MVS | 98.8% | 100.0% | 0.0% | 99.4% |
| 2026-02-04 | v2.5.0 | S3 | P95 | MVS | 99.1% | 87.5% | 14.3% | 92.9% |
| 2026-01-23 | v2.4.0 | C6 | P95 | MVS | 99.2% | 99.8% | 0.2% | 99.5% |
| 2026-01-23 | v2.4.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2026-01-23 | v2.4.0 | S3 | P95  | MVS | 99.1% | 87.5% | 14.3% | 92.9% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | MVS | 96.4% | 100.0% | 0.0% | 98.2% |
| 2025-12-27 | v2.3.0 | C6 | Fixed | MVS | 98.1% | 100.0% | 0.0% | 99.0% |

### MVS Detector Configuration

| Parameter | ESP32-C3 | ESP32-C6 | ESP32-S3 |
|-----------|----------|----------|----------|
| Window Size | 75 | 50 | 100 |
| Subcarriers | [20-31] | [11-22] | [18-29] |
| Hampel Filter | OFF | OFF | ON |
| Threshold | P95 × 1.1 | P95 × 1.4 | P95 × 1.4 |

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
