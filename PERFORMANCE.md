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

| Test | C3 | C6 | S3 | ESP32 |
|------|-----|-----|-----|-------|
| MVS + NBVI auto-calibration | ✅ | ✅ | ✅ | ✅ |
| MVS + Optimal subcarriers | ✅ | ✅ | ✅ | ✅ |
| ML detection | ✅ | ✅ | ✅ | ✅ |
| Threshold sensitivity (C++) | ✅ | ✅ | ✅ | ✅ |
| Window size sensitivity (C++) | ✅ | ✅ | ✅ | ✅ |

**MVS + Optimal subcarriers**: Uses offline-tuned subcarriers (best case reference).
**MVS + NBVI auto-calibration**: Uses NBVI for runtime subcarrier selection (production case).

Tests run on both C++ (PlatformIO) and Python (pytest) platforms with identical results.

## Running Tests

```bash
source venv/bin/activate

# C++
cd test && pio test -f test_motion_detection -v

# Python
cd micro-espectre && pytest tests/test_validation_real_data.py -v
```

---

## Current Results

Results from C++ and Python tests are identical (same algorithms, same data, same methodology).

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS Optimal | 99.6% | 100.0% | 0.0% | 99.8% |
| ESP32-C3 | MVS + NBVI | 99.6% | 100.0% | 0.0% | 99.8% |
| ESP32-C3 | ML | 99.7% | 99.8% | 0.3% | 99.8% |
| ESP32-C6 | MVS Optimal | 98.8% | 99.8% | 0.3% | 99.3% |
| ESP32-C6 | MVS + NBVI | 99.9% | 99.9% | 0.1% | 99.9% |
| ESP32-C6 | ML | 99.7% | 100.0% | 0.0% | 99.8% |
| ESP32-S3 | MVS Optimal | 98.6% | 99.3% | 0.9% | 99.0% |
| ESP32-S3 | MVS + NBVI | 95.8% | 100.0% | 0.0% | 97.9% |
| ESP32-S3 | ML | 98.1% | 100.0% | 0.0% | 99.0% |
| ESP32 | MVS Optimal | 100.0% | 98.7% | 2.3% | 99.3% |
| ESP32 | MVS + NBVI | 100.0% | 98.7% | 2.3% | 99.3% |
| ESP32 | ML | 100.0% | 100.0% | 0.0% | 100.0% |

**MVS Optimal**: Uses offline-tuned subcarriers (best case reference).
**MVS + NBVI**: Uses NBVI auto-calibration (production case).

ESP32 (original) is excluded from ML training data due to lack of gain lock support. CV normalization is applied to compensate. Despite being excluded from training, it achieves 100.0% recall and F1-score, demonstrating strong cross-chip generalization.

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

## System Resources

Resource usage benchmarks for ESPectre with full ESPHome stack (WiFi, API, OTA, debug sensors).

### Flash Usage

| Chip | Firmware Size | Flash Used | Available (OTA) |
|------|---------------|------------|-----------------|
| ESP32-C3 | 856 KB | 45% | 1.0 MB |

Partition layout uses dual OTA partitions (1.81 MB each) for safe updates.

### RAM Usage

| Chip | Phase | Free Heap | Notes |
|------|-------|-----------|-------|
| ESP32-C3 | Post-setup | 214 KB | After ESPectre init |
| ESP32-C3 | Post-calibration | 174 KB | After NBVI completes |
| ESP32-C6 | Post-setup | 311 KB | After ESPectre init |
| ESP32-C6 | Post-calibration | 269 KB | After NBVI completes |

ESPectre runtime footprint: ~40 KB (calibration buffers released after completion).

### Detection Timing

Time to process one CSI packet (feature extraction + detection, measured on hardware):

| Chip | Algorithm | Detection Time | CPU @ 100 pps |
|------|-----------|----------------|---------------|
| ESP32-C3 | MVS | ~440 µs | ~4.4% |
| ESP32-C3 | ML | ~3300 µs | ~33% |
| ESP32-C6 | MVS | ~250 µs | ~2.5% |
| ESP32-C6 | ML | ~2100 µs | ~21% |

At 100 pps, each packet has a 10 ms budget. MVS uses ~0.44 ms (4.4%) and ML uses ~3.3 ms (33%), leaving ample headroom for WiFi, ESPHome, and Home Assistant communication.

**MVS**: Extracts a single feature (spatial turbulence) and its moving variance.

**ML**: Extracts 12 statistical features from sliding window, then runs MLP inference (12 → 16 → 8 → 1 = 328 MACs). The MLP itself is lightweight; most time is spent on feature extraction. For ML architecture details, see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md#architecture-selection).

### Monitoring

Development YAML files (`-dev.yaml`) include ESPHome debug sensors for runtime monitoring of free heap, max block size, and loop time. These sensors are available in Home Assistant for continuous monitoring.

Additional performance logs are available at DEBUG level (`logger.level: DEBUG`):
- `[resources]` - Free heap at startup and post-calibration
- `[perf]` - Detection time per packet (logged every ~10 seconds)

---

## Result History

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-02-15 | v2.5.0 | C6 | - | ML | 99.9% | 100.0% | 0.0% | 99.9% |
| 2026-02-15 | v2.5.0 | C6 | NBVI | MVS | 99.9% | 99.9% | 0.1% | 99.9% |
| 2026-01-23 | v2.4.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | MVS | 96.4% | 100.0% | 0.0% | 98.2% |

### Test Configuration

Configuration used for all test results (unified across chips):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 75 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | NBVI | Auto-selects 12 non-consecutive subcarriers |
| Hampel Filter | OFF | Can be enabled for noisy environments |
| Adaptive Threshold | Percentile-based | P95 × 1.1 (`DEFAULT_ADAPTIVE_FACTOR`) |
| CV Normalization | Per-file | Based on `use_cv_normalization` in `dataset_info.json` |

CV normalization is applied per-file based on whether data was collected with AGC gain lock enabled. See Test Data section for details.

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
