# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection system based on Moving Variance Segmentation (MVS).

## Test Methodology

### Test Data

| Dataset | Baseline Packets | Movement Packets | Total |
|---------|------------------|------------------|-------|
| ESP32-C6 | 1000 | 1000 | 2000 |
| ESP32-S3 | 1353 | 1366 | 2719 |

- **Packet Rate**: 100 packets/second

### Configuration

| Parameter | ESP32-C6 | ESP32-S3 |
|-----------|----------|----------|
| Window Size | 50 packets | 100 packets |
| Threshold | Adaptive (default P95 × 1.4) | Adaptive (default P95 × 1.4) |
| Subcarriers | [11-22] | [48-59] |
| Hampel Filter | OFF | ON |

### Test Environment
- **Platform**: ESP32-C6, ESP32-S3
- **Distance from router**: ~3 meters
- **Environment**: Indoor residential

---

## Results

Both platforms produce **identical results** using the same test methodology:
- Process all baseline packets first (expecting IDLE)
- Then process all movement packets (expecting MOTION)
- Continuous context (no reset between baseline and movement)
- Filters disabled (lowpass, hampel off by default)
- Adaptive threshold: P95(baseline_mv) × 1.4

### 64 SC (HT20) - ESP32-C6

```
CONFUSION MATRIX (1000 baseline + 1000 movement packets):
                    Predicted
                IDLE        MOTION
Actual IDLE     1000 (TN)   0 (FP)
Actual MOTION   12 (FN)     988 (TP)
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Recall** | 98.8% | >90% | ✅ |
| **Precision** | 100.0% | - | ✅ |
| **FP Rate** | 0.0% | <10% | ✅ |
| **F1-Score** | 99.4% | - | ✅ |

### 64 SC (HT20) - ESP32-S3

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
| **Recall** | 99.1% | >90% | ✅ |
| **Precision** | 87.5% | - | ✅ |
| **FP Rate** | 14.3% | <15% | ✅ |
| **F1-Score** | 92.9% | - | ✅ |

**Note**: S3 has higher baseline noise leading to more false positives. The Hampel filter helps reduce spikes, and a larger window (100 vs 50) provides more stable variance estimation.

---

## Interpretation

### Strengths
- **Zero false positives**: The system never triggers false alarms during idle periods
- **High recall (98.8%)**: Detects 98.8% of all movement events
- **Perfect precision**: When motion is reported, it's always real motion

### False Negatives Analysis
The 12 false negatives (1.2% of movement packets) are typically caused by:
1. **Transition periods**: Packets at the very start/end of movement sequences
2. **Micro-movements**: Very subtle movements that don't exceed the threshold
3. **Buffer warm-up**: First few packets after state transitions

These missed detections are acceptable for most use cases (home automation, presence detection) where brief detection gaps don't impact functionality.

---

## How to Verify Performance

### Monitor Detection in Real-Time

```bash
# View ESPHome logs (choose your platform)
esphome logs <your-config>.yaml
```

Watch for state transitions:
- `state=MOTION` when movement occurs
- `state=IDLE` when room is quiet

### Home Assistant History

Use Home Assistant's History panel to visualize:
- **binary_sensor.espectre_motion_detected** - Motion events over time
- **sensor.espectre_movement_score** - Movement intensity graph

---

## Reproducing These Results

### Test Data Location

Both C++ and Python tests use the **same real CSI data** captured from ESP32-C6:

| Chip | Baseline | Movement |
|------|----------|----------|
| ESP32-C6 | `baseline_c6_64sc_20251212_142443.npz` | `movement_c6_64sc_20251212_142443.npz` |
| ESP32-S3 | `baseline_s3_64sc_20260117_222606.npz` | `movement_s3_64sc_20260117_222626.npz` |

Files are located in `micro-espectre/data/`.

### Running the Tests

**C++ (ESPHome component)**:

```bash
# Activate virtual environment
source venv/bin/activate

# Run motion detection test suite (shows confusion matrix)
cd test
pio test -f test_motion_detection -vvv
```

**Python (Micro-ESPectre)**:

```bash
# Activate virtual environment
source venv/bin/activate

# Run performance test
cd micro-espectre/tests
pytest test_validation_real_data.py::TestPerformanceMetrics::test_mvs_detection_accuracy -v -s
```

### Test Implementation

| Platform | Test File | Test Function |
|----------|-----------|---------------|
| **C++** | `test/test/test_motion_detection/test_motion_detection.cpp` | `test_mvs_detection_accuracy()` |
| **Python** | `micro-espectre/tests/test_validation_real_data.py` | `TestPerformanceMetrics::test_mvs_detection_accuracy()` |

Both tests use identical methodology with chip-specific parameters:
1. Initialize MVS with chip-specific window size (C6: 50, S3: 100)
2. Select chip-specific subcarriers (C6: [11-22], S3: [48-59])
3. Enable Hampel filter for S3 (reduces baseline noise spikes)
4. Process all baseline packets (no reset)
5. Continue processing all movement packets (same context)
6. Count TP, TN, FP, FN based on detected state vs expected state
7. Assert: Recall > 90%, FP Rate < target (C6: 10%, S3: 15%)

Tests run with **multiple chip datasets** (C6, S3) using 64 SC (HT20).

---

## Performance Targets

ESPectre is designed for **security and presence detection** applications where:

| Priority | Metric | C6 Target | S3 Target | Rationale |
|----------|--------|-----------|-----------|-----------|
| **High** | Recall | >90% | >90% | Minimize missed detections |
| **High** | FP Rate | <10% | <15% | Avoid false alarms |
| **Medium** | Precision | >90% | >85% | Ensure reported motion is real |
| **Medium** | F1-Score | >90% | >90% | Balance precision and recall |

The current configuration exceeds all targets. S3 has relaxed FP rate target due to higher baseline noise.

---

## Tuning for Your Environment

Real performances may vary based on:
- **Distance from router**: Optimal 3-8 meters
- **Room layout**: Open spaces vs. cluttered rooms
- **Wall materials**: Drywall vs. concrete
- **Interference**: Other Wi-Fi devices, microwave ovens

See [TUNING.md](TUNING.md) for detailed tuning instructions.

---

## P95 Automatic Band Selection with Adaptive Threshold

When using P95 Band Selection for automatic subcarrier optimization, the algorithm:
1. Selects a contiguous 12-subcarrier band that minimizes false positive rate
2. Calculates an adaptive threshold: `P95(baseline_mv) × 1.4`

This ensures **zero false positives** while maintaining high recall across all environments.

### C6 Results

| Metric | Fixed Band [11-22] | P95 Auto-Calibration + adaptive threshold |
|--------|--------------------|-----------------------|
| **Selected Band** | [11-22] | [11-22] |
| **Threshold** | 1.0 (fixed) | 0.8377 (P95 × 1.4) |
| **Recall** | 98.1% | 98.8% |
| **Precision** | 100.0% | 100.0% |
| **FP Rate** | 0.0% | 0.0% |
| **F1-Score** | 99.0% | 99.4% |

**Why use P95 instead of fixed band?**

Fixed bands achieve slightly better performance in the reference test environment, but **subcarrier quality varies significantly between environments** due to:
- Room geometry and materials (walls, furniture, metal objects)
- WiFi interference from neighboring networks
- Distance and orientation relative to the access point
- ESP32 variant and antenna characteristics

**P95 Band Selection automatically finds the optimal band for each specific environment**, making it the recommended choice for production deployments. The algorithm evaluates all candidate bands and selects the one with the best balance of low false positive rate and motion sensitivity.

With **adaptive threshold** (P95 × 1.4), the system achieves **zero false positives** and **>98.8% recall**. The threshold automatically adapts to the baseline noise level of the selected band.

---

## Chip Comparison

Real-time calibration results from the same environment (same room, same router position).

| Chip | Brand | Clone | RSSI | P95 MV | FP Rate | Threshold | Band | Stability |
|------|-------|-------|------|--------|---------|-----------|------|-----------|
| **ESP32-C6** | Waveshare | No | -46 dB | **0.73** | 0.0% | 1.03 | [11-22] | ✅ 1-5% |
| ESP32-S3 | Waveshare | No | -64 dB | 0.80 | 0.0% | 1.12 | [15-26] | ✅ 12-78% |
| ESP32 | Waveshare | No | -49 dB | 0.82 | 0.8% | 1.14 | [19-30] | ⚠️ 40-60% |
| ESP32-C3 super mini| GERUI | No | -73 dB | 0.83 | 0.0% | 1.17 | [37-48] | ⚠️ 38-53% |
| ESP32-S3 | DUBEUYEW | Yes | -80 dB | 1.71 | 65.2% | 2.40 | [37-48] | ❌ 25-106% |


### Key Findings

1. **ESP32-C6 is the optimal chip for CSI sensing** - lowest P95, most stable baseline
2. **Clone chips have inferior RF quality** - S3 clone had 65% FP rate vs 0% for original
3. **Single-core RISC-V chips perform well** - C6 outperforms dual-core Xtensa chips
4. **RSSI correlates with stability** - chips with better signal have more stable baselines

### Recommendation

For new ESPectre deployments:
- **Best choice**: ESP32-C6 (best CSI stability, lowest cost)
- **Good alternative**: ESP32-S3 with Espressif original chip (if PSRAM needed)
- **Avoid**: Clone chips with non-Espressif silicon

---

## Version History

| Date | Version | Dataset | Mode | Threshold | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|------|-----------|--------|-----------|---------|----------|
| 2026-01-21 | v2.4.0 | C6 | NBVI | 0.62 (P95×1.4) | 99.8% | 96.3% | 3.8% | 98.0% |
| 2026-01-21 | v2.4.0 | C6 | P95 | 0.84 (P95×1.4) | 98.8% | 100.0% | 0.0% | 99.4% |
| 2026-01-21 | v2.4.0 | S3 | P95 + Hampel | 1.16 (P95×1.4) | 99.1% | 87.5% | 14.3% | 92.9% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | 1.0 (fixed) | 96.4% | 100.0% | 0.0% | 98.2% |
| 2025-12-27 | v2.3.0 | C6 | Fixed | 1.0 (fixed) | 98.1% | 100.0% | 0.0% | 99.0% |

**Notes**:
- v2.4.0 uses adaptive threshold (P95 × 1.4), v2.3.0 used fixed threshold (1.0)
- Lower threshold = higher recall but more false positives

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
