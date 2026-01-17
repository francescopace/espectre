# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection system based on Moving Variance Segmentation (MVS).

## Test Methodology

### Test Data
- **Baseline packets**: 1000 CSI packets captured during idle state (no movement)
- **Movement packets**: 1000 CSI packets captured during active movement
- **Total**: 2000 packets
- **Packet Rate**: 100 packets/second

### Configuration
| Parameter | 64 SC (HT20) |
|-----------|--------------|
| Window Size | 50 packets |
| Threshold | 1.0 |
| Subcarriers | [11-22] |

### Test Environment
- **Platform**: ESP32-C6 (results expected to be similar on other ESP32 variants)
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

### 64 SC (HT20) - Fixed Band [11-22]

```
CONFUSION MATRIX (1000 baseline + 1000 movement packets):
                    Predicted
                IDLE        MOTION
Actual IDLE     1000 (TN)   0 (FP)
Actual MOTION   19 (FN)     981 (TP)
```

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Recall** | 98.1% | >90% | ✅ |
| **Precision** | 100.0% | - | ✅ |
| **FP Rate** | 0.0% | <10% | ✅ |
| **F1-Score** | 99.0% | - | ✅ |

---

## Interpretation

### Strengths
- **Zero false positives**: The system never triggers false alarms during idle periods
- **High recall (98.1%)**: Detects 98.1% of all movement events
- **Perfect precision**: When motion is reported, it's always real motion

### False Negatives Analysis
The 19 false negatives (1.9% of movement packets) are typically caused by:
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

| Dataset | Baseline | Movement |
|---------|----------|----------|
| 64 SC | `baseline_c6_64sc_20251212_142443.npz` | `movement_c6_64sc_20251212_142443.npz` |
| 256 SC | `baseline_c6_256sc_20260110_182357.npz` | `movement_c6_256sc_20260110_182443.npz` |

Files are located in `micro-espectre/data/baseline/` and `micro-espectre/data/movement/`.

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

Both tests use identical methodology:
1. Initialize MVS with `window_size=50`, `threshold=1.0`
2. Select subcarriers based on dataset (64 SC: [11-22], 256 SC: [147-158])
3. Process all baseline packets (no reset)
4. Continue processing all movement packets (same context)
5. Count TP, TN, FP, FN based on detected state vs expected state
6. Assert: Recall > 95%, FP Rate < 10%

Tests run automatically with **both 64 SC and 256 SC datasets**.

---

## Performance Targets

ESPectre is designed for **security and presence detection** applications where:

| Priority | Metric | Target | Rationale |
|----------|--------|--------|-----------|
| **High** | Recall | >90% | Minimize missed detections |
| **High** | FP Rate | <10% | Avoid false alarms |
| **Medium** | Precision | >90% | Ensure reported motion is real |
| **Medium** | F1-Score | >90% | Balance precision and recall |

The current configuration exceeds all targets.

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

### 64 SC (HT20)

| Metric | Fixed Band [11-22] | P95 Auto-Calibration |
|--------|--------------------|-----------------------|
| **Selected Band** | [11-22] | [11-22] |
| **Recall** | 98.8% | 98.8% |
| **Precision** | 100.0% | 100.0% |
| **FP Rate** | 0.0% | 0.0% |
| **F1-Score** | 99.4% | 99.4% |

### 256 SC (HE20)

| Metric | Fixed Band [147-158] | P95 Auto-Calibration |
|--------|----------------------|-----------------------|
| **Selected Band** | [147-158] | [147-158] |
| **Recall** | 98.7% | 98.7% |
| **Precision** | 100.0% | 100.0% |
| **FP Rate** | 0.0% | 0.0% |
| **F1-Score** | 99.3% | 99.3% |

> **Note**: With adaptive threshold (P95 × 1.4), both 64 SC and 256 SC achieve **zero false positives** and **>98% recall**. The threshold automatically adapts to the baseline noise level of the selected band.

**Why use P95 instead of fixed band?**

Fixed bands achieve slightly better performance in the reference test environment, but **subcarrier quality varies significantly between environments** due to:
- Room geometry and materials (walls, furniture, metal objects)
- WiFi interference from neighboring networks
- Distance and orientation relative to the access point
- ESP32 variant and antenna characteristics

**P95 Band Selection automatically finds the optimal band for each specific environment**, making it the recommended choice for production deployments. The algorithm evaluates all candidate bands and selects the one with the best balance of low false positive rate and motion sensitivity.

---

## Version History

| Date | Version | Dataset | Mode | Recall | Precision | FP Rate | F1-Score | Notes |
|------|---------|---------|------|--------|-----------|---------|----------|-------|
| 2026-01-13 | v2.4.0 | 64 SC | P95+AT | 98.8% | 100.0% | 0.0% | 99.4% | Adaptive Threshold (P95×1.4) |
| 2026-01-11 | v2.4.0 | 64 SC | P95 | 98.1% | 100.0% | 0.0% | 99.0% | P95 Band Selection |
| 2026-01-11 | v2.4.0 | 64 SC | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | P95 Band Selection |
| 2025-12-27 | v2.3.0 | 64 SC | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | Multi-window validation |
| 2025-12-27 | v2.3.0 | 64 SC | NBVI | 96.4% | 100.0% | 0.0% | 98.2% | Multi-window validation |
| 2025-12-13 | v2.2.0 | 64 SC | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | ESPHome Port |
| 2025-12-13 | v2.2.0 | 64 SC | NBVI | 96.5% | 100.0% | 0.0% | 98.2% | ESPHome Port |
| 2025-11-28 | v1.4.0 | 64 SC | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | Initial MVS implementation |

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
