# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection system based on Moving Variance Segmentation (MVS).

## Test Methodology

### Test Data
- **Baseline packets**: 1000 CSI packets captured during idle state (no movement)
- **Movement packets**: 1000 CSI packets captured during active movement
- **Total**: 2000 packets
- **Packet Rate**: 100 packets/second

### Configuration
| Parameter | Value |
|-----------|-------|
| Window Size | 50 packets |
| Threshold | 1.0 |
| Subcarriers | [11-22] (12 subcarriers) |

### Test Environment
- **Platform**: ESP32-C6 (results expected to be similar on other ESP32 variants)
- **Distance from router**: ~3 meters
- **Environment**: Indoor residential

---

## Results

Both platforms produce **identical results** using the same test methodology:
- Process all 1000 baseline packets first (expecting IDLE)
- Then process all 1000 movement packets (expecting MOTION)
- Continuous context (no reset between baseline and movement)
- Same parameters: window_size=50, threshold=1.0, subcarriers=[11-22]
- Filters disabled (lowpass, hampel off by default), normalization always enabled

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

### Detailed Counts
| Metric | Count | Description |
|--------|-------|-------------|
| True Positives (TP) | 981 | Movement correctly detected |
| True Negatives (TN) | 1000 | Idle correctly identified |
| False Positives (FP) | 0 | No false alarms |
| False Negatives (FN) | 19 | Missed movement detections |

> **Note**: These tests were performed with optional filters disabled (lowpass, hampel). Normalization is always enabled for cross-device consistency. See [TUNING.md](TUNING.md) for filter configuration options.

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

Both platforms use the **same real CSI data** captured from ESP32-C6:

| Platform | Baseline Data | Movement Data |
|----------|---------------|---------------|
| **C++** | `test/data/real_csi_data_esp32.h` | `test/data/real_csi_arrays.inc` |
| **Python** | `micro-espectre/data/baseline/baseline_c6_001.npz` | `micro-espectre/data/movement/movement_c6_001.npz` |

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
1. Initialize MVS with `window_size=50`, `threshold=1.0`, `subcarriers=[11-22]`
2. Process all 1000 baseline packets (no reset)
3. Continue processing all 1000 movement packets (same context)
4. Count TP, TN, FP, FN based on detected state vs expected state
5. Assert: Recall > 95%, FP Rate < 1%

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

## NBVI Automatic Calibration

When using NBVI (Normalized Baseline Variability Index) for automatic subcarrier selection instead of the fixed band [11-22], performance is slightly lower but still excellent:

| Metric | Fixed Band [11-22] | NBVI Auto-Calibration |
|--------|--------------------|-----------------------|
| **Recall** | 98.1% | 96.4% |
| **Precision** | 100.0% | 100.0% |
| **FP Rate** | 0.0% | 0.0% |
| **F1-Score** | 99.0% | 98.2% |

**Why use NBVI instead of fixed band?**

The fixed band [11-22] achieves slightly better performance in the reference test environment, but **subcarrier quality varies significantly between environments** due to:
- Room geometry and materials (walls, furniture, metal objects)
- WiFi interference from neighboring networks
- Distance and orientation relative to the access point
- ESP32 variant and antenna characteristics

**NBVI automatically selects the optimal subcarriers for each specific environment**, making it the recommended choice for production deployments. The fixed band is useful only for controlled test environments where optimal subcarriers have been manually identified.

---

## Version History

| Date | Version | Mode | Recall | Precision | FP Rate | F1-Score | Notes |
|------|---------|------|--------|-----------|---------|----------|-------|
| 2025-12-27 | v2.3.0 | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | Multi-window validation |
| 2025-12-27 | v2.3.0 | NBVI | 96.4% | 100.0% | 0.0% | 98.2% | Multi-window validation |
| 2025-12-13 | v2.2.0 | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | ESPHome Port |
| 2025-12-13 | v2.2.0 | NBVI | 96.5% | 100.0% | 0.0% | 98.2% | ESPHome Port |
| 2025-11-28 | v1.4.0 | Fixed | 98.1% | 100.0% | 0.0% | 99.0% | Initial MVS implementation |

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
