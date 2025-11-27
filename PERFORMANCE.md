# 📊 ESPectre Performance Metrics

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
- **Platform**: ESP32-C6
- **Distance from router**: ~3 meters
- **Environment**: Indoor residential

---

## Confusion Matrix

```
CONFUSION MATRIX (1000 baseline + 1000 movement packets):
                    Predicted
                IDLE        MOTION
Actual IDLE     1000 (TN)   0 (FP)
Actual MOTION   19 (FN)     981 (TP)
```

### Metrics Breakdown

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

## How to Reproduce

### Using ESP32 Test Suite (C)
```bash
cd test_app
idf.py build flash monitor
# Run: performance_suite_comprehensive test
```

### Using Python Tools
```bash
cd micro-espectre/tools
python 2_analyze_system_tuning.py --confusion-matrix
```

### Requirements
- Baseline data: `micro-espectre/tools/baseline_data.bin`
- Movement data: `micro-espectre/tools/movement_data.bin`

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

See [CALIBRATION.md](CALIBRATION.md) for detailed tuning instructions.

---

## Version History

| Date | Version | Recall | FP Rate | Notes |
|------|---------|--------|---------|-------|
| 2025-11-28 | v1.4.0 | 98.1% | 0.0% | Current release |
