# ESPectre Test Suite

Automated test system for ESPectre focused on **Home Assistant integration performance** for security and presence detection.

## Overview

The test suite has been reorganized to focus on **real-world performance metrics**:
- **90% Recall** (detect 9 out of 10 movements)
- **1-5 False Positives per hour** (balanced false alarm rate)
- **Stable state transitions** (reliable detection)

**Total Tests**: 17 (reduced from 38 - 55% reduction)
- **Performance Tests (CORE)**: 4
- **Calibration & Optimization**: 3 (real data only)
- **Segmentation**: 6
- **Component**: 4

---

## Quick Start

### Build and Run Tests

```bash
cd test_app
idf.py build flash monitor
```

### Capture Test Output for Analysis

```bash
cd test_app
idf.py monitor | tee test_output.log
```

Press `Ctrl+]` to exit monitor when tests complete.

### Analyze Results with Python

```bash
cd test_app
python analyze_test_results.py test_output.log
open test_results/report.html
```

---

## Test Suite Structure

### 📊 Performance Tests (CORE) - 4 tests

Primary tests for validating Home Assistant integration:

#### 1. `test_performance_suite.c`
- Evaluates all 10 features individually
- Calculates: Accuracy, Precision, **Recall**, F1-Score, FP/FN rates
- Ranks features by recall (priority for security)
- Generates confusion matrix
- **Output**: JSON with best feature and metrics

#### 2. `test_threshold_optimization.c`
- Generates 50-point ROC curve
- Calculates AUC (Area Under Curve)
- Finds optimal threshold for 90% recall target
- Trade-off analysis (80%, 85%, 90%, 95%, 99% recall)
- **Output**: JSON with ROC data and optimal threshold

#### 3. `test_temporal_robustness.c`
- Scenario 1: Prolonged baseline (empty room stability)
- Scenario 2: Baseline → Movement (intrusion detection)
- Scenario 3: Movement → Baseline (person leaves)
- Scenario 4: Continuous movement
- Measures detection latency and persistence
- **Output**: JSON with scenario results

#### 4. `test_home_assistant_integration.c`
- Simulates: Empty room → Person enters → Moves → Leaves
- Includes debouncing (3 consecutive detections)
- Includes persistence (15 packets ~1 second)
- State machine validation
- **Output**: JSON with production readiness assessment

### 🎯 Calibration & Optimization - 3 tests

- `test_features_differ_between_baseline_and_movement` - Feature separation validation
- `test_calibration_with_real_csi_data` - Complete calibration flow (1000+1000 packets)
- `test_pca_subcarrier_analysis_on_real_data` - PCA subcarrier optimization

### 📈 Segmentation - 6 tests

Temporal event segmentation for activity recognition:
- `test_segmentation_init`
- `test_spatial_turbulence_calculation`
- `test_segmentation_calibration`
- `test_segmentation_no_false_positives`
- `test_segmentation_movement_detection`
- `test_segmentation_reset`

### 🔧 Component - 4 tests

Filter/wavelet integration tests:
- `test_filter_pipeline_with_wavelet_integration`
- `test_filter_pipeline_with_wavelet_disabled`
- `test_wavelet_denoising_reduces_noise`
- `test_wavelet_streaming_mode`

---

## Interpreting Results

### Performance Suite

**Key Metrics:**
- **Recall**: Percentage of movements detected (target: ≥90%)
- **FP Rate**: False positive rate (target: ≤10%)
- **F1-Score**: Harmonic mean of precision and recall

**Example Output:**
```
Rank  Feature                   Recall    FN Rate   FP Rate   F1-Score
────────────────────────────────────────────────────────────────────────
✅  1  temporal_delta_mean      63.13%    36.87%    32.53%    61.46%
✅  2  spatial_gradient         63.18%    36.82%    21.72%    56.63%
⚠️   3  entropy                  59.70%    40.30%    67.98%    52.44%
```

### Home Assistant Integration

**Production Readiness Criteria:**
- ✅ Recall ≥ 90%
- ✅ FP rate: 1-5 per hour
- ✅ State transitions: 2-6 (stable)

**Status:**
- 🎉 **READY FOR PRODUCTION**: All criteria met
- ⚠️ **NEEDS TUNING**: Some criteria not met

---

## Python Analysis Tools

The `analyze_test_results.py` script generates:
- `test_results/roc_curve.png` - ROC curve
- `test_results/precision_recall_curve.png` - Precision-Recall curve
- `test_results/confusion_matrix.png` - Confusion matrix heatmap
- `test_results/temporal_scenarios.png` - Scenario comparison
- `test_results/home_assistant_summary.png` - Integration assessment
- `test_results/report.html` - **Comprehensive HTML report**

---

## Troubleshooting

### Low Recall (<90%)

**Solutions:**
1. Run `test_threshold_optimization` to find optimal threshold
2. Check `test_performance_suite` for best features
3. Consider combining multiple features
4. Adjust debouncing (reduce from 3 to 2)

### High False Positive Rate (>10%)

**Solutions:**
1. Increase threshold slightly
2. Enable Hampel filter (outlier removal)
3. Enable Adaptive Normalizer (drift compensation)
4. Increase debouncing (from 3 to 5)

### Unstable State Transitions (>6)

**Solutions:**
1. Increase debouncing count
2. Increase persistence timeout
3. Add hysteresis (dual thresholds)

---

## Best Practices

1. **Always Run Performance Tests First** - Start with `test_performance_suite`
2. **Optimize Threshold** - Use `test_threshold_optimization`
3. **Validate Temporal Behavior** - Run `test_temporal_robustness`
4. **Final Validation** - Run `test_home_assistant_integration` before deployment
5. **Analyze with Python** - Generate visualizations for better insights

---

## Example Workflow

```bash
# 1. Build and run tests
cd test_app
idf.py build flash monitor | tee test_output.log

# 2. Analyze results
cd ../test
python analyze_test_results.py test_output.log

# 3. View report
open test_results/report.html  # macOS
# or xdg-open test_results/report.html  # Linux
# or start test_results/report.html  # Windows

# 4. Adjust configuration based on recommendations
# Edit main/config_manager.c or use MQTT commands

# 5. Re-run tests to validate improvements
```

---

## Real CSI Data

The `real_csi_data.h` file contains real CSI packets:
- 1000 baseline packets (empty room)
- 1000 movement packets (person walking)

These are used for realistic calibration and detection testing.

---

## Adding New Tests

### 1. Write the Test

Add your test in the appropriate file:

```c
TEST_CASE_ESP(my_new_test_name, "[tag]")
{
    // Test code
    TEST_ASSERT_EQUAL(expected, actual);
}
```

**Important:** Use a **symbolic name** (lowercase with underscores)!

### 2. Update test_app_main.c

Add to `test_app_main.c`:
```c
extern test_desc_t test_desc_my_new_test_name;
// ...
unity_testcase_register(&test_desc_my_new_test_name);
```

### 3. Update CMakeLists.txt

Add your test file to `main/CMakeLists.txt` if it's a new file.

### 4. Build and Test

```bash
idf.py build flash monitor
```

---

## Technical Notes

### Why TEST_CASE_ESP with Symbolic Names?

The standard `TEST_CASE` macro uses `__attribute__((constructor))` which doesn't work correctly on ESP32-S3.

Solution: **symbolic names**:
```c
TEST_CASE_ESP(test_name, "[tag]")  // → test_desc_test_name
```

### Preventing Infinite Reboot Loops

The final `while(1)` loop in `app_main()` prevents infinite reboot loops.

---

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/francescopace/espectre/issues
- Email: francesco.pace@gmail.com
