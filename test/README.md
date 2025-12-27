# Test Suite

Test suite based on **PlatformIO Unity** to validate ESPectre CSI algorithms.

## Quick Start

```bash
# Activate virtualenv
source ../venv/bin/activate

# Run all tests (native is the default environment)
cd test && pio test

# Run specific suite
pio test -f test_motion_detection

# Run on ESP32-C6 device
pio test -e esp32c6
```

---

## Test Suites

| Suite | Type | Data | Focus |
|-------|------|------|-------|
| `test_csi_processor` | Unit | **Real** | API, getters, state machine, normalization, low-pass filter |
| `test_hampel_filter` | Unit | **Real** | Outlier removal filter |
| `test_calibration` | Unit | **Real** | NBVI, magnitude, turbulence, normalization scale, fallback |
| `test_calibration_manager` | Integration | **Real** | CalibrationManager API, file I/O, NBVI ranking |
| `test_csi_manager` | Integration | **Real** | CSIManager API, callbacks, motion detection |
| `test_calibration_file_storage` | Unit | Synthetic | File-based magnitude storage |
| `test_traffic_generator` | Unit | Synthetic | Error handling, rate limiting, adaptive backoff |
| `test_motion_detection` | Integration | **Real** | MVS performance, NBVI end-to-end |


### Target Metrics (Motion Detection)
- **Recall**: ≥95% (detect real movements)
- **FP Rate**: <1% (avoid false alarms)

---

## Real CSI Data

The `data/` folder contains **2000 real CSI packets**:
- 1000 baseline (empty room)
- 1000 movement (person walking)

---

## Code Coverage

Run tests with coverage instrumentation:

```bash
./run_coverage.sh
```

### Current Coverage

| File | Lines | Functions | Branches |
|------|-------|-----------|----------|
| `csi_manager.cpp` | 92% | 100% | 85% |
| `csi_processor.cpp` | 89% | 100% | 81% |
| `calibration_manager.cpp` | 74% | 100% | 65% |
| `utils.h` | 92% | 100% | 69% |
| `gain_controller.cpp` | 75% | 50% | - |
| **Total** | **76%** | **75%** | **70%** |

> **Note**: Coverage measured on Codecov (CI). Tests use real CSI data from ESP32-C6 captures.

---

## Project Structure

```
test/
├── mocks/              # Mock implementations
│   ├── esp_idf/        # ESP-IDF mocks (native only)
│   └── esphome/        # ESPHome mocks (native only)
├── data/               # Real CSI test data
├── test/               # Test suites (one folder per suite)
├── platformio.ini      # PlatformIO configuration
└── run_coverage.sh     # Coverage script
```

---

## Adding New Tests

Create `test/test_my_feature/test_my_feature.cpp`:

```cpp
#include <unity.h>

void setUp(void) {}
void tearDown(void) {}

void test_example(void) {
    TEST_ASSERT_EQUAL(1, 1);
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_example);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
```

> **Note**: PlatformIO requires each suite in a separate folder.

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.
