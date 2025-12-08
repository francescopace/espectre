# 🛜 ESPectre 👻 - Test Suite

Test suite based on **PlatformIO Unity** to validate ESPectre CSI algorithms.

## 🚀 Quick Start

```bash
# Activate virtualenv
source ../venv/bin/activate

# Run all tests locally (native)
cd test && pio test -e native

# Run specific suite
pio test -e native -f test_motion_detection

# Run on ESP32-C6 device
pio test -e esp32c6
```

---

## 🧪 Test Suites

| Suite | Type | Data | Tests | Focus |
|-------|------|------|-------|-------|
| `test_csi_processor` | Unit | **Real** | 39 | API, getters, state machine |
| `test_hampel_filter` | Unit | **Real** | 20 | Outlier removal filter |
| `test_calibration` | Unit | **Real** | 33 | NBVI, magnitude, turbulence, compare, end-to-end CalibrationManager |
| `test_calibration_manager` | Integration | **Real** | 24 | CalibrationManager API, file I/O, NBVI ranking, edge cases |
| `test_csi_manager` | Integration | **Real** | 29 | CSIManager API, callbacks, motion detection, error paths |
| `test_calibration_file_storage` | Unit | Synthetic | 9 | File-based magnitude storage |
| `test_motion_detection` | Integration | **Real** | 3 | MVS performance metrics |

### Test Counts

| Environment | Tests | Notes |
|-------------|-------|-------|
| **Native** | 157 | Full suite with WiFiCSIMock |
| **ESP32-C6** | 157 | Full suite with WiFiCSIMock (dependency injection) |

### Target Metrics (Motion Detection)
- **Recall**: ≥95% (detect real movements)
- **FP Rate**: <1% (avoid false alarms)

---

## 📦 Real CSI Data

The `data/` folder contains **2000 real CSI packets**:
- 1000 baseline (empty room)
- 1000 movement (person walking)

---

## 📊 Code Coverage

Run tests with coverage instrumentation:

```bash
./run_coverage.sh
```

### Current Coverage

| File | Lines | Functions | Branches |
|------|-------|-----------|----------|
| `csi_manager.cpp` | **100%** | 100% | 94% |
| `utils.h` | 96% | 100% | 69% |
| `csi_processor.cpp` | 91% | 100% | 84% |
| `calibration_manager.cpp` | 86% | 100% | 63% |
| **Total** | **90%** | **96%** | **63%** |

> **Note**: Coverage measured on Codecov (CI). Tests use real CSI data from ESP32-C6 captures.

---

## 📁 Project Structure

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

## ➕ Adding New Tests

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

## 📄 License

GPLv3 - See [LICENSE](../LICENSE) for details.
