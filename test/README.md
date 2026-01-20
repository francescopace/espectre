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
```

---

## Test Suites

| Suite | Type | Data | Focus |
|-------|------|------|-------|
| `test_csi_processor` | Unit | **Real** | API, getters, state machine, adaptive threshold, low-pass filter |
| `test_hampel_filter` | Unit | **Real** | Outlier removal filter |
| `test_calibration` | Unit | **Real** | P95 band selection, magnitude, turbulence, adaptive threshold, fallback |
| `test_calibration_manager` | Integration | **Real** | CalibrationManager API, file I/O, P95 band ranking |
| `test_csi_manager` | Integration | **Real** | CSIManager API, callbacks, motion detection |
| `test_calibration_file_storage` | Unit | Synthetic | File-based magnitude storage |
| `test_traffic_generator` | Unit | Synthetic | Error handling, rate limiting, adaptive backoff |
| `test_motion_detection` | Integration | **Real** | MVS performance, P95 calibration end-to-end |


### Target Metrics (Motion Detection)
- **Recall**: ≥95% (detect real movements)
- **FP Rate**: <1% (avoid false alarms)

---

## Real CSI Data

Tests load real CSI data from NPZ files in `micro-espectre/data/` using the [cnpy](https://github.com/rogersce/cnpy) library.

### Datasets

| Chip | Baseline | Movement | Packets |
|------|----------|----------|---------|
| ESP32-C6 | `baseline_c6_64sc_*.npz` | `movement_c6_64sc_*.npz` | 1000 each |
| ESP32-S3 | `baseline_s3_64sc_*.npz` | `movement_s3_64sc_*.npz` | 1000 each |

Tests run with **multiple chip datasets** (C6, S3) using 64 SC (HT20 mode).

Both Python and C++ tests use the same NPZ files, eliminating duplication.

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

## Smoke Tests (QEMU)

Smoke tests run automatically in CI using the composite action `.github/actions/qemu-smoke-test/`.

To run locally with `act`:

```bash
# Install act (https://github.com/nektos/act)
brew install act  # macOS

# Run a specific smoke test
act -j build --matrix chip:"QEMU ESP32-C3" -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

### What it detects

- Kernel panics
- Guru Meditation errors
- Assertion failures
- Stack smashing

### Supported chips

| Chip | Architecture | QEMU Machine |
|------|--------------|--------------|
| ESP32-S3 | Xtensa | esp32s3 |
| ESP32-C3 | RISC-V | esp32c3 |
| ESP32-C6 | RISC-V | esp32c6 |

> Note: ESP32 original is excluded - QEMU doesn't emulate PHY registers correctly, causing false crashes.

> **Note**: Smoke tests appear as "Smoke Test ESP32-C3" etc. in CI.

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
