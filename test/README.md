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
| `test_mvs_detector` | Unit | **Real** | MVS algorithm, threshold, filters, state machine, lowpass |
| `test_pca_detector` | Unit | **Real** | PCA algorithm, jitter/wander metrics |
| `test_csi_manager` | Unit | Synthetic | CSIManager API, enable/disable, callbacks |
| `test_utils` | Unit | **Real** | Variance, magnitude, turbulence, compare functions |
| `test_hampel_filter` | Unit | **Real** | Outlier removal filter |
| `test_p95_calibrator` | Unit | **Real** | P95 band selection, magnitude, turbulence, adaptive threshold |
| `test_nbvi_calibrator` | Unit | **Real** | NBVI subcarrier selection, configuration |
| `test_pca_calibrator` | Unit | **Real** | PCA calibration, correlation threshold calculation |
| `test_calibration_file_storage` | Unit | Synthetic | File-based magnitude storage |
| `test_traffic_generator` | Unit | Synthetic | Error handling, rate limiting, adaptive backoff |
| `test_serial_streamer` | Unit | Synthetic | Serial streaming API |
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
