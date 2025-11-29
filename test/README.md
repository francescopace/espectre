# 🛜 ESPectre 👻 - Test Suite

Test suite based on **PlatformIO Unity** to validate ESPectre CSI algorithms.

## 🚀 Quick Start

```bash
# Activate virtualenv
source ../venv/bin/activate

# Run all tests locally
cd test && pio test -e native

# Run specific suite
pio test -e native -f test_motion_detection

# Run on ESP32 device
pio test -e esp32dev
```

---

## 🧪 Test Suites

| Suite | Type | Data | Tests | Focus |
|-------|------|------|-------|-------|
| `test_csi_processor` | Unit | Real | 19 | API, validation, memory |
| `test_hampel_filter` | Unit | Synthetic | 16 | Outlier removal filter |
| `test_calibration` | Unit | Synthetic | 21 | NBVI calibration algorithm |
| `test_calibration_file_storage` | Unit | Synthetic | 9 | File-based magnitude storage |
| `test_motion_detection` | Integration | **Real** | 3 | MVS performance |

**Total: 68 test cases**

### Target Metrics (Motion Detection)
- **Recall**: ≥95% (detect real movements)
- **FP Rate**: <1% (avoid false alarms)

---

## 📦 Real CSI Data

The `data/` folder contains **2000 real CSI packets**:
- 1000 baseline (empty room)
- 1000 movement (person walking)

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

#ifdef ARDUINO
void setup() { delay(2000); process(); }
void loop() {}
#else
int main(int argc, char **argv) { return process(); }
#endif
```

> **Note**: PlatformIO requires each suite in a separate folder.

---

## 📄 License

GPLv3 - See [LICENSE](../LICENSE) for details.
