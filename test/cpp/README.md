# Test Suite

Host-side **CMake + CTest** suite for validating the ESPectre `core / runtime / frontend` layers.

## Quick Start

```bash
# Activate virtualenv (from repo root)
source .venv/bin/activate

# Configure and run the full host-side suite
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build --output-on-failure

# Run specific suite
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
```

---

## Test Suites

The registered targets are grouped by the layer they exercise:

- Core: `test_utils`, `test_core_helpers`, `test_hampel_filter`,
  `test_classic_detector`, and `test_ml_detector`
- Runtime: `test_traffic_generator`, `test_runtime_helpers`,
  `test_runtime_frontend_controller`, `test_runtime_detector_switch`,
  `test_wifi_lifecycle`, `test_pending_event`,
  `test_wifi_provisioning_service`, `test_device_config_store`,
  `test_espectre_protocol`, `test_csi_pipeline`, `test_csi_frame_identity`,
  `test_csi_traffic_service`, and `test_udp_listener`
- Integration with real CSI: `test_motion_detection`,
  `test_long_recordings`, `test_low_rssi`, `test_empty_rooms`, and
  `test_packet_rate_adaptation`
- Frontend: `test_sensor_publisher`, `test_frontend_controls`,
  `test_native_frontend`, and `test_matter_frontend`

`test/cpp/suites/CMakeLists.txt` is the executable registration source of truth;
this list is the human-readable catalog.


### Target Metrics (Motion Detection)
- **Recall**: >95% for all chips (detect motion in motion datasets)
- **FP Rate**: <5% for all chips (avoid false alarms)

See [docs/performance](../../docs/performance/README.md) for detailed targets per chip and algorithm.

### Performance Report Parity Gate
- `tools/generate_performance_report.py` now depends on the host-side C++ integration suites staying aligned with the published Python replay metrics.
- The report command builds `test/cpp/build` when needed, runs `test_motion_detection` and `test_long_recordings`, and compares their structured aggregate outputs against the Python report data before writing `docs/performance/README.md`.
- If the paired or long-recording aggregates drift, the report generation fails and prints the mismatched chip/algorithm/metric entries instead of publishing stale documentation.

---

## Real CSI Data

Tests load real CSI data from NPZ files in `data/` using the [cnpy](https://github.com/rogersce/cnpy) library.

### Datasets

| Chip | Static Presence | Motion |
|------|-----------------|--------|
| ESP32-C3 | `static_presence_c3_64sc_*.npz` | `motion_c3_64sc_*.npz` |
| ESP32-C5 | `static_presence_c5_64sc_*.npz` | `motion_c5_64sc_*.npz` |
| ESP32-C6 | `static_presence_c6_64sc_*.npz` | `motion_c6_64sc_*.npz` |
| ESP32-S3 | `static_presence_s3_64sc_*.npz` | `motion_s3_64sc_*.npz` |
| ESP32 | `static_presence_esp32_64sc_*.npz` | `motion_esp32_64sc_*.npz` |

Tests run with **multiple chip datasets** (C3, C5, C6, S3, and ESP32) using
64 SC (HT20 mode).

Both Python and C++ tests use the same NPZ files, eliminating duplication.

---

## Code Coverage

Run the host-side suite with coverage instrumentation:

```bash
./run_coverage.sh
```

The coverage script prints both the aggregate report and the per-layer breakdown used during development (`core`, `runtime`, `frontend`).

Recent local snapshot (2026-05-30):

- Total line coverage: `87.09%`
- `core`: `92.25%`
- `runtime`: `80.92%`
- `frontend`: `97.69%`

---

## Project Structure

```
test/
├── cmake/              # Shared CMake modules for the host-side suite
├── mocks/              # ESP-IDF / ESPHome host-side fakes
├── suites/             # Test suites grouped by layer
│   ├── core/
│   ├── runtime/
│   ├── integration/
│   └── frontend/
├── support/            # Harness and shared test-side support (cnpy, dataset loader, runtime shim)
├── CMakeLists.txt      # Host-side test entrypoint
└── run_coverage.sh     # Coverage script
```

Production code under test lives outside `test/`:

- `../src/cpp/core/` for reusable detection logic
- `../src/cpp/runtime/` for the shared runtime contract and `../src/cpp/runtime/esp_idf/` for the current runtime orchestration
- `../src/cpp/frontend/esphome/espectre/` for the ESPHome component manifest and adapter layer
- `../src/cpp/frontend/matter/espectre/` for the Matter adapter and surface mapping

---

## Adding New Tests

Create `test/suites/core/test_my_feature.cpp`:

```cpp
#include "test_harness.h"

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

Register the file in `test/suites/CMakeLists.txt` and run it with `ctest -R test_my_feature`.
