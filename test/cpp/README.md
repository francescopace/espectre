# Test suite

These CMake and CTest tests run the ESPectre `core`, `runtime`, and `frontend` code on your computer, without a board.

You need the OpenSSL development library (`libssl-dev` on Debian/Ubuntu, `openssl@3` with Homebrew). The device-identity tests use it for real SHA-256 behind both the mbedTLS and PSA interfaces, with known test vectors and error cases; they do not run ESP-IDF's hardware crypto drivers. Run them with `./test/cpp/run_all_tests.sh -R test_device_identity`.

## Quick start

```bash
# Activate virtualenv (from repo root)
source .venv/bin/activate

# Configure and run the full host-side suite
./test/cpp/run_all_tests.sh

# Run specific suite
./test/cpp/run_all_tests.sh -R test_motion_detection

# Override automatic logical-CPU detection when needed
CTEST_PARALLEL_LEVEL=2 ./test/cpp/run_all_tests.sh
```

## Test suites

The registered targets are grouped by the layer they exercise:

- Core: `test_utils`, `test_core_helpers`, `test_hampel_filter`, `test_lightweight_detector`, and `test_high_accuracy_detector`
- Runtime: `test_traffic_generator`, `test_traffic_generator_s2`, `test_traffic_generator_ampdu`, `test_traffic_generator_auto`, `test_traffic_generator_s2_auto`, `test_traffic_generator_c5_auto`, `test_traffic_generator_c5_fixed`, `test_runtime_helpers`, `test_frontend_sysinfo_helpers`, `test_periodic_sensing_status_logger`, `test_device_identity`, `test_device_identity_psa`, `test_ota_service_https`, `test_runtime_frontend_controller`, `test_sdk_surface`, `test_runtime_detector_switch`, `test_wifi_lifecycle`, `test_wifi_lifecycle_s2`, `test_wifi_lifecycle_auto`, `test_wifi_lifecycle_ampdu`, `test_wifi_lifecycle_6m`, `test_wifi_lifecycle_dual_band`, `test_pending_event`, `test_wifi_provisioning_service`, `test_device_config_store`, `test_espectre_protocol`, `test_csi_pipeline`, `test_csi_frame_identity`, `test_csi_traffic_service`, and `test_udp_listener`
- Integration with real CSI: `test_motion_detection`, `test_long_recordings`, `test_low_rssi`, `test_empty_rooms`, and `test_packet_rate_adaptation`
- Frontend: `test_sensor_publisher`, `test_frontend_controls`, `test_native_frontend_lifecycle`, `test_native_direct_frontend`, `test_native_mqtt_frontend`, `test_home_assistant_mqtt_frontend`, `test_native_frontend_ota`, `test_recovery_button_service`, and `test_matter_frontend`

The authoritative list is `test/cpp/suites/CMakeLists.txt`; the list above is for reading.

- Normal runs and the performance-report parity gate build `RelWithDebInfo`, so replay-heavy tests run fast but keep debug info and assertions. Coverage builds use `Debug` for accurate line mapping.
- The launchers use all CPU cores. Set `CTEST_PARALLEL_LEVEL` to use fewer.
- `test_packet_rate_adaptation` replays the first 60 seconds at 120, 100, and 80 pps. These are test rates, not a minimum supported rate.

### Dataset test matrix

Each of the `normal`, `reserved`, `weak`, `empty`, `long`, and `packet_rate` checks runs for every chip (ESP32, C3, S2, S3, C5, C6), named `test_<suite>.<gate>.<chip>`. Filter them with the labels `performance`, `chip:<chip>`, and `dataset:<gate>`.

A case is skipped (exit code 77) only when there is no dataset for that chip and check. A broken catalog, pair, or file, or a failed replay, is a failure. Run the executables without arguments for a summary of all chips.

### Source ownership

Each source file has one main test suite, listed in `coverage_ownership.json`; a Python test checks that no file is missing, duplicated, or outdated. Integration and parity suites use that code without repeating its constants.

| Production source | Primary test owner | Separate integration or parity gate |
|---|---|---|
| `src/cpp/core/utils.*`, feature helpers, CSI format helpers, and `temporal_csi_sampler.*` | `test_utils`, `test_core_helpers` | `test_motion_detection` only for replay metrics |
| `src/cpp/core/hampel_filter.*` | `test_hampel_filter` | Detector replay suites run it without duplicating filter expectations |
| `src/cpp/core/lightweight_detector.*` | `test_lightweight_detector` | `test_motion_detection`, `test_long_recordings`, `test_low_rssi`, and `test_empty_rooms` |
| `src/cpp/core/high_accuracy_detector.*` and generated weights | `test_high_accuracy_detector` | `test_motion_detection`, `test_long_recordings`, and `test_empty_rooms` |
| Shared runtime contracts, policies, configuration, CSI pipeline, and protocol | The matching `test_runtime_*`, `test_device_config_store`, `test_espectre_protocol`, `test_csi_*`, or service suite | `test_packet_rate_adaptation` for quantified cadence behavior |
| Published SDK facade | `test_sdk_surface` | Python `test_sdk_surface_invariants.py` checks facade and documentation registration |
| ESPHome, Native, and Matter adapters | The matching frontend suite; shared controls stay in `test_frontend_controls` | Full firmware builds validate SDK-specific integration |

Add new regression tests to the owning suite. Create a new suite only for a new subsystem, or an integration with its own setup and failure modes.

### Target metrics (motion detection)

- **Recall** above 95% on every chip (motion is detected).
- **False-positive rate** below 5% on every chip (no false alarms).

See the [performance report](../../docs/performance/README.md) for results per chip and detector.

### Performance report parity gate

`tools/generate_performance_report.py` builds `test/cpp/build`, runs `test_motion_detection` and `test_long_recordings`, and compares their `selection + holdout` results with the Python numbers before writing the report. If they differ, it fails and lists the mismatched chip, detector, and metric. (The suites also replay `train` recordings, but those are not part of the comparison.)

These two suites only check that replay counts and output are correct. The numeric targets are enforced by `test_validation_real_data.py::TestPerformanceMetrics` and by the report.

## Real CSI data

Tests load real CSI data from NPZ files in `data/` using the [cnpy](https://github.com/rogersce/cnpy) library.

### Datasets

| Chip | Static Presence | Motion |
|------|-----------------|--------|
| ESP32-C3 | `static_presence_c3_64sc_*.npz` | `motion_c3_64sc_*.npz` |
| ESP32-C5 | `static_presence_c5_64sc_*.npz` | `motion_c5_64sc_*.npz` |
| ESP32-C6 | `static_presence_c6_64sc_*.npz` | `motion_c6_64sc_*.npz` |
| ESP32-S2 | Pending | Pending |
| ESP32-S3 | `static_presence_s3_64sc_*.npz` | `motion_s3_64sc_*.npz` |
| ESP32 | `static_presence_esp32_64sc_*.npz` | `motion_esp32_64sc_*.npz` |

Python and C++ read the same HT20 files for all six chips. ESP32-S2 has no dataset yet, so its cases are reported as skipped.

## Code coverage

Run the host-side suite with coverage instrumentation:

```bash
./test/cpp/run_coverage.sh
```

It uses your compiler (Apple Clang and `llvm-cov` on macOS) and prints line, function, and branch coverage overall and for `core`, `runtime`, and `frontend`.

To get the same numbers as CI (GCC 13 on Linux/amd64, in Docker):

```bash
./test/cpp/run_gcc13_coverage.sh --ci
```

CI requires at least 85% line, 85% function, and 50% branch coverage (`coverage-thresholds.json`). These are fixed targets, not taken from the last run. The Docker image is built locally and cached, and always uses `linux/amd64`, so Apple Silicon matches CI.

## Project structure

```
test/cpp/
├── cmake/              # Shared CMake modules for the host-side suite
├── mocks/              # ESP-IDF / ESPHome host-side fakes
├── suites/             # Test suites grouped by layer
│   ├── core/
│   ├── runtime/
│   ├── integration/
│   └── frontend/
├── support/            # Harness, datasets, runtime shims, and in-memory traffic fakes
├── CMakeLists.txt      # Host-side test entrypoint
├── coverage-thresholds.json  # Fixed canonical runtime coverage gates
├── gcc13-coverage.Dockerfile  # Canonical Linux/GCC coverage environment
├── run_all_tests.sh    # Parallel build and test launcher
├── run_gcc13_coverage.sh      # Docker-backed canonical GCC 13 coverage runner
└── run_coverage.sh     # Coverage script
```

Production code under test lives outside `test/cpp/`:

- `src/cpp/core/` for reusable detection logic
- `src/cpp/runtime/` for the shared runtime contract and `src/cpp/runtime/esp_idf/` for the current runtime orchestration
- `src/cpp/frontend/esphome/components/espectre/` for the ESPHome component manifest and adapter layer
- `src/cpp/frontend/matter/espectre/` for the Matter adapter and surface mapping

Traffic and UDP-listener tests use the in-memory fakes in `support/csi_traffic_fakes.h`. Tests must never open real UDP sockets; the real network code is checked by the firmware builds.

## Adding new tests

Create `test/cpp/suites/core/test_my_feature.cpp`:

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

Register the file in `test/cpp/suites/CMakeLists.txt` and run it with `ctest --test-dir test/cpp/build -R test_my_feature`.
