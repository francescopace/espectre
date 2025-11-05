# ESPectre Unit Tests

Automated test system for ESPectre that runs 19 tests at device boot.

## Structure

```
test_app/
├── main/
│   ├── test_app_main.c      # Entry point - registers and runs all tests
│   ├── test_case_esp.h      # TEST_CASE_ESP macro (custom for ESP-IDF)
│   ├── test_filters.c       # 5 tests for filters
│   ├── test_features.c      # 14 tests for CSI feature extraction
│   └── mock_csi_data.h      # Synthetic data generator
├── CMakeLists.txt
├── sdkconfig.defaults       # Configuration (watchdog disabled)
└── README.md                # This file
```

## How to Run Tests

### 1. Build and Flash

```bash
cd test_app
source ../esp-idf/export.sh
idf.py build flash monitor
```

### 2. Expected Output

#### All tests pass:
```
╔═════════════════════════════════════╗
║       🛜  E S P e c t r e 👻         ║
║              Unit Tests             ║
╚═════════════════════════════════════╝

Running all tests automatically...

Running Adaptive normalizer respects enabled flag...
./main/test_filters.c:16:Adaptive normalizer respects enabled flag:PASS
Running Butterworth filter initialization...
./main/test_filters.c:54:Butterworth filter initialization:PASS
...
(all tests are executed)
...

-----------------------
19 Tests 0 Failures 0 Ignored
OK

✅ All tests passed!
Press Ctrl+] to exit monitor
```

#### If tests fail:
```
...
Running Mock CSI data generation...
./main/test_features.c:181:Mock CSI data generation:FAIL: Expected 0 to be greater than 0

-----------------------
19 Tests 1 Failures 0 Ignored
FAIL

❌ Tests failed: 1 failure(s)
Press Ctrl+] to exit monitor
```

## Synthetic Data

The `mock_csi_data.h` file provides 4 types of synthetic CSI data:

- **MOCK_CSI_STATIC**: Static environment with low noise
- **MOCK_CSI_WALKING**: Walking movement (~2Hz, moderate amplitude)
- **MOCK_CSI_RUNNING**: Running movement (~4Hz, high amplitude)
- **MOCK_CSI_OUTLIERS**: Data with outliers to test Hampel filter

### Random Seed Initialization

The mock data generator uses a fixed random seed (12345) to ensure reproducible test results. This guarantees that tests produce consistent results across different runs.

## Adding New Tests

1. Add the test in `test_filters.c` or `test_features.c`:

```c
TEST_CASE_ESP("Test name", "[tag]")
{
    // Test code
    TEST_ASSERT_EQUAL(expected, actual);
}
```

2. Find the generated line number:
```bash
grep -n "TEST_CASE_ESP.*Test name" test_app/main/test_*.c
```

3. Add declaration and registration in `test_app_main.c`:
```c
extern test_desc_t test_desc_XXX;  // Test name

// In app_main():
unity_testcase_register(&test_desc_XXX);
```

## Technical Notes

### Why TEST_CASE_ESP instead of TEST_CASE?

The standard `TEST_CASE` macro from ESP-IDF uses `__attribute__((constructor))` to automatically register tests, but this doesn't work correctly in the ESP32-S3 toolchain.

`TEST_CASE_ESP` solves the problem by exporting test descriptors as global symbols that are then manually registered in `app_main()`.

### Preventing Infinite Reboot Loops

The final `while(1)` loop in `app_main()` prevents infinite reboot loops in case of crashes during tests. The system remains waiting after execution, allowing you to see all results.

### Test Result Reporting

The test runner checks Unity's `TestFailures` counter after running all tests and displays:
- ✅ "All tests passed!" when all tests succeed
- ❌ "Tests failed: X failure(s)" when tests fail

This provides immediate feedback on test status without needing to parse the detailed output.

## References

- [ESP-IDF Unit Testing Guide](https://docs.espressif.com/projects/esp-idf/en/release-v6.0/esp32s3/api-guides/unit-tests.html)
- [Unity Test Framework](http://www.throwtheswitch.org/unity)
