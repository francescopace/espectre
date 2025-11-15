# ESPectre Unit Tests

Automated test system for ESPectre that runs tests at device boot.

## Structure

```
test_app/
├── main/
│   ├── test_app_main.c           # Entry point (AUTO-GENERATED)
│   ├── test_case_esp.h           # TEST_CASE_ESP macro with symbolic names
│   ├── ...
│   ├── mock_csi_data.h           # Synthetic data generator
│   └── real_csi_data.h           # Real CSI data samples
└── README.md                     # This file
```

## How to Run Tests

### Manual Build and Flash

```bash
cd test_app
source ../esp-idf/export.sh
idf.py build flash monitor
```
## Expected Output

```
╔═════════════════════════════════════╗
║       🛜  E S P e c t r e 👻         ║
║              Unit Tests             ║
╚═════════════════════════════════════╝

Running all tests automatically...

Running adaptive_normalizer_respects_enabled_flag...
./main/test_filters.c:17:adaptive_normalizer_respects_enabled_flag:PASS
...
(all 30 tests are executed)
...

-----------------------
30 Tests 0 Failures 0 Ignored
OK

✅ All tests passed!
Press Ctrl+] to exit monitor
```

## Adding New Tests

### 1. Write the Test

Add your test in the appropriate file (e.g., `test_filters.c`):

```c
TEST_CASE_ESP(my_new_test_name, "[filters]")
{
    // Test code
    TEST_ASSERT_EQUAL(expected, actual);
}
```

**Important:** Use a **symbolic name** (lowercase with underscores), not a string!

### 2. Update test_app_main.c

Update `test_app_main.c` with:
   - `extern test_desc_t test_desc_name;` declarations
   - `unity_testcase_register(&test_desc_name);` calls

### 3. Build and Test

```bash
idf.py build flash monitor
```

## Technical Notes

### Why TEST_CASE_ESP with Symbolic Names?

The standard `TEST_CASE` macro from ESP-IDF uses `__attribute__((constructor))` to automatically register tests, but this doesn't work correctly on ESP32-S3.

My solution uses **symbolic names**:

```c
TEST_CASE_ESP(test_name, "[tag]")    // → test_desc_test_name (symbolic!)
```

### Preventing Infinite Reboot Loops

The final `while(1)` loop in `app_main()` prevents infinite reboot loops in case of crashes during tests.

## Synthetic Data

The `mock_csi_data.h` file provides 4 types of synthetic CSI data:

- **MOCK_CSI_STATIC**: Static environment with low noise
- **MOCK_CSI_WALKING**: Walking movement (~2Hz, moderate amplitude)
- **MOCK_CSI_RUNNING**: Running movement (~4Hz, high amplitude)
- **MOCK_CSI_OUTLIERS**: Data with outliers to test Hampel filter

The mock data generator uses a fixed random seed (12345) to ensure reproducible test results.

## Real CSI Data

The `real_csi_data.h` file contains 100 real CSI packets:
- 50 baseline packets (empty room)
- 50 movement packets (person walking)

These are used for realistic calibration and detection testing.

## References

- [ESP-IDF Unit Testing Guide](https://docs.espressif.com/projects/esp-idf/en/release-v6.0/esp32s3/api-guides/unit-tests.html)
- [Unity Test Framework](http://www.throwtheswitch.org/unity)
