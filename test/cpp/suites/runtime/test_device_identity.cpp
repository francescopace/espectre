/*
 * ESPectre - Device Identity Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "device_identity.h"
#include "esp_mac.h"
#include "mbedtls/sha256.h"

using namespace espectre;

void setUp(void) {}
void tearDown(void) {}

void test_device_identity_uses_station_mac_sha256_pseudonym(void) {
  const uint64_t expected = 0x3cf79180d3a0aca4ULL;

  TEST_ASSERT_EQUAL(expected, derive_runtime_device_id());
  TEST_ASSERT_EQUAL_STRING("3cf79180d3a0aca4", derive_runtime_device_id_string().c_str());
  TEST_ASSERT_EQUAL(1, g_esp_mac_mock.call_count);
  TEST_ASSERT_EQUAL(1, g_mbedtls_sha256_mock.call_count);
  TEST_ASSERT_EQUAL(sizeof("espectre-device-id-v1") - 1U + 6U,
                    g_mbedtls_sha256_mock.input_len);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_device_identity_uses_station_mac_sha256_pseudonym);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return process();
}
#endif
