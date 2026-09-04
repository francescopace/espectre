/*
 * ESPectre - Invalid Runtime Sensing Kconfig Tests
 *
 * Verifies that malformed generated configuration cannot escape into setup.
 *
 * Author: Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "runtime_config_utils.h"
#include "runtime_sensing_kconfig.h"
#include "test_harness.h"

using namespace espectre;

void test_invalid_kconfig_values_fall_back_to_valid_defaults(void) {
  const RuntimeConfig config = make_runtime_sensing_config_from_kconfig();

  TEST_ASSERT_EQUAL(RUNTIME_MOTION_ON_HITS_DEFAULT, config.motion_on_hits);
  TEST_ASSERT_EQUAL(RUNTIME_MOTION_OFF_HITS_DEFAULT, config.motion_off_hits);
  TEST_ASSERT_EQUAL(RUNTIME_HAMPEL_WINDOW_DEFAULT, config.hampel_window);
  TEST_ASSERT_EQUAL_STRING(RUNTIME_CSI_TRAFFIC_MULTICAST_GROUP_DEFAULT,
                           config.csi_traffic_multicast_group.c_str());
  TEST_ASSERT_EQUAL(static_cast<int>(RuntimeConfigError::NONE),
                    static_cast<int>(validate_runtime_config(config)));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_invalid_kconfig_values_fall_back_to_valid_defaults);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
