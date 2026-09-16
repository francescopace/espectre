/*
 * ESPectre - Native Frontend Lifecycle Tests
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend_test_support.h"
#include "frontend_loop_timer.h"

void test_frontend_loop_timer_records_complete_bodies_and_excludes_idle_time(void) {
  esp_timer_mock::reset(0, 0);
  float latest_ms = 0.0f;
  const auto run_loop = [&](int64_t duration_us, bool early_return) {
    FrontendLoopTimer loop_timer(latest_ms);
    esp_timer_mock::advance(duration_us);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, latest_ms);
    if (early_return) return;
    esp_timer_mock::advance(500);
  };
  run_loop(6250, true);
  TEST_ASSERT_EQUAL_FLOAT(6.25f, latest_ms);
  esp_timer_mock::advance(2000000);
  TEST_ASSERT_EQUAL_FLOAT(6.25f, latest_ms);
  latest_ms = 0.0f;
  run_loop(1250, false);
  TEST_ASSERT_EQUAL_FLOAT(1.75f, latest_ms);
  esp_timer_mock::reset();
}

void test_native_frontend_setup_registers_runtime_listener(void) {
  frontend_runtime_shim::state.snapshot.threshold = 3.25f;

  NativeFrontend frontend;
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.is_setup_complete());
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_listener);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener != &frontend);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_live_telemetry_enabled_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, frontend.snapshot().threshold);
}

void test_native_frontend_setup_fails_when_runtime_setup_fails(void) {
  frontend_runtime_shim::state.setup_result = false;
  NativeFrontend frontend;
  TEST_ASSERT_FALSE(frontend.setup());
}

void test_native_frontend_loop_and_shutdown_forward_to_runtime(void) {
  {
    NativeFrontend frontend;
    TEST_ASSERT_TRUE(frontend.setup());
    frontend.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
}

void test_native_frontend_defers_wifi_reconfigure_resume_until_after_runtime_loop(
    void) {
  NativeFrontend frontend;
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  const int initial_set_services_armed_calls =
      frontend_runtime_shim::state.set_services_armed_calls;

  frontend.prepare_for_wifi_reconfigure();
  TEST_ASSERT_TRUE(frontend.wifi_reconfigure_quiesced_);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  frontend.resume_after_wifi_reconfigure();
  TEST_ASSERT_TRUE(frontend.wifi_reconfigure_resume_pending_);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  frontend.loop();
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  TEST_ASSERT_FALSE(frontend.wifi_reconfigure_resume_pending_);
  TEST_ASSERT_FALSE(frontend.wifi_reconfigure_quiesced_);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(initial_set_services_armed_calls + 2,
                    frontend_runtime_shim::state.set_services_armed_calls);
}

void test_native_frontend_allows_sensing_when_mqtt_is_missing(void) {
  NativeFrontend frontend;
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}


int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_frontend_loop_timer_records_complete_bodies_and_excludes_idle_time);
  RUN_TEST(test_native_frontend_setup_registers_runtime_listener);
  RUN_TEST(test_native_frontend_setup_fails_when_runtime_setup_fails);
  RUN_TEST(test_native_frontend_loop_and_shutdown_forward_to_runtime);
  RUN_TEST(test_native_frontend_defers_wifi_reconfigure_resume_until_after_runtime_loop);
  RUN_TEST(test_native_frontend_allows_sensing_when_mqtt_is_missing);
  return UNITY_END();
}
