/*
 * ESPectre - Runtime Helper Unit Tests
 *
 * Covers lightweight runtime helpers that are easy to exercise host-side.
 */

#include "test_harness.h"

#include "gain_controller.h"
#include "wifi_csi_interface.h"

using namespace esphome::espectre;

namespace {

void dummy_csi_callback(void *, wifi_csi_info_t *) {}

}  // namespace

void test_gain_controller_unsupported_platform_initializes_as_locked(void) {
    GainController controller;
    controller.init();

    TEST_ASSERT_TRUE(controller.is_locked());
    TEST_ASSERT_TRUE(controller.needs_cv_normalization());
    TEST_ASSERT_TRUE(controller.was_skipped_due_to_strong_signal() == false);
    TEST_ASSERT_TRUE(controller.get_mode() == GainLockMode::AUTO);
    TEST_ASSERT_FALSE(GainController::is_supported());
}

void test_gain_controller_disabled_mode_reports_cv_normalization(void) {
    GainController controller;
    controller.init(GainLockMode::DISABLED);

    TEST_ASSERT_TRUE(controller.is_locked());
    TEST_ASSERT_TRUE(controller.needs_cv_normalization());
    TEST_ASSERT_TRUE(controller.get_mode() == GainLockMode::DISABLED);
    TEST_ASSERT_EQUAL(0, controller.get_packet_count());
    TEST_ASSERT_EQUAL(0, controller.get_agc_gain());
    TEST_ASSERT_EQUAL(0, controller.get_fft_gain());
}

void test_gain_controller_callback_fires_immediately_when_skipped(void) {
    GainController controller;
    controller.init(GainLockMode::AUTO);

    bool callback_called = false;
    controller.set_lock_complete_callback([&callback_called]() { callback_called = true; });

    TEST_ASSERT_TRUE(callback_called);
}

void test_gain_controller_process_packet_keeps_state_on_unsupported_platform(void) {
    GainController controller;
    controller.init(GainLockMode::ENABLED);

    wifi_csi_info_t info{};
    controller.process_packet(&info);

    TEST_ASSERT_TRUE(controller.is_locked());
    TEST_ASSERT_TRUE(controller.needs_cv_normalization());
    TEST_ASSERT_TRUE(controller.get_mode() == GainLockMode::ENABLED);
    TEST_ASSERT_EQUAL(GainController::CALIBRATION_PACKETS, GainController::get_calibration_packets());
    TEST_ASSERT_EQUAL(64, GainController::get_subcarrier_count());
}

void test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi(void) {
    WiFiCSIReal wifi;
    wifi_csi_config_t config{};

    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_config(&config));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_rx_cb(dummy_csi_callback, nullptr));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(true));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(false));
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_gain_controller_unsupported_platform_initializes_as_locked);
    RUN_TEST(test_gain_controller_disabled_mode_reports_cv_normalization);
    RUN_TEST(test_gain_controller_callback_fires_immediately_when_skipped);
    RUN_TEST(test_gain_controller_process_packet_keeps_state_on_unsupported_platform);
    RUN_TEST(test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
