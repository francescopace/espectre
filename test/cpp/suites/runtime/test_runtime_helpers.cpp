/*
 * ESPectre - Runtime Helper Unit Tests
 *
 * Covers lightweight runtime helpers that are easy to exercise host-side.
 */

#include "test_harness.h"

#include "gain_controller.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "wifi_csi_interface.h"

#include <algorithm>
#include <string>
#include <vector>

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

void test_runtime_config_utils_validate_and_name_modes(void) {
    RuntimeConfig config;
    TEST_ASSERT_TRUE(validate_runtime_threshold(0.0f));
    TEST_ASSERT_TRUE(validate_runtime_threshold(10.0f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(-0.1f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(10.1f));

    set_manual_threshold(config, 4.25f);
    TEST_ASSERT_EQUAL_FLOAT(4.25f, config.segmentation_threshold);
    TEST_ASSERT_TRUE(config.threshold_mode == ThresholdMode::MANUAL);
    TEST_ASSERT_EQUAL_STRING("manual", threshold_mode_name(config.threshold_mode));
    TEST_ASSERT_EQUAL_STRING("Manual", threshold_mode_display_name(config.threshold_mode));
    TEST_ASSERT_EQUAL_STRING("ping", traffic_mode_name(RuntimeTrafficMode::PING));
    TEST_ASSERT_EQUAL_STRING("dns", traffic_mode_name(RuntimeTrafficMode::DNS));
    TEST_ASSERT_EQUAL_STRING("enabled", gain_lock_mode_name(RuntimeGainLockMode::ENABLED));
    TEST_ASSERT_EQUAL_STRING("ml", detection_algorithm_name(DetectionAlgorithm::ML));
    TEST_ASSERT_EQUAL_STRING("fixed", subcarrier_source_name(RuntimeSubcarrierSource::FIXED_DEFAULT));
    TEST_ASSERT_TRUE(parse_threshold_mode("min") == ThresholdMode::MIN);
    TEST_ASSERT_TRUE(parse_threshold_mode("auto") == ThresholdMode::AUTO);
    TEST_ASSERT_TRUE(parse_traffic_mode("ping") == RuntimeTrafficMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_mode("dns") == RuntimeTrafficMode::DNS);
    TEST_ASSERT_TRUE(parse_gain_lock_mode("enabled") == RuntimeGainLockMode::ENABLED);
    TEST_ASSERT_TRUE(parse_gain_lock_mode("disabled") == RuntimeGainLockMode::DISABLED);
    TEST_ASSERT_TRUE(parse_gain_lock_mode("auto") == RuntimeGainLockMode::AUTO);
    TEST_ASSERT_TRUE(parse_detection_algorithm("ml") == DetectionAlgorithm::ML);
    TEST_ASSERT_TRUE(parse_detection_algorithm("mvs") == DetectionAlgorithm::MVS);
}

void test_runtime_diagnostics_emit_expected_key_value_pairs(void) {
    RuntimeConfig config;
    RuntimeSnapshot snapshot;
    config.threshold_mode = ThresholdMode::MANUAL;
    config.lowpass_enabled = true;
    snapshot.threshold = 2.5f;
    snapshot.detector_name = "mvs";
    snapshot.best_pxx = 0.125f;

    std::vector<std::string> lines;
    visit_runtime_diagnostics(config, snapshot, [&lines](const char *key, const char *value) {
        lines.emplace_back(std::string(key) + "=" + value);
    });

    TEST_ASSERT_TRUE(!lines.empty());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "threshold=2.50 (manual)") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "detector=mvs") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "lowpass=on") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "best_pxx=0.1250") != lines.end());
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_gain_controller_unsupported_platform_initializes_as_locked);
    RUN_TEST(test_gain_controller_disabled_mode_reports_cv_normalization);
    RUN_TEST(test_gain_controller_callback_fires_immediately_when_skipped);
    RUN_TEST(test_gain_controller_process_packet_keeps_state_on_unsupported_platform);
    RUN_TEST(test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi);
    RUN_TEST(test_runtime_config_utils_validate_and_name_modes);
    RUN_TEST(test_runtime_diagnostics_emit_expected_key_value_pairs);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
