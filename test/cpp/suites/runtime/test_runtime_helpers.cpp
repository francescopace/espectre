/*
 * ESPectre - Runtime Helper Unit Tests
 *
 * Covers lightweight runtime helpers that are easy to exercise host-side.
 */

#include "test_harness.h"

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
    TEST_ASSERT_EQUAL_STRING("ml", detection_algorithm_name(DetectionAlgorithm::ML));
    TEST_ASSERT_EQUAL_STRING("classic", detection_algorithm_name(DetectionAlgorithm::CLASSIC));
    TEST_ASSERT_EQUAL_STRING("fixed", subcarrier_source_name(RuntimeSubcarrierSource::FIXED_DEFAULT));
    TEST_ASSERT_EQUAL_STRING("Auto (adaptive)", threshold_mode_display_name(ThresholdMode::AUTO));
    TEST_ASSERT_TRUE(parse_threshold_mode("min") == ThresholdMode::MIN);
    TEST_ASSERT_TRUE(parse_threshold_mode("auto") == ThresholdMode::AUTO);
    TEST_ASSERT_TRUE(parse_traffic_mode("ping") == RuntimeTrafficMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_mode("dns") == RuntimeTrafficMode::DNS);
    TEST_ASSERT_TRUE(parse_detection_algorithm("ml") == DetectionAlgorithm::ML);
    TEST_ASSERT_TRUE(parse_detection_algorithm("classic") == DetectionAlgorithm::CLASSIC);
}

void test_runtime_diagnostics_emit_expected_key_value_pairs(void) {
    RuntimeConfig config;
    RuntimeSnapshot snapshot;
    config.threshold_mode = ThresholdMode::MANUAL;
    config.lowpass_enabled = true;
    snapshot.threshold = 2.5f;
    snapshot.detector_name = "classic";
    snapshot.startup_threshold = 0.125f;

    std::vector<std::string> lines;
    visit_runtime_diagnostics(config, snapshot, [&lines](const char *key, const char *value) {
        lines.emplace_back(std::string(key) + "=" + value);
    });

    TEST_ASSERT_TRUE(!lines.empty());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "threshold=2.500000 (manual)") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "detector=classic") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "lowpass=on") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "startup_threshold=0.125000") != lines.end());
}

int process(void) {
    UNITY_BEGIN();
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
