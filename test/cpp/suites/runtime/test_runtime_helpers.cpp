/*
 * ESPectre - Runtime Helper Unit Tests
 *
 * Covers lightweight runtime helpers that are easy to exercise host-side.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "csi_capture_service.h"
#include "csi_format_classifier.h"
#include "csi_format.h"
#include "csi_platform_config.h"
#include "runtime_config_utils.h"
#include "mqtt_payload_assembler.h"
#include "runtime_diagnostics.h"
#include "runtime_performance_diagnostics.h"
#include "runtime_time.h"
#include "wifi_csi_interface.h"

#include <algorithm>
#include <string>
#include <vector>

#include "esp_timer.h"

#define private public
#undef private

using namespace espectre;

namespace {

void dummy_csi_callback(void *, wifi_csi_info_t *) {}

struct CapturedCsiPacket {
  uint32_t callback_count{0U};
  int8_t first_value{0};
  uint16_t info_len{0U};
  size_t normalized_len{0U};
  bool first_word_invalid{true};
};

struct CapturedChannelChange {
  uint32_t callback_count{0U};
  uint8_t previous_channel{0U};
  uint8_t current_channel{0U};
};

class CaptureWiFiMock final : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t *config) override {
    (void)config;
    configure_calls++;
    return ESP_OK;
  }

  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx) override {
    callback = cb;
    callback_context = ctx;
    callback_registration_calls++;
    return ESP_OK;
  }

  esp_err_t set_csi(bool enable) override {
    enabled = enable;
    if (enable) {
      enable_calls++;
    } else {
      disable_calls++;
    }
    return ESP_OK;
  }

  int configure_calls{0};
  int callback_registration_calls{0};
  int enable_calls{0};
  int disable_calls{0};
  bool enabled{false};
  wifi_csi_cb_t callback{nullptr};
  void *callback_context{nullptr};
};

void capture_csi_packet(void *context, const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) {
  auto *captured = static_cast<CapturedCsiPacket *>(context);
  captured->callback_count++;
  captured->first_value = normalized.valid() ? normalized.data[0] : 0;
  captured->info_len = info != nullptr ? info->len : 0U;
  captured->normalized_len = normalized.len;
  captured->first_word_invalid = info == nullptr || info->first_word_invalid;
}

void capture_channel_change(void *context, uint8_t previous_channel, uint8_t current_channel) {
  auto *captured = static_cast<CapturedChannelChange *>(context);
  captured->callback_count++;
  captured->previous_channel = previous_channel;
  captured->current_channel = current_channel;
}

}  // namespace

void test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi(void) {
    WiFiCSIReal wifi;
    wifi_csi_config_t config{};

    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_config(&config));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_rx_cb(dummy_csi_callback, nullptr));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(true));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(false));
}

void test_original_esp32_csi_config_captures_ht_ltf_only(void) {
    const wifi_csi_config_t config = build_ht20_csi_config();

    TEST_ASSERT_FALSE(config.lltf_en);
    TEST_ASSERT_TRUE(config.htltf_en);
    TEST_ASSERT_FALSE(config.stbc_htltf2_en);
}

void test_csi_capture_service_filters_duplicate_and_stale_timestamps(void) {
    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init();
    service.set_packet_callback(&capture_csi_packet, &captured);

    std::array<int8_t, HT20_CSI_LEN> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.len = HT20_CSI_LEN;
    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 0U;

    const uint32_t timestamps[] = {100U, 101U, 101U, 50U, 102U};
    for (uint32_t timestamp : timestamps) {
        info.rx_ctrl.timestamp = timestamp;
        service.process_packet(&info);
    }

    TEST_ASSERT_EQUAL(3U, captured.callback_count);
    TEST_ASSERT_EQUAL(3U, service.valid_packets());
    TEST_ASSERT_EQUAL(2U, service.filtered_packets());
    TEST_ASSERT_EQUAL(2U, service.rejected_out_of_order_packets());

    service.reset_session();
    info.rx_ctrl.timestamp = 50U;
    service.process_packet(&info);

    TEST_ASSERT_EQUAL(4U, captured.callback_count);
    TEST_ASSERT_EQUAL(1U, service.valid_packets());
    TEST_ASSERT_EQUAL(0U, service.filtered_packets());
    TEST_ASSERT_EQUAL(0U, service.rejected_out_of_order_packets());
}

void test_csi_capture_service_defers_channel_change_and_resets_session_baseline(void) {
    CaptureWiFiMock wifi;
    CsiCaptureService service;
    CapturedCsiPacket packets;
    CapturedChannelChange channel_change;
    service.init(&wifi);
    service.set_packet_callback(&capture_csi_packet, &packets);
    service.set_channel_change_callback(&capture_channel_change, &channel_change);

    std::array<int8_t, HT20_CSI_LEN> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.len = HT20_CSI_LEN;
    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 0U;
    info.rx_ctrl.channel = 8U;
    info.rx_ctrl.timestamp = 100U;

    TEST_ASSERT_EQUAL(ESP_OK, service.enable());
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(1U, packets.callback_count);

    info.rx_ctrl.channel = 10U;
    info.rx_ctrl.timestamp = 101U;
    service.process_packet(&info);
    info.rx_ctrl.timestamp = 102U;
    service.process_packet(&info);

    TEST_ASSERT_EQUAL(1U, packets.callback_count);
    TEST_ASSERT_EQUAL(0U, channel_change.callback_count);
    service.loop();
    TEST_ASSERT_EQUAL(1U, channel_change.callback_count);
    TEST_ASSERT_EQUAL(8U, channel_change.previous_channel);
    TEST_ASSERT_EQUAL(10U, channel_change.current_channel);

    TEST_ASSERT_EQUAL(ESP_OK, service.disable());
    TEST_ASSERT_EQUAL(ESP_OK, service.enable());
    info.rx_ctrl.channel = 11U;
    info.rx_ctrl.timestamp = 1U;
    service.process_packet(&info);
    service.loop();

    TEST_ASSERT_EQUAL(2U, packets.callback_count);
    TEST_ASSERT_EQUAL(1U, channel_change.callback_count);
}

void test_csi_format_classifier_rejects_ht40_before_normalization(void) {
    std::array<int8_t, HT20_CSI_LEN_DOUBLE> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.len = HT20_CSI_LEN_DOUBLE;
    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 1U;

    const CsiFormatAssessment assessment = assess_ht20_sensing_format(&info);

    TEST_ASSERT_FALSE(assessment.is_sensing_accepted());
    TEST_ASSERT_TRUE(assessment.reason_code == CsiFormatReasonCode::UNSUPPORTED_WIDTH);
    TEST_ASSERT_TRUE(assessment.normalization_tag == NormalizedCSIPayloadTag::NONE);
}

void test_csi_capture_service_tracks_format_drop_reasons(void) {
    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init();
    service.set_packet_callback(&capture_csi_packet, &captured);

    std::array<int8_t, HT20_CSI_LEN> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.len = HT20_CSI_LEN;

    info.rx_ctrl.sig_mode = 0U;
    info.rx_ctrl.cwb = 0U;
    service.process_packet(&info);

    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 1U;
    service.process_packet(&info);

    info.len = 64U;
    info.rx_ctrl.cwb = 0U;
    service.process_packet(&info);

    TEST_ASSERT_EQUAL(0U, captured.callback_count);
    TEST_ASSERT_EQUAL(1U, service.unsupported_phy_packets());
    TEST_ASSERT_EQUAL(1U, service.unsupported_width_packets());
    TEST_ASSERT_EQUAL(1U, service.unknown_layout_packets());
    TEST_ASSERT_EQUAL(3U, service.filtered_packets());
}

void test_runtime_config_utils_validate_and_name_values(void) {
    TEST_ASSERT_TRUE(validate_runtime_threshold(0.0f));
    TEST_ASSERT_TRUE(validate_runtime_threshold(1.0f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(-0.1f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(1.1f));
    TEST_ASSERT_EQUAL_STRING("ping", traffic_mode_name(RuntimeTrafficMode::PING));
    TEST_ASSERT_EQUAL_STRING("dns", traffic_mode_name(RuntimeTrafficMode::DNS));
    TEST_ASSERT_EQUAL_STRING("internal", csi_traffic_mode_name(CsiTrafficMode::INTERNAL));
    TEST_ASSERT_EQUAL_STRING("external", csi_traffic_mode_name(CsiTrafficMode::EXTERNAL));
    TEST_ASSERT_EQUAL_STRING("high_accuracy", detection_algorithm_name(DetectionAlgorithm::HIGH_ACCURACY));
    TEST_ASSERT_EQUAL_STRING("lightweight", detection_algorithm_name(DetectionAlgorithm::LIGHTWEIGHT));
    TEST_ASSERT_EQUAL_STRING("fixed", subcarrier_source_name(RuntimeSubcarrierSource::FIXED_DEFAULT));
    TEST_ASSERT_TRUE(parse_traffic_mode("ping") == RuntimeTrafficMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_mode("dns") == RuntimeTrafficMode::DNS);
    TEST_ASSERT_TRUE(parse_csi_traffic_mode("internal") == CsiTrafficMode::INTERNAL);
    TEST_ASSERT_TRUE(parse_csi_traffic_mode("external") == CsiTrafficMode::EXTERNAL);
    TEST_ASSERT_TRUE(parse_csi_traffic_mode("pacing") == CsiTrafficMode::INTERNAL);
    TEST_ASSERT_TRUE(parse_csi_traffic_mode("disabled") == CsiTrafficMode::INTERNAL);
    TEST_ASSERT_TRUE(parse_csi_traffic_mode("unsupported") == CsiTrafficMode::INTERNAL);
    TEST_ASSERT_TRUE(csi_traffic_mode_is_sensing_control(CsiTrafficMode::INTERNAL));
    TEST_ASSERT_TRUE(csi_traffic_mode_is_sensing_control(CsiTrafficMode::EXTERNAL));
    TEST_ASSERT_TRUE(csi_traffic_mode_is_sensing_control(CsiTrafficMode::EXTERNAL));
    TEST_ASSERT_TRUE(normalize_sensing_csi_traffic_mode(CsiTrafficMode::INTERNAL) == CsiTrafficMode::INTERNAL);
    TEST_ASSERT_TRUE(parse_detection_algorithm("high_accuracy") == DetectionAlgorithm::HIGH_ACCURACY);
    TEST_ASSERT_TRUE(parse_detection_algorithm("lightweight") == DetectionAlgorithm::LIGHTWEIGHT);
    TEST_ASSERT_EQUAL_STRING("2g", wifi_band_policy_name(WifiBandPolicy::BAND_2G));
    TEST_ASSERT_EQUAL_STRING("5g", wifi_band_policy_name(WifiBandPolicy::BAND_5G));
    TEST_ASSERT_EQUAL_STRING("auto", wifi_band_policy_name(WifiBandPolicy::AUTO));
    TEST_ASSERT_TRUE(parse_wifi_band_policy("2g") == WifiBandPolicy::BAND_2G);
    TEST_ASSERT_TRUE(parse_wifi_band_policy("5g") == WifiBandPolicy::BAND_5G);
    TEST_ASSERT_TRUE(parse_wifi_band_policy("auto") == WifiBandPolicy::AUTO);
    TEST_ASSERT_TRUE(parse_wifi_band_policy("unsupported") == WifiBandPolicy::BAND_2G);
}

void test_runtime_config_validator_covers_the_public_schema(void) {
    RuntimeConfig config;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::NONE);

    config.runtime_profile = static_cast<RuntimeProfile>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::RUNTIME_PROFILE);
    config = RuntimeConfig{};
    config.wifi_band_policy = static_cast<WifiBandPolicy>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::WIFI_BAND_POLICY);
    config = RuntimeConfig{};
    config.detection_algorithm = static_cast<DetectionAlgorithm>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::DETECTION_ALGORITHM);
    config = RuntimeConfig{};
    config.segmentation_threshold = 2.0f;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::SEGMENTATION_THRESHOLD);
    config = RuntimeConfig{};
    config.segmentation_window_size_ms = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::SEGMENTATION_WINDOW_SIZE_MS);
    config = RuntimeConfig{};
    config.csi_target_pps = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TARGET_PPS);
    config = RuntimeConfig{};
    config.traffic_generator_mode = static_cast<RuntimeTrafficMode>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::TRAFFIC_GENERATOR_MODE);
    config = RuntimeConfig{};
    config.csi_traffic_mode = static_cast<CsiTrafficMode>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TRAFFIC_MODE);
    config = RuntimeConfig{};
    config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
    config.csi_traffic_udp_port = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TRAFFIC_UDP_PORT);
    config = RuntimeConfig{};
    config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
    config.csi_traffic_multicast_group = "192.168.1.2";
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TRAFFIC_MULTICAST_GROUP);
    config = RuntimeConfig{};
    config.publish_interval_ms = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::PUBLISH_INTERVAL_MS);
    config = RuntimeConfig{};
    config.evaluation_interval_ms = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::EVALUATION_INTERVAL_MS);
    config = RuntimeConfig{};
    config.motion_on_hits = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::MOTION_HITS);
    config = RuntimeConfig{};
    config.lowpass_enabled = true;
    config.lowpass_cutoff = 1.0f;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::LOWPASS_CUTOFF);
    config = RuntimeConfig{};
    config.hampel_enabled = true;
    config.hampel_window = 2U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::HAMPEL_WINDOW);
    config = RuntimeConfig{};
    config.hampel_threshold = 0.0f;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::HAMPEL_THRESHOLD);
    config = RuntimeConfig{};
    config.lowpass_enabled = false;
    config.lowpass_cutoff = 1.0f;
    config.hampel_enabled = false;
    config.hampel_window = 0U;
    config.hampel_threshold = 0.0f;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::NONE);
    TEST_ASSERT_EQUAL_STRING("invalid Hampel window",
                             runtime_config_error_message(RuntimeConfigError::HAMPEL_WINDOW));
}

void test_runtime_diagnostics_emit_expected_key_value_pairs(void) {
    RuntimeConfig config;
    RuntimeSnapshot snapshot;
    config.lowpass_enabled = true;
    snapshot.threshold = 2.5f;
    snapshot.detector_name = "lightweight";
    snapshot.startup_threshold = 0.125f;

    std::vector<std::string> lines;
    visit_runtime_diagnostics(config, snapshot, [&lines](const char *key, const char *value) {
        lines.emplace_back(std::string(key) + "=" + value);
    });

    TEST_ASSERT_TRUE(!lines.empty());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "threshold=2.500000") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "detector=lightweight") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "lowpass=on") != lines.end());
    TEST_ASSERT_TRUE(std::none_of(lines.begin(), lines.end(), [](const std::string &line) {
        return line.rfind("subcarriers=", 0U) == 0U || line.rfind("startup_threshold=", 0U) == 0U;
    }));
}

void test_runtime_diagnostics_sampler_derives_five_second_rates(void) {
    RuntimeDiagnosticsSnapshot baseline;
    baseline.traffic_packets_total = 100U;
    baseline.csi_callbacks_total = 100U;
    baseline.csi_accepted_total = 90U;
    baseline.csi_admitted_total = 80U;
    baseline.csi_filtered_total = 10U;

    RuntimeDiagnosticsSampler sampler;
    sampler.reset(baseline, 1000U);

    RuntimeDiagnosticsSnapshot current = baseline;
    current.traffic_packets_total = 600U;
    current.csi_callbacks_total = 580U;
    current.csi_accepted_total = 540U;
    current.csi_admitted_total = 505U;
    current.csi_filtered_total = 40U;
    current.csi_missing_slots_total = 25U;
    current.csi_excess_total = 15U;
    current.csi_stale_total = 5U;
    current.csi_out_of_order_total = 10U;
    current.csi_occupancy_slots = 82U;
    current.csi_window_slots = 100U;
    current.wifi_channel = 10U;
    current.wifi_rssi_dbm = -55;

    const RuntimeDiagnosticsSample sample = sampler.sample(current, 6000U);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, sample.traffic_tx_pps);
    TEST_ASSERT_EQUAL_FLOAT(96.0f, sample.csi_callback_pps);
    TEST_ASSERT_EQUAL_FLOAT(90.0f, sample.csi_accepted_pps);
    TEST_ASSERT_EQUAL_FLOAT(85.0f, sample.csi_admitted_pps);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, sample.csi_filtered_pps);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, sample.csi_missing_slots_pps);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, sample.csi_excess_pps);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, sample.csi_stale_pps);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, sample.csi_out_of_order_pps);
    TEST_ASSERT_EQUAL_FLOAT(0.82f, sample.csi_occupancy_ratio);
    TEST_ASSERT_EQUAL_UINT8(10U, sample.wifi_channel);
    TEST_ASSERT_EQUAL_INT8(-55, sample.wifi_rssi_dbm);
}

void test_runtime_performance_diagnostics_publish_complete_windows(void) {
    esp_timer_mock::reset(1, 0);
    RuntimePerformanceDiagnostics diagnostics;
    diagnostics.reset();
    diagnostics.update_if_due();
    diagnostics.record_loop_duration(100U);
    diagnostics.record_loop_duration(300U);
    diagnostics.record_detection_timing(900U, 3U, 200U, 400U);

    esp_timer_mock::advance(10000000);
    diagnostics.update_if_due();
    const RuntimePerformanceDiagnosticsSnapshot snapshot = diagnostics.snapshot();

    TEST_ASSERT_TRUE(snapshot.window_ready);
    TEST_ASSERT_EQUAL(10000000U, snapshot.window_duration_us);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.004f, snapshot.runtime_load_percent);
    TEST_ASSERT_EQUAL(2U, snapshot.loop_samples);
    TEST_ASSERT_EQUAL(200U, snapshot.loop_average_us);
    TEST_ASSERT_EQUAL(300U, snapshot.loop_maximum_us);
    TEST_ASSERT_EQUAL(3U, snapshot.detection_samples);
    TEST_ASSERT_EQUAL(900U, snapshot.detection_sum_us);
    TEST_ASSERT_EQUAL(300U, snapshot.detection_average_us);
    TEST_ASSERT_EQUAL(200U, snapshot.detection_minimum_us);
    TEST_ASSERT_EQUAL(400U, snapshot.detection_maximum_us);
}

void test_runtime_performance_diagnostics_json_marks_unready_and_unsupported_values(void) {
    RuntimeDiagnosticsSnapshot diagnostics;
    diagnostics.free_memory_bytes = 4096U;
    diagnostics.minimum_free_memory_bytes = 2048U;
    diagnostics.largest_free_memory_block_bytes = 1024U;
    diagnostics.cpu_frequency_mhz = 160U;

    std::string json{"{\"existing\":1"};
    append_runtime_performance_diagnostics_json(&json, diagnostics);
    json += "}";

    TEST_ASSERT_TRUE(json.find("\"free_memory_kb\":4") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"minimum_free_memory_kb\":2") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"largest_free_memory_kb\":1") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"cpu_frequency_mhz\":160") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"performance_window_ready\":false") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"runtime_load_percent\":null") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"detection_timing_supported\":false") != std::string::npos);
    TEST_ASSERT_TRUE(json.find("\"detection_samples\":null") != std::string::npos);
}

void test_mqtt_payload_assembler_accepts_complete_and_fragmented_payloads(void) {
    MqttPayloadAssembler assembler;

    TEST_ASSERT_TRUE(assembler.append("ping", 4, 4, 0) == MqttPayloadAssembler::Result::COMPLETE);
    TEST_ASSERT_TRUE(assembler.payload() == "ping");
    assembler.reset();

    TEST_ASSERT_TRUE(assembler.append("calib", 5, 9, 0) == MqttPayloadAssembler::Result::INCOMPLETE);
    TEST_ASSERT_TRUE(assembler.append("rate", 4, 9, 5) == MqttPayloadAssembler::Result::COMPLETE);
    TEST_ASSERT_TRUE(assembler.payload() == "calibrate");
}

void test_mqtt_payload_assembler_rejects_invalid_fragments(void) {
    MqttPayloadAssembler assembler;

    TEST_ASSERT_TRUE(assembler.append("abc", 3, 6, 0) == MqttPayloadAssembler::Result::INCOMPLETE);
    TEST_ASSERT_TRUE(assembler.append("def", 3, 6, 2) == MqttPayloadAssembler::Result::INVALID);
    TEST_ASSERT_TRUE(assembler.payload().empty());

    std::string oversized(MqttPayloadAssembler::MAX_PAYLOAD_SIZE + 1U, 'x');
    TEST_ASSERT_TRUE(assembler.append(oversized.data(), oversized.size(), oversized.size(), 0) ==
                     MqttPayloadAssembler::Result::INVALID);
    TEST_ASSERT_TRUE(assembler.payload().empty());
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi);
    RUN_TEST(test_original_esp32_csi_config_captures_ht_ltf_only);
    RUN_TEST(test_csi_capture_service_filters_duplicate_and_stale_timestamps);
    RUN_TEST(test_csi_capture_service_defers_channel_change_and_resets_session_baseline);
    RUN_TEST(test_csi_format_classifier_rejects_ht40_before_normalization);
    RUN_TEST(test_csi_capture_service_tracks_format_drop_reasons);
    RUN_TEST(test_runtime_config_utils_validate_and_name_values);
    RUN_TEST(test_runtime_config_validator_covers_the_public_schema);
    RUN_TEST(test_runtime_diagnostics_emit_expected_key_value_pairs);
    RUN_TEST(test_runtime_diagnostics_sampler_derives_five_second_rates);
    RUN_TEST(test_runtime_performance_diagnostics_publish_complete_windows);
    RUN_TEST(test_runtime_performance_diagnostics_json_marks_unready_and_unsupported_values);
    RUN_TEST(test_mqtt_payload_assembler_accepts_complete_and_fragmented_payloads);
    RUN_TEST(test_mqtt_payload_assembler_rejects_invalid_fragments);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
