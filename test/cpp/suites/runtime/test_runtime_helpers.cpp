/*
 * ESPectre - Runtime Helper Unit Tests
 *
 * Covers lightweight runtime helpers that are easy to exercise host-side.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include "csi_capture_service.h"
#include "csi_format.h"
#include "csi_platform_config.h"
#include "runtime_config_utils.h"
#include "mqtt_payload_assembler.h"
#include "runtime_diagnostics.h"
#include "runtime_time.h"
#include "wifi_csi_interface.h"

#include <algorithm>
#include <string>
#include <vector>

#define private public
#include "csi_stream_transport.h"
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

void test_csi_stream_transport_serializes_v7_phy_metadata(void) {
    CsiStreamTransport transport;
    transport.configure(0x1122334455667788ULL, 5001U, 1000U, 1U);
    transport.reset_session();

    std::array<int8_t, HT20_CSI_LEN> csi{};
    csi[0] = 17;
    wifi_csi_info_t csi_info{};
    csi_info.buf = csi.data();
    csi_info.len = HT20_CSI_LEN;
    csi_info.rx_ctrl.channel = 8U;
    const NormalizedCSIPayload normalized{csi.data(), csi.size(), NormalizedCSIPayloadTag::NONE};
    std::array<uint8_t, sizeof(CsiStreamHeaderV7) + HT20_CSI_LEN> record{};

    csi_info.rx_ctrl.sig_mode = 1U;
    csi_info.rx_ctrl.cwb = 0U;
    transport.handle_csi_packet(&csi_info, normalized, true);
    transport.latest_csi_.captured_at_us = monotonic_now_us();
    size_t record_len = transport.build_stream_packet_(record.data(), record.size());
    const auto *header = reinterpret_cast<const CsiStreamHeaderV7 *>(record.data());

    TEST_ASSERT_EQUAL(sizeof(record), record_len);
    TEST_ASSERT_EQUAL(STREAM_VERSION, header->version);
    TEST_ASSERT_EQUAL(sizeof(CsiStreamHeaderV7), header->header_len);
    TEST_ASSERT_EQUAL(static_cast<uint8_t>(StreamPhyMode::HT), header->phy_mode);
    TEST_ASSERT_EQUAL(static_cast<uint8_t>(StreamLtfType::HT_LTF), header->ltf_type);
    TEST_ASSERT_EQUAL(static_cast<uint8_t>(StreamChannelWidth::MHZ_20), header->channel_width);
    TEST_ASSERT_EQUAL_INT8(17, static_cast<int8_t>(record[sizeof(CsiStreamHeaderV7)]));
}

void test_csi_stream_transport_prefers_latest_fresh_sample(void) {
    CsiStreamTransport transport;
    transport.configure(0x1122334455667788ULL, 5001U, 1000U, 1U);
    transport.reset_session();

    std::array<int8_t, HT20_CSI_LEN> csi_a{};
    std::array<int8_t, HT20_CSI_LEN> csi_b{};
    csi_a[0] = 11;
    csi_b[0] = 22;

    wifi_csi_info_t csi_info{};
    csi_info.len = HT20_CSI_LEN;
    const NormalizedCSIPayload normalized_a{csi_a.data(), csi_a.size(), NormalizedCSIPayloadTag::NONE};
    const NormalizedCSIPayload normalized_b{csi_b.data(), csi_b.size(), NormalizedCSIPayloadTag::NONE};

    csi_info.buf = csi_a.data();
    transport.handle_csi_packet(&csi_info, normalized_a, true);
    csi_info.buf = csi_b.data();
    transport.handle_csi_packet(&csi_info, normalized_b, true);
    transport.latest_csi_.captured_at_us = monotonic_now_us();

    std::array<uint8_t, sizeof(CsiStreamHeaderV7) + HT20_CSI_LEN> record{};
    const size_t record_len = transport.build_stream_packet_(record.data(), record.size());

    TEST_ASSERT_EQUAL(sizeof(record), record_len);
    TEST_ASSERT_EQUAL_INT8(22, static_cast<int8_t>(record[sizeof(CsiStreamHeaderV7)]));
    TEST_ASSERT_EQUAL(2U, transport.latest_csi_sent_total_);
}

void test_csi_stream_transport_drops_stale_latest_sample(void) {
    CsiStreamTransport transport;
    transport.configure(0x1122334455667788ULL, 5001U, 1000U, 1U);
    transport.reset_session();

    std::array<int8_t, HT20_CSI_LEN> csi{};
    csi[0] = 33;
    wifi_csi_info_t csi_info{};
    csi_info.buf = csi.data();
    csi_info.len = HT20_CSI_LEN;
    const NormalizedCSIPayload normalized{csi.data(), csi.size(), NormalizedCSIPayloadTag::NONE};

    transport.handle_csi_packet(&csi_info, normalized, true);
    TEST_ASSERT_TRUE(transport.latest_csi_.valid);
    transport.latest_csi_.captured_at_us = 0U;

    std::array<uint8_t, sizeof(CsiStreamHeaderV7) + HT20_CSI_LEN> record{};
    const size_t record_len = transport.build_stream_packet_(record.data(), record.size());

    TEST_ASSERT_EQUAL(0U, record_len);
    TEST_ASSERT_EQUAL(1U, transport.latest_csi_sent_total_);
    TEST_ASSERT_EQUAL(1ULL, transport.stream_repeat_total_.load(std::memory_order_relaxed));
}

void test_runtime_config_utils_validate_and_name_values(void) {
    TEST_ASSERT_TRUE(validate_runtime_threshold(0.0f));
    TEST_ASSERT_TRUE(validate_runtime_threshold(1.0f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(-0.1f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(1.1f));
    TEST_ASSERT_EQUAL_STRING("ping", traffic_mode_name(RuntimeTrafficMode::PING));
    TEST_ASSERT_EQUAL_STRING("dns", traffic_mode_name(RuntimeTrafficMode::DNS));
    TEST_ASSERT_EQUAL_STRING("ml", detection_algorithm_name(DetectionAlgorithm::ML));
    TEST_ASSERT_EQUAL_STRING("classic", detection_algorithm_name(DetectionAlgorithm::CLASSIC));
    TEST_ASSERT_EQUAL_STRING("fixed", subcarrier_source_name(RuntimeSubcarrierSource::FIXED_DEFAULT));
    TEST_ASSERT_TRUE(parse_traffic_mode("ping") == RuntimeTrafficMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_mode("dns") == RuntimeTrafficMode::DNS);
    TEST_ASSERT_TRUE(parse_detection_algorithm("ml") == DetectionAlgorithm::ML);
    TEST_ASSERT_TRUE(parse_detection_algorithm("classic") == DetectionAlgorithm::CLASSIC);
}

void test_runtime_diagnostics_emit_expected_key_value_pairs(void) {
    RuntimeConfig config;
    RuntimeSnapshot snapshot;
    config.lowpass_enabled = true;
    snapshot.threshold = 2.5f;
    snapshot.detector_name = "classic";
    snapshot.startup_threshold = 0.125f;

    std::vector<std::string> lines;
    visit_runtime_diagnostics(config, snapshot, [&lines](const char *key, const char *value) {
        lines.emplace_back(std::string(key) + "=" + value);
    });

    TEST_ASSERT_TRUE(!lines.empty());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "threshold=2.500000") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "detector=classic") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "lowpass=on") != lines.end());
    TEST_ASSERT_TRUE(std::find(lines.begin(), lines.end(), "startup_threshold=0.125000") != lines.end());
}

void test_mqtt_payload_assembler_accepts_complete_and_fragmented_payloads(void) {
    MqttPayloadAssembler assembler;

    TEST_ASSERT_TRUE(assembler.append("ping", 4, 4, 0) == MqttPayloadAssembler::Result::COMPLETE);
    TEST_ASSERT_EQUAL_STRING("ping", assembler.payload().c_str());
    assembler.reset();

    TEST_ASSERT_TRUE(assembler.append("calib", 5, 9, 0) == MqttPayloadAssembler::Result::INCOMPLETE);
    TEST_ASSERT_TRUE(assembler.append("rate", 4, 9, 5) == MqttPayloadAssembler::Result::COMPLETE);
    TEST_ASSERT_EQUAL_STRING("calibrate", assembler.payload().c_str());
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
    RUN_TEST(test_csi_stream_transport_serializes_v7_phy_metadata);
    RUN_TEST(test_csi_stream_transport_prefers_latest_fresh_sample);
    RUN_TEST(test_csi_stream_transport_drops_stale_latest_sample);
    RUN_TEST(test_runtime_config_utils_validate_and_name_values);
    RUN_TEST(test_runtime_diagnostics_emit_expected_key_value_pairs);
    RUN_TEST(test_mqtt_payload_assembler_accepts_complete_and_fragmented_payloads);
    RUN_TEST(test_mqtt_payload_assembler_rejects_invalid_fragments);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
