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
#include "csi_features.h"
#include "csi_platform_config.h"
#include "runtime_config_utils.h"
#include "mqtt_payload_assembler.h"
#include "runtime_diagnostics.h"
#include "runtime_diagnostics_protocol.h"
#include "runtime_performance_diagnostics.h"
#include "runtime_time.h"
#include "sensing_readiness_gate.h"
#include "sta_socket_helpers.h"
#include "wifi_csi_interface.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#include "espectre_log.h"

#include "esp_timer.h"
#include "esp_netif.h"
#include "network_traffic.h"

#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

#define private public
#undef private

using namespace espectre;

namespace {

#if !CONFIG_SOC_WIFI_HE_SUPPORT
void dummy_csi_callback(void *, wifi_csi_info_t *) {}
#endif

struct CapturedCsiPacket {
  std::array<int8_t, HT20_CSI_LEN> payload{};
  uint32_t callback_count{0U};
  int8_t first_value{0};
  int8_t retained_value{0};
  uint16_t info_len{0U};
  size_t normalized_len{0U};
  NormalizedCSIPayloadTag normalization_tag{NormalizedCSIPayloadTag::NONE};
  bool first_word_invalid{true};
  bool missing_lltf_bins_zero{false};
  bool rotated_to_centered{false};
  bool reset_detector{false};
};

#if !CONFIG_SOC_WIFI_HE_SUPPORT
struct CapturedChannelChange {
  uint32_t callback_count{0U};
  uint8_t previous_channel{0U};
  uint8_t current_channel{0U};
};
#endif

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
  captured->reset_detector = normalized.reset_detector_before_consume;
  captured->first_value = normalized.valid() ? normalized.data[0] : 0;
  captured->retained_value = normalized.valid() ? normalized.data[16] : 0;
  captured->info_len = info != nullptr ? info->len : 0U;
  captured->normalized_len = normalized.len;
  captured->normalization_tag = normalized.tag;
  captured->first_word_invalid = info == nullptr || info->first_word_invalid;
  captured->missing_lltf_bins_zero = normalized.valid();
  if (normalized.valid()) {
    std::copy_n(normalized.data, normalized.len, captured->payload.data());
    for (uint8_t bin : HT20_LLTF_MISSING_BINS) {
      const size_t byte_index = static_cast<size_t>(bin) * 2U;
      captured->missing_lltf_bins_zero &=
          normalized.data[byte_index] == 0 && normalized.data[byte_index + 1U] == 0;
    }
  }
  captured->rotated_to_centered = normalized.rotated_to_centered;
}

#if !CONFIG_SOC_WIFI_HE_SUPPORT
void capture_channel_change(void *context, uint8_t previous_channel, uint8_t current_channel) {
  auto *captured = static_cast<CapturedChannelChange *>(context);
  captured->callback_count++;
  captured->previous_channel = previous_channel;
  captured->current_channel = current_channel;
}
#endif

}  // namespace

void test_csi_quality_rejects_hardware_errors_and_preserves_valid_tones(void) {
    CaptureWiFiMock wifi;
    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init(&wifi);
    service.set_packet_callback(capture_csi_packet, &captured);
    TEST_ASSERT_EQUAL(ESP_OK, service.enable());
    std::array<int8_t, 256> payload;
    payload.fill(12);
    for (uint8_t bin : HT20_CENTERED_ONLY_NULL_BINS) {
        payload[bin * 2] = payload[bin * 2 + 1] = 0;
    }
    wifi_csi_info_t info{};
    info.buf = payload.data();
    info.len = 128;
    info.rx_ctrl.channel = 1;
#if CONFIG_SOC_WIFI_HE_SUPPORT
    info.rx_ctrl.cur_bb_format = RX_BB_FORMAT_HT;
    info.rx_ctrl.rx_channel_estimate_info_vld = 1;
    info.rx_ctrl.rx_channel_estimate_len = 128;
#else
    info.rx_ctrl.sig_mode = 1;
#endif
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(1, captured.callback_count);
    TEST_ASSERT_TRUE(std::equal(payload.begin(), payload.begin() + 128, captured.payload.begin()));
    info.rx_ctrl.rx_state = 1;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    TEST_ASSERT_EQUAL(9, service.rx_error_packets());
    info.len = 127;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    TEST_ASSERT_EQUAL(18, service.rx_error_packets());
    info.buf = nullptr;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    TEST_ASSERT_EQUAL(27, service.rx_error_packets());
    info.buf = payload.data();
    info.len = 128;
    info.rx_ctrl.rx_state = 0;
#if CONFIG_SOC_WIFI_HE_SUPPORT
    info.rx_ctrl.rxend_state = 1;
    info.rx_ctrl.rx_channel_estimate_info_vld = 0;
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(1, service.rx_end_error_packets());
    TEST_ASSERT_EQUAL(0, service.invalid_estimate_packets());
    info.rx_ctrl.rxend_state = 0;
    info.len = 127;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    info.len = 0;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    info.len = 128;
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(19, service.invalid_estimate_packets());
    info.rx_ctrl.rx_channel_estimate_info_vld = 1;
#endif
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(2, captured.callback_count);
    TEST_ASSERT_FALSE(captured.reset_detector);
    info.first_word_invalid = true;
    for (uint16_t length : {126, 127}) {
        info.len = length;
        for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    }
    info.len = 128;
    info.buf = nullptr;
    for (unsigned i = 0; i < 9; ++i) service.process_packet(&info);
    info.buf = payload.data();
    TEST_ASSERT_EQUAL(27, service.invalid_first_word_packets());
    TEST_ASSERT_EQUAL(2, captured.callback_count);
    for (uint16_t length : {128, 256}) {
        info.len = length;
        payload[0] = payload[1] = payload[2] = payload[3] = 127;
        service.process_packet(&info);
        if (length == 128) TEST_ASSERT_FALSE(captured.reset_detector);
        TEST_ASSERT_TRUE(std::equal(payload.begin() + 4, payload.begin() + 128, captured.payload.begin() + 4));
        for (unsigned i = 0; i < 4; ++i) TEST_ASSERT_EQUAL_INT8(0, captured.payload[i]);
        TEST_ASSERT_EQUAL_INT8(127, payload[0]);
    }
    TEST_ASSERT_EQUAL(4, captured.callback_count);
    TEST_ASSERT_EQUAL(2, service.sanitized_first_word_packets());
    for (uint16_t length : {114, 228}) {
        info.len = length;
        service.process_packet(&info);
    }
    TEST_ASSERT_EQUAL(4, captured.callback_count);
    TEST_ASSERT_EQUAL(29, service.invalid_first_word_packets());
    // No latched layout is available after a session reset. Corrupted guard
    // bytes must not prevent a centered layout being established safely.
    service.reset_session();
    info.len = 128;
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(5, captured.callback_count);
    service.reset_session();
    payload.fill(0);
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(5, captured.callback_count);
    TEST_ASSERT_EQUAL(1, service.invalid_first_word_packets());
    for (unsigned i = 0; i < payload.size(); ++i) payload[i] = static_cast<int8_t>(i % 119 + 1);
    for (uint8_t bin : HT20_CLASSIC_ONLY_NULL_BINS) {
        payload[bin * 2] = payload[bin * 2 + 1] = 0;
    }
    const auto required_bins = detail::required_amplitude_bins(
        DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE, TURB_IQR_AGGREGATION_WIDTH);
    TEST_ASSERT_FALSE(required_bins[32]);
    TEST_ASSERT_FALSE(required_bins[33]);
    const auto original_classic = payload;
    for (uint16_t length : {128, 256}) {
        info.len = length;
        service.process_packet(&info);
        TEST_ASSERT_TRUE(captured.first_word_invalid);
        for (unsigned i = 0; i < 128; ++i) {
            const int8_t expected = i >= 64 && i < 68 ? 0 : original_classic[(i + 64) % 128];
            TEST_ASSERT_EQUAL_INT8(expected, captured.payload[i]);
        }
        TEST_ASSERT_TRUE(payload == original_classic);
    }
    TEST_ASSERT_EQUAL(7, captured.callback_count);
    TEST_ASSERT_EQUAL(2, service.sanitized_first_word_packets());
    TEST_ASSERT_EQUAL(1, service.invalid_first_word_packets());
    TEST_ASSERT_EQUAL(1, service.filtered_packets());
    TEST_ASSERT_EQUAL(ESP_OK, service.disable());
    TEST_ASSERT_EQUAL(ESP_OK, service.enable(CsiCaptureProfile::LLTF20));
    info.len = 106;
#if CONFIG_SOC_WIFI_HE_SUPPORT
    info.rx_ctrl.cur_bb_format = RX_BB_FORMAT_11G;
#else
    info.rx_ctrl.sig_mode = 0;
#endif
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(7, captured.callback_count);
    TEST_ASSERT_EQUAL(2, service.invalid_first_word_packets());
}

#if !CONFIG_SOC_WIFI_HE_SUPPORT
void test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi(void) {
    WiFiCSIReal wifi;
    wifi_csi_config_t config{};

    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_config(&config));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi_rx_cb(dummy_csi_callback, nullptr));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(true));
    TEST_ASSERT_EQUAL(ESP_OK, wifi.set_csi(false));
}

void test_lltf_preference_and_vht_capability_resolve_capture_profile(void) {
    TEST_ASSERT_TRUE(resolve_csi_capture_profile(true, true, 36U) ==
                     CsiCaptureProfile::LLTF20);
    TEST_ASSERT_TRUE(resolve_csi_capture_profile(false, true, 6U) ==
                     CsiCaptureProfile::HT20);
    TEST_ASSERT_TRUE(resolve_csi_capture_profile(false, true, 36U) ==
                     CsiCaptureProfile::VHT20);
    TEST_ASSERT_TRUE(resolve_csi_capture_profile(false, false, 36U) ==
                     CsiCaptureProfile::HT20);

    const wifi_csi_config_t config =
        build_csi_config(CsiCaptureProfile::LLTF20);

    TEST_ASSERT_TRUE(config.lltf_en);
    TEST_ASSERT_FALSE(config.htltf_en);
    TEST_ASSERT_FALSE(config.stbc_htltf2_en);
}

void test_lltf20_profile_inherits_supported_ht20_payload_layouts(void) {
    std::array<int8_t, HT20_CSI_LEN_DOUBLE> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 0U;

    struct LayoutCase {
      uint16_t raw_len;
      CsiLayoutId layout_id;
      NormalizedCSIPayloadTag normalization_tag;
      bool requires_normalization;
    };
    const LayoutCase cases[] = {
        {HT20_CSI_LEN, CsiLayoutId::HT20_64,
         NormalizedCSIPayloadTag::NONE, false},
        {HT20_CSI_LEN_SHORT, CsiLayoutId::HT20_57,
         NormalizedCSIPayloadTag::HT57_TO_64, true},
        {HT20_CSI_LEN_DOUBLE, CsiLayoutId::HT20_64_DOUBLE,
         NormalizedCSIPayloadTag::DOUBLE_HT20, true},
        {HT20_CSI_LEN_SHORT_DOUBLE, CsiLayoutId::HT20_57_DOUBLE,
         NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64, true},
    };

    for (const LayoutCase &layout : cases) {
      info.len = layout.raw_len;
      const CsiFormatAssessment assessment =
          assess_ht20_sensing_format(&info, CsiCaptureProfile::LLTF20);

      TEST_ASSERT_TRUE(assessment.is_sensing_accepted());
      TEST_ASSERT_TRUE(assessment.layout_id == layout.layout_id);
      TEST_ASSERT_TRUE(assessment.normalization_tag ==
                       layout.normalization_tag);
      TEST_ASSERT_EQUAL(layout.requires_normalization,
                        assessment.requires_normalization());
      TEST_ASSERT_EQUAL(HT20_CSI_LEN, assessment.normalized_len);
      TEST_ASSERT_EQUAL(HT20_NUM_SUBCARRIERS,
                        assessment.normalized_num_subcarriers);
    }
}

void test_compact_lltf_preserves_tones_and_bypasses_latched_classic_order(void) {
    std::array<int8_t, LLTF20_CSI_LEN_SHORT> compact{};
    std::array<int8_t, HT20_CSI_LEN> expected{};
    for (int tone = -26; tone <= 26; ++tone) {
      const size_t source = static_cast<size_t>(tone + 26) * 2U;
      compact[source] = static_cast<int8_t>(tone);
      compact[source + 1U] = static_cast<int8_t>(-tone);
      expected[(tone + 32) * 2] = static_cast<int8_t>(tone);
      expected[(tone + 32) * 2 + 1] = static_cast<int8_t>(-tone);
    }
    std::array<int8_t, HT20_CSI_LEN> scratch{};
    scratch.fill(127);
    const auto normalized = normalize_ht20_csi_payload(
        compact.data(), compact.size(), scratch.data(), scratch.size());
    TEST_ASSERT_TRUE(normalized.valid());
    TEST_ASSERT_EQUAL(HT20_CSI_LEN, normalized.len);
    TEST_ASSERT_TRUE(normalized.tag == NormalizedCSIPayloadTag::LLTF53_TO_64);
    TEST_ASSERT_TRUE(std::equal(expected.begin(), expected.end(), normalized.data));
    TEST_ASSERT_FALSE(normalize_ht20_csi_payload(
        compact.data(), compact.size(), nullptr, scratch.size()).valid());
    TEST_ASSERT_FALSE(normalize_ht20_csi_payload(
        compact.data(), compact.size(), scratch.data(), scratch.size() - 1U).valid());

    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init();
    service.set_packet_callback(&capture_csi_packet, &captured);
    TEST_ASSERT_EQUAL(ESP_OK, service.enable(CsiCaptureProfile::LLTF20));
    // Establish classic ordering from a full-width packet before compact LLTF.
    scratch.fill(7);
    for (uint8_t bin : HT20_CLASSIC_ONLY_NULL_BINS) {
      scratch[bin * 2U] = scratch[bin * 2U + 1U] = 0;
    }
    wifi_csi_info_t info{};
    info.buf = scratch.data();
    info.len = scratch.size();
    info.rx_ctrl.channel = 6U;
    info.rx_ctrl.timestamp = 100U;
    service.process_packet(&info);
    TEST_ASSERT_TRUE(captured.rotated_to_centered);
    info.buf = compact.data();
    info.len = compact.size();
    info.rx_ctrl.timestamp++;
    service.process_packet(&info);
    TEST_ASSERT_EQUAL(2U, captured.callback_count);
    TEST_ASSERT_EQUAL(LLTF20_CSI_LEN_SHORT, captured.info_len);
    TEST_ASSERT_TRUE(expected == captured.payload);
    TEST_ASSERT_FALSE(captured.rotated_to_centered);
    TEST_ASSERT_TRUE(captured.missing_lltf_bins_zero);
    TEST_ASSERT_EQUAL(ESP_OK, service.disable());

    for (auto profile : {CsiCaptureProfile::LLTF20, CsiCaptureProfile::HT20,
                         CsiCaptureProfile::VHT20}) {
      for (uint8_t sig_mode : {0U, 1U, 3U}) {
        info.rx_ctrl.sig_mode = sig_mode;
        const auto assessment = assess_ht20_sensing_format(&info, profile);
        const bool accepted = profile == CsiCaptureProfile::LLTF20 && sig_mode == 0U;
        TEST_ASSERT_EQUAL(accepted, assessment.is_sensing_accepted());
        if (accepted) {
          TEST_ASSERT_TRUE(assessment.layout_id == CsiLayoutId::LLTF20_53);
          TEST_ASSERT_TRUE(assessment.requires_normalization());
        }
      }
    }
}

void test_lltf20_legacy_frames_reject_unrelated_payload_lengths(void) {
    std::array<int8_t, HT20_CSI_LEN_DOUBLE> csi{};
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.rx_ctrl.sig_mode = 0U;
    info.rx_ctrl.cwb = 0U;

    info.len = HT20_CSI_LEN;
    TEST_ASSERT_TRUE(assess_ht20_sensing_format(
                         &info, CsiCaptureProfile::LLTF20)
                         .is_sensing_accepted());

    const uint16_t unsupported_lengths[] = {
        HT20_CSI_LEN_SHORT,
        HT20_CSI_LEN_DOUBLE,
        HT20_CSI_LEN_SHORT_DOUBLE,
    };
    for (uint16_t raw_len : unsupported_lengths) {
      info.len = raw_len;
      const CsiFormatAssessment assessment =
          assess_ht20_sensing_format(&info, CsiCaptureProfile::LLTF20);

      TEST_ASSERT_FALSE(assessment.is_sensing_accepted());
      TEST_ASSERT_TRUE(assessment.reason_code ==
                       CsiFormatReasonCode::UNKNOWN_LAYOUT);
      TEST_ASSERT_EQUAL(0U, assessment.normalized_len);
    }
}

void test_csi_capture_service_normalizes_ht_layouts_under_lltf20(void) {
    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init();
    service.set_packet_callback(&capture_csi_packet, &captured);
    TEST_ASSERT_EQUAL(ESP_OK, service.enable(CsiCaptureProfile::LLTF20));

    std::array<int8_t, HT20_CSI_LEN_DOUBLE> csi{};
    csi.fill(7);
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 0U;
    info.rx_ctrl.channel = 6U;

    struct NormalizationCase {
      uint16_t raw_len;
      NormalizedCSIPayloadTag normalization_tag;
    };
    const NormalizationCase cases[] = {
        {HT20_CSI_LEN, NormalizedCSIPayloadTag::NONE},
        {HT20_CSI_LEN_SHORT, NormalizedCSIPayloadTag::HT57_TO_64},
        {HT20_CSI_LEN_DOUBLE, NormalizedCSIPayloadTag::DOUBLE_HT20},
        {HT20_CSI_LEN_SHORT_DOUBLE,
         NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64},
    };

    uint32_t timestamp = 100U;
    for (const NormalizationCase &layout : cases) {
      captured = {};
      info.len = layout.raw_len;
      info.rx_ctrl.timestamp = timestamp++;
      service.process_packet(&info);

      TEST_ASSERT_EQUAL(1U, captured.callback_count);
      TEST_ASSERT_EQUAL(layout.raw_len, captured.info_len);
      TEST_ASSERT_EQUAL(HT20_CSI_LEN, captured.normalized_len);
      TEST_ASSERT_TRUE(captured.normalization_tag ==
                       layout.normalization_tag);
      TEST_ASSERT_TRUE(captured.missing_lltf_bins_zero);
    }

    TEST_ASSERT_EQUAL(ESP_OK, service.disable());
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
    const uint32_t delivered[] = {1U, 2U, 2U, 2U, 3U};
    for (size_t i = 0; i < 5U; ++i) {
        info.rx_ctrl.timestamp = timestamps[i];
        service.process_packet(&info);
        TEST_ASSERT_EQUAL(delivered[i], captured.callback_count);
    }

    TEST_ASSERT_EQUAL(3U, captured.callback_count);
    TEST_ASSERT_EQUAL(2U, service.filtered_packets());

    service.reset_session();
    info.rx_ctrl.timestamp = 50U;
    service.process_packet(&info);

    TEST_ASSERT_EQUAL(4U, captured.callback_count);
    TEST_ASSERT_EQUAL(0U, service.filtered_packets());
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

    const CsiFormatAssessment assessment =
        assess_ht20_sensing_format(&info, CsiCaptureProfile::HT20);

    TEST_ASSERT_FALSE(assessment.is_sensing_accepted());
    TEST_ASSERT_TRUE(assessment.reason_code == CsiFormatReasonCode::UNSUPPORTED_WIDTH);
    TEST_ASSERT_TRUE(assessment.normalization_tag == NormalizedCSIPayloadTag::NONE);
}

void test_csi_capture_service_zero_fills_lltf_after_layout_detection(void) {
    CsiCaptureService service;
    CapturedCsiPacket captured;
    service.init();
    service.set_packet_callback(&capture_csi_packet, &captured);
    TEST_ASSERT_EQUAL(ESP_OK, service.enable(CsiCaptureProfile::LLTF20));

    std::array<int8_t, HT20_CSI_LEN> csi{};
    csi.fill(7);
    for (uint8_t bin : HT20_CLASSIC_ONLY_NULL_BINS) {
        const size_t byte_index = static_cast<size_t>(bin) * 2U;
        csi[byte_index] = 0;
        csi[byte_index + 1U] = 0;
    }
    wifi_csi_info_t info{};
    info.buf = csi.data();
    info.len = HT20_CSI_LEN;
    info.rx_ctrl.sig_mode = 0U;
    info.rx_ctrl.cwb = 0U;

    service.process_packet(&info);

    TEST_ASSERT_EQUAL(1U, captured.callback_count);
    TEST_ASSERT_TRUE(captured.rotated_to_centered);
    TEST_ASSERT_TRUE(captured.missing_lltf_bins_zero);
    TEST_ASSERT_EQUAL(7, captured.retained_value);
    TEST_ASSERT_EQUAL(7, csi[HT20_LLTF_MISSING_BINS[0] * 2U]);
    TEST_ASSERT_TRUE(service.last_assessment().format_id == CsiFormatId::HT20);
    TEST_ASSERT_EQUAL(ESP_OK, service.disable());
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

    info.rx_ctrl.sig_mode = 3U;
    info.rx_ctrl.cwb = 0U;
    service.process_packet(&info);
    TEST_ASSERT_TRUE(service.last_assessment().reason_code == CsiFormatReasonCode::UNSUPPORTED_PHY);
    TEST_ASSERT_EQUAL(0U, captured.callback_count);

    info.rx_ctrl.sig_mode = 1U;
    info.rx_ctrl.cwb = 1U;
    service.process_packet(&info);
    TEST_ASSERT_TRUE(service.last_assessment().reason_code == CsiFormatReasonCode::UNSUPPORTED_WIDTH);
    TEST_ASSERT_EQUAL(0U, captured.callback_count);

    info.len = 64U;
    info.rx_ctrl.cwb = 0U;
    service.process_packet(&info);
    TEST_ASSERT_TRUE(service.last_assessment().reason_code == CsiFormatReasonCode::UNKNOWN_LAYOUT);
    TEST_ASSERT_EQUAL(0U, captured.callback_count);
    TEST_ASSERT_EQUAL(3U, service.filtered_packets());
}

void test_runtime_config_utils_validate_and_name_values(void) {
    TEST_ASSERT_TRUE(validate_runtime_threshold(0.0f));
    TEST_ASSERT_TRUE(validate_runtime_threshold(1.0f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(-0.1f));
    TEST_ASSERT_FALSE(validate_runtime_threshold(1.1f));
    TEST_ASSERT_EQUAL_STRING("ping", traffic_generator_mode_name(TrafficGeneratorMode::PING));
    TEST_ASSERT_EQUAL_STRING("dns", traffic_generator_mode_name(TrafficGeneratorMode::DNS));
    TEST_ASSERT_EQUAL_STRING("dns_tcp", traffic_generator_mode_name(TrafficGeneratorMode::DNS_TCP));
    TEST_ASSERT_EQUAL_STRING("external", traffic_generator_mode_name(TrafficGeneratorMode::EXTERNAL));
    TEST_ASSERT_EQUAL_STRING("high_accuracy", detection_algorithm_name(DetectionAlgorithm::HIGH_ACCURACY));
    TEST_ASSERT_EQUAL_STRING("lightweight", detection_algorithm_name(DetectionAlgorithm::LIGHTWEIGHT));
    TEST_ASSERT_TRUE(parse_traffic_generator_mode("ping") == TrafficGeneratorMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_generator_mode("dns") == TrafficGeneratorMode::DNS);
    TEST_ASSERT_TRUE(parse_traffic_generator_mode("dns_tcp") == TrafficGeneratorMode::DNS_TCP);
    TEST_ASSERT_TRUE(parse_traffic_generator_mode("unsupported") == TrafficGeneratorMode::PING);
    TEST_ASSERT_TRUE(parse_traffic_generator_mode("external") == TrafficGeneratorMode::EXTERNAL);
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

void test_runtime_traffic_target_resolves_unicast_ipv4_and_rejects_invalid_addresses(void) {
    RuntimeConfig config;
    const uint8_t gateway_bytes[] = {192, 168, 1, 1};
    uint32_t gateway;
    std::memcpy(&gateway, gateway_bytes, sizeof(gateway));
    TEST_ASSERT_EQUAL(gateway, runtime_traffic_target_addr(config, gateway));
    TEST_ASSERT_EQUAL(0U, runtime_traffic_target_addr(config, 0U));
    for (const char *target : {"192.168.1.53", "10.0.0.1", "1.1.1.1"}) {
        config.traffic_generator_target_ip = target;
        TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::NONE);
        TEST_ASSERT_TRUE(runtime_traffic_target_addr(config, 0U) != 0U);
    }
    config.traffic_generator_target_ip = "192.168.1.53";
    const uint8_t expected[] = {192, 168, 1, 53};
    const uint32_t address = runtime_traffic_target_addr(config, gateway);
    TEST_ASSERT_EQUAL(0, std::memcmp(expected, &address, sizeof(address)));
    for (const char *target : {"router.local", "::1", "192.168.1", "192.168.1.256",
                              "192.168.1.1:53", "192.168.1.1 ", "192.168.01.1",
                              "192.168..1", "192.168.1.1.", "0.0.0.0", "0.1.2.3",
                              "127.0.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255"}) {
        config.traffic_generator_target_ip = target;
        TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::TRAFFIC_GENERATOR_TARGET_IP);
        TEST_ASSERT_EQUAL(0U, runtime_traffic_target_addr(config, gateway));
    }
}

void test_runtime_control_update_applies_fields_as_the_setters_do(void) {
    RuntimeConfig base;
    base.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
    base.threshold = 0.42f;

    RuntimeControlUpdate detector_only;
    detector_only.has_detection_algorithm = true;
    detector_only.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
    RuntimeConfig switched = apply_runtime_control_update(base, detector_only);
    TEST_ASSERT_TRUE(switched.detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY);
    TEST_ASSERT_EQUAL_FLOAT(runtime_default_threshold(DetectionAlgorithm::HIGH_ACCURACY), switched.threshold);

    RuntimeControlUpdate same_detector = detector_only;
    same_detector.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
    TEST_ASSERT_EQUAL_FLOAT(0.42f, apply_runtime_control_update(base, same_detector).threshold);

    RuntimeControlUpdate combined = detector_only;
    combined.has_threshold = true;
    combined.threshold = 0.3f;
    combined.has_motion_hits = true;
    combined.motion_on_hits = 4U;
    combined.motion_off_hits = 2U;
    combined.has_traffic_generator_mode = true;
    combined.traffic_generator_mode = TrafficGeneratorMode::DNS;
    const RuntimeConfig updated = apply_runtime_control_update(base, combined);
    TEST_ASSERT_EQUAL_FLOAT(0.3f, updated.threshold);
    TEST_ASSERT_EQUAL_UINT8(4U, updated.motion_on_hits);
    TEST_ASSERT_EQUAL_UINT8(2U, updated.motion_off_hits);
    TEST_ASSERT_TRUE(updated.traffic_generator_mode == TrafficGeneratorMode::DNS);
    TEST_ASSERT_TRUE(validate_runtime_config(updated) == RuntimeConfigError::NONE);

    base.csi_capture_policy = CsiCapturePolicy::HT_VHT;
    RuntimeControlUpdate raw_traffic;
    raw_traffic.has_traffic_generator_mode = true;
    raw_traffic.traffic_generator_mode = TrafficGeneratorMode::WIFI_RAW;
    TEST_ASSERT_TRUE(validate_runtime_config(apply_runtime_control_update(base, raw_traffic)) ==
                     RuntimeConfigError::CSI_CAPTURE_PROFILE_TRAFFIC);
}

void test_runtime_config_validator_covers_the_public_schema(void) {
    RuntimeConfig config;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::NONE);

    config.wifi_band_policy = static_cast<WifiBandPolicy>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::WIFI_BAND_POLICY);
    config = RuntimeConfig{};
    config.detection_algorithm = static_cast<DetectionAlgorithm>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::DETECTION_ALGORITHM);
    config = RuntimeConfig{};
    config.threshold = 2.0f;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::SEGMENTATION_THRESHOLD);
    config = RuntimeConfig{};
    config.window_size_ms = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::SEGMENTATION_WINDOW_SIZE_MS);
    config = RuntimeConfig{};
    config.csi_target_pps = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TARGET_PPS);
    config = RuntimeConfig{};
    config.traffic_generator_mode = static_cast<TrafficGeneratorMode>(0x7f);
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::TRAFFIC_GENERATOR_MODE);
    config = RuntimeConfig{};
    config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
    config.csi_traffic_udp_port = 0U;
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TRAFFIC_UDP_PORT);
    config = RuntimeConfig{};
    config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
    config.csi_traffic_multicast_group = "192.168.1.2";
    TEST_ASSERT_TRUE(validate_runtime_config(config) == RuntimeConfigError::CSI_TRAFFIC_MULTICAST_GROUP);
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

void test_capture_profile_selection_and_source_constraints(void) {
    RuntimeConfig config;
    TEST_ASSERT_EQUAL(CsiCapturePolicy::AUTO, config.csi_capture_policy);
    for (bool prefers_lltf : {false, true}) {
        for (bool supports_vht : {false, true}) {
            for (uint8_t channel : {6U, 36U}) {
                TEST_ASSERT_EQUAL(CsiCaptureProfile::LLTF20,
                    resolve_csi_capture_profile(prefers_lltf, supports_vht, channel, CsiCapturePolicy::LLTF));
                TEST_ASSERT_EQUAL(supports_vht && channel > 14U ? CsiCaptureProfile::VHT20 : CsiCaptureProfile::HT20,
                    resolve_csi_capture_profile(prefers_lltf, supports_vht, channel, CsiCapturePolicy::HT_VHT));
            }
        }
    }
    TEST_ASSERT_EQUAL(CsiCaptureProfile::HT20, select_csi_capture_profile(6U));
    TEST_ASSERT_EQUAL(CsiCaptureProfile::LLTF20, select_csi_capture_profile(6U, true));
    TEST_ASSERT_EQUAL(CsiCaptureProfile::LLTF20, select_csi_capture_profile(6U, false, CsiCapturePolicy::LLTF));
    TEST_ASSERT_EQUAL(CsiCaptureProfile::HT20, select_csi_capture_profile(6U, false, CsiCapturePolicy::HT_VHT));

    for (auto profile : {CsiCapturePolicy::AUTO, CsiCapturePolicy::LLTF, CsiCapturePolicy::HT_VHT}) {
        config.csi_capture_policy = profile;
        config.traffic_generator_mode = TrafficGeneratorMode::PING;
        for (auto band : {WifiBandPolicy::BAND_2G, WifiBandPolicy::BAND_5G, WifiBandPolicy::AUTO}) {
            config.wifi_band_policy = band;
            TEST_ASSERT_EQUAL(RuntimeConfigError::NONE, validate_runtime_config(config));
        }
        config.traffic_generator_mode = TrafficGeneratorMode::WIFI_RAW;
        const bool compatible = profile != CsiCapturePolicy::HT_VHT;
        TEST_ASSERT_EQUAL(compatible ? RuntimeConfigError::NONE : RuntimeConfigError::CSI_CAPTURE_PROFILE_TRAFFIC,
                          validate_runtime_config(config));
    }
    config = RuntimeConfig{};
    config.csi_capture_policy = static_cast<CsiCapturePolicy>(0x7f);
    TEST_ASSERT_EQUAL(RuntimeConfigError::CSI_CAPTURE_PROFILE, validate_runtime_config(config));
    WiFiCSIReal wifi;
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, configure_csi(&wifi, static_cast<CsiCaptureProfile>(0x7f)));
    TEST_ASSERT_TRUE(csi_capture_profile_supported(CsiCaptureProfile::LLTF20));
    TEST_ASSERT_FALSE(csi_capture_profile_supported(static_cast<CsiCaptureProfile>(0x7f)));
#if !CONFIG_IDF_TARGET_ESP32C5
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, configure_csi(&wifi, CsiCaptureProfile::VHT20));
#endif
}

void test_station_network_traffic_counts_delivery_and_successful_sends(void) {
    esp_netif_mock_reset();
    const NetworkTrafficSnapshot baseline = read_network_traffic();
    const int lookups = g_esp_netif_mock.get_handle_call_count;
    esp_netif_inherent_config_t station_key{};
    station_key.if_key = "WIFI_STA_DEF";
    esp_netif_config_t station_config{};
    station_config.base = &station_key;
    esp_netif_t *sta = __wrap_esp_netif_new(&station_config);
    esp_netif_inherent_config_t other_key{};
    other_key.if_key = "WIFI_AP_DEF";
    esp_netif_config_t other_config{};
    other_config.base = &other_key;
    esp_netif_t *other = __wrap_esp_netif_new(&other_config);
    uint8_t buffer[] = {1U, 2U, 3U};
    int extra = 42;
    TEST_ASSERT_NOT_NULL(sta);
    TEST_ASSERT_NOT_NULL(other);

    TEST_ASSERT_EQUAL(ESP_OK, __wrap_esp_netif_receive(sta, buffer, sizeof(buffer), &extra));
    TEST_ASSERT_TRUE(sta == g_esp_netif_mock.last_netif);
    TEST_ASSERT_TRUE(buffer == g_esp_netif_mock.last_buffer);
    TEST_ASSERT_TRUE(&extra == g_esp_netif_mock.last_extra);
    TEST_ASSERT_EQUAL(sizeof(buffer), g_esp_netif_mock.last_len);
    g_esp_netif_mock.receive_result = ESP_FAIL;
    TEST_ASSERT_EQUAL(ESP_FAIL, __wrap_esp_netif_receive(sta, buffer, sizeof(buffer), &extra));
    __wrap_esp_netif_receive(other, buffer, sizeof(buffer), nullptr);
    __wrap_esp_netif_receive(nullptr, buffer, sizeof(buffer), nullptr);

    TEST_ASSERT_EQUAL(ESP_OK, __wrap_esp_netif_transmit_wrap(sta, buffer, sizeof(buffer), &extra));
    TEST_ASSERT_TRUE(sta == g_esp_netif_mock.last_netif);
    TEST_ASSERT_TRUE(buffer == g_esp_netif_mock.last_buffer);
    TEST_ASSERT_TRUE(&extra == g_esp_netif_mock.last_extra);
    TEST_ASSERT_EQUAL(sizeof(buffer), g_esp_netif_mock.last_len);
    __wrap_esp_netif_transmit_wrap(other, buffer, sizeof(buffer), nullptr);
    __wrap_esp_netif_transmit_wrap(nullptr, buffer, sizeof(buffer), nullptr);
    g_esp_netif_mock.transmit_result = ESP_FAIL;
    TEST_ASSERT_EQUAL(ESP_FAIL, __wrap_esp_netif_transmit_wrap(sta, buffer, sizeof(buffer), &extra));
    TEST_ASSERT_EQUAL(lookups, g_esp_netif_mock.get_handle_call_count);
    TEST_ASSERT_EQUAL(4, g_esp_netif_mock.receive_call_count);
    TEST_ASSERT_EQUAL(4, g_esp_netif_mock.transmit_call_count);
    const NetworkTrafficSnapshot after = read_network_traffic();
    TEST_ASSERT_EQUAL(2U, after.rx_packets - baseline.rx_packets);
    TEST_ASSERT_EQUAL(1U, after.tx_packets - baseline.tx_packets);
    // The runtime loop reads these counters, so a read must never resolve the
    // station handle: that lookup waits for the lwIP core lock.
    TEST_ASSERT_EQUAL(lookups, g_esp_netif_mock.get_handle_call_count);

    __wrap_esp_netif_destroy(sta);
    esp_netif_t *recreated = __wrap_esp_netif_new(&station_config);
    TEST_ASSERT_NOT_NULL(recreated);
    const NetworkTrafficSnapshot before_recreate = read_network_traffic();
    __wrap_esp_netif_receive(sta, buffer, sizeof(buffer), nullptr);
    __wrap_esp_netif_receive(recreated, buffer, sizeof(buffer), nullptr);
    const NetworkTrafficSnapshot after_recreate = read_network_traffic();
    TEST_ASSERT_EQUAL(1U, after_recreate.rx_packets - before_recreate.rx_packets);
    TEST_ASSERT_EQUAL(lookups, g_esp_netif_mock.get_handle_call_count);
    __wrap_esp_netif_destroy(recreated);
    __wrap_esp_netif_destroy(other);
    esp_netif_mock_reset();
}

void test_network_rates_wrap_independently_of_generator_resets(void) {
    RuntimeDiagnosticsSnapshot counters;
    counters.traffic.generator_packets_total = 100U;
    counters.traffic.tx_packets_total = UINT32_MAX - 2U;
    counters.traffic.rx_packets_total = UINT32_MAX - 3U;
    RuntimeDiagnosticsSampler sampler;
    sampler.reset(counters, UINT32_MAX - 499U);
    counters.traffic.generator_packets_total = 0U;
    counters.traffic.tx_packets_total = 7U;
    counters.traffic.rx_packets_total = 16U;
    const RuntimeDiagnosticsSample sample = sampler.sample(counters, 500U);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sample.generator_pps);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, sample.traffic_tx_pps);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, sample.traffic_rx_pps);
    const RuntimeDiagnosticsSample idle = sampler.sample(counters, 1500U);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, idle.traffic_tx_pps);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, idle.traffic_rx_pps);
}

void test_runtime_diagnostics_sampler_derives_five_second_rates(void) {
    RuntimeDiagnosticsSnapshot baseline;
    baseline.traffic.generator_packets_total = 100U;
    baseline.traffic.tx_packets_total = 100U;
    baseline.traffic.rx_packets_total = 200U;
    baseline.csi.callbacks_total = 100U;
    baseline.csi.accepted_total = 90U;
    baseline.csi.admitted_total = 80U;
    baseline.csi.filtered_total = 10U;

    RuntimeDiagnosticsSampler sampler;
    sampler.reset(baseline, 1000U);

    RuntimeDiagnosticsSnapshot current = baseline;
    current.traffic.generator_packets_total = 600U;
    current.traffic.tx_packets_total = 700U;
    current.traffic.rx_packets_total = 750U;
    current.csi.callbacks_total = 580U;
    current.csi.accepted_total = 540U;
    current.csi.admitted_total = 505U;
    current.csi.filtered_total = 40U;
    current.csi.rx_error_total = 1U;
    current.csi.rx_end_error_total = 2U;
    current.csi.invalid_estimate_total = 3U;
    current.csi.invalid_first_word_total = 4U;
    current.csi.sanitized_first_word_total = 500U;
    current.csi.missing_slots_total = 25U;
    current.csi.excess_total = 15U;
    current.csi.stale_total = 5U;
    current.csi.out_of_order_total = 10U;
    current.csi.occupancy_slots = 82U;
    current.csi.window_slots = 100U;
    current.link.channel = 10U;
    current.link.rssi_dbm = -55;

    const RuntimeDiagnosticsSample sample = sampler.sample(current, 6000U);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, sample.generator_pps);
    TEST_ASSERT_EQUAL_FLOAT(120.0f, sample.traffic_tx_pps);
    TEST_ASSERT_EQUAL_FLOAT(110.0f, sample.traffic_rx_pps);
    TEST_ASSERT_EQUAL_FLOAT(96.0f, sample.csi_callback_pps);
    TEST_ASSERT_EQUAL_FLOAT(90.0f, sample.csi_accepted_pps);
    TEST_ASSERT_EQUAL_FLOAT(85.0f, sample.csi_admitted_pps);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, sample.csi_filtered_pps);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, sample.csi_hw_error_pps);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, sample.csi_missing_slots_pps);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, sample.csi_excess_pps);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, sample.csi_stale_pps);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, sample.csi_out_of_order_pps);
    TEST_ASSERT_EQUAL_FLOAT(0.82f, sample.csi_occupancy_ratio);
    TEST_ASSERT_EQUAL_UINT8(10U, sample.wifi_channel);
    TEST_ASSERT_EQUAL_INT8(-55, sample.wifi_rssi_dbm);
}

void test_runtime_hardware_error_rate_handles_counter_epochs_and_clock_wrap(void) {
    RuntimeDiagnosticsSnapshot counters;
    counters.csi.rx_error_total = 100U;
    RuntimeDiagnosticsSampler sampler;
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sampler.sample(counters, 0xffffff00U).csi_hw_error_pps);
    counters.csi.rx_error_total = 102U;
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sampler.sample(counters, 0xffffff00U).csi_hw_error_pps);
    // 500 ms cross the 32-bit monotonic-clock boundary.
    TEST_ASSERT_EQUAL_FLOAT(4.0f, sampler.sample(counters, 244U).csi_hw_error_pps);
    counters.csi.rx_error_total = 1U;
    TEST_ASSERT_EQUAL_FLOAT(2.0f, sampler.sample(counters, 744U).csi_hw_error_pps);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sampler.sample(counters, 1244U).csi_hw_error_pps);
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

struct LoopStepTimerLog {
    int64_t write_delay_us{0};
    int warnings{0};
    std::string warning;
};

bool loop_step_timer_log_enabled(void *, LogLevel, const char *) { return true; }

void loop_step_timer_log_write(void *context, LogLevel level, const char *, int, const char *format,
                               va_list args) {
    auto *log = static_cast<LoopStepTimerLog *>(context);
    esp_timer_mock::advance(log->write_delay_us);
    if (level != LogLevel::WARNING) {
        return;
    }
    char text[1024];
    std::vsnprintf(text, sizeof(text), format, args);
    log->warnings++;
    log->warning = text;
}

void test_runtime_loop_step_timer_reports_slow_steps_and_separates_frontend_time(void) {
    esp_timer_mock::reset(1000, 0);
    LoopStepTimerLog log;
    TEST_ASSERT_TRUE(set_log_sink({&log, &loop_step_timer_log_enabled, &loop_step_timer_log_write}));
    RuntimeLoopStepTimer timer;
    timer.reset();

    timer.begin();
    esp_timer_mock::advance(2000);
    timer.mark("pipeline");
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(0, log.warnings);

    // A step held by a blocking sink write and a listener callback names both.
    timer.begin();
    esp_timer_mock::advance(3000);
    timer.mark("pipeline");
    log.write_delay_us = 240000;
    ESPECTRE_LOGI("test.loop", "heartbeat");
    log.write_delay_us = 0;
    esp_timer_mock::advance(4000);
    RuntimeLoopStepTimer::record_listener_time(4000U);
    timer.mark("heartbeat");
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(1, log.warnings);
    TEST_ASSERT_TRUE(log.warning.find("Runtime loop took 247 ms") != std::string::npos);
    TEST_ASSERT_TRUE(log.warning.find("pipeline 3 ms") != std::string::npos);
    TEST_ASSERT_TRUE(log.warning.find("heartbeat 244 ms (log sink 240 ms, listener 4 ms)") !=
                     std::string::npos);
    TEST_ASSERT_TRUE(log.warning.find("log sink 240 ms in 1 write (max 240 ms)") != std::string::npos);

    // A loop that is not called in time is reported even when it runs fast.
    esp_timer_mock::advance(150000);
    timer.begin();
    timer.mark("pipeline");
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(2, log.warnings);
    TEST_ASSERT_TRUE(log.warning.find("took 0 ms, 150 ms after the previous one") != std::string::npos);

    // A restart does not report the time the runtime was stopped.
    esp_timer_mock::advance(500000);
    timer.reset();
    timer.begin();
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(2, log.warnings);

    // A blocking log between iterations stays in the gap and in the summary.
    timer.begin();
    timer.finish("test.loop");
    log.write_delay_us = 180000;
    ESPECTRE_LOGI("test.loop", "between");
    log.write_delay_us = 0;
    timer.begin();
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(3, log.warnings);
    TEST_ASSERT_TRUE(log.warning.find("took 0 ms, 180 ms after the previous one (log sink 180 ms, listener 0 ms)") !=
                     std::string::npos);
    TEST_ASSERT_TRUE(log.warning.find("log sink 180 ms in 1 write (max 180 ms), listener 0 ms") !=
                     std::string::npos);

    // The report keeps the last step and the summary when every step is slow.
    timer.reset();
    esp_timer_mock::reset(5000, 0);
    timer.begin();
    for (const char *step : {"wifi_events", "calibration", "readiness", "pipeline", "threshold", "traffic",
                             "receive_path", "heartbeat"}) {
        log.write_delay_us = 2000;
        ESPECTRE_LOGI("test.loop", "step");
        log.write_delay_us = 0;
        RuntimeLoopStepTimer::record_listener_time(2000U);
        esp_timer_mock::advance(20000);
        timer.mark(step);
    }
    timer.finish("test.loop");
    TEST_ASSERT_EQUAL(4, log.warnings);
    TEST_ASSERT_TRUE(log.warning.find("heartbeat 22 ms (log sink 2 ms, listener 2 ms)") != std::string::npos);
    TEST_ASSERT_TRUE(log.warning.find("log sink 16 ms in 8 writes (max 2 ms), listener 16 ms") !=
                     std::string::npos);
    clear_log_sink();
}

void test_runtime_performance_diagnostics_json_marks_unready_and_unsupported_values(void) {
    RuntimeDiagnosticsSnapshot diagnostics;
    diagnostics.platform.free_memory_bytes = 4096U;
    diagnostics.platform.minimum_free_memory_bytes = 2048U;
    diagnostics.platform.largest_free_memory_block_bytes = 1024U;
    diagnostics.platform.cpu_frequency_mhz = 160U;

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

void test_sta_socket_binding_rejects_missing_or_invalid_interface(void) {
    esp_netif_mock_reset();
    g_esp_netif_mock.handle_available = false;
    TEST_ASSERT_FALSE(bind_socket_to_sta_interface(-1, "test", "udp"));

    esp_netif_mock_reset();
    g_esp_netif_mock.impl_index = 0;
    TEST_ASSERT_FALSE(bind_socket_to_sta_interface(-1, "test", "udp"));
}

void test_sta_socket_binding_uses_resolved_interface(void) {
    esp_netif_mock_reset();
    unsigned interface_index = if_nametoindex("lo");
    if (interface_index == 0U) {
        interface_index = if_nametoindex("lo0");
    }
    TEST_ASSERT_TRUE(interface_index > 0U);
    g_esp_netif_mock.impl_index = static_cast<int>(interface_index);
    const int sock = socket(AF_INET, SOCK_DGRAM, 0);
    TEST_ASSERT_TRUE(sock >= 0);
    const bool bound = bind_socket_to_sta_interface(sock, "test", "udp");
#if defined(__linux__)
    TEST_ASSERT_TRUE(bound);
#else
    (void)bound;
#endif
    close(sock);
}

#endif

void test_sensing_readiness_gate_holds_only_brief_coverage_dips(void) {
    using Edge = SensingReadinessGate::Edge;
    SensingReadinessGate gate;
    SensingReadinessInputs inputs;
    TEST_ASSERT_TRUE(gate.update(inputs, 0U, 1000U) == Edge::NONE);
    TEST_ASSERT_TRUE(gate.reason() == SensingReadinessReason::SENSING_STOPPED);
    inputs = {true, false, true, true, true, false};
    // Coverage below the floor never makes sensing ready in the first place.
    TEST_ASSERT_TRUE(gate.update(inputs, 10U, 1000U) == Edge::NONE);
    TEST_ASSERT_FALSE(gate.ready());
    inputs.detector_ready = true;
    TEST_ASSERT_TRUE(gate.update(inputs, 20U, 1000U) == Edge::READY);

    // A dip shorter than the grace period stays invisible.
    inputs.detector_ready = false;
    TEST_ASSERT_TRUE(gate.update(inputs, 100U, 1000U) == Edge::NONE);
    TEST_ASSERT_TRUE(gate.update(inputs, 1099U, 1000U) == Edge::NONE);
    TEST_ASSERT_TRUE(gate.ready());
    TEST_ASSERT_TRUE(gate.reason() == SensingReadinessReason::LOW_COVERAGE);
    inputs.detector_ready = true;
    TEST_ASSERT_TRUE(gate.update(inputs, 1099U, 1000U) == Edge::DIP_ABSORBED);
    TEST_ASSERT_EQUAL(999, gate.last_dip_ms());

    // A dip that outlasts the grace period clears readiness.
    inputs.detector_ready = false;
    TEST_ASSERT_TRUE(gate.update(inputs, 2000U, 1000U) == Edge::NONE);
    TEST_ASSERT_TRUE(gate.update(inputs, 3000U, 1000U) == Edge::UNREADY);
    TEST_ASSERT_TRUE(gate.reason() == SensingReadinessReason::LOW_COVERAGE);
    inputs.detector_ready = true;
    TEST_ASSERT_TRUE(gate.update(inputs, 3100U, 1000U) == Edge::READY);

    // Every other condition clears readiness at once, even mid-dip.
    const struct {
        SensingReadinessInputs inputs;
        SensingReadinessReason reason;
    } immediate[] = {
        {{false, false, true, true, true, false}, SensingReadinessReason::SENSING_STOPPED},
        {{true, true, true, true, true, false}, SensingReadinessReason::CALIBRATING},
        {{true, false, false, true, true, false}, SensingReadinessReason::NO_DETECTOR},
        {{true, false, true, false, true, false}, SensingReadinessReason::INPUT_STALE},
        {{true, false, true, true, false, false}, SensingReadinessReason::WINDOW_FILLING},
    };
    uint32_t now_ms = 4000U;
    for (const auto &item : immediate) {
        inputs = {true, false, true, true, true, true};
        (void) gate.update(inputs, now_ms, 1000U);
        TEST_ASSERT_TRUE(gate.ready());
        inputs.detector_ready = false;
        TEST_ASSERT_TRUE(gate.update(inputs, now_ms + 10U, 1000U) == Edge::NONE);
        TEST_ASSERT_TRUE(gate.update(item.inputs, now_ms + 20U, 1000U) == Edge::UNREADY);
        TEST_ASSERT_TRUE(gate.reason() == item.reason);
        TEST_ASSERT_NOT_NULL(sensing_readiness_reason_name(item.reason));
        now_ms += 100U;
    }
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_csi_quality_rejects_hardware_errors_and_preserves_valid_tones);
    RUN_TEST(test_sensing_readiness_gate_holds_only_brief_coverage_dips);
#if !CONFIG_SOC_WIFI_HE_SUPPORT
    RUN_TEST(test_wifi_csi_real_forwards_calls_to_mocked_esp_wifi);
    RUN_TEST(test_lltf_preference_and_vht_capability_resolve_capture_profile);
    RUN_TEST(test_lltf20_profile_inherits_supported_ht20_payload_layouts);
    RUN_TEST(test_compact_lltf_preserves_tones_and_bypasses_latched_classic_order);
    RUN_TEST(test_lltf20_legacy_frames_reject_unrelated_payload_lengths);
    RUN_TEST(test_csi_capture_service_normalizes_ht_layouts_under_lltf20);
    RUN_TEST(test_csi_capture_service_filters_duplicate_and_stale_timestamps);
    RUN_TEST(test_csi_capture_service_defers_channel_change_and_resets_session_baseline);
    RUN_TEST(test_csi_format_classifier_rejects_ht40_before_normalization);
    RUN_TEST(test_csi_capture_service_zero_fills_lltf_after_layout_detection);
    RUN_TEST(test_csi_capture_service_tracks_format_drop_reasons);
    RUN_TEST(test_runtime_config_utils_validate_and_name_values);
    RUN_TEST(test_runtime_control_update_applies_fields_as_the_setters_do);
    RUN_TEST(test_runtime_config_validator_covers_the_public_schema);
    RUN_TEST(test_capture_profile_selection_and_source_constraints);
    RUN_TEST(test_runtime_traffic_target_resolves_unicast_ipv4_and_rejects_invalid_addresses);
    RUN_TEST(test_runtime_diagnostics_sampler_derives_five_second_rates);
    RUN_TEST(test_station_network_traffic_counts_delivery_and_successful_sends);
    RUN_TEST(test_network_rates_wrap_independently_of_generator_resets);
    RUN_TEST(test_runtime_hardware_error_rate_handles_counter_epochs_and_clock_wrap);
    RUN_TEST(test_runtime_performance_diagnostics_publish_complete_windows);
    RUN_TEST(test_runtime_loop_step_timer_reports_slow_steps_and_separates_frontend_time);
    RUN_TEST(test_runtime_performance_diagnostics_json_marks_unready_and_unsupported_values);
    RUN_TEST(test_mqtt_payload_assembler_accepts_complete_and_fragmented_payloads);
    RUN_TEST(test_mqtt_payload_assembler_rejects_invalid_fragments);
    RUN_TEST(test_sta_socket_binding_rejects_missing_or_invalid_interface);
    RUN_TEST(test_sta_socket_binding_uses_resolved_interface);
#endif
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
