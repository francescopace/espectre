/*
 * ESPectre - Emulator CSI Replay Unit Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "csi_format_classifier.h"
#include "csi_frame_identity.h"
#include "csi_replay_frame.h"
#include "driver/uart.h"
#include "emulator_csi_replay.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "lwip/inet.h"

using namespace espectre;

extern "C" {
// The firmware links with --wrap; the host build forwards the real calls to the Wi-Fi mock.
esp_err_t __real_esp_wifi_set_csi_config(const wifi_csi_config_t *config) {
  return esp_wifi_set_csi_config(config);
}
esp_err_t __real_esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx) {
  return esp_wifi_set_csi_rx_cb(cb, ctx);
}
esp_err_t __real_esp_wifi_set_csi(bool en) { return esp_wifi_set_csi(en); }

esp_err_t __wrap_esp_wifi_set_csi_config(const wifi_csi_config_t *config);
esp_err_t __wrap_esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx);
esp_err_t __wrap_esp_wifi_set_csi(bool en);
}

namespace {

constexpr uint8_t kLocalMac[6] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
constexpr uint32_t kRxTimestampUs = 0x12345678U;

std::vector<uint8_t> encode(int8_t rssi, const std::vector<int8_t> &csi, uint32_t rx_timestamp_us) {
  std::vector<uint8_t> body = {
      static_cast<uint8_t>(rssi),
      static_cast<uint8_t>(csi.size()),
      static_cast<uint8_t>(rx_timestamp_us),
      static_cast<uint8_t>(rx_timestamp_us >> 8U),
      static_cast<uint8_t>(rx_timestamp_us >> 16U),
      static_cast<uint8_t>(rx_timestamp_us >> 24U),
  };
  body.insert(body.end(), reinterpret_cast<const uint8_t *>(csi.data()),
              reinterpret_cast<const uint8_t *>(csi.data()) + csi.size());
  const uint32_t sum = std::accumulate(body.begin(), body.end(), 0U);
  std::vector<uint8_t> frame = {'C', 'S', 'I', 'R'};
  frame.insert(frame.end(), body.begin(), body.end());
  frame.push_back(static_cast<uint8_t>(sum));
  return frame;
}

std::vector<int8_t> ht20_csi() {
  // Centered HT20 layout: guards and DC stay zero, every data tone is nonzero.
  std::vector<int8_t> csi(HT20_CSI_LEN, 0);
  for (int bin = 4; bin <= 60; ++bin) {
    if (bin == 32) continue;
    csi[static_cast<size_t>(bin) * 2U] = static_cast<int8_t>(8 + (bin % 5));
    csi[static_cast<size_t>(bin) * 2U + 1U] = static_cast<int8_t>(-3 - (bin % 4));
  }
  return csi;
}

int feed(CsiReplayFrameParser &parser, const std::vector<uint8_t> &bytes, CsiReplayFrame *out) {
  int frames = 0;
  for (uint8_t byte : bytes) {
    if (parser.push(byte, out)) frames++;
  }
  return frames;
}

CsiReplayIdentity station() {
  CsiReplayIdentity identity;
  identity.local_ip_addr = htonl(0xC0A80402U);
  identity.peer_ip_addr = htonl(0xC0A80401U);
  std::memcpy(identity.local_mac_addr, kLocalMac, sizeof(kLocalMac));
  return identity;
}

CsiFrameFilterConfig external_filter() {
  CsiFrameFilterConfig config;
  config.traffic_mode = TrafficGeneratorMode::EXTERNAL_HOST;
  config.local_ip_addr = station().local_ip_addr;
  std::memcpy(config.local_mac_addr, kLocalMac, sizeof(kLocalMac));
  return config;
}

CsiReplayFrame decoded_frame() {
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  TEST_ASSERT_EQUAL(1, feed(parser, encode(-57, ht20_csi(), kRxTimestampUs), &frame));
  return frame;
}

void record_csi(void *, wifi_csi_info_t *) {}

}  // namespace

void test_parser_decodes_a_complete_frame(void) {
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  const auto csi = ht20_csi();
  TEST_ASSERT_EQUAL(1, feed(parser, encode(-57, csi, kRxTimestampUs), &frame));
  TEST_ASSERT_EQUAL_INT(-57, frame.rssi_dbm);
  TEST_ASSERT_EQUAL(HT20_CSI_LEN, frame.len);
  TEST_ASSERT_EQUAL(kRxTimestampUs, frame.rx_timestamp_us);
  TEST_ASSERT_EQUAL_UINT8_ARRAY(reinterpret_cast<const uint8_t *>(csi.data()),
                                reinterpret_cast<const uint8_t *>(frame.csi.data()), csi.size());
}

void test_parser_drops_a_bad_checksum_and_recovers_on_the_next_frame(void) {
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  auto corrupt = encode(-40, ht20_csi(), 1U);
  corrupt.back() = static_cast<uint8_t>(corrupt.back() + 1U);
  TEST_ASSERT_EQUAL(0, feed(parser, corrupt, &frame));
  TEST_ASSERT_EQUAL(1, feed(parser, encode(-41, ht20_csi(), 2U), &frame));
  TEST_ASSERT_EQUAL(2U, frame.rx_timestamp_us);
}

void test_parser_resynchronizes_after_noise_and_a_partial_magic(void) {
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  std::vector<uint8_t> stream = {0x00, 'C', 'S', 'X', 'C', 'S'};
  const auto valid = encode(-50, ht20_csi(), 3U);
  stream.insert(stream.end(), valid.begin(), valid.end());
  TEST_ASSERT_EQUAL(1, feed(parser, stream, &frame));
  TEST_ASSERT_EQUAL(3U, frame.rx_timestamp_us);
}

void test_parser_rejects_lengths_outside_the_csi_buffer(void) {
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  const std::vector<uint8_t> empty = {'C', 'S', 'I', 'R', 0xC0, 0x00, 1, 0, 0, 0, 0xC1};
  TEST_ASSERT_EQUAL(0, feed(parser, empty, &frame));
  std::vector<uint8_t> oversized = {'C', 'S', 'I', 'R', 0xC0,
                                    static_cast<uint8_t>(CsiReplayFrame::kMaxCsiBytes + 1U), 1, 0, 0, 0};
  TEST_ASSERT_EQUAL(0, feed(parser, oversized, &frame));
  TEST_ASSERT_EQUAL(1, feed(parser, encode(-50, ht20_csi(), 4U), &frame));
}

void test_replayed_packet_passes_the_external_traffic_filter_and_format_classifier(void) {
  const CsiReplayFrame frame = decoded_frame();
  CsiReplayPacket packet;
  TEST_ASSERT_TRUE(build_csi_replay_packet(frame, station(), 6U, 7U, &packet));

  CsiAckCredit ack_credit;
  CsiRetransmissionFilter retransmissions;
  TEST_ASSERT_TRUE(csi_frame_matches_traffic(&packet.info, external_filter(), CsiCaptureProfile::HT20,
                                             1000000U, ack_credit, retransmissions));
  const CsiFormatAssessment assessment = assess_ht20_sensing_format(&packet.info);
  TEST_ASSERT_TRUE(assessment.is_sensing_accepted());
  TEST_ASSERT_EQUAL(kRxTimestampUs, packet.info.rx_ctrl.timestamp);
  TEST_ASSERT_EQUAL_INT(-57, packet.info.rx_ctrl.rssi);
  TEST_ASSERT_EQUAL(6U, packet.info.rx_ctrl.channel);
}

void test_consecutive_replayed_packets_are_not_taken_for_retransmissions(void) {
  const CsiReplayFrame frame = decoded_frame();
  const CsiFrameFilterConfig filter = external_filter();
  CsiAckCredit ack_credit;
  CsiRetransmissionFilter retransmissions;
  for (uint16_t sequence = 0U; sequence < 3U; ++sequence) {
    CsiReplayPacket packet;
    TEST_ASSERT_TRUE(build_csi_replay_packet(frame, station(), 6U, sequence, &packet));
    TEST_ASSERT_TRUE(csi_frame_matches_traffic(&packet.info, filter, CsiCaptureProfile::HT20,
                                               1000000U, ack_credit, retransmissions));
  }
}

void test_packet_waits_for_the_station_address(void) {
  const CsiReplayFrame frame = decoded_frame();
  CsiReplayPacket packet;
  CsiReplayIdentity no_address = station();
  no_address.local_ip_addr = 0U;
  TEST_ASSERT_FALSE(build_csi_replay_packet(frame, no_address, 6U, 0U, &packet));
}

void test_pacer_releases_frames_at_their_recorded_spacing_behind_the_buffer(void) {
  CsiReplayPacer pacer;
  const int64_t first = pacer.due(5000000U, 1000000);
  TEST_ASSERT_EQUAL(1000000 + CsiReplayPacer::kJitterBufferUs, first);
  // Frames that arrive together keep their recorded 10 ms spacing.
  TEST_ASSERT_EQUAL(first + 10000, pacer.due(5010000U, 1000500));
  TEST_ASSERT_EQUAL(first + 20000, pacer.due(5020000U, 1000500));
  TEST_ASSERT_EQUAL(first, pacer.release_at(first));
  pacer.mark_released(first, first);
  TEST_ASSERT_EQUAL(first + 10000, pacer.release_at(first + 10000));
}

void test_pacer_spaces_out_a_backlog_after_a_stall(void) {
  CsiReplayPacer pacer;
  const int64_t first = pacer.due(0U, 0);
  pacer.mark_released(first, first);
  // The task woke 200 ms late: the overdue frames leave one catch-up interval apart.
  const int64_t now = first + 200000;
  const int64_t second = pacer.release_at(first + 10000);
  TEST_ASSERT_TRUE(second <= now);
  pacer.mark_released(second, now);
  const int64_t third = pacer.release_at(first + 20000);
  TEST_ASSERT_TRUE(third <= now);
  pacer.mark_released(third, now);
  TEST_ASSERT_EQUAL(now + CsiReplayPacer::kCatchUpSpacingUs, pacer.release_at(first + 30000));
}

void test_pacer_keeps_spacing_across_a_timestamp_wrap_and_resyncs_after_a_gap(void) {
  CsiReplayPacer pacer;
  const int64_t first = pacer.due(0xFFFFF000U, 0);
  TEST_ASSERT_EQUAL(first + 0x2000, pacer.due(0x00001000U, 0));
  // A frame far from the timeline starts a new one behind the buffer.
  const int64_t later = 30000000;
  TEST_ASSERT_EQUAL(later + CsiReplayPacer::kJitterBufferUs, pacer.due(0x00002000U, later));
}

void test_marker_needs_the_locally_administered_prefix(void) {
  const uint8_t marked[6] = {0x02, 0x45, 0x53, 0x50, 0x00, 0x01};
  TEST_ASSERT_TRUE(is_emulator_replay_mac(marked));
  // esp-emu's own default MAC and a real Espressif OUI stay unmarked.
  const uint8_t emulator_default[6] = {0x24, 0x0A, 0xC4, 0x00, 0x00, 0x01};
  TEST_ASSERT_FALSE(is_emulator_replay_mac(emulator_default));
  const uint8_t universal[6] = {0x00, 0x45, 0x53, 0x50, 0x00, 0x01};
  TEST_ASSERT_FALSE(is_emulator_replay_mac(universal));
  TEST_ASSERT_FALSE(is_emulator_replay_mac(nullptr));
}

void test_silicon_passes_every_csi_call_to_esp_idf(void) {
  // The default mock factory MAC is an Espressif OUI, as on a real chip.
  g_esp_wifi_mock = {};
  TEST_ASSERT_FALSE(emulator_csi_replay_active());
  wifi_csi_config_t config{};
  TEST_ASSERT_EQUAL(ESP_OK, __wrap_esp_wifi_set_csi_config(&config));
  int context = 0;
  TEST_ASSERT_EQUAL(ESP_OK, __wrap_esp_wifi_set_csi_rx_cb(&record_csi, &context));
  TEST_ASSERT_EQUAL(ESP_OK, __wrap_esp_wifi_set_csi(true));
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_callback == &record_csi);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_callback_context == &context);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(0, g_uart_mock.driver_install_count);
}

int main() {
  using namespace espectre::test;
  begin_suite();
  RUN_TEST(test_parser_decodes_a_complete_frame);
  RUN_TEST(test_parser_drops_a_bad_checksum_and_recovers_on_the_next_frame);
  RUN_TEST(test_parser_resynchronizes_after_noise_and_a_partial_magic);
  RUN_TEST(test_parser_rejects_lengths_outside_the_csi_buffer);
  RUN_TEST(test_replayed_packet_passes_the_external_traffic_filter_and_format_classifier);
  RUN_TEST(test_consecutive_replayed_packets_are_not_taken_for_retransmissions);
  RUN_TEST(test_packet_waits_for_the_station_address);
  RUN_TEST(test_pacer_releases_frames_at_their_recorded_spacing_behind_the_buffer);
  RUN_TEST(test_pacer_spaces_out_a_backlog_after_a_stall);
  RUN_TEST(test_pacer_keeps_spacing_across_a_timestamp_wrap_and_resyncs_after_a_gap);
  RUN_TEST(test_marker_needs_the_locally_administered_prefix);
  RUN_TEST(test_silicon_passes_every_csi_call_to_esp_idf);
  return end_suite();
}
