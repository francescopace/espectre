/*
 * ESPectre - CSI Frame Identity Unit Tests
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "csi_frame_identity.h"
#include "null_data_frame.h"
#include "lwip/inet.h"

using namespace espectre;

namespace {

constexpr uint8_t kLocalMac[6] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
constexpr uint8_t kOtherMac[6] = {0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
constexpr uint8_t kZeroMac[6] = {};
constexpr uint8_t kMulticastMac[6] = {0x01, 0x00, 0x5E, 0x7F, 0x00, 0x01};
constexpr uint32_t kGateway = 0xC0A80101U;
constexpr uint32_t kLocal = 0xC0A80111U;
constexpr uint32_t kOther = 0xC0A80112U;
constexpr uint32_t kMulticast = 0xEFFF0001U;

void write_be16(uint8_t *target, uint16_t value) {
  target[0] = static_cast<uint8_t>(value >> 8U);
  target[1] = static_cast<uint8_t>(value);
}

void write_be32(uint8_t *target, uint32_t value) {
  target[0] = static_cast<uint8_t>(value >> 24U);
  target[1] = static_cast<uint8_t>(value >> 16U);
  target[2] = static_cast<uint8_t>(value >> 8U);
  target[3] = static_cast<uint8_t>(value);
}

std::vector<uint8_t> ipv4_frame(uint8_t protocol,
                                uint32_t source,
                                uint32_t destination,
                                const std::vector<uint8_t> &transport) {
  std::vector<uint8_t> frame(8U + 20U + transport.size(), 0U);
  const uint8_t snap[8] = {0xAAU, 0xAAU, 0x03U, 0U, 0U, 0U, 0x08U, 0U};
  std::memcpy(frame.data(), snap, sizeof(snap));
  uint8_t *ip = frame.data() + 8U;
  ip[0] = 0x45U;
  write_be16(ip + 2U, static_cast<uint16_t>(20U + transport.size()));
  ip[8] = 64U;
  ip[9] = protocol;
  write_be32(ip + 12U, source);
  write_be32(ip + 16U, destination);
  std::memcpy(ip + 20U, transport.data(), transport.size());
  return frame;
}

std::vector<uint8_t> canonical_marker() {
  return std::vector<uint8_t>(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                              RUNTIME_CSI_TRAFFIC_MARKER_BYTES +
                                  RUNTIME_CSI_TRAFFIC_MARKER_LENGTH);
}

std::vector<uint8_t> udp_frame(uint32_t destination,
                               uint16_t port,
                               std::vector<uint8_t> marker = canonical_marker()) {
  std::vector<uint8_t> udp(8U + marker.size(), 0U);
  write_be16(udp.data(), 40000U);
  write_be16(udp.data() + 2U, port);
  write_be16(udp.data() + 4U, static_cast<uint16_t>(udp.size()));
  std::copy(marker.begin(), marker.end(), udp.begin() + 8U);
  return ipv4_frame(17U, kOther, destination, udp);
}

std::vector<uint8_t> ping_reply(uint32_t source, uint16_t identifier) {
  std::vector<uint8_t> icmp(8U, 0U);
  write_be16(icmp.data() + 4U, identifier);
  return ipv4_frame(1U, source, kLocal, icmp);
}

std::vector<uint8_t> ping_request(uint32_t source,
                                  uint32_t destination = kLocal,
                                  uint8_t code = 0U) {
  std::vector<uint8_t> icmp(8U, 0U);
  icmp[0] = 8U;
  icmp[1] = code;
  write_be16(icmp.data() + 4U, 0x4321U);
  return ipv4_frame(1U, source, destination, icmp);
}

std::vector<uint8_t> dns_tcp_reply(bool with_payload = true) {
  std::vector<uint8_t> tcp(with_payload ? 34U : 20U, 0U);
  write_be16(tcp.data(), 53U);
  write_be16(tcp.data() + 2U, 40000U);
  tcp[12] = 0x50U;
  tcp[13] = 0x10U;
  if (with_payload) {
    tcp[20] = 0U;
    tcp[21] = 12U;
    tcp[24] = 0x81U;
    tcp[25] = 0x80U;
  }
  return ipv4_frame(6U, kGateway, kLocal, tcp);
}

std::vector<uint8_t> dns_udp_reply() {
  std::vector<uint8_t> udp(20U, 0U);
  write_be16(udp.data(), 53U);
  write_be16(udp.data() + 2U, 40000U);
  write_be16(udp.data() + 4U, static_cast<uint16_t>(udp.size()));
  udp[10U] = 0x81U;
  udp[11U] = 0x80U;
  return ipv4_frame(17U, kGateway, kLocal, udp);
}

wifi_csi_info_t csi_info(const std::vector<uint8_t> &payload, const uint8_t *destination_mac = kLocalMac) {
  wifi_csi_info_t info{};
  info.payload = const_cast<uint8_t *>(payload.data());
  info.payload_len = static_cast<uint16_t>(payload.size());
  std::memcpy(info.dmac, destination_mac, 6U);
  return info;
}

constexpr uint32_t kNowUs = 1000000U;

bool matches_once(const wifi_csi_info_t *info,
                  const CsiFrameFilterConfig &config,
                  CsiCaptureProfile profile,
                  CsiAckCredit &ack_credit) {
  CsiRetransmissionFilter retransmissions;
  return csi_frame_matches_traffic(info, config, profile, kNowUs, ack_credit, retransmissions);
}

bool matches_once(const wifi_csi_info_t *info,
                  const CsiFrameFilterConfig &config,
                  CsiCaptureProfile profile) {
  CsiAckCredit ack_credit;
  return matches_once(info, config, profile, ack_credit);
}

bool matches(const std::vector<uint8_t> &payload,
             const CsiFrameFilterConfig &config,
             const uint8_t *destination_mac = kLocalMac) {
  const wifi_csi_info_t info = csi_info(payload, destination_mac);
  return matches_once(&info, config, CsiCaptureProfile::HT20);
}

wifi_csi_info_t local_ack_info(uint8_t *header) {
  wifi_csi_info_t info{};
  info.hdr = header;
  info.rx_ctrl.sig_len = 14U;
  return info;
}

std::vector<uint8_t> radio_frame(bool espnow = false, uint8_t version = 1U) {
  std::vector<uint8_t> frame(24U, 0U);
  frame[0] = espnow ? 0xD0U : 0x48U;
  std::memcpy(frame.data() + 4U, kLocalMac, 6U);
  std::memcpy(frame.data() + 10U, kOtherMac, 6U);
  std::memset(frame.data() + 16U, 0xFFU, 6U);
  if (espnow) {
    const uint8_t body[] = {127U, 0x18U, 0xFEU, 0x34U, 1U, 2U, 3U, 4U,
                           221U, 6U, 0x18U, 0xFEU, 0x34U, 4U, version, 42U};
    frame.insert(frame.end(), std::begin(body), std::end(body));
  }
  return frame;
}

wifi_csi_info_t radio_info(std::vector<uint8_t> &frame) {
  wifi_csi_info_t info{};
  info.hdr = frame.data();
  info.payload = frame.data() + 24U;
  info.payload_len = static_cast<uint16_t>(frame.size() - 24U);
  info.rx_ctrl.sig_len = frame.size() + 4U;
  std::memcpy(info.mac, frame.data() + 10U, 6U);
  std::memcpy(info.dmac, frame.data() + 4U, 6U);
  return info;
}

CsiFrameFilterConfig filter(TrafficGeneratorMode mode) {
  CsiFrameFilterConfig config;
  config.traffic_mode = mode;
  config.local_ip_addr = inet_addr("192.168.1.17");
  config.internal_target_ip_addr = inet_addr("192.168.1.1");
  config.multicast_ip_addr = inet_addr("239.255.0.1");
  config.external_udp_port = 5555U;
  config.internal_icmp_identifier = 0x1234U;
  std::memcpy(config.local_mac_addr, kLocalMac, 6U);
  return config;
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_local_ack_requires_lltf20_and_a_generator_credit_in_every_traffic_mode(void) {
  uint8_t header[10] = {0xD4U, 0U, 0U, 0U};
  std::memcpy(header + 4U, kLocalMac, 6U);
  const wifi_csi_info_t info = local_ack_info(header);

  for (const auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS,
                          TrafficGeneratorMode::DNS_TCP, TrafficGeneratorMode::EXTERNAL_HOST,
                          TrafficGeneratorMode::WIFI_RAW}) {
    const auto config = filter(mode);
    CsiAckCredit credit;
    // An ACK for the device's own traffic, such as the CSI stream, carries no credit.
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::LLTF20, credit));
    for (const auto profile : {CsiCaptureProfile::HT20, CsiCaptureProfile::VHT20}) {
      credit.arm(kNowUs);
      TEST_ASSERT_FALSE(matches_once(&info, config, profile, credit));
    }
    credit.arm(kNowUs);
    TEST_ASSERT_TRUE(matches_once(&info, config, CsiCaptureProfile::LLTF20, credit));
  }
}

void test_ack_credit_admits_one_ack_per_arm_within_its_window(void) {
  CsiAckCredit credit;
  TEST_ASSERT_FALSE(credit.consume(kNowUs));
  credit.arm(kNowUs);
  TEST_ASSERT_TRUE(credit.consume(kNowUs + CsiAckCredit::kWindowUs));
  TEST_ASSERT_FALSE(credit.consume(kNowUs + CsiAckCredit::kWindowUs));

  credit.arm(kNowUs);
  TEST_ASSERT_FALSE(credit.consume(kNowUs + CsiAckCredit::kWindowUs + 1U));
  TEST_ASSERT_FALSE(credit.consume(kNowUs - 1U));  // Read before the arm.
  credit.clear();
  TEST_ASSERT_FALSE(credit.consume(kNowUs));

  credit.arm(0U);
  TEST_ASSERT_TRUE(credit.consume(1U));
  credit.arm(0xFFFFFFF0U);
  TEST_ASSERT_TRUE(credit.consume(0x10U));  // The timer wraps between arm and ACK.
}

void test_retransmission_of_the_last_admitted_frame_is_dropped(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  const auto payload = udp_frame(kLocal, 5555U);
  uint8_t header[24] = {0x88U, 0x02U};
  wifi_csi_info_t info = csi_info(payload);
  info.hdr = header;
  info.rx_seq = 100U;
  std::memcpy(info.mac, kOtherMac, 6U);
  CsiAckCredit credit;
  CsiRetransmissionFilter retransmissions;
  const auto accepted = [&]() {
    return csi_frame_matches_traffic(&info, config, CsiCaptureProfile::HT20, kNowUs, credit,
                                     retransmissions);
  };

  // A retry is the only copy when the original never decoded.
  header[1] |= 0x08U;
  TEST_ASSERT_TRUE(accepted());
  TEST_ASSERT_FALSE(accepted());
  header[1] &= static_cast<uint8_t>(~0x08U);
  TEST_ASSERT_TRUE(accepted());  // Without the retry bit, a repeated number is a new frame.
  header[1] |= 0x08U;
  TEST_ASSERT_FALSE(accepted());
  info.rx_seq = 101U;
  TEST_ASSERT_TRUE(accepted());
  // Same number and transmitter but another length: a frame from another TID.
  info.rx_ctrl.sig_len = 120U;
  TEST_ASSERT_TRUE(accepted());
  TEST_ASSERT_FALSE(accepted());
  std::memcpy(info.mac, kMulticastMac, 6U);
  TEST_ASSERT_TRUE(accepted());  // Same number from another transmitter.

  // Rejected traffic never becomes the frame that later retries are compared to.
  auto other_port = udp_frame(kLocal, 5556U);
  info.payload = other_port.data();
  info.rx_seq = 102U;
  TEST_ASSERT_FALSE(accepted());
  info.payload = const_cast<uint8_t *>(payload.data());
  TEST_ASSERT_TRUE(accepted());
}

void test_lltf20_rejects_ack_without_valid_local_receiver_or_header(void) {
  auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  uint8_t header[10] = {0xD4U, 0U, 0U, 0U};
  std::memcpy(header + 4U, kLocalMac, 6U);
  wifi_csi_info_t info = local_ack_info(header);
  const auto accepted = [&]() {
    CsiAckCredit credit;
    credit.arm(kNowUs);
    return matches_once(&info, config, CsiCaptureProfile::LLTF20, credit);
  };
  TEST_ASSERT_TRUE(accepted());
  std::memcpy(header + 4U, kOtherMac, 6U);
  std::memcpy(info.dmac, kLocalMac, 6U);
  TEST_ASSERT_FALSE(accepted());
  std::memcpy(header + 4U, kLocalMac, 6U);
  std::memset(config.local_mac_addr, 0, 6U);
  TEST_ASSERT_FALSE(accepted());
  std::memcpy(config.local_mac_addr, kMulticastMac, 6U);
  std::memcpy(header + 4U, kMulticastMac, 6U);
  TEST_ASSERT_FALSE(accepted());
  std::memcpy(config.local_mac_addr, kLocalMac, 6U);
  std::memcpy(header + 4U, kLocalMac, 6U);
  for (const uint8_t control : {0xC4U, 0x94U, 0x80U, 0x08U, 0xD5U}) {
    header[0] = control;  // CTS, Block ACK, beacon, data, and invalid version.
    TEST_ASSERT_FALSE(accepted());
  }
  header[0] = 0xD4U;
  header[1] = 0x01U;
  TEST_ASSERT_FALSE(accepted());
  header[1] = 0U;
  for (const uint16_t length : {0U, 9U, 13U, 15U}) {
    info.rx_ctrl.sig_len = length;
    TEST_ASSERT_FALSE(accepted());
  }
  info.rx_ctrl.sig_len = sizeof(header) + 4U;
  info.rx_ctrl.rx_state = 1U;
  TEST_ASSERT_FALSE(accepted());
  info.rx_ctrl.rx_state = 0U;
  info.hdr = nullptr;
  TEST_ASSERT_FALSE(accepted());
  TEST_ASSERT_FALSE(matches_once(nullptr, config, CsiCaptureProfile::LLTF20));
}

void test_lltf20_preserves_ip_traffic_provenance_filter(void) {
  const auto config = filter(TrafficGeneratorMode::DNS_TCP);
  const auto reply = dns_tcp_reply(true);
  const auto tcp_ack = dns_tcp_reply(false);
  const auto unrelated = ping_reply(kGateway, 0x1234U);
  for (const auto *payload : {&reply, &tcp_ack, &unrelated}) {
    const auto info = csi_info(*payload);
    TEST_ASSERT_EQUAL(payload == &reply, matches_once(&info, config, CsiCaptureProfile::LLTF20));
  }
}

void test_external_accepts_only_canonical_unicast_and_multicast_marker(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  const auto unicast = udp_frame(kLocal, 5555U);
  const auto multicast = udp_frame(kMulticast, 5555U);
  TEST_ASSERT_TRUE(matches(unicast, config));
  TEST_ASSERT_TRUE(matches(multicast, config, kMulticastMac));

  const auto wrong_marker = udp_frame(kLocal, 5555U, {'.'});
  const auto truncated_marker = udp_frame(kLocal, 5555U, {0xF0U, 0x9FU, 0x91U});
  const auto malformed_marker = udp_frame(kLocal, 5555U, {0xF0U, 0x28U, 0x8CU, 0xBCU});
  auto marker_with_trailing_byte = canonical_marker();
  marker_with_trailing_byte.push_back('x');
  const auto oversized_marker = udp_frame(kLocal, 5555U, marker_with_trailing_byte);
  const auto wrong_port = udp_frame(kLocal, 5556U);
  const auto wrong_device = udp_frame(kOther, 5555U);
  TEST_ASSERT_FALSE(matches(wrong_marker, config));
  TEST_ASSERT_FALSE(matches(truncated_marker, config));
  TEST_ASSERT_FALSE(matches(malformed_marker, config));
  TEST_ASSERT_FALSE(matches(oversized_marker, config));
  TEST_ASSERT_FALSE(matches(wrong_port, config));
  TEST_ASSERT_FALSE(matches(wrong_device, config));
}

void test_external_rejects_other_mac_fragments_and_truncation(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  const auto valid = udp_frame(kLocal, 5555U);
  auto fragmented = valid;
  fragmented[8U + 6U] = 0x20U;
  auto truncated = valid;
  truncated.resize(20U);
  auto invalid_header = valid;
  invalid_header[8U] = 0x44U;
  auto incomplete_ip = valid;
  write_be16(incomplete_ip.data() + 8U + 2U,
             static_cast<uint16_t>(incomplete_ip.size()));
  TEST_ASSERT_FALSE(matches(valid, config, kOtherMac));
  TEST_ASSERT_FALSE(matches(fragmented, config));
  TEST_ASSERT_FALSE(matches(truncated, config));
  TEST_ASSERT_FALSE(matches(invalid_header, config));
  TEST_ASSERT_FALSE(matches(incomplete_ip, config));
  TEST_ASSERT_FALSE(matches(valid, config, kMulticastMac));
  TEST_ASSERT_FALSE(matches(udp_frame(kMulticast, 5555U), config, kOtherMac));
}

void test_external_accepts_multicast_to_local_mac_with_canonical_identity(void) {
  CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  const auto multicast = udp_frame(kMulticast, config.external_udp_port);
  TEST_ASSERT_TRUE(matches(multicast, config));
  TEST_ASSERT_FALSE(matches(multicast, config, kOtherMac));
  TEST_ASSERT_FALSE(matches(udp_frame(kMulticast, config.external_udp_port, {'.'}), config));
  TEST_ASSERT_FALSE(matches(udp_frame(kMulticast, config.external_udp_port + 1U), config));

  // Different multicast groups can map to the same Ethernet multicast MAC.
  const auto other_group = udp_frame(kMulticast ^ 0x01000000U, config.external_udp_port);
  TEST_ASSERT_FALSE(matches(other_group, config));
  TEST_ASSERT_FALSE(matches(other_group, config, kMulticastMac));

  config.multicast_ip_addr = 0U;
  TEST_ASSERT_FALSE(matches(multicast, config));
  TEST_ASSERT_FALSE(matches(multicast, config, kMulticastMac));
}

void test_external_rejects_udp_length_mismatch_and_data_after_marker(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  auto short_udp = udp_frame(kLocal, 5555U);
  write_be16(short_udp.data() + 8U + 20U + 4U, 8U);
  auto trailing = udp_frame(kLocal, 5555U);
  trailing.push_back('x');
  write_be16(trailing.data() + 8U + 2U,
             static_cast<uint16_t>(trailing.size() - 8U));
  TEST_ASSERT_FALSE(matches(short_udp, config));
  TEST_ASSERT_FALSE(matches(trailing, config));
}

void test_external_accepts_bounded_shifted_llc_frame(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  std::vector<uint8_t> shifted(31U, 0x48U);
  const auto valid = udp_frame(kLocal, 5555U);
  shifted.insert(shifted.end(), valid.begin(), valid.end());
  TEST_ASSERT_TRUE(matches(shifted, config));

  std::vector<uint8_t> beyond_scan(64U, 0x48U);
  beyond_scan.insert(beyond_scan.end(), valid.begin(), valid.end());
  TEST_ASSERT_FALSE(matches(beyond_scan, config));
}

void test_external_accepts_unicast_ping_requests(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  TEST_ASSERT_TRUE(matches(ping_request(kGateway), config));
  TEST_ASSERT_TRUE(matches(ping_request(kOther), config));
}

void test_external_accepts_direct_null_and_espnow_without_ip_identity(void) {
  auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  config.local_ip_addr = 0U;
  for (bool espnow : {false, true}) {
    for (uint8_t version : {1U, 2U}) {
      auto frame = radio_frame(espnow, version);
      auto info = radio_info(frame);
      for (auto profile : {CsiCaptureProfile::HT20, CsiCaptureProfile::LLTF20}) {
        TEST_ASSERT_TRUE(matches_once(&info, config, profile));
        for (auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS,
                          TrafficGeneratorMode::DNS_TCP, TrafficGeneratorMode::WIFI_RAW}) {
          TEST_ASSERT_FALSE(matches_once(&info, filter(mode), profile));
        }
      }
    }
  }
}

void test_external_radio_requires_direct_local_unicast_and_valid_header(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  for (bool espnow : {false, true}) {
    const auto original = radio_frame(espnow);
    for (const auto &change : std::vector<std::pair<size_t, uint8_t>>{
             {0U, 0x08U}, {0U, 0xC8U}, {0U, 0xD1U}, {0U, 0xE0U}, {1U, 0x01U}, {1U, 0x02U},
             {1U, 0x04U}, {1U, 0x40U}, {1U, 0x80U}, {22U, 1U},
             {4U, 0xFFU}, {4U, kOtherMac[0]}, {10U, 0xFFU}}) {
      auto frame = original;
      frame[change.first] = change.second;
      auto info = radio_info(frame);
      TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    }
    for (const auto *source : {kLocalMac, kZeroMac}) {
      auto frame = original;
      std::memcpy(frame.data() + 10U, source, 6U);
      auto info = radio_info(frame);
      TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    }
    auto frame = original;
    auto info = radio_info(frame);
    info.dmac[0] ^= 2U;
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    info = radio_info(frame);
    info.rx_ctrl.rx_state = 1U;
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    info = radio_info(frame);
    info.hdr = nullptr;
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    info = radio_info(frame);
    info.rx_ctrl.sig_len = 27U;
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
    auto no_mac = config;
    std::memset(no_mac.local_mac_addr, 0, 6U);
    info = radio_info(frame);
    TEST_ASSERT_FALSE(matches_once(&info, no_mac, CsiCaptureProfile::HT20));
  }
}

void check_external_padded_null(CsiCaptureProfile profile) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  for (size_t length : {TRAFFIC_NULL_DATA_FRAME_SIZE + 1U,
                        TRAFFIC_NULL_DATA_PEER_FRAME_SIZE,
                        TRAFFIC_NULL_DATA_PEER_FRAME_SIZE + 1U}) {
    auto frame = radio_frame();
    frame.resize(length, 0U);
    for (bool matching_phy : {false, true}) {
      auto info = radio_info(frame);
#if CONFIG_SOC_WIFI_HE_SUPPORT
      info.rx_ctrl.cur_bb_format = matching_phy ? (profile == CsiCaptureProfile::VHT20 ? RX_BB_FORMAT_VHT : RX_BB_FORMAT_HT) : RX_BB_FORMAT_11G;
#else
      info.rx_ctrl.sig_mode = matching_phy ? 1U : 0U;
#endif
      TEST_ASSERT_EQUAL(matching_phy && length == TRAFFIC_NULL_DATA_PEER_FRAME_SIZE,
                        matches_once(&info, config, profile));
    }
  }
  auto frame = radio_frame();
  frame.resize(TRAFFIC_NULL_DATA_PEER_FRAME_SIZE, 0U);
  auto valid = radio_info(frame);
#if CONFIG_SOC_WIFI_HE_SUPPORT
  valid.rx_ctrl.cur_bb_format = profile == CsiCaptureProfile::VHT20 ? RX_BB_FORMAT_VHT : RX_BB_FORMAT_HT;
#else
  valid.rx_ctrl.sig_mode = 1U;
#endif
  for (auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS,
                    TrafficGeneratorMode::DNS_TCP, TrafficGeneratorMode::WIFI_RAW}) {
    TEST_ASSERT_FALSE(matches_once(&valid, filter(mode), profile));
  }
  for (size_t i = TRAFFIC_NULL_DATA_FRAME_SIZE; i < frame.size(); ++i) {
    frame[i] = 1U;
    TEST_ASSERT_FALSE(matches_once(&valid, config, profile));
    frame[i] = 0U;
  }
  auto info = valid;
  info.payload_len = 1U;
  TEST_ASSERT_FALSE(matches_once(&info, config, profile));
  info = valid;
  info.payload = nullptr;
  TEST_ASSERT_FALSE(matches_once(&info, config, profile));
  info = valid;
  info.payload = frame.data();  // A driver exposing the MAC header must fail closed.
  TEST_ASSERT_FALSE(matches_once(&info, config, profile));
  info = valid;
  info.payload_len = frame.size();  // Some drivers include the MAC header in this length.
  TEST_ASSERT_TRUE(matches_once(&info, config, profile));
  info = valid;
#if CONFIG_SOC_WIFI_HE_SUPPORT
  info.rx_ctrl.second = 1U;
#else
  info.rx_ctrl.cwb = 1U;
#endif
  TEST_ASSERT_FALSE(matches_once(&info, config, profile));
  info = valid;
  info.dmac[0] ^= 2U;
  TEST_ASSERT_FALSE(matches_once(&info, config, profile));
  frame[1] = 1U;
  TEST_ASSERT_FALSE(matches_once(&valid, config, profile));
}

void test_external_padded_null_requires_ht20_or_vht20_and_exact_zero_padding(void) {
  check_external_padded_null(CsiCaptureProfile::HT20);
#if CONFIG_SOC_WIFI_HE_SUPPORT
  check_external_padded_null(CsiCaptureProfile::VHT20);
#endif
}

void test_external_espnow_rejects_malformed_or_truncated_vendor_actions(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  const auto original = radio_frame(true);
  for (const auto &change : std::vector<std::pair<size_t, uint8_t>>{
           {0U, 126U}, {1U, 0U}, {8U, 220U}, {9U, 4U}, {9U, 255U},
           {10U, 0U}, {13U, 3U}, {14U, 0U}, {14U, 3U}, {14U, 0x11U}, {14U, 0x12U},
           {14U, 0x22U}}) {
    auto frame = original;
    frame[24U + change.first] = change.second;
    auto info = radio_info(frame);
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  }
  for (size_t length = 24U; length < original.size(); ++length) {
    auto frame = original;
    frame.resize(length);
    auto info = radio_info(frame);
    TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  }
  auto frame = original;
  auto info = radio_info(frame);
  info.payload = nullptr;
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  info = radio_info(frame);
  ++info.rx_ctrl.sig_len;
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  frame.push_back(0U);
  info = radio_info(frame);
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
}

void test_external_espnow_v2_accepts_multiple_complete_elements(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  auto frame = radio_frame(true, 2U);
  const std::vector<uint8_t> second(frame.begin() + 32U, frame.end());
  frame[38U] |= 0x10U;
  frame.insert(frame.end(), second.begin(), second.end());
  auto info = radio_info(frame);
  TEST_ASSERT_TRUE(matches_once(&info, config, CsiCaptureProfile::HT20));
  frame.back() = 0U;  // Application bytes are not interpreted as control messages.
  TEST_ASSERT_TRUE(matches_once(&info, config, CsiCaptureProfile::HT20));
  frame[38U] = 2U;  // A further element requires the v2 More Data bit.
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  frame[38U] = 0x12U;
  frame[46U] = 1U;  // Versions must not change within a v2 frame.
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
  frame[46U] = 2U;
  frame[40U] = 0U;
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));

  // The largest v2 frame carries 1470 data bytes: five 250-byte elements and
  // one 220-byte element. One more data byte exceeds the protocol limit.
  for (const size_t last_body : {220U, 221U}) {
    auto largest = radio_frame(true, 2U);
    largest.resize(32U);
    for (size_t element = 0U; element < 6U; ++element) {
      const size_t body = element < 5U ? 250U : last_body;
      const uint8_t control = element < 5U ? 0x12U : 0x02U;
      const uint8_t head[] = {221U, static_cast<uint8_t>(5U + body), 0x18U, 0xFEU, 0x34U, 4U,
                              control};
      largest.insert(largest.end(), std::begin(head), std::end(head));
      largest.insert(largest.end(), body, 0x5AU);
    }
    auto largest_info = radio_info(largest);
    TEST_ASSERT_EQUAL(last_body == 220U,
                      matches_once(&largest_info, config, CsiCaptureProfile::HT20));
  }
}

void test_external_espnow_bounds_driver_payload_length_by_mpdu_length(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  auto frame = radio_frame(true, 2U);
  auto info = radio_info(frame);
  // On an ESP32-S3 receiver, ESP-IDF 5.5.5 advertises the entire MPDU length
  // while payload points at the action body. The bytes after the real body are
  // not part of this frame.
  info.payload_len = info.rx_ctrl.sig_len;
  TEST_ASSERT_TRUE(matches_once(&info, config, CsiCaptureProfile::HT20));
  info.payload_len = static_cast<uint16_t>(frame.size() - 25U);
  TEST_ASSERT_FALSE(matches_once(&info, config, CsiCaptureProfile::HT20));
}

void test_external_radio_retries_are_counted_once_and_do_not_consume_ack_credit(void) {
  const auto config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  for (bool espnow : {false, true}) {
    auto frame = radio_frame(espnow);
    auto info = radio_info(frame);
    CsiRetransmissionFilter retransmissions;
    CsiAckCredit credit;
    credit.arm(kNowUs);
    TEST_ASSERT_TRUE(csi_frame_matches_traffic(&info, config, CsiCaptureProfile::LLTF20,
                                               kNowUs, credit, retransmissions));
    frame[1U] |= 0x08U;
    TEST_ASSERT_FALSE(csi_frame_matches_traffic(&info, config, CsiCaptureProfile::LLTF20,
                                                kNowUs, credit, retransmissions));
    ++info.rx_seq;
    TEST_ASSERT_TRUE(csi_frame_matches_traffic(&info, config, CsiCaptureProfile::LLTF20,
                                               kNowUs, credit, retransmissions));
    TEST_ASSERT_TRUE(credit.consume(kNowUs));
  }
}

void test_external_rejects_other_icmp_traffic(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::EXTERNAL_HOST);
  TEST_ASSERT_FALSE(matches(ping_reply(kOther, 0x4321U), config));
  TEST_ASSERT_FALSE(matches(ping_request(kOther, kLocal, 1U), config));
  TEST_ASSERT_FALSE(matches(ping_request(kOther, kOther), config));
  TEST_ASSERT_FALSE(matches(ping_request(kOther), config, kOtherMac));
  TEST_ASSERT_FALSE(matches(ping_request(kOther, kMulticast), config, kMulticastMac));
}

void test_internal_ping_requires_gateway_echo_reply_and_active_identifier(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::PING);
  const auto valid = ping_reply(kGateway, 0x1234U);
  const auto wrong_gateway = ping_reply(kOther, 0x1234U);
  const auto wrong_identifier = ping_reply(kGateway, 0x1235U);
  auto echo_request = valid;
  echo_request[8U + 20U] = 8U;
  TEST_ASSERT_TRUE(matches(valid, config));
  TEST_ASSERT_FALSE(matches(wrong_gateway, config));
  TEST_ASSERT_FALSE(matches(wrong_identifier, config));
  TEST_ASSERT_FALSE(matches(echo_request, config));
}

void test_internal_dns_requires_gateway_tcp_53_payload_and_rejects_ack_only(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::DNS_TCP);
  const auto valid = dns_tcp_reply(true);
  const auto ack_only = dns_tcp_reply(false);
  auto length_mismatch = dns_tcp_reply(true);
  length_mismatch[8U + 20U + 21U] = 11U;
  auto query = dns_tcp_reply(true);
  query[8U + 20U + 24U] = 0x01U;
  auto http = dns_tcp_reply(true);
  write_be16(http.data() + 8U + 20U, 80U);
  TEST_ASSERT_TRUE(matches(valid, config));
  TEST_ASSERT_FALSE(matches(ack_only, config));
  TEST_ASSERT_FALSE(matches(length_mismatch, config));
  TEST_ASSERT_FALSE(matches(query, config));
  TEST_ASSERT_FALSE(matches(http, config));
}

void test_internal_dns_udp_requires_gateway_udp_53_response(void) {
  const CsiFrameFilterConfig config = filter(TrafficGeneratorMode::DNS);
  const auto valid = dns_udp_reply();
  auto wrong_length = valid;
  write_be16(wrong_length.data() + 8U + 20U + 4U, 19U);
  auto query = valid;
  query[8U + 20U + 10U] = 0x01U;
  auto wrong_port = valid;
  write_be16(wrong_port.data() + 8U + 20U, 54U);
  auto wrong_gateway = valid;
  write_be32(wrong_gateway.data() + 8U + 12U, kOther);
  TEST_ASSERT_TRUE(matches(valid, config));
  TEST_ASSERT_FALSE(matches(wrong_length, config));
  TEST_ASSERT_FALSE(matches(query, config));
  TEST_ASSERT_FALSE(matches(wrong_port, config));
  TEST_ASSERT_FALSE(matches(wrong_gateway, config));
  TEST_ASSERT_FALSE(matches(dns_tcp_reply(true), config));
}

void test_internal_ip_modes_accept_only_the_configured_target(void) {
  for (const auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS, TrafficGeneratorMode::DNS_TCP}) {
    auto config = filter(mode);
    auto reply = mode == TrafficGeneratorMode::PING ? ping_reply(kGateway, 0x1234U)
                 : mode == TrafficGeneratorMode::DNS ? dns_udp_reply() : dns_tcp_reply(true);
    config.internal_target_ip_addr = inet_addr("192.168.1.77");
    TEST_ASSERT_FALSE(matches(reply, config));
    write_be32(reply.data() + 8U + 12U, 0xC0A8014DU);
    TEST_ASSERT_TRUE(matches(reply, config));
    write_be32(reply.data() + 8U + 12U, kOther);
    TEST_ASSERT_FALSE(matches(reply, config));
  }
}

int main() {
  using namespace espectre::test;
  begin_suite();
  RUN_TEST(test_local_ack_requires_lltf20_and_a_generator_credit_in_every_traffic_mode);
  RUN_TEST(test_ack_credit_admits_one_ack_per_arm_within_its_window);
  RUN_TEST(test_retransmission_of_the_last_admitted_frame_is_dropped);
  RUN_TEST(test_lltf20_rejects_ack_without_valid_local_receiver_or_header);
  RUN_TEST(test_lltf20_preserves_ip_traffic_provenance_filter);
  RUN_TEST(test_external_accepts_only_canonical_unicast_and_multicast_marker);
  RUN_TEST(test_external_rejects_other_mac_fragments_and_truncation);
  RUN_TEST(test_external_accepts_multicast_to_local_mac_with_canonical_identity);
  RUN_TEST(test_external_rejects_udp_length_mismatch_and_data_after_marker);
  RUN_TEST(test_external_accepts_bounded_shifted_llc_frame);
  RUN_TEST(test_external_accepts_unicast_ping_requests);
  RUN_TEST(test_external_accepts_direct_null_and_espnow_without_ip_identity);
  RUN_TEST(test_external_radio_requires_direct_local_unicast_and_valid_header);
  RUN_TEST(test_external_padded_null_requires_ht20_or_vht20_and_exact_zero_padding);
  RUN_TEST(test_external_espnow_rejects_malformed_or_truncated_vendor_actions);
  RUN_TEST(test_external_espnow_v2_accepts_multiple_complete_elements);
  RUN_TEST(test_external_espnow_bounds_driver_payload_length_by_mpdu_length);
  RUN_TEST(test_external_radio_retries_are_counted_once_and_do_not_consume_ack_credit);
  RUN_TEST(test_external_rejects_other_icmp_traffic);
  RUN_TEST(test_internal_ping_requires_gateway_echo_reply_and_active_identifier);
  RUN_TEST(test_internal_dns_requires_gateway_tcp_53_payload_and_rejects_ack_only);
  RUN_TEST(test_internal_dns_udp_requires_gateway_udp_53_response);
  RUN_TEST(test_internal_ip_modes_accept_only_the_configured_target);
  return end_suite();
}
