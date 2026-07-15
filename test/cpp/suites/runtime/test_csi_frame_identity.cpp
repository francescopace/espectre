/*
 * ESPectre - CSI Frame Identity Unit Tests
 *
 * Unit tests for CSI Frame Identity.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "csi_frame_identity.h"
#include "lwip/inet.h"

using namespace espectre;

namespace {

constexpr uint8_t kCollectorMac[6] = {0x7C, 0x2C, 0x67, 0x42, 0xBB, 0xAC};
constexpr uint8_t kLocalMac[6] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
constexpr uint8_t kOtherMac[6] = {0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
constexpr size_t kUdpPacingPayloadBytes = 1U;
constexpr size_t kLlcSnapUdpFrameBytes = 8U + 20U + 8U + kUdpPacingPayloadBytes;
constexpr size_t kShiftedCapturePrefixBytes = 32U + kLlcSnapUdpFrameBytes;

std::array<uint8_t, kLlcSnapUdpFrameBytes> build_llc_snap_payload_(uint32_t src_ip_host_order, uint32_t dst_ip_host_order) {
  std::array<uint8_t, kLlcSnapUdpFrameBytes> payload{};
  payload[0] = 0xAAU;
  payload[1] = 0xAAU;
  payload[2] = 0x03U;
  payload[3] = 0x00U;
  payload[4] = 0x00U;
  payload[5] = 0x00U;
  payload[6] = 0x08U;
  payload[7] = 0x00U;

  uint8_t *ip = payload.data() + 8U;
  ip[0] = 0x45U;
  ip[2] = 0x00U;
  ip[3] = static_cast<uint8_t>(20U + 8U + kUdpPacingPayloadBytes);
  ip[8] = 64U;
  ip[9] = 17U;
  ip[12] = static_cast<uint8_t>((src_ip_host_order >> 24U) & 0xFFU);
  ip[13] = static_cast<uint8_t>((src_ip_host_order >> 16U) & 0xFFU);
  ip[14] = static_cast<uint8_t>((src_ip_host_order >> 8U) & 0xFFU);
  ip[15] = static_cast<uint8_t>(src_ip_host_order & 0xFFU);
  ip[16] = static_cast<uint8_t>((dst_ip_host_order >> 24U) & 0xFFU);
  ip[17] = static_cast<uint8_t>((dst_ip_host_order >> 16U) & 0xFFU);
  ip[18] = static_cast<uint8_t>((dst_ip_host_order >> 8U) & 0xFFU);
  ip[19] = static_cast<uint8_t>(dst_ip_host_order & 0xFFU);
  payload[8U + 20U + 8U] = 0xA5U;
  return payload;
}

std::array<uint8_t, kShiftedCapturePrefixBytes> build_shifted_llc_snap_payload_(uint32_t src_ip_host_order,
                                                                                uint32_t dst_ip_host_order) {
  std::array<uint8_t, kShiftedCapturePrefixBytes> payload{};
  const auto llc_payload = build_llc_snap_payload_(src_ip_host_order, dst_ip_host_order);
  constexpr size_t kShiftBytes = 31U;
  static_assert(kShiftBytes + kLlcSnapUdpFrameBytes <= kShiftedCapturePrefixBytes, "shifted LLC payload must fit");
  std::memcpy(payload.data() + kShiftBytes, llc_payload.data(), llc_payload.size());
  return payload;
}

wifi_csi_info_t build_csi_info_(const uint8_t *payload,
                                size_t payload_len,
                                const uint8_t *src_mac,
                                const uint8_t *dst_mac) {
  wifi_csi_info_t info{};
  info.payload = const_cast<uint8_t *>(payload);
  info.payload_len = static_cast<uint16_t>(payload_len);
  if (src_mac != nullptr) {
    std::memcpy(info.mac, src_mac, 6U);
  }
  if (dst_mac != nullptr) {
    std::memcpy(info.dmac, dst_mac, 6U);
  }
  return info;
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_csi_frame_matches_local_identity_accepts_llc_snap_for_local_ip(void) {
  const uint32_t local_ip = inet_addr("192.168.1.17");
  const auto payload = build_llc_snap_payload_(0xC0A80116U, 0xC0A80111U);
  const wifi_csi_info_t info = build_csi_info_(payload.data(), payload.size(), kCollectorMac, kLocalMac);

  TEST_ASSERT_TRUE(csi_frame_matches_local_identity(&info, local_ip, kLocalMac));
}

void test_csi_frame_matches_local_identity_accepts_shifted_llc_snap_with_capture_prefix(void) {
  const uint32_t local_ip = inet_addr("192.168.1.17");
  const auto payload = build_shifted_llc_snap_payload_(0xC0A80116U, 0xC0A80111U);
  const wifi_csi_info_t info = build_csi_info_(payload.data(), payload.size(), kCollectorMac, kLocalMac);

  TEST_ASSERT_TRUE(csi_frame_matches_local_identity(&info, local_ip, kLocalMac));
}

void test_csi_frame_matches_local_identity_rejects_other_unicast_mac(void) {
  const auto payload = build_llc_snap_payload_(0xC0A80116U, 0xC0A80111U);
  const wifi_csi_info_t info = build_csi_info_(payload.data(), payload.size(), kCollectorMac, kOtherMac);

  TEST_ASSERT_FALSE(csi_frame_matches_local_identity(&info, inet_addr("192.168.1.17"), kLocalMac));
}

void test_csi_frame_matches_local_identity_accepts_broadcast_mac(void) {
  const uint8_t broadcast_mac[6] = {0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU};
  const auto payload = build_llc_snap_payload_(0xC0A80116U, 0xC0A80111U);
  const wifi_csi_info_t info = build_csi_info_(payload.data(), payload.size(), kCollectorMac, broadcast_mac);

  TEST_ASSERT_TRUE(csi_frame_matches_local_identity(&info, inet_addr("192.168.1.17"), kLocalMac));
}

void test_csi_frame_matches_local_identity_rejects_llc_snap_for_other_ip(void) {
  const uint32_t local_ip = inet_addr("192.168.1.17");
  const auto payload = build_llc_snap_payload_(0xC0A80116U, 0xC0A80118U);
  const wifi_csi_info_t info = build_csi_info_(payload.data(), payload.size(), kCollectorMac, kOtherMac);

  TEST_ASSERT_FALSE(csi_frame_matches_local_identity(&info, local_ip, kLocalMac));
}

void test_csi_frame_matches_local_identity_accepts_unparsed_payload(void) {
  const uint8_t payload[3] = {0x01U, 0x02U, 0x03U};
  const wifi_csi_info_t info = build_csi_info_(payload, sizeof(payload), kCollectorMac, kLocalMac);

  TEST_ASSERT_TRUE(csi_frame_matches_local_identity(&info, inet_addr("192.168.1.17"), kLocalMac));
}

int main() {
  using namespace espectre::test;
  begin_suite();
  RUN_TEST(test_csi_frame_matches_local_identity_rejects_other_unicast_mac);
  RUN_TEST(test_csi_frame_matches_local_identity_accepts_broadcast_mac);
  RUN_TEST(test_csi_frame_matches_local_identity_accepts_llc_snap_for_local_ip);
  RUN_TEST(test_csi_frame_matches_local_identity_accepts_shifted_llc_snap_with_capture_prefix);
  RUN_TEST(test_csi_frame_matches_local_identity_rejects_llc_snap_for_other_ip);
  RUN_TEST(test_csi_frame_matches_local_identity_accepts_unparsed_payload);
  return end_suite();
}
