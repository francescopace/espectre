/*
 * ESPectre - CSI Frame Identity
 *
 * Matches CSI frames against the local device identity when filtering
 * traffic.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "csi_frame_identity.h"

#include <algorithm>
#include <cstring>

#include "lwip/inet.h"

namespace espectre {

namespace {

constexpr size_t kLlcSnapHeaderBytes = 8U;
constexpr uint16_t kEtherTypeIpv4 = 0x0800U;
constexpr uint8_t kIpProtoUdp = 17U;
constexpr uint8_t kLlcSnapPrefix[6] = {0xAAU, 0xAAU, 0x03U, 0x00U, 0x00U, 0x00U};

uint32_t read_be32(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24U) | (static_cast<uint32_t>(data[1]) << 16U) |
         (static_cast<uint32_t>(data[2]) << 8U) | static_cast<uint32_t>(data[3]);
}

bool local_ip_matches(uint32_t local_ip_addr, uint32_t packet_dst_ipv4) {
  if (local_ip_addr == 0U || packet_dst_ipv4 == ntohl(local_ip_addr)) {
    return true;
  }
  return (packet_dst_ipv4 >> 28) == 0xEU || packet_dst_ipv4 == 0xFFFFFFFFU;
}

bool is_broadcast_mac(const uint8_t *mac) {
  if (mac == nullptr) {
    return false;
  }
  for (size_t i = 0; i < 6U; i++) {
    if (mac[i] != 0xFFU) {
      return false;
    }
  }
  return true;
}

bool is_zero_mac(const uint8_t *mac) {
  if (mac == nullptr) {
    return true;
  }
  for (size_t i = 0; i < 6U; i++) {
    if (mac[i] != 0U) {
      return false;
    }
  }
  return true;
}

bool destination_mac_matches(const uint8_t *local_mac_addr, const uint8_t *frame_dmac) {
  if (frame_dmac == nullptr) {
    return true;
  }
  if (is_broadcast_mac(frame_dmac) || (frame_dmac[0] & 0x01U) != 0U) {
    return true;
  }
  if (is_zero_mac(local_mac_addr)) {
    return true;
  }
  return std::memcmp(local_mac_addr, frame_dmac, 6U) == 0;
}

bool parse_ipv4_udp_from_llc_snap(const uint8_t *payload,
                                  size_t payload_len,
                                  uint32_t *out_src_ipv4,
                                  uint32_t *out_dst_ipv4) {
  if (payload == nullptr || out_src_ipv4 == nullptr || out_dst_ipv4 == nullptr || payload_len < kLlcSnapHeaderBytes + 20U + 8U) {
    return false;
  }
  if (payload[0] != 0xAAU || payload[1] != 0xAAU || payload[2] != 0x03U || payload[3] != 0x00U ||
      payload[4] != 0x00U || payload[5] != 0x00U ||
      ((static_cast<uint16_t>(payload[6]) << 8U) | static_cast<uint16_t>(payload[7])) != kEtherTypeIpv4) {
    return false;
  }

  const uint8_t *ip = payload + kLlcSnapHeaderBytes;
  const size_t ip_len = payload_len - kLlcSnapHeaderBytes;
  if (ip_len < 20U || (ip[0] >> 4U) != 4U || ip[9] != kIpProtoUdp) {
    return false;
  }

  const size_t ip_header_len = static_cast<size_t>(ip[0] & 0x0FU) * 4U;
  if (ip_header_len < 20U || ip_len < ip_header_len + 8U) {
    return false;
  }

  const uint16_t fragment_field = static_cast<uint16_t>((static_cast<uint16_t>(ip[6]) << 8U) | static_cast<uint16_t>(ip[7]));
  if ((fragment_field & 0x3FFFU) != 0U) {
    return false;
  }

  *out_src_ipv4 = read_be32(ip + 12U);
  *out_dst_ipv4 = read_be32(ip + 16U);
  return true;
}

}  // namespace

bool csi_frame_matches_local_identity(const wifi_csi_info_t *info,
                                      uint32_t local_ip_addr,
                                      const uint8_t *local_mac_addr) {
  if (info == nullptr) {
    return false;
  }

  if (!destination_mac_matches(local_mac_addr, info->dmac)) {
    return false;
  }

  if (info->payload == nullptr || info->payload_len == 0U) {
    return true;
  }

  const uint8_t *payload = reinterpret_cast<const uint8_t *>(info->payload);
  uint32_t src_ipv4 = 0U;
  uint32_t dst_ipv4 = 0U;
  if (parse_ipv4_udp_from_llc_snap(payload, info->payload_len, &src_ipv4, &dst_ipv4)) {
    (void)src_ipv4;
    return local_ip_matches(local_ip_addr, dst_ipv4);
  }

  const size_t scan_limit = info->payload_len;
  for (size_t offset = 1U; offset < scan_limit; offset++) {
    if (info->payload_len - offset < sizeof(kLlcSnapPrefix)) {
      continue;
    }
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_ipv4_udp_from_llc_snap(payload + offset, info->payload_len - offset, &src_ipv4, &dst_ipv4)) {
      (void)src_ipv4;
      return local_ip_matches(local_ip_addr, dst_ipv4);
    }
  }

  return true;
}

}  // namespace espectre
