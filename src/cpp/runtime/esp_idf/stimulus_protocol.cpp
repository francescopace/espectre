#include "stimulus_protocol.h"

#include <algorithm>
#include <cstring>

#include "lwip/inet.h"

namespace esphome {
namespace espectre {

namespace {

constexpr uint8_t kStimulusMagic[4] = {'E', 'S', 'T', 'M'};
constexpr size_t kLlcSnapHeaderBytes = 8U;
constexpr uint16_t kEtherTypeIpv4 = 0x0800U;
constexpr uint8_t kIpProtoUdp = 17U;
constexpr uint8_t kLlcSnapPrefix[6] = {0xAAU, 0xAAU, 0x03U, 0x00U, 0x00U, 0x00U};

uint16_t read_be16(const uint8_t *data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8U) | static_cast<uint16_t>(data[1]));
}

uint32_t read_be32(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24U) | (static_cast<uint32_t>(data[1]) << 16U) |
         (static_cast<uint32_t>(data[2]) << 8U) | static_cast<uint32_t>(data[3]);
}

bool collector_matches(uint32_t collector_ip_addr, uint32_t packet_src_ipv4) {
  return collector_ip_addr == 0U || packet_src_ipv4 == ntohl(collector_ip_addr);
}

bool local_ip_matches(uint32_t local_ip_addr, uint32_t packet_dst_ipv4) {
  return local_ip_addr == 0U || packet_dst_ipv4 == ntohl(local_ip_addr);
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

  StimulusMetadata metadata{};
  const uint8_t *payload = reinterpret_cast<const uint8_t *>(info->payload);
  if (parse_stimulus_from_llc_snap(payload, info->payload_len, &metadata)) {
    return local_ip_matches(local_ip_addr, metadata.dst_ipv4_addr);
  }

  const size_t scan_limit = std::min<size_t>(info->payload_len, 32U);
  for (size_t offset = 1U; offset + sizeof(kLlcSnapPrefix) <= scan_limit; offset++) {
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_stimulus_from_llc_snap(payload + offset, info->payload_len - offset, &metadata)) {
      return local_ip_matches(local_ip_addr, metadata.dst_ipv4_addr);
    }
  }

  return true;
}

bool parse_stimulus_datagram(const uint8_t *payload, size_t payload_len, StimulusMetadata *metadata) {
  if (payload == nullptr || metadata == nullptr || payload_len < STIMULUS_HEADER_BYTES) {
    return false;
  }
  if (std::memcmp(payload, kStimulusMagic, sizeof(kStimulusMagic)) != 0 || payload[4] != STIMULUS_VERSION) {
    return false;
  }
  if (payload[5] != STIMULUS_ROLE_MEASUREMENT && payload[5] != STIMULUS_ROLE_REFERENCE) {
    return false;
  }

  metadata->stimulus_id = read_be32(payload + 6U);
  metadata->is_reference = (payload[5] == STIMULUS_ROLE_REFERENCE);
  metadata->src_ipv4_addr = 0U;
  metadata->dst_ipv4_addr = 0U;
  return true;
}

bool parse_stimulus_from_llc_snap(const uint8_t *payload, size_t payload_len, StimulusMetadata *metadata) {
  if (payload == nullptr || metadata == nullptr ||
      payload_len < kLlcSnapHeaderBytes + 20U + 8U + STIMULUS_HEADER_BYTES) {
    return false;
  }
  if (payload[0] != 0xAAU || payload[1] != 0xAAU || payload[2] != 0x03U || payload[3] != 0x00U ||
      payload[4] != 0x00U || payload[5] != 0x00U || read_be16(payload + 6U) != kEtherTypeIpv4) {
    return false;
  }

  const uint8_t *ip = payload + kLlcSnapHeaderBytes;
  const size_t ip_len = payload_len - kLlcSnapHeaderBytes;
  if (ip_len < 20U || (ip[0] >> 4U) != 4U || ip[9] != kIpProtoUdp) {
    return false;
  }

  const size_t ip_header_len = static_cast<size_t>(ip[0] & 0x0FU) * 4U;
  if (ip_header_len < 20U || ip_len < ip_header_len + 8U + STIMULUS_HEADER_BYTES) {
    return false;
  }

  const uint16_t fragment_field = read_be16(ip + 6U);
  if ((fragment_field & 0x3FFFU) != 0U) {
    return false;
  }

  if (!parse_stimulus_datagram(ip + ip_header_len + 8U, ip_len - ip_header_len - 8U, metadata)) {
    return false;
  }

  metadata->src_ipv4_addr = read_be32(ip + 12U);
  metadata->dst_ipv4_addr = read_be32(ip + 16U);
  return true;
}

bool extract_stimulus_metadata_from_payload(const uint8_t *payload,
                                            size_t payload_len,
                                            uint32_t collector_ip_addr,
                                            uint32_t local_ip_addr,
                                            StimulusMetadata *metadata) {
  if (payload == nullptr || metadata == nullptr || payload_len == 0U) {
    return false;
  }

  if (parse_stimulus_datagram(payload, payload_len, metadata)) {
    return true;
  }

  if (parse_stimulus_from_llc_snap(payload, payload_len, metadata)) {
    return collector_matches(collector_ip_addr, metadata->src_ipv4_addr) &&
           local_ip_matches(local_ip_addr, metadata->dst_ipv4_addr);
  }

  const size_t scan_limit = std::min<size_t>(payload_len, 32U);
  for (size_t offset = 1U; offset + sizeof(kLlcSnapPrefix) <= scan_limit; offset++) {
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_stimulus_from_llc_snap(payload + offset, payload_len - offset, metadata)) {
      return collector_matches(collector_ip_addr, metadata->src_ipv4_addr) &&
             local_ip_matches(local_ip_addr, metadata->dst_ipv4_addr);
    }
  }

  const size_t magic_scan_limit = std::min<size_t>(payload_len, 96U);
  for (size_t offset = 1U; offset + STIMULUS_HEADER_BYTES <= magic_scan_limit; offset++) {
    if (std::memcmp(payload + offset, kStimulusMagic, sizeof(kStimulusMagic)) != 0) {
      continue;
    }
    if (parse_stimulus_datagram(payload + offset, payload_len - offset, metadata)) {
      return true;
    }
  }

  return false;
}

bool extract_stimulus_metadata_from_csi(const wifi_csi_info_t *info,
                                        uint32_t collector_ip_addr,
                                        uint32_t local_ip_addr,
                                        const uint8_t *local_mac_addr,
                                        StimulusMetadata *metadata) {
  if (info == nullptr || metadata == nullptr || info->payload == nullptr || info->payload_len == 0U) {
    return false;
  }
  if (!csi_frame_matches_local_identity(info, local_ip_addr, local_mac_addr)) {
    return false;
  }

  const uint8_t *payload = reinterpret_cast<const uint8_t *>(info->payload);
  const size_t payload_len = info->payload_len;

  if (parse_stimulus_datagram(payload, payload_len, metadata)) {
    return destination_mac_matches(local_mac_addr, info->dmac);
  }

  if (parse_stimulus_from_llc_snap(payload, payload_len, metadata)) {
    return collector_matches(collector_ip_addr, metadata->src_ipv4_addr) &&
           local_ip_matches(local_ip_addr, metadata->dst_ipv4_addr);
  }

  const size_t scan_limit = std::min<size_t>(payload_len, 32U);
  for (size_t offset = 1U; offset + sizeof(kLlcSnapPrefix) <= scan_limit; offset++) {
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_stimulus_from_llc_snap(payload + offset, payload_len - offset, metadata)) {
      return collector_matches(collector_ip_addr, metadata->src_ipv4_addr) &&
             local_ip_matches(local_ip_addr, metadata->dst_ipv4_addr);
    }
  }

  const size_t magic_scan_limit = std::min<size_t>(payload_len, 96U);
  for (size_t offset = 1U; offset + STIMULUS_HEADER_BYTES <= magic_scan_limit; offset++) {
    if (std::memcmp(payload + offset, kStimulusMagic, sizeof(kStimulusMagic)) != 0) {
      continue;
    }
    if (parse_stimulus_datagram(payload + offset, payload_len - offset, metadata)) {
      return destination_mac_matches(local_mac_addr, info->dmac);
    }
  }

  return false;
}

}  // namespace espectre
}  // namespace esphome
