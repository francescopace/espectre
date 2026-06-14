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

}  // namespace

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
  return true;
}

bool extract_stimulus_metadata_from_payload(const uint8_t *payload,
                                            size_t payload_len,
                                            uint32_t collector_ip_addr,
                                            StimulusMetadata *metadata) {
  if (payload == nullptr || metadata == nullptr || payload_len == 0U) {
    return false;
  }

  if (parse_stimulus_datagram(payload, payload_len, metadata)) {
    return true;
  }

  if (parse_stimulus_from_llc_snap(payload, payload_len, metadata)) {
    return collector_matches(collector_ip_addr, metadata->src_ipv4_addr);
  }

  const size_t scan_limit = std::min<size_t>(payload_len, 32U);
  for (size_t offset = 1U; offset + sizeof(kLlcSnapPrefix) <= scan_limit; offset++) {
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_stimulus_from_llc_snap(payload + offset, payload_len - offset, metadata)) {
      return collector_matches(collector_ip_addr, metadata->src_ipv4_addr);
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
                                        StimulusMetadata *metadata) {
  if (info == nullptr || metadata == nullptr || info->payload == nullptr || info->payload_len == 0U) {
    return false;
  }

  return extract_stimulus_metadata_from_payload(reinterpret_cast<const uint8_t *>(info->payload),
                                                info->payload_len,
                                                collector_ip_addr,
                                                metadata);
}

}  // namespace espectre
}  // namespace esphome
