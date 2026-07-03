#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_wifi.h"

namespace esphome {
namespace espectre {

struct StimulusMetadata {
  uint32_t stimulus_id{0U};
  bool is_reference{false};
  uint32_t src_ipv4_addr{0U};  // host byte order when available
  uint32_t dst_ipv4_addr{0U};  // host byte order when available
};

constexpr uint8_t STIMULUS_VERSION = 1U;
constexpr uint8_t STIMULUS_ROLE_MEASUREMENT = 0U;
constexpr uint8_t STIMULUS_ROLE_REFERENCE = 1U;
constexpr size_t STIMULUS_HEADER_BYTES = 10U;

bool csi_frame_matches_local_identity(const wifi_csi_info_t *info,
                                      uint32_t local_ip_addr,
                                      const uint8_t *local_mac_addr);
bool parse_stimulus_datagram(const uint8_t *payload, size_t payload_len, StimulusMetadata *metadata);
bool parse_stimulus_from_llc_snap(const uint8_t *payload, size_t payload_len, StimulusMetadata *metadata);
bool extract_stimulus_metadata_from_payload(const uint8_t *payload,
                                            size_t payload_len,
                                            uint32_t collector_ip_addr,
                                            uint32_t local_ip_addr,
                                            StimulusMetadata *metadata);
bool extract_stimulus_metadata_from_csi(const wifi_csi_info_t *info,
                                        uint32_t collector_ip_addr,
                                        uint32_t local_ip_addr,
                                        const uint8_t *local_mac_addr,
                                        StimulusMetadata *metadata);

}  // namespace espectre
}  // namespace esphome
