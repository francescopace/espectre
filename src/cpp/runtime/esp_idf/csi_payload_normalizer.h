/*
 * ESPectre - CSI Payload Normalizer
 *
 * Shared helpers for normalizing ESP-IDF CSI payloads to the internal HT20
 * layout expected by ESPectre components and streamers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace esphome {
namespace espectre {

enum class NormalizedCSIPayloadTag : uint8_t {
  NONE = 0,
  DOUBLE_HT20,
  HT57_TO_64,
  DOUBLE_HT57_TO_64,
};

struct NormalizedCSIPayload {
  const int8_t *data{nullptr};
  size_t len{0};
  NormalizedCSIPayloadTag tag{NormalizedCSIPayloadTag::NONE};

  bool valid() const { return data != nullptr; }
};

NormalizedCSIPayload normalize_ht20_csi_payload(const int8_t *csi_data,
                                                size_t csi_len,
                                                int8_t *remap_buffer,
                                                size_t remap_buffer_len);

const char *normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag tag);

}  // namespace espectre
}  // namespace esphome
