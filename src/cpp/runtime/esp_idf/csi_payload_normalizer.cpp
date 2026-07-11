/*
 * ESPectre - CSI Payload Normalizer
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_payload_normalizer.h"

#include <cstring>

#include "csi_format.h"

namespace espectre {

NormalizedCSIPayload normalize_ht20_csi_payload(const int8_t *csi_data,
                                                size_t csi_len,
                                                int8_t *remap_buffer,
                                                size_t remap_buffer_len) {
  if (csi_data == nullptr) {
    return {};
  }

  NormalizedCSIPayloadTag tag = NormalizedCSIPayloadTag::NONE;
  if (csi_len == HT20_CSI_LEN_DOUBLE) {
    csi_len = HT20_CSI_LEN;
    tag = NormalizedCSIPayloadTag::DOUBLE_HT20;
  } else if (csi_len == HT20_CSI_LEN_SHORT_DOUBLE) {
    csi_len = HT20_CSI_LEN_SHORT;
    tag = NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64;
  }

  if (csi_len == HT20_CSI_LEN) {
    return {csi_data, HT20_CSI_LEN, tag};
  }

  if (csi_len != HT20_CSI_LEN_SHORT || remap_buffer == nullptr || remap_buffer_len < HT20_CSI_LEN) {
    return {};
  }

  std::memset(remap_buffer, 0, HT20_CSI_LEN);
  std::memcpy(&remap_buffer[HT20_CSI_LEN_SHORT_LEFT_PAD], csi_data, HT20_CSI_LEN_SHORT);
  if (tag == NormalizedCSIPayloadTag::NONE) {
    tag = NormalizedCSIPayloadTag::HT57_TO_64;
  }

  return {remap_buffer, HT20_CSI_LEN, tag};
}

const char *normalized_csi_payload_tag_to_string(NormalizedCSIPayloadTag tag) {
  switch (tag) {
    case NormalizedCSIPayloadTag::NONE:
      return "none";
    case NormalizedCSIPayloadTag::DOUBLE_HT20:
      return "double_ht20";
    case NormalizedCSIPayloadTag::HT57_TO_64:
      return "ht57_to_64";
    case NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64:
      return "double_ht57_to_64";
    default:
      return "unknown";
  }
}

}  // namespace espectre
