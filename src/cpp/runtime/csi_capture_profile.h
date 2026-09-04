/*
 * ESPectre - CSI Capture Profile
 *
 * Describes the physical training field and 20 MHz OFDM geometry selected by
 * the runtime for CSI capture.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

enum class CsiCaptureProfile : uint8_t {
  HT20 = 0,
  LLTF20 = 1,
  VHT20 = 2,
};

constexpr const char *csi_capture_profile_name(CsiCaptureProfile profile) {
  switch (profile) {
    case CsiCaptureProfile::LLTF20:
      return "lltf20";
    case CsiCaptureProfile::VHT20:
      return "vht20";
    case CsiCaptureProfile::HT20:
    default:
      return "ht20";
  }
}

constexpr bool csi_capture_profile_uses_lltf(CsiCaptureProfile profile) {
  return profile == CsiCaptureProfile::LLTF20;
}

/** Resolve the automatic capture policy from target capabilities and link channel. */
constexpr CsiCaptureProfile resolve_csi_capture_profile(bool prefers_lltf20,
                                                        bool supports_vht20,
                                                        uint8_t wifi_channel) {
  if (prefers_lltf20) {
    return CsiCaptureProfile::LLTF20;
  }
  if (supports_vht20 && wifi_channel > 14U) {
    return CsiCaptureProfile::VHT20;
  }
  return CsiCaptureProfile::HT20;
}

}  // namespace espectre
