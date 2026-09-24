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

/** Training field and PHY the runtime captures CSI from, always at 20 MHz. */
enum class CsiCaptureProfile : uint8_t {
  /** HT long training field. */
  HT20 = 0,
  /** Legacy long training field; lacks the edge tones. */
  LLTF20 = 1,
  /** VHT long training field, on supported 5 GHz links. */
  VHT20 = 2,
};

/** Build-time policy resolved to a physical capture profile after association. */
enum class CsiCapturePolicy : uint8_t {
  /** LLTF20 for internal `wifi_raw` traffic, otherwise as `HT_VHT`. */
  AUTO = 0,
  /** Always LLTF20. */
  LLTF = 1,
  /** VHT20 on a supported 5 GHz link, HT20 otherwise. */
  HT_VHT = 2,
};

/** Protocol name of a profile: `ht20`, `lltf20`, or `vht20`. */
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

/** Whether the profile captures the legacy training field. */
constexpr bool csi_capture_profile_uses_lltf(CsiCaptureProfile profile) {
  return profile == CsiCaptureProfile::LLTF20;
}

/** Resolve the configured capture policy from target capabilities and link channel. */
constexpr CsiCaptureProfile resolve_csi_capture_profile(bool prefers_lltf20,
                                                        bool supports_vht20,
                                                        uint8_t wifi_channel,
                                                        CsiCapturePolicy requested = CsiCapturePolicy::AUTO) {
  if (requested == CsiCapturePolicy::LLTF ||
      (requested == CsiCapturePolicy::AUTO && prefers_lltf20)) {
    return CsiCaptureProfile::LLTF20;
  }
  if (supports_vht20 && wifi_channel > 14U) {
    return CsiCaptureProfile::VHT20;
  }
  return CsiCaptureProfile::HT20;
}

}  // namespace espectre
