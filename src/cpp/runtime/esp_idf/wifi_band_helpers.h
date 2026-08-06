/*
 * ESPectre - Wi-Fi Band Helpers
 *
 * Declares which bands the radio can use for HT20 sensing and validates the
 * optional channel hint against them.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "sdkconfig.h"
#include "runtime_interface.h"

// Dual-band parts expose the per-band protocol and bandwidth APIs
// (esp_wifi_set_protocols, esp_wifi_set_bandwidths), which are the only ones
// valid while the radio may associate on either band. The single-band APIs
// return ESP_ERR_NOT_SUPPORTED under WIFI_BAND_MODE_AUTO, so the two paths
// cannot be merged.
#if defined(CONFIG_SOC_WIFI_SUPPORT_5G) && CONFIG_SOC_WIFI_SUPPORT_5G
#define ESPECTRE_WIFI_DUAL_BAND 1
#else
#define ESPECTRE_WIFI_DUAL_BAND 0
#endif

namespace espectre {

// Channel 0 means "let the station follow the AP", which is the default.
constexpr int WIFI_CHANNEL_AUTO = 0;
constexpr int WIFI_CHANNEL_2G_MAX = 14;

/** Return whether this build can honor the requested band policy. */
constexpr bool wifi_band_policy_is_supported(WifiBandPolicy policy) {
#if ESPECTRE_WIFI_DUAL_BAND
  return policy == WifiBandPolicy::BAND_2G || policy == WifiBandPolicy::BAND_5G ||
         policy == WifiBandPolicy::AUTO;
#else
  return policy == WifiBandPolicy::BAND_2G;
#endif
}

/** Return whether a channel hint is compatible with the requested band. */
constexpr bool wifi_channel_matches_band_policy(int channel, WifiBandPolicy policy) {
  if (channel == WIFI_CHANNEL_AUTO) {
    return true;
  }
  if (channel > WIFI_CHANNEL_AUTO && channel <= WIFI_CHANNEL_2G_MAX) {
    return policy != WifiBandPolicy::BAND_5G;
  }
  return policy != WifiBandPolicy::BAND_2G;
}

/**
 * Report whether an optional channel hint is usable on this build.
 *
 * 2.4 GHz channels are contiguous; the 5 GHz channel numbers are the 20 MHz
 * centers of the UNII bands, which are spaced four channels apart from 36 and
 * from 149. Passing a 5 GHz channel a 2.4 GHz-only radio cannot tune would
 * fail silently at association time, so those numbers are rejected there.
 *
 * @param channel Channel number, or WIFI_CHANNEL_AUTO for no hint
 * @return true when the channel can be configured on this build
 */
constexpr bool wifi_channel_is_supported(int channel) {
  if (channel >= WIFI_CHANNEL_AUTO && channel <= WIFI_CHANNEL_2G_MAX) {
    return true;
  }
#if ESPECTRE_WIFI_DUAL_BAND
  if (channel >= 36 && channel <= 64) {
    return (channel % 4) == 0;
  }
  if (channel >= 100 && channel <= 144) {
    return (channel % 4) == 0;
  }
  if (channel >= 149 && channel <= 177) {
    return (channel % 4) == 1;
  }
#endif
  return false;
}

/**
 * Describe the accepted channel hint values for operator-facing errors.
 *
 * @return A short human-readable range description
 */
inline const char *wifi_channel_supported_description() {
#if ESPECTRE_WIFI_DUAL_BAND
  return "0..14, or a 5 GHz channel (36..64, 100..144, 149..177)";
#else
  return "0..14";
#endif
}

inline const char *wifi_channel_supported_description(WifiBandPolicy policy) {
#if ESPECTRE_WIFI_DUAL_BAND
  if (policy == WifiBandPolicy::BAND_5G) {
    return "0, or a 5 GHz channel (36..64, 100..144, 149..177)";
  }
  if (policy == WifiBandPolicy::AUTO) {
    return wifi_channel_supported_description();
  }
#else
  (void)policy;
#endif
  return "0..14";
}

}  // namespace espectre
