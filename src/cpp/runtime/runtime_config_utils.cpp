/*
 * ESPectre - Runtime Config Utils
 *
 * Helpers for normalizing and applying runtime configuration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_config_utils.h"
#include "runtime_config_validation.h"

#if __has_include("sdkconfig.h")
#include "sdkconfig.h"
#endif

#include <cmath>
#include <cstring>

namespace espectre {

bool validate_runtime_threshold(float threshold) {
  return std::isfinite(threshold) && threshold >= RUNTIME_THRESHOLD_MIN && threshold <= RUNTIME_THRESHOLD_MAX;
}

bool validate_runtime_threshold_for_algorithm(float threshold, DetectionAlgorithm algorithm) {
  return std::isfinite(threshold) && threshold >= RUNTIME_THRESHOLD_MIN &&
         threshold <= runtime_threshold_max(algorithm);
}

namespace {

bool wifi_band_policy_valid(WifiBandPolicy policy) {
  return policy == WifiBandPolicy::BAND_2G || policy == WifiBandPolicy::BAND_5G ||
         policy == WifiBandPolicy::AUTO;
}

bool parse_ipv4(const std::string &group, unsigned (&octets)[4], bool canonical = false) {
  size_t index = 0U;
  size_t offset = 0U;
  while (index < 4U && offset < group.size()) {
    unsigned value = 0U;
    size_t digits = 0U;
    while (offset < group.size() && group[offset] >= '0' && group[offset] <= '9') {
      value = value * 10U + static_cast<unsigned>(group[offset] - '0');
      if (value > 255U) {
        return false;
      }
      ++offset;
      ++digits;
    }
    if (digits == 0U || (canonical && digits > 1U && group[offset - digits] == '0')) {
      return false;
    }
    octets[index++] = value;
    if (index < 4U) {
      if (offset >= group.size() || group[offset] != '.') {
        return false;
      }
      ++offset;
    }
  }
  return index == 4U && offset == group.size();
}

bool multicast_group_valid(const std::string &group) {
  unsigned octets[4]{};
  return group.empty() ||
         (parse_ipv4(group, octets) && octets[0] >= 224U && octets[0] <= 239U);
}

}  // namespace

uint32_t runtime_traffic_target_addr(const RuntimeConfig &config, uint32_t gateway_addr) {
  if (config.traffic_generator_target_ip.empty()) return gateway_addr;
  unsigned octets[4]{};
  if (!parse_ipv4(config.traffic_generator_target_ip, octets, true) ||
      octets[0] == 0U || octets[0] == 127U || octets[0] >= 224U) {
    return 0U;
  }
  const uint8_t bytes[] = {static_cast<uint8_t>(octets[0]), static_cast<uint8_t>(octets[1]),
                           static_cast<uint8_t>(octets[2]), static_cast<uint8_t>(octets[3])};
  uint32_t address;
  std::memcpy(&address, bytes, sizeof(address));
  return address;
}

bool runtime_traffic_generator_mode_supported(TrafficGeneratorMode mode) {
#if defined(CONFIG_IDF_TARGET_ESP32C6) && CONFIG_IDF_TARGET_ESP32C6
  if (mode == TrafficGeneratorMode::WIFI_RAW) return false;
#endif
  return runtime_traffic_generator_mode_valid(mode);
}

bool runtime_capture_profile_supports_traffic(CsiCapturePolicy profile, TrafficGeneratorMode mode) {
  return mode != TrafficGeneratorMode::WIFI_RAW || profile == CsiCapturePolicy::AUTO ||
         profile == CsiCapturePolicy::LLTF;
}

RuntimeConfigError validate_runtime_config(const RuntimeConfig &config) {
  if (!wifi_band_policy_valid(config.wifi_band_policy)) return RuntimeConfigError::WIFI_BAND_POLICY;
  if (config.csi_capture_policy != CsiCapturePolicy::AUTO &&
      config.csi_capture_policy != CsiCapturePolicy::LLTF &&
      config.csi_capture_policy != CsiCapturePolicy::HT_VHT) {
    return RuntimeConfigError::CSI_CAPTURE_PROFILE;
  }
  if (!validate_runtime_uint32(config.csi_target_pps, RUNTIME_CSI_TARGET_PPS_MIN,
                               RUNTIME_CSI_TARGET_PPS_MAX)) {
    return RuntimeConfigError::CSI_TARGET_PPS;
  }
  if (!runtime_traffic_generator_mode_supported(config.traffic_generator_mode)) {
    return RuntimeConfigError::TRAFFIC_GENERATOR_MODE;
  }
  if (!runtime_capture_profile_supports_traffic(config.csi_capture_policy, config.traffic_generator_mode)) {
    return RuntimeConfigError::CSI_CAPTURE_PROFILE_TRAFFIC;
  }
  if (!config.traffic_generator_target_ip.empty() && runtime_traffic_target_addr(config, 0U) == 0U) {
    return RuntimeConfigError::TRAFFIC_GENERATOR_TARGET_IP;
  }
  if (!runtime_csi_traffic_source_valid(config.csi_traffic_source)) {
    return RuntimeConfigError::CSI_TRAFFIC_MODE;
  }
  if (config.csi_traffic_source == CsiTrafficSource::EXTERNAL) {
    if (config.csi_traffic_udp_port < RUNTIME_NETWORK_PORT_MIN) {
      return RuntimeConfigError::CSI_TRAFFIC_UDP_PORT;
    }
    if (!multicast_group_valid(config.csi_traffic_multicast_group)) {
      return RuntimeConfigError::CSI_TRAFFIC_MULTICAST_GROUP;
    }
  }

  {
    if (!runtime_detection_algorithm_valid(config.detection_algorithm)) {
      return RuntimeConfigError::DETECTION_ALGORITHM;
    }
    if (!validate_runtime_threshold_for_algorithm(config.threshold,
                                                  config.detection_algorithm)) {
      return RuntimeConfigError::SEGMENTATION_THRESHOLD;
    }
    if (!validate_runtime_uint32(config.window_size_ms,
                                 RUNTIME_WINDOW_SIZE_MS_MIN,
                                 RUNTIME_WINDOW_SIZE_MS_MAX)) {
      return RuntimeConfigError::SEGMENTATION_WINDOW_SIZE_MS;
    }
    if (!validate_runtime_uint32(config.evaluation_interval_ms,
                                 RUNTIME_EVALUATION_INTERVAL_MS_MIN,
                                 RUNTIME_EVALUATION_INTERVAL_MS_MAX)) {
      return RuntimeConfigError::EVALUATION_INTERVAL_MS;
    }
    if (!validate_runtime_uint8(config.motion_on_hits, RUNTIME_MOTION_HITS_MIN,
                                RUNTIME_MOTION_HITS_MAX) ||
        !validate_runtime_uint8(config.motion_off_hits, RUNTIME_MOTION_HITS_MIN,
                                RUNTIME_MOTION_HITS_MAX)) {
      return RuntimeConfigError::MOTION_HITS;
    }
    if (config.lowpass_enabled &&
        !validate_runtime_float(config.lowpass_cutoff, RUNTIME_LOWPASS_CUTOFF_MIN,
                                RUNTIME_LOWPASS_CUTOFF_MAX)) {
      return RuntimeConfigError::LOWPASS_CUTOFF;
    }
    if (config.hampel_enabled &&
        !validate_runtime_uint8(config.hampel_window, RUNTIME_HAMPEL_WINDOW_MIN,
                                RUNTIME_HAMPEL_WINDOW_MAX)) {
      return RuntimeConfigError::HAMPEL_WINDOW;
    }
    if (config.hampel_enabled &&
        !validate_runtime_float(config.hampel_threshold, RUNTIME_HAMPEL_THRESHOLD_MIN,
                                RUNTIME_HAMPEL_THRESHOLD_MAX)) {
      return RuntimeConfigError::HAMPEL_THRESHOLD;
    }
  }
  return RuntimeConfigError::NONE;
}

const char *runtime_config_error_message(RuntimeConfigError error) {
  switch (error) {
    case RuntimeConfigError::CSI_CAPTURE_PROFILE: return "invalid CSI capture profile";
    case RuntimeConfigError::CSI_CAPTURE_PROFILE_TRAFFIC: return "wifi_raw requires auto or lltf CSI capture profile";
    case RuntimeConfigError::NONE: return "valid configuration";
    case RuntimeConfigError::WIFI_BAND_POLICY: return "invalid Wi-Fi band policy";
    case RuntimeConfigError::DETECTION_ALGORITHM: return "invalid detection algorithm";
    case RuntimeConfigError::SEGMENTATION_THRESHOLD: return "invalid segmentation threshold";
    case RuntimeConfigError::SEGMENTATION_WINDOW_SIZE_MS: return "invalid segmentation window duration";
    case RuntimeConfigError::CSI_TARGET_PPS: return "invalid CSI target PPS";
    case RuntimeConfigError::TRAFFIC_GENERATOR_MODE: return "invalid traffic generator mode";
    case RuntimeConfigError::TRAFFIC_GENERATOR_TARGET_IP: return "invalid traffic generator target IPv4 address";
    case RuntimeConfigError::CSI_TRAFFIC_MODE: return "invalid CSI traffic mode for runtime profile";
    case RuntimeConfigError::CSI_TRAFFIC_UDP_PORT: return "invalid CSI traffic UDP port";
    case RuntimeConfigError::CSI_TRAFFIC_MULTICAST_GROUP: return "invalid CSI multicast group";
    case RuntimeConfigError::EVALUATION_INTERVAL_MS: return "invalid evaluation interval";
    case RuntimeConfigError::MOTION_HITS: return "invalid motion hit counts";
    case RuntimeConfigError::LOWPASS_CUTOFF: return "invalid low-pass cutoff";
    case RuntimeConfigError::HAMPEL_WINDOW: return "invalid Hampel window";
    case RuntimeConfigError::HAMPEL_THRESHOLD: return "invalid Hampel threshold";
  }
  return "unknown configuration error";
}

const char *wifi_band_policy_name(WifiBandPolicy policy) {
  switch (policy) {
    case WifiBandPolicy::BAND_5G:
      return "5g";
    case WifiBandPolicy::AUTO:
      return "auto";
    case WifiBandPolicy::BAND_2G:
    default:
      return "2g";
  }
}

const char *traffic_generator_mode_name(TrafficGeneratorMode mode) {
  switch (mode) {
    case TrafficGeneratorMode::PING:
      return RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME;
    case TrafficGeneratorMode::DNS:
      return RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME;
    case TrafficGeneratorMode::DNS_TCP:
      return RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_TCP_NAME;
    case TrafficGeneratorMode::WIFI_RAW:
      return RUNTIME_TRAFFIC_GENERATOR_MODE_WIFI_RAW_NAME;
    default:
      return RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME;
  }
}

const char *csi_traffic_source_name(CsiTrafficSource mode) {
  switch (mode) {
    case CsiTrafficSource::EXTERNAL:
      return RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME;
    case CsiTrafficSource::INTERNAL:
    default:
      return RUNTIME_CSI_TRAFFIC_MODE_INTERNAL_NAME;
  }
}

const char *detection_algorithm_name(DetectionAlgorithm algorithm) {
  switch (algorithm) {
    case DetectionAlgorithm::HIGH_ACCURACY:
      return RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME;
    case DetectionAlgorithm::LIGHTWEIGHT:
    default:
      return RUNTIME_DETECTION_ALGORITHM_LIGHTWEIGHT_NAME;
  }
}

TrafficGeneratorMode parse_traffic_generator_mode(const char *mode) {
  if (mode != nullptr && std::strcmp(mode, RUNTIME_TRAFFIC_GENERATOR_MODE_WIFI_RAW_NAME) == 0) {
    return TrafficGeneratorMode::WIFI_RAW;
  }
  if (mode != nullptr && std::strcmp(mode, RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME) == 0) {
    return TrafficGeneratorMode::DNS;
  }
  if (mode != nullptr && std::strcmp(mode, RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_TCP_NAME) == 0) {
    return TrafficGeneratorMode::DNS_TCP;
  }
  return TrafficGeneratorMode::PING;
}

CsiTrafficSource parse_csi_traffic_source(const char *mode) {
  if (mode != nullptr && std::strcmp(mode, RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME) == 0) {
    return CsiTrafficSource::EXTERNAL;
  }
  return CsiTrafficSource::INTERNAL;
}

DetectionAlgorithm parse_detection_algorithm(const char *algorithm) {
  return (algorithm != nullptr && std::strcmp(algorithm, RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME) == 0)
             ? DetectionAlgorithm::HIGH_ACCURACY
             : DetectionAlgorithm::LIGHTWEIGHT;
}

WifiBandPolicy parse_wifi_band_policy(const char *policy) {
  if (policy != nullptr && std::strcmp(policy, "5g") == 0) {
    return WifiBandPolicy::BAND_5G;
  }
  if (policy != nullptr && std::strcmp(policy, "auto") == 0) {
    return WifiBandPolicy::AUTO;
  }
  return WifiBandPolicy::BAND_2G;
}

}  // namespace espectre
