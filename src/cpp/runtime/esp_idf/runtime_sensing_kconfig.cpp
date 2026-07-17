/*
 * ESPectre - Runtime Sensing Kconfig
 *
 * Builds the default sensing runtime configuration from Kconfig values.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "runtime_sensing_kconfig.h"

#include <cerrno>
#include <cstdlib>

#include "espectre_log.h"
#include "runtime_config_utils.h"
#include "sdkconfig.h"

#ifndef CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC
#define CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC 1
#endif
#ifndef CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML
#define CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML 0
#endif
#ifndef CONFIG_ESPECTRE_SEGMENTATION_WINDOW_SIZE
#define CONFIG_ESPECTRE_SEGMENTATION_WINDOW_SIZE 100
#endif
#ifndef CONFIG_ESPECTRE_TRAFFIC_GENERATOR_RATE
#define CONFIG_ESPECTRE_TRAFFIC_GENERATOR_RATE 100
#endif
#ifndef CONFIG_ESPECTRE_TRAFFIC_GENERATOR_ADAPTIVE
#define CONFIG_ESPECTRE_TRAFFIC_GENERATOR_ADAPTIVE 1
#endif
#ifndef CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS
#define CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS 0
#endif
#ifndef CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_PING
#define CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_PING 1
#endif
#ifndef CONFIG_ESPECTRE_PUBLISH_INTERVAL
#define CONFIG_ESPECTRE_PUBLISH_INTERVAL 100
#endif
#ifndef CONFIG_ESPECTRE_EVALUATION_INTERVAL
#define CONFIG_ESPECTRE_EVALUATION_INTERVAL 25
#endif
#ifndef CONFIG_ESPECTRE_MOTION_ON_HITS
#define CONFIG_ESPECTRE_MOTION_ON_HITS 4
#endif
#ifndef CONFIG_ESPECTRE_MOTION_OFF_HITS
#define CONFIG_ESPECTRE_MOTION_OFF_HITS 3
#endif
#ifndef CONFIG_ESPECTRE_LOWPASS_ENABLED
#define CONFIG_ESPECTRE_LOWPASS_ENABLED 0
#endif
#ifndef CONFIG_ESPECTRE_LOWPASS_CUTOFF
#define CONFIG_ESPECTRE_LOWPASS_CUTOFF "11.0"
#endif
#ifndef CONFIG_ESPECTRE_HAMPEL_ENABLED
#define CONFIG_ESPECTRE_HAMPEL_ENABLED 1
#endif
#ifndef CONFIG_ESPECTRE_HAMPEL_WINDOW
#define CONFIG_ESPECTRE_HAMPEL_WINDOW 7
#endif
#ifndef CONFIG_ESPECTRE_HAMPEL_THRESHOLD
#define CONFIG_ESPECTRE_HAMPEL_THRESHOLD "5.0"
#endif

namespace espectre {

namespace {

static const char *const TAG = "espectre.runtime.cfg";

float parse_float_or_default_(const char *value, float default_value, float min_value, float max_value,
                              const char *key) {
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char *end_ptr = nullptr;
  errno = 0;
  const float parsed = std::strtof(value, &end_ptr);
  const bool parsed_ok = end_ptr != value && end_ptr != nullptr && *end_ptr == '\0' && errno != ERANGE &&
                         validate_runtime_float(parsed, min_value, max_value);
  if (!parsed_ok) {
    ESP_LOGW(TAG, "Invalid %s=\"%s\", using default %.3f", key, value, static_cast<double>(default_value));
    return default_value;
  }
  return parsed;
}

}  // namespace

RuntimeConfig make_runtime_sensing_config_from_kconfig() {
  RuntimeConfig config = make_runtime_sensing_config();

#if CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML
  config.detection_algorithm = DetectionAlgorithm::ML;
#else
  config.detection_algorithm = DetectionAlgorithm::CLASSIC;
#endif

  config.segmentation_threshold = runtime_default_threshold(config.detection_algorithm);

  config.segmentation_window_size = static_cast<uint16_t>(CONFIG_ESPECTRE_SEGMENTATION_WINDOW_SIZE);
  config.traffic_generator_rate = static_cast<uint32_t>(CONFIG_ESPECTRE_TRAFFIC_GENERATOR_RATE);
  config.traffic_generator_adaptive = CONFIG_ESPECTRE_TRAFFIC_GENERATOR_ADAPTIVE;
#if CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS
  config.traffic_generator_mode = RuntimeTrafficMode::DNS;
#else
  config.traffic_generator_mode = RuntimeTrafficMode::PING;
#endif
  config.publish_interval = static_cast<uint32_t>(CONFIG_ESPECTRE_PUBLISH_INTERVAL);
  config.evaluation_interval = static_cast<uint32_t>(CONFIG_ESPECTRE_EVALUATION_INTERVAL);
  config.motion_on_hits = static_cast<uint8_t>(CONFIG_ESPECTRE_MOTION_ON_HITS);
  config.motion_off_hits = static_cast<uint8_t>(CONFIG_ESPECTRE_MOTION_OFF_HITS);
  config.lowpass_enabled = CONFIG_ESPECTRE_LOWPASS_ENABLED;
  config.lowpass_cutoff = parse_float_or_default_(CONFIG_ESPECTRE_LOWPASS_CUTOFF,
                                                  RUNTIME_LOWPASS_CUTOFF_DEFAULT,
                                                  RUNTIME_LOWPASS_CUTOFF_MIN,
                                                  RUNTIME_LOWPASS_CUTOFF_MAX,
                                                  "CONFIG_ESPECTRE_LOWPASS_CUTOFF");
  config.hampel_enabled = CONFIG_ESPECTRE_HAMPEL_ENABLED;
  config.hampel_window = static_cast<uint8_t>(CONFIG_ESPECTRE_HAMPEL_WINDOW);
  config.hampel_threshold = parse_float_or_default_(CONFIG_ESPECTRE_HAMPEL_THRESHOLD,
                                                    RUNTIME_HAMPEL_THRESHOLD_DEFAULT,
                                                    RUNTIME_HAMPEL_THRESHOLD_MIN,
                                                    RUNTIME_HAMPEL_THRESHOLD_MAX,
                                                    "CONFIG_ESPECTRE_HAMPEL_THRESHOLD");

  return config;
}

}  // namespace espectre
