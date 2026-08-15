/*
 * ESPectre - Runtime Sensing Schema
 *
 * Shared sensing schema enums and defaults for runtime config.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "base_detector.h"
#include "lightweight_detector.h"
#include "csi_traffic_types.h"
#include "filters.h"
#include "high_accuracy_detector.h"
#include "threshold.h"

/**
 * @file runtime_sensing_schema.h
 * @brief The schema behind `RuntimeConfig`: enums, defaults, and valid ranges.
 *
 * This is the single source of truth for what a sensing configuration may
 * contain. Every tunable is declared as a `RUNTIME_<FIELD>_DEFAULT` plus, where
 * a range applies, `_MIN` and `_MAX`. Read them instead of hardcoding limits,
 * so a UI, a provisioning flow, or a config parser stays correct across SDK
 * releases.
 *
 * The `static_assert` block at the end holds these values in lockstep with the
 * detector and filter constants they mirror, so a drift between the runtime
 * schema and `core/` fails the build rather than the device.
 */

namespace espectre {

/** Which detector runs. See `docs/ALGORITHMS.md` for how they differ. */
enum class DetectionAlgorithm {
  /** Lightweight feature fusion. Self-calibrates, and needs no training data. Default. */
  LIGHTWEIGHT,
  /** High-accuracy neural detector using the trained weights in `core/ml_weights.h`. */
  HIGH_ACCURACY,
};

/** Which runtime backend the controller builds. */
enum class RuntimeProfile {
  /** Detect motion on-device and report state. The normal profile. */
  SENSING,
  /**
   * Ship raw CSI to a host collector instead of detecting.
   *
   * For dataset collection and offline analysis. Requires a build with the
   * stream runtime compiled in; `setup()` fails otherwise.
   */
  STREAM,
};

/** Which packet the internal generator sends to solicit CSI from the AP. */
enum class RuntimeTrafficMode {
  /** DNS queries. Useful where ICMP is filtered. */
  DNS,
  /** ICMP echo. Default. */
  PING,
};

constexpr const char *const RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME = "dns";
constexpr const char *const RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME = "ping";
constexpr const char *const RUNTIME_TRAFFIC_GENERATOR_MODE_DEFAULT_NAME = "ping";

constexpr const char *const RUNTIME_CSI_TRAFFIC_MODE_INTERNAL_NAME = "internal";
constexpr const char *const RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME = "external";
constexpr const char *const RUNTIME_CSI_TRAFFIC_MODE_PACING_NAME = "pacing";
constexpr const char *const RUNTIME_CSI_TRAFFIC_MODE_DISABLED_NAME = "disabled";
constexpr const char *const RUNTIME_CSI_TRAFFIC_MODE_DEFAULT_NAME = "internal";

constexpr const char *const RUNTIME_DETECTION_ALGORITHM_LIGHTWEIGHT_NAME = "lightweight";
constexpr const char *const RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME = "high_accuracy";
constexpr const char *const RUNTIME_DETECTION_ALGORITHM_DEFAULT_NAME = "lightweight";

constexpr float RUNTIME_THRESHOLD_MIN = 0.0f;
constexpr float RUNTIME_THRESHOLD_MAX = 1.0f;
constexpr float RUNTIME_HIGH_ACCURACY_THRESHOLD_MAX = 1.0f;
constexpr float RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT = 1.0f;

constexpr uint32_t RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MIN = 1000U;
constexpr uint32_t RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MAX = 2000U;
constexpr uint32_t RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT = 1000U;

constexpr uint32_t RUNTIME_CSI_TARGET_PPS_MIN = 1U;
// The maximum public two-second window resolves to the detector's 1000-slot
// storage ceiling at this rate. Supported hardware normally runs near 100 pps.
constexpr uint32_t RUNTIME_CSI_TARGET_PPS_MAX = 500U;
constexpr uint32_t RUNTIME_CSI_TARGET_PPS_DEFAULT = 100U;
constexpr bool RUNTIME_TRAFFIC_GENERATOR_ADAPTIVE_DEFAULT = false;

constexpr uint32_t RUNTIME_PUBLISH_INTERVAL_MS_MIN = 100;
constexpr uint32_t RUNTIME_PUBLISH_INTERVAL_MS_MAX = 60000;
constexpr uint32_t RUNTIME_PUBLISH_INTERVAL_MS_DEFAULT = 1000;
constexpr uint32_t RUNTIME_EVALUATION_INTERVAL_MS_MIN = 10;
constexpr uint32_t RUNTIME_EVALUATION_INTERVAL_MS_MAX = 10000;
constexpr uint32_t RUNTIME_EVALUATION_INTERVAL_MS_DEFAULT = 250;

constexpr uint8_t RUNTIME_MOTION_HITS_MIN = 1;
constexpr uint8_t RUNTIME_MOTION_HITS_MAX = 20;
constexpr uint8_t RUNTIME_MOTION_ON_HITS_DEFAULT = 4;
constexpr uint8_t RUNTIME_MOTION_OFF_HITS_DEFAULT = 3;

constexpr bool RUNTIME_LOWPASS_ENABLED_DEFAULT = false;
constexpr float RUNTIME_LOWPASS_CUTOFF_MIN = 5.0f;
constexpr float RUNTIME_LOWPASS_CUTOFF_MAX = 20.0f;
constexpr float RUNTIME_LOWPASS_CUTOFF_DEFAULT = 11.0f;

constexpr bool RUNTIME_HAMPEL_ENABLED_DEFAULT = true;
constexpr uint8_t RUNTIME_HAMPEL_WINDOW_MIN = 3;
constexpr uint8_t RUNTIME_HAMPEL_WINDOW_MAX = 11;
constexpr uint8_t RUNTIME_HAMPEL_WINDOW_DEFAULT = 7;
constexpr float RUNTIME_HAMPEL_THRESHOLD_MIN = 1.0f;
constexpr float RUNTIME_HAMPEL_THRESHOLD_MAX = 10.0f;
constexpr float RUNTIME_HAMPEL_THRESHOLD_DEFAULT = 5.0f;

constexpr uint16_t RUNTIME_STREAM_COLLECTOR_PORT_DEFAULT = 5001;
constexpr uint32_t RUNTIME_STREAM_LOG_INTERVAL_MS_DEFAULT = 1000;
constexpr uint8_t RUNTIME_STREAM_TX_BATCH_RECORDS_DEFAULT = 4;

constexpr uint16_t RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT = 5555;

constexpr float runtime_threshold_max(DetectionAlgorithm algorithm) {
  return algorithm == DetectionAlgorithm::LIGHTWEIGHT ? LIGHTWEIGHT_MAX_THRESHOLD
                                                      : RUNTIME_HIGH_ACCURACY_THRESHOLD_MAX;
}

constexpr bool runtime_detection_algorithm_valid(DetectionAlgorithm algorithm) {
  return algorithm == DetectionAlgorithm::LIGHTWEIGHT ||
         algorithm == DetectionAlgorithm::HIGH_ACCURACY;
}

constexpr float runtime_default_threshold(DetectionAlgorithm algorithm) {
  return algorithm == DetectionAlgorithm::HIGH_ACCURACY
             ? HIGH_ACCURACY_DEFAULT_THRESHOLD
             : LIGHTWEIGHT_DEFAULT_THRESHOLD;
}

static_assert(RUNTIME_THRESHOLD_MIN == 0.0f, "Runtime threshold min must stay at zero");
static_assert(RUNTIME_HIGH_ACCURACY_THRESHOLD_MAX == HIGH_ACCURACY_MAX_THRESHOLD,
              "Runtime High Accuracy threshold max drifted from high_accuracy_detector.h");
static_assert(RUNTIME_HIGH_ACCURACY_THRESHOLD_MAX == LIGHTWEIGHT_MAX_THRESHOLD,
              "Lightweight and High Accuracy probability scales must stay aligned");
static_assert(RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT == SEGMENTATION_DEFAULT_THRESHOLD,
              "Runtime segmentation threshold default drifted from threshold.h");
static_assert(RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MIN == DETECTOR_WINDOW_SIZE_MS_MIN,
              "Runtime segmentation window duration min drifted from detector_limits.h");
static_assert(RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MAX == DETECTOR_WINDOW_SIZE_MS_MAX,
              "Runtime segmentation window duration max drifted from detector_limits.h");
static_assert(RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT == DETECTOR_WINDOW_SIZE_MS_DEFAULT,
              "Runtime segmentation window duration default drifted from detector_limits.h");
static_assert(RUNTIME_LOWPASS_CUTOFF_MIN == LOWPASS_CUTOFF_MIN, "Runtime lowpass cutoff min drifted from filters.h");
static_assert(RUNTIME_LOWPASS_CUTOFF_MAX == LOWPASS_CUTOFF_MAX, "Runtime lowpass cutoff max drifted from filters.h");
static_assert(RUNTIME_LOWPASS_CUTOFF_DEFAULT == LOWPASS_CUTOFF_DEFAULT,
              "Runtime lowpass cutoff default drifted from filters.h");
static_assert(RUNTIME_HAMPEL_WINDOW_MIN == HAMPEL_TURBULENCE_WINDOW_MIN,
              "Runtime Hampel window min drifted from filters.h");
static_assert(RUNTIME_HAMPEL_WINDOW_MAX == HAMPEL_TURBULENCE_WINDOW_MAX,
              "Runtime Hampel window max drifted from filters.h");
static_assert(RUNTIME_HAMPEL_WINDOW_DEFAULT == HAMPEL_TURBULENCE_WINDOW_DEFAULT,
              "Runtime Hampel window default drifted from filters.h");
static_assert(RUNTIME_HAMPEL_THRESHOLD_DEFAULT == HAMPEL_TURBULENCE_THRESHOLD_DEFAULT,
              "Runtime Hampel threshold default drifted from filters.h");

}  // namespace espectre
