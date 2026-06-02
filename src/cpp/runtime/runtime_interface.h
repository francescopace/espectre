#pragma once

#include <array>
#include <cstdint>

#include "threshold.h"
#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

enum class DetectionAlgorithm {
  MVS,
  ML,
};

enum class RuntimeTrafficMode {
  DNS,
  PING,
};

enum class RuntimeGainLockMode {
  AUTO,
  ENABLED,
  DISABLED,
};

struct RuntimeConfig {
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::MVS};
  ThresholdMode threshold_mode{ThresholdMode::AUTO};
  float segmentation_threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  uint16_t segmentation_window_size{DETECTOR_DEFAULT_WINDOW_SIZE};
  uint32_t traffic_generator_rate{100};
  RuntimeTrafficMode traffic_generator_mode{RuntimeTrafficMode::PING};
  RuntimeGainLockMode gain_lock_mode{RuntimeGainLockMode::AUTO};
  uint32_t publish_interval{100};
  uint32_t evaluation_interval{25};
  uint8_t motion_on_hits{3};
  uint8_t motion_off_hits{3};
  bool lowpass_enabled{false};
  float lowpass_cutoff{11.0f};
  bool hampel_enabled{true};
  uint8_t hampel_window{7};
  float hampel_threshold{5.0f};
  bool user_specified_subcarriers{false};
  std::array<uint8_t, HT20_SELECTED_BAND_SIZE> selected_subcarriers{
      DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
      DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
      DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11],
  };
};

class IEspectreRuntime {
 public:
  virtual ~IEspectreRuntime() = default;

  virtual bool setup() = 0;
  virtual void shutdown() = 0;
  virtual void loop() = 0;

  virtual bool set_threshold_runtime(float threshold) = 0;
  virtual bool trigger_recalibration() = 0;
  virtual bool is_calibrating() const = 0;

  virtual RuntimeSnapshot get_snapshot() const = 0;
  virtual RuntimeCapabilities get_capabilities() const = 0;

  virtual void set_listener(IRuntimeListener *listener) = 0;
};

}  // namespace espectre
}  // namespace esphome
