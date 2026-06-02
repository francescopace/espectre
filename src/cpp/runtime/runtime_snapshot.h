#pragma once

#include <array>

#include "base_detector.h"
#include "utils.h"

namespace esphome {
namespace espectre {

enum class RuntimeSubcarrierSource {
  AUTO_CALIBRATED,
  USER_CONFIGURED,
  MODEL_DEFAULT,
};

struct RuntimeSnapshot {
  MotionState motion_state{MotionState::IDLE};
  float movement_metric{0.0f};
  float threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  bool calibrating{false};
  bool ready_to_publish{false};
  bool gain_locked{false};
  float best_pxx{0.0f};
  const char *detector_name{"unknown"};
  RuntimeSubcarrierSource subcarrier_source{RuntimeSubcarrierSource::AUTO_CALIBRATED};
  std::array<uint8_t, HT20_SELECTED_BAND_SIZE> selected_subcarriers{
      DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
      DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
      DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11],
  };
};

}  // namespace espectre
}  // namespace esphome
