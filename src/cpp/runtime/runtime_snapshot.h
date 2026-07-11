#pragma once

#include "base_detector.h"
#include "csi_format.h"
#include "threshold.h"

namespace esphome {
namespace espectre {

enum class RuntimeSubcarrierSource {
  FIXED_DEFAULT,
};

struct RuntimeSnapshot {
  MotionState motion_state{MotionState::IDLE};
  float movement_metric{0.0f};
  float threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  bool calibrating{false};
  bool ready_to_publish{false};
  float startup_threshold{0.0f};
  const char *detector_name{"unknown"};
  RuntimeSubcarrierSource subcarrier_source{RuntimeSubcarrierSource::FIXED_DEFAULT};
  SelectedSubcarriers fixed_subcarriers{make_default_subcarriers()};
};

}  // namespace espectre
}  // namespace esphome
