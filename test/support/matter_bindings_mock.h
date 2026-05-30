#pragma once

#include <string>
#include <vector>

#include "matter_bindings.h"

namespace esphome {
namespace espectre {
namespace matter_bindings_mock {

struct MotionPublish {
  uint16_t endpoint_id{0};
  bool motion_detected{false};
};

struct PeriodicPublish {
  uint16_t endpoint_id{0};
  MatterPeriodicState state{};
};

struct ThresholdPublish {
  uint16_t endpoint_id{0};
  float threshold{0.0f};
};

struct CalibratingPublish {
  uint16_t endpoint_id{0};
  bool calibrating{false};
};

struct State {
  std::vector<MotionPublish> motion_events;
  std::vector<PeriodicPublish> periodic_events;
  std::vector<ThresholdPublish> threshold_events;
  std::vector<CalibratingPublish> calibrating_events;
  std::vector<std::string> faults;
};

extern State state;

void reset();

class MockMatterBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void publish_periodic_state(uint16_t endpoint_id, const MatterPeriodicState &state) override;
  void publish_threshold(uint16_t endpoint_id, float threshold) override;
  void publish_calibrating(uint16_t endpoint_id, bool calibrating) override;
  void report_fault(const char *message) override;
};

}  // namespace matter_bindings_mock
}  // namespace espectre
}  // namespace esphome
