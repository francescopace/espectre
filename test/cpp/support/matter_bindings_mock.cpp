#include "matter_bindings_mock.h"

namespace espectre {
namespace matter_bindings_mock {

State state{};

void reset() { state = State{}; }

void MockMatterBindings::publish_motion(uint16_t endpoint_id, bool motion_detected) {
  state.motion_events.push_back(MotionPublish{endpoint_id, motion_detected});
}

void MockMatterBindings::publish_periodic_state(uint16_t endpoint_id, const MatterPeriodicState &periodic_state) {
  state.periodic_events.push_back(PeriodicPublish{endpoint_id, periodic_state});
}

void MockMatterBindings::publish_threshold(uint16_t endpoint_id, float threshold) {
  state.threshold_events.push_back(ThresholdPublish{endpoint_id, threshold});
}

void MockMatterBindings::publish_calibrating(uint16_t endpoint_id, bool calibrating) {
  state.calibrating_events.push_back(CalibratingPublish{endpoint_id, calibrating});
}

void MockMatterBindings::report_fault(const char *message) {
  if (message != nullptr) {
    state.faults.emplace_back(message);
  }
}

}  // namespace matter_bindings_mock
}  // namespace espectre
