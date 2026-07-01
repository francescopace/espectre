#pragma once

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

class EspIdfRuntime;

namespace frontend_runtime_shim {

struct State {
  bool setup_result{true};
  RuntimeSnapshot snapshot{};
  RuntimeCapabilities capabilities{};
  bool shutdown_called{false};
  int loop_calls{0};
  int set_threshold_calls{0};
  float last_threshold{0.0f};
  int trigger_recalibration_calls{0};
  bool calibrating{false};
  bool services_armed{true};
  bool live_telemetry_enabled{true};
  int set_live_telemetry_enabled_calls{0};
  IRuntimeListener *last_listener{nullptr};
  EspIdfRuntime *last_instance{nullptr};
};

extern State state;

void reset();

}  // namespace frontend_runtime_shim

}  // namespace espectre
}  // namespace esphome
