#pragma once

#include <memory>

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

class RuntimeFrontendController {
 public:
  void set_config(const RuntimeConfig &config);
  RuntimeConfig &config() { return config_; }
  const RuntimeConfig &config() const { return config_; }
  const RuntimeSnapshot &snapshot() const { return snapshot_; }
  const RuntimeCapabilities &capabilities() const { return capabilities_; }
  bool is_setup_complete() const { return setup_complete_; }

  bool setup(IRuntimeListener *listener);
  void loop();
  void shutdown();

  void set_services_armed(bool armed);
  void set_live_telemetry_enabled(bool enabled);
  bool services_armed() const { return services_armed_; }

  bool set_threshold_runtime(float threshold);
  bool trigger_recalibration();
  bool is_calibrating() const;

  void record_snapshot(const RuntimeSnapshot &snapshot);

 private:
  RuntimeConfig config_{};
  RuntimeSnapshot snapshot_{};
  RuntimeCapabilities capabilities_{};
  std::unique_ptr<IEspectreRuntime> runtime_;
  bool setup_complete_{false};
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
};

}  // namespace espectre
}  // namespace esphome
