#pragma once

#include <string>

#include "base_detector.h"
#include "csi_manager.h"
#include "gain_controller.h"
#include "ml_detector.h"
#include "mvs_detector.h"
#include "nbvi_calibrator.h"
#include "runtime_interface.h"
#include "traffic_generator_manager.h"
#include "udp_listener.h"
#include "wifi_lifecycle.h"

namespace esphome {
namespace espectre {

class EspIdfRuntime : public IEspectreRuntime {
 public:
  explicit EspIdfRuntime(const RuntimeConfig &config);

  bool setup() override;
  void shutdown() override;
  void loop() override;

  bool set_threshold_runtime(float threshold) override;
  bool trigger_recalibration() override;
  bool is_calibrating() const override;

  RuntimeSnapshot get_snapshot() const override;
  RuntimeCapabilities get_capabilities() const override;

  void set_listener(IRuntimeListener *listener) override;

 private:
  bool configure_detector_();
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  bool start_calibration_();
  void notify_fault_(const char *message);

  RuntimeConfig config_;
  RuntimeSnapshot snapshot_;
  RuntimeCapabilities capabilities_{};
  IRuntimeListener *listener_{nullptr};

  BaseDetector *detector_{nullptr};
  MVSDetector mvs_detector_;
  MLDetector ml_detector_;

  CSIManager csi_manager_;
  WiFiLifecycleManager wifi_lifecycle_;
  NBVICalibrator nbvi_calibrator_;
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;

  bool setup_complete_{false};
  std::string last_fault_;
};

}  // namespace espectre
}  // namespace esphome
