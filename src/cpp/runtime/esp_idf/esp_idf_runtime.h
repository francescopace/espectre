#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "base_detector.h"
#include "classic_detector.h"
#include "csi_pipeline.h"
#include "ml_detector.h"
#include "runtime_interface.h"
#include "csi_traffic_service.h"
#include "wifi_lifecycle.h"

namespace esphome {
namespace espectre {

class EspIdfRuntime : public IEspectreRuntime {
 public:
  explicit EspIdfRuntime(const RuntimeConfig &config);

  bool setup() override;
  void shutdown() override;
  void loop() override;
  void set_services_armed(bool armed) override;
  void set_live_telemetry_enabled(bool enabled) override;

  bool set_threshold_runtime(float threshold) override;
  bool trigger_recalibration() override;
  bool is_calibrating() const override;

  RuntimeSnapshot get_snapshot() const override;
  RuntimeCapabilities get_capabilities() const override;

  void set_listener(IRuntimeListener *listener) override;

 private:
  void update_live_telemetry_callback_();
  bool configure_detector_();
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  bool start_calibration_();
  bool handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len);
  void finish_threshold_calibration_(bool success);
  void notify_fault_(const char *message);
  bool has_wifi_ip_() const;
  uint32_t local_wifi_ip_addr_() const;
  void refresh_csi_local_identity_();

  RuntimeConfig config_;
  RuntimeSnapshot snapshot_;
  RuntimeCapabilities capabilities_{};
  IRuntimeListener *listener_{nullptr};

  BaseDetector *detector_{nullptr};
  ClassicDetector classic_detector_;
  MLDetector ml_detector_;

  CsiPipeline csi_pipeline_;
  WiFiLifecycleManager wifi_lifecycle_;
  CsiTrafficService csi_traffic_service_;

  StartupThresholdCalibrator threshold_calibrator_;
  bool threshold_calibration_active_{false};
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
  bool wifi_ready_{false};
  bool csi_wifi_lifecycle_ready_{false};
  bool setup_complete_{false};
  std::string last_fault_;
};

}  // namespace espectre
}  // namespace esphome
