#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "base_detector.h"
#include "csi_pipeline.h"
#include "pending_event.h"
#include "runtime_interface.h"
#include "runtime_debug_telemetry.h"
#include "csi_traffic_service.h"
#include "wifi_lifecycle.h"

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
  void log_calibration_progress_(uint8_t percent, uint32_t packets, uint16_t target_packets);
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  bool start_calibration_();
  bool handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len);
  static bool threshold_calibration_packet_callback_(void *context,
                                                     const int8_t *csi_data,
                                                     size_t csi_len);
  void finish_threshold_calibration_(bool success);
  void notify_fault_(const char *message);
  bool has_wifi_ip_() const;
  uint32_t local_wifi_ip_addr_() const;
  void refresh_csi_local_identity_();

  RuntimeConfig config_;
  RuntimeSnapshot snapshot_;
  RuntimeCapabilities capabilities_{};
  IRuntimeListener *listener_{nullptr};

  std::unique_ptr<BaseDetector> detector_;

  CsiPipeline csi_pipeline_;
  WiFiLifecycleManager wifi_lifecycle_;
  CsiTrafficService csi_traffic_service_;
  RuntimeDebugTelemetry debug_telemetry_;

  std::unique_ptr<StartupThresholdCalibrator> threshold_calibrator_;
  std::atomic<bool> threshold_calibration_active_{false};
  uint32_t calibration_packets_since_evaluation_{0};
  std::atomic<uint8_t> next_calibration_progress_percent_{25U};
  PendingEvent<uint8_t, uint32_t, uint16_t> calibration_progress_event_;
  // Posted from the CSI callback with the outcome, completed from the loop.
  PendingEvent<bool> calibration_finished_event_;
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
  bool wifi_ready_{false};
  bool csi_wifi_lifecycle_ready_{false};
  bool setup_complete_{false};
  std::string last_fault_;
};

}  // namespace espectre
