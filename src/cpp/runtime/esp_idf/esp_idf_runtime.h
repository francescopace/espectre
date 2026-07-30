/*
 * ESPectre - ESP-IDF Runtime
 *
 * ESP-IDF runtime that wires Wi-Fi, CSI capture, calibration, and
 * detection together.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "base_detector.h"
#include "csi_pipeline.h"
#include "esp_idf_runtime_base.h"
#include "pending_event.h"
#include "runtime_interface.h"
#include "runtime_debug_telemetry.h"
#include "csi_traffic_service.h"
#include "wifi_lifecycle.h"

namespace espectre {

class EspIdfRuntime : public EspIdfRuntimeBase {
 public:
  explicit EspIdfRuntime(const RuntimeConfig &config);

  bool setup() override;
  void shutdown() override;
  void loop() override;
  void set_services_armed(bool armed) override;
  void set_live_telemetry_enabled(bool enabled) override;

  bool set_threshold_runtime(float threshold) override;
  bool set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits) override;
  bool set_detection_algorithm_runtime(DetectionAlgorithm algorithm) override;
  bool trigger_recalibration() override;
  bool is_calibrating() const override;

 private:
  void update_live_telemetry_callback_();
  bool configure_detector_();
  std::unique_ptr<BaseDetector> make_detector_(DetectionAlgorithm algorithm, float threshold);
  void cancel_calibration_(bool notify_listener);
  void log_calibration_progress_(uint8_t percent, uint32_t packets, uint16_t target_packets);
  void on_wifi_connected_(const esp_netif_ip_info_t &ip_info);
  void on_wifi_disconnected_();
  bool start_calibration_();
  bool handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len,
                                            int8_t rssi_dbm, bool evaluation_due,
                                            uint32_t packets_in_window);
  static bool threshold_calibration_packet_callback_(void *context,
                                                     const int8_t *csi_data,
                                                     size_t csi_len,
                                                     int8_t rssi_dbm,
                                                     bool evaluation_due,
                                                     uint32_t packets_in_window);
  void finish_threshold_calibration_(bool success);
  void refresh_csi_local_identity_(uint32_t local_ip_addr);

  std::unique_ptr<BaseDetector> detector_;

  CsiPipeline csi_pipeline_;
  WiFiLifecycleManager wifi_lifecycle_;
  CsiTrafficService csi_traffic_service_;

  std::unique_ptr<StartupThresholdCalibrator> threshold_calibrator_;
  std::atomic<bool> threshold_calibration_active_{false};
  std::atomic<uint8_t> next_calibration_progress_percent_{25U};
  PendingEvent<uint8_t, uint32_t, uint16_t> calibration_progress_event_;
  // Posted from the CSI callback with the outcome, completed from the loop.
  PendingEvent<bool> calibration_finished_event_;
  bool wifi_ready_{false};
  esp_netif_ip_info_t wifi_ip_info_{};
};

}  // namespace espectre
