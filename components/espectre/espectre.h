/*
 * ESPectre - Main Component
 * 
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/core/log.h"
#include "esphome/core/preferences.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/number/number.h"

// Include ESP-IDF WiFi headers
#include "esp_wifi.h"
#include "esp_err.h"
#include "esp_event.h"

// Include C++ modules
#include "csi_processor.h"  // For SEGMENTATION_MAX_WINDOW_SIZE
#include "sensor_publisher.h"
#include "csi_manager.h"
#include "wifi_lifecycle.h"
#include "config_manager.h"
#include "calibration_manager.h"
#include "traffic_generator_manager.h"
#include "serial_streamer.h"

namespace esphome {
namespace espectre {

static const char *const TAG = "espectre";

// Default subcarrier selection (top 12 most informative)
constexpr uint8_t DEFAULT_SUBCARRIERS[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

class ESpectreComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  ~ESpectreComponent();
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }
  
  // Setters for YAML configuration
  void set_segmentation_threshold(float threshold) { this->segmentation_threshold_ = threshold; }
  void set_segmentation_window_size(uint16_t size) { this->segmentation_window_size_ = size; }
  void set_traffic_generator_rate(uint32_t rate) { this->traffic_generator_rate_ = rate; }
  void set_lowpass_enabled(bool enabled) { this->lowpass_enabled_ = enabled; }
  void set_lowpass_cutoff(float cutoff) { this->lowpass_cutoff_ = cutoff; }
  void set_hampel_enabled(bool enabled) { this->hampel_enabled_ = enabled; }
  void set_hampel_window(uint8_t window) { this->hampel_window_ = window; }
  void set_hampel_threshold(float threshold) { this->hampel_threshold_ = threshold; }
  
  // Subcarrier selection (optional, defaults to auto-calibrated or DEFAULT_SUBCARRIERS)
  void set_selected_subcarriers(const std::vector<uint8_t> &subcarriers) {
    size_t count = std::min(subcarriers.size(), (size_t)12);
    for (size_t i = 0; i < count; i++) {
      this->selected_subcarriers_[i] = subcarriers[i];
    }
    this->user_specified_subcarriers_ = true;  // Mark as user-specified
  }
  
  // Setters for ESPHome sensors (delegated to SensorPublisher)
  void set_movement_sensor(sensor::Sensor *sensor) { this->sensor_publisher_.set_movement_sensor(sensor); }
  void set_motion_binary_sensor(binary_sensor::BinarySensor *sensor) { this->sensor_publisher_.set_motion_binary_sensor(sensor); }
  
  // Setter for threshold number control
  void set_threshold_number(number::Number *num) { this->threshold_number_ = num; }
  
  // Runtime threshold adjustment (called from HA via number component)
  void set_threshold_runtime(float threshold);
  float get_threshold() const { return this->segmentation_threshold_; }
  
 protected:
  // WiFi lifecycle callbacks
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  
  // Send system info over serial (for game display)
  void send_system_info_();
  
  // C state (core modules)
  csi_processor_context_t csi_processor_{};
  csi_motion_state_t motion_state_{};
  
  // Configuration from YAML
  float segmentation_threshold_{1.0f};
  uint16_t segmentation_window_size_{50};
  uint32_t traffic_generator_rate_{100};
  bool lowpass_enabled_{true};      // Low-pass filter enabled by default
  float lowpass_cutoff_{11.0f};     // Default cutoff frequency in Hz
  bool hampel_enabled_{false};
  uint8_t hampel_window_{7};
  float hampel_threshold_{4.0f};
  float normalization_scale_{1.0f};   // Normalization scale (attenuate if baseline > 0.25)
  uint8_t selected_subcarriers_[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
  
  bool user_specified_subcarriers_{false};  // True if user specified in YAML
  
  // Managers (handle specific responsibilities)
  SensorPublisher sensor_publisher_;
  CSIManager csi_manager_;
  WiFiLifecycleManager wifi_lifecycle_;
  ConfigurationManager config_manager_;
  CalibrationManager calibration_manager_;
  TrafficGeneratorManager traffic_generator_;
  SerialStreamer serial_streamer_;
  
  // Number controls
  number::Number *threshold_number_{nullptr};
  
  // Calibration results (for diagnostics)
  float baseline_variance_{0.0f};
  
  // State flags
  bool ready_to_publish_{false};      // True when CSI is ready and calibration done
  bool threshold_republished_{false}; // True after threshold has been re-published to HA
};

}  // namespace espectre
}  // namespace esphome
