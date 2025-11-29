#pragma once

#include "esphome/core/component.h"
#include "esphome/core/log.h"
#include "esphome/core/preferences.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"

// Include ESP-IDF WiFi headers
#include "esp_wifi.h"
#include "esp_err.h"
#include "esp_event.h"

// Include C++ modules
#include "csi_processor.h"
#include "filters.h"
#include "csi_features.h"
#include "nbvi_calibrator.h"
#include "traffic_generator.h"

namespace esphome {
namespace espectre {

static const char *const TAG = "espectre";

// Default subcarrier selection (top 12 most informative)
constexpr uint8_t DEFAULT_SUBCARRIERS[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

// Preferences structure (replaces NVS)
struct ESpectrePreferences {
  float segmentation_threshold;
  uint16_t segmentation_window_size;
  uint32_t traffic_generator_rate;
  bool features_enabled;
  bool butterworth_enabled;
  bool wavelet_enabled;
  uint8_t wavelet_level;
  float wavelet_threshold;
  bool hampel_enabled;
  float hampel_threshold;
  bool savgol_enabled;
};

class ESpectreComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }
  
  // Setters for YAML configuration
  void set_segmentation_threshold(float threshold) { this->segmentation_threshold_ = threshold; }
  void set_segmentation_window_size(uint16_t size) { this->segmentation_window_size_ = size; }
  void set_traffic_generator_rate(uint32_t rate) { this->traffic_generator_rate_ = rate; }
  void set_features_enabled(bool enabled) { this->features_enabled_ = enabled; }
  void set_butterworth_enabled(bool enabled) { this->butterworth_enabled_ = enabled; }
  void set_wavelet_enabled(bool enabled) { this->wavelet_enabled_ = enabled; }
  void set_wavelet_level(uint8_t level) { this->wavelet_level_ = level; }
  void set_wavelet_threshold(float threshold) { this->wavelet_threshold_ = threshold; }
  void set_hampel_enabled(bool enabled) { this->hampel_enabled_ = enabled; }
  void set_hampel_threshold(float threshold) { this->hampel_threshold_ = threshold; }
  void set_savgol_enabled(bool enabled) { this->savgol_enabled_ = enabled; }
  
  // Subcarrier selection (optional, defaults to auto-calibrated or DEFAULT_SUBCARRIERS)
  void set_selected_subcarriers(const std::vector<uint8_t> &subcarriers) {
    size_t count = std::min(subcarriers.size(), (size_t)12);
    for (size_t i = 0; i < count; i++) {
      this->selected_subcarriers_[i] = subcarriers[i];
    }
    this->num_selected_subcarriers_ = count;
    this->user_specified_subcarriers_ = true;  // Mark as user-specified
  }
  
  // Setters for ESPHome sensors
  void set_movement_sensor(sensor::Sensor *sensor) { this->movement_sensor_ = sensor; }
  void set_threshold_sensor(sensor::Sensor *sensor) { this->threshold_sensor_ = sensor; }
  void set_turbulence_sensor(sensor::Sensor *sensor) { this->turbulence_sensor_ = sensor; }
  void set_motion_binary_sensor(binary_sensor::BinarySensor *sensor) { this->motion_binary_sensor_ = sensor; }
  
  // Feature sensors (optional)
  void set_variance_sensor(sensor::Sensor *sensor) { this->variance_sensor_ = sensor; }
  void set_skewness_sensor(sensor::Sensor *sensor) { this->skewness_sensor_ = sensor; }
  void set_kurtosis_sensor(sensor::Sensor *sensor) { this->kurtosis_sensor_ = sensor; }
  void set_entropy_sensor(sensor::Sensor *sensor) { this->entropy_sensor_ = sensor; }
  void set_iqr_sensor(sensor::Sensor *sensor) { this->iqr_sensor_ = sensor; }
  void set_spatial_variance_sensor(sensor::Sensor *sensor) { this->spatial_variance_sensor_ = sensor; }
  void set_spatial_correlation_sensor(sensor::Sensor *sensor) { this->spatial_correlation_sensor_ = sensor; }
  void set_spatial_gradient_sensor(sensor::Sensor *sensor) { this->spatial_gradient_sensor_ = sensor; }
  void set_temporal_delta_mean_sensor(sensor::Sensor *sensor) { this->temporal_delta_mean_sensor_ = sensor; }
  void set_temporal_delta_variance_sensor(sensor::Sensor *sensor) { this->temporal_delta_variance_sensor_ = sensor; }
  
 protected:
 
 // WiFi event handler (static for C API)
 static void wifi_event_handler_(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data);
  void handle_wifi_event_(esp_event_base_t event_base, int32_t event_id, void* event_data);

  // CSI callback (static for C API)
  static void csi_callback_wrapper_(void *ctx, wifi_csi_info_t *data);
  void handle_csi_data_(wifi_csi_info_t *data);
  
  // Setup helpers
  void setup_csi_();
  void run_nbvi_calibration_();
  bool load_preferences_();
  void save_preferences_();
  
  // Update sensors
  void update_sensors_();
  
  // C state (core modules)
  csi_processor_context_t csi_processor_{};
  csi_features_t current_features_{};
  csi_motion_state_t motion_state_{};
  
  // Filters
  butterworth_filter_t butterworth_{};
  filter_buffer_t filter_buffer_{};
  wavelet_state_t wavelet_{};
  
  // Configuration from YAML
  float segmentation_threshold_{1.0f};
  uint16_t segmentation_window_size_{50};
  uint32_t traffic_generator_rate_{100};
  bool features_enabled_{true};
  bool butterworth_enabled_{true};
  bool wavelet_enabled_{false};
  uint8_t wavelet_level_{3};
  float wavelet_threshold_{1.0f};
  bool hampel_enabled_{true};
  float hampel_threshold_{2.0f};
  bool savgol_enabled_{true};
  
  uint8_t selected_subcarriers_[12];
  uint8_t num_selected_subcarriers_{12};
  bool user_specified_subcarriers_{false};  // True if user specified in YAML
  
  // ESPHome sensors
  sensor::Sensor *movement_sensor_{nullptr};
  sensor::Sensor *threshold_sensor_{nullptr};
  sensor::Sensor *turbulence_sensor_{nullptr};
  binary_sensor::BinarySensor *motion_binary_sensor_{nullptr};
  
  // Feature sensors
  sensor::Sensor *variance_sensor_{nullptr};
  sensor::Sensor *skewness_sensor_{nullptr};
  sensor::Sensor *kurtosis_sensor_{nullptr};
  sensor::Sensor *entropy_sensor_{nullptr};
  sensor::Sensor *iqr_sensor_{nullptr};
  sensor::Sensor *spatial_variance_sensor_{nullptr};
  sensor::Sensor *spatial_correlation_sensor_{nullptr};
  sensor::Sensor *spatial_gradient_sensor_{nullptr};
  sensor::Sensor *temporal_delta_mean_sensor_{nullptr};
  sensor::Sensor *temporal_delta_variance_sensor_{nullptr};
  
  // Preferences (replaces NVS)
  ESPPreferenceObject pref_;
  
  // Statistics
  uint32_t packets_processed_{0};
  uint32_t packets_dropped_{0};
  
  // NBVI calibration state
  nbvi_calibrator_t *nbvi_calibrator_{nullptr};
  
  // State flags
  bool csi_enabled_{false};
  bool traffic_gen_started_{false};
  bool ready_to_publish_{false};  // True when CSI is ready and calibration done
  
  // ESP-IDF event handler instances
  esp_event_handler_instance_t wifi_event_instance_connected_{nullptr};
  esp_event_handler_instance_t wifi_event_instance_disconnected_{nullptr};
};

}  // namespace espectre
}  // namespace esphome
