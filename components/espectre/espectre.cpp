/*
 * ESPectre - Main Component Implementation
 * 
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "espectre.h"
#include "threshold_number.h"
#include "calibrate_switch.h"
#include "utils.h"
#include "threshold.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"
#include "esp_wifi.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <vector>
#include <string>
#include <span>

#include "sdkconfig.h"

#ifdef USE_ESP32_BLE_SERVER
#include "esphome/components/esp32_ble_server/ble_server.h"
#include "esphome/components/esp32_ble_server/ble_characteristic.h"
#endif

namespace esphome {
namespace espectre {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  
  // 0. Initialize WiFi for optimal CSI capture
  esp_err_t wifi_init_err = this->wifi_lifecycle_.init();
  if (wifi_init_err != ESP_OK) {
    ESP_LOGE(TAG, "WiFi lifecycle init failed: %s. ESPectre setup aborted.",
             esp_err_to_name(wifi_init_err));
    this->mark_failed();
    return;
  }
  
  // 1. Configure the motion detector based on algorithm selection
  if (this->detection_algorithm_ == DetectionAlgorithm::ML) {
    // ML uses probability threshold (0.0-1.0), default 0.5 if not manually specified
    float ml_threshold = (this->threshold_mode_ == ThresholdMode::MANUAL) 
                         ? this->segmentation_threshold_ 
                         : ML_DEFAULT_THRESHOLD;
    this->segmentation_threshold_ = ml_threshold;  // Update for consistency
    this->ml_detector_ = MLDetector(this->segmentation_window_size_, ml_threshold);
    this->ml_detector_.configure_lowpass(this->lowpass_enabled_, this->lowpass_cutoff_);
    this->ml_detector_.configure_hampel(this->hampel_enabled_, this->hampel_window_, this->hampel_threshold_);
    this->detector_ = &this->ml_detector_;
    ESP_LOGI(TAG, "Using ML detector (window=%d, threshold=%.2f)", 
             this->segmentation_window_size_, ml_threshold);
  } else {
    this->mvs_detector_ = MVSDetector(this->segmentation_window_size_, this->segmentation_threshold_);
    this->mvs_detector_.configure_lowpass(this->lowpass_enabled_, this->lowpass_cutoff_);
    this->mvs_detector_.configure_hampel(this->hampel_enabled_, this->hampel_window_, this->hampel_threshold_);
    this->detector_ = &this->mvs_detector_;
    ESP_LOGI(TAG, "Using MVS detector (window=%d, threshold=%.2f)", 
             this->segmentation_window_size_, this->segmentation_threshold_);
  }
  
  // 2. Initialize managers (each manager handles its own internal initialization)
  this->nbvi_calibrator_.init(&this->csi_manager_);
  this->nbvi_calibrator_.set_mvs_window_size(this->segmentation_window_size_);
  this->nbvi_calibrator_.configure_lowpass(this->lowpass_enabled_, this->lowpass_cutoff_);
  this->nbvi_calibrator_.configure_hampel(this->hampel_enabled_, this->hampel_window_, this->hampel_threshold_);
  // Buffer size = 10 windows (matches CALIBRATION_NUM_WINDOWS constant)
  this->nbvi_calibrator_.set_buffer_size(this->segmentation_window_size_ * CALIBRATION_NUM_WINDOWS);
  this->traffic_generator_.init(this->traffic_generator_rate_, this->traffic_generator_mode_);
  this->udp_listener_.init(5555);  // UDP listener for external traffic mode

#ifdef USE_ESP32_BLE_SERVER
  if (this->ble_channel_enabled_) {
    if (this->ble_server_ == nullptr || this->ble_telemetry_char_ == nullptr ||
        this->ble_sysinfo_char_ == nullptr || this->ble_control_char_ == nullptr) {
      ESP_LOGW(TAG, "BLE channel enabled but server/characteristics are not configured; disabling BLE channel");
      this->ble_channel_enabled_ = false;
    } else {
      this->ble_server_->on_connect([this](uint16_t conn_id) { this->on_ble_client_connected_(conn_id); });
      this->ble_server_->on_disconnect([this](uint16_t conn_id) { this->on_ble_client_disconnected_(conn_id); });
      this->ble_control_char_->on_write([this](std::span<const uint8_t> value, uint16_t) {
        std::string command(reinterpret_cast<const char *>(value.data()), value.size());
        this->handle_ble_control_command_(command);
      });
    }
  }
#elif !defined(USE_ESP32_BLE_SERVER)
  if (this->ble_channel_enabled_) {
    ESP_LOGW(TAG, "BLE channel requested but esp32_ble_server is not available");
    this->ble_channel_enabled_ = false;
  }
#endif
  
  // 3. Initialize CSI manager with detector
  this->csi_manager_.init(
    this->detector_,
    this->selected_subcarriers_,
    this->publish_interval_,
    this->gain_lock_mode_
  );
  this->csi_manager_.set_game_mode_callback([this](float movement, float threshold) {
    if (!this->ble_channel_enabled_ || !this->ble_client_connected_ || this->ble_telemetry_char_ == nullptr) {
      return;
    }
    const uint32_t now = millis();
    if (now - this->last_ble_telemetry_ms_ < this->ble_telemetry_interval_ms_) {
      return;
    }
    this->last_ble_telemetry_ms_ = now;

    std::vector<uint8_t> payload(sizeof(float) * 2);
    memcpy(payload.data(), &movement, sizeof(float));
    memcpy(payload.data() + sizeof(float), &threshold, sizeof(float));
#ifdef USE_ESP32_BLE_SERVER
    this->ble_telemetry_char_->set_value(std::move(payload));
    this->ble_telemetry_char_->notify();
#endif
  });
  
  // 4. Register WiFi lifecycle handlers
  esp_err_t handlers_err = this->wifi_lifecycle_.register_handlers(
      [this]() { this->on_wifi_connected_(); },
      [this]() { this->on_wifi_disconnected_(); }
  );
  if (handlers_err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to register WiFi handlers: %s. ESPectre setup aborted.",
             esp_err_to_name(handlers_err));
    this->mark_failed();
    return;
  }
  
  ESP_LOGI(TAG, "ESPectre initialized successfully");
  ESP_LOGD(TAG, "[resources] Free heap: %lu bytes, largest block: %lu bytes",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_DEFAULT),
           (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));
}

ESpectreComponent::~ESpectreComponent() {
  // Detector cleanup is handled by destructor of member objects
}

void ESpectreComponent::on_wifi_connected_() {
  
  // Enable CSI using CSI Manager with periodic callback
  if (!this->csi_manager_.is_enabled()) {
    ESP_ERROR_CHECK(this->csi_manager_.enable(
      [this](MotionState state, uint32_t packets_received) {

        // Don't publish until ready
        if (!this->ready_to_publish_) return;
        
        // Re-publish threshold on first sensor update (HA is now connected)
        if (!this->threshold_republished_ && this->threshold_number_ != nullptr) {
          auto *threshold_num = static_cast<ESpectreThresholdNumber *>(this->threshold_number_);
          threshold_num->republish_state();
          this->threshold_republished_ = true;
        }
        
        // Log status with progress bar and actual CSI rate
        this->sensor_publisher_.log_status(TAG, this->detector_, state, packets_received);
        
        // Publish all sensors
        this->sensor_publisher_.publish_all(this->detector_, state);
      }
    ));
  }
  
  // Start traffic generator or UDP listener (external traffic mode)
  if (this->traffic_generator_rate_ > 0) {
    ESP_LOGD(TAG, "Starting traffic generator (rate: %u pps)...", this->traffic_generator_rate_);
    if (!this->traffic_generator_.is_running()) {
      if (!this->traffic_generator_.start()) {
        ESP_LOGW(TAG, "Failed to start traffic generator");
        return;
      }
      ESP_LOGI(TAG, "Traffic generator started successfully");
    } else {
      ESP_LOGI(TAG, "Traffic generator already running");
    }
  } else {
    // External traffic mode: start UDP listener
    ESP_LOGI(TAG, "Traffic generator disabled (rate: 0) - starting UDP listener for external traffic");
    if (!this->udp_listener_.is_running()) {
      if (!this->udp_listener_.start()) {
        ESP_LOGW(TAG, "Failed to start UDP listener");
      }
    }
  }
  
  // Two-phase calibration:
  // 1. Gain Lock (~3 seconds, 300 packets) - locks AGC/FFT for stable CSI
  // 2. Baseline Calibration (~7.5 seconds, 750 packets) - calculates normalization scale
  this->csi_manager_.set_gain_lock_callback([this]() {
    auto& gc = this->csi_manager_.get_gain_controller();
    auto mode = gc.get_mode();
    if (mode == GainLockMode::DISABLED) {
      ESP_LOGI(TAG, "Gain calibration complete (CV normalization enabled)");
    } else if (this->csi_manager_.is_gain_locked()) {
      ESP_LOGI(TAG, "Gain locked");
    } else {
      ESP_LOGI(TAG, "Gain calibration complete (strong signal, CV normalization enabled)");
    }
    
    // CV normalization: only needed when gain is not locked (AGC varies)
    // When gain is locked, raw std provides better sensitivity for all band types
    bool need_cv = gc.needs_cv_normalization();
    this->detector_->set_cv_normalization(need_cv);
    this->nbvi_calibrator_.set_cv_normalization(need_cv);
    
    this->start_calibration_();
  });
  
  // Ready to publish sensors (with internal or external traffic)
  this->ready_to_publish_ = true;
  this->threshold_republished_ = false;
}

void ESpectreComponent::on_wifi_disconnected_() {
  // Disable CSI using CSI Manager
  this->csi_manager_.disable();
  
  // Stop traffic generator
  if (this->traffic_generator_.is_running()) {
    this->traffic_generator_.stop();
  }
  
  // Stop UDP listener
  if (this->udp_listener_.is_running()) {
    this->udp_listener_.stop();
  }
  
  // Reset flags
  this->ready_to_publish_ = false;
}

void ESpectreComponent::loop() {
  // Drain UDP packets in external traffic mode
  if (this->udp_listener_.is_running()) {
    this->udp_listener_.loop();
  }
}

void ESpectreComponent::set_threshold_runtime(float threshold) {
  // Update internal state
  this->segmentation_threshold_ = threshold;
  
  // Update CSI manager (which updates the detector internally)
  this->csi_manager_.set_threshold(threshold);
  
  // Publish to Home Assistant
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(threshold);
  }
  
  ESP_LOGD(TAG, "Threshold updated to %.2f (session-only, recalculated at boot)", threshold);
}

void ESpectreComponent::start_calibration_() {
  // ML detector uses fixed subcarriers from training - no calibration needed
  if (this->detection_algorithm_ == DetectionAlgorithm::ML) {
    ESP_LOGI(TAG, "ML detector uses fixed subcarriers - skipping calibration");
    
    // Use unified default subcarriers
    memcpy(this->selected_subcarriers_, DEFAULT_SUBCARRIERS, 12);
    this->csi_manager_.update_subcarrier_selection(DEFAULT_SUBCARRIERS);
    
    // Update switch state
    if (this->calibrate_switch_ != nullptr) {
      static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
    }
    
    ESP_LOGI(TAG, "Using default subcarriers: [%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]",
             DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
             DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
             DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11]);
    return;
  }
  
  if (this->user_specified_subcarriers_) {
    ESP_LOGI(TAG, "Starting baseline calibration (fixed subcarriers)...");
  } else {
    ESP_LOGD(TAG, "Starting NBVI band calibration...");
  }
  
  // Update switch state to ON (calibrating)
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
  
  if (this->threshold_mode_ == ThresholdMode::MIN) {
    ESP_LOGW(TAG, "Threshold mode: min - maximum sensitivity, may cause false positives");
  }
  
  // Common callback for all calibrators
  auto calibration_callback = [this](const uint8_t* band, uint8_t size, 
                                     const std::vector<float>& cal_values, bool success) {
    if (success) {
      // Only update subcarriers if auto-selected (not user-specified)
      if (!this->user_specified_subcarriers_) {
        memcpy(this->selected_subcarriers_, band, size);
        this->csi_manager_.update_subcarrier_selection(band);
      }
    }
    
    // Apply adaptive threshold if calibration produced valid data
    if (band != nullptr && !cal_values.empty()) {
      float adaptive_threshold;
      uint8_t percentile;
      calculate_adaptive_threshold(cal_values, this->threshold_mode_, adaptive_threshold, percentile);
      
      this->best_pxx_ = adaptive_threshold;
      
      if (this->threshold_mode_ != ThresholdMode::MANUAL) {
        this->set_threshold_runtime(adaptive_threshold);
        ESP_LOGD(TAG, "Adaptive threshold: %.4f (P%d)", adaptive_threshold, percentile);
      } else {
        ESP_LOGD(TAG, "Using manual threshold: %.2f (adaptive would be: %.2f)", 
                 this->segmentation_threshold_, adaptive_threshold);
      }
      
      // Clear detector buffer
      this->csi_manager_.clear_detector_buffer();
      this->sensor_publisher_.reset_rate_counter();
    }

    this->traffic_generator_.resume();
    
    if (this->calibrate_switch_ != nullptr) {
      static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
    }
    
    ESP_LOGD(TAG, "Calibration %s", success ? "completed successfully" : "failed");
    ESP_LOGD(TAG, "[resources] Post-calibration heap: %lu bytes",
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_DEFAULT));
  };
  
  // Start calibration using NBVI
  this->nbvi_calibrator_.set_collection_complete_callback([this]() {
    this->traffic_generator_.pause();
  });
  
  esp_err_t cal_start_err = this->nbvi_calibrator_.start_calibration(
    this->selected_subcarriers_,
    12,
    calibration_callback
  );
  if (cal_start_err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to start calibration: %s", esp_err_to_name(cal_start_err));
    if (this->calibrate_switch_ != nullptr) {
      static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
    }
  }
}

void ESpectreComponent::trigger_recalibration() {
  // Check if calibration already in progress
  if (this->nbvi_calibrator_.is_calibrating()) {
    ESP_LOGW(TAG, "Calibration already in progress");
    return;
  }
  
  // Check if gain is locked (required for calibration)
  if (!this->csi_manager_.is_gain_locked()) {
    ESP_LOGW(TAG, "Cannot recalibrate: gain not yet locked");
    return;
  }
  
  ESP_LOGI(TAG, "Manual recalibration triggered");
  this->start_calibration_();
}

void ESpectreComponent::send_system_info_ble_() {
#ifndef USE_ESP32_BLE_SERVER
  return;
#else
  if (!this->ble_channel_enabled_ || this->ble_sysinfo_char_ == nullptr) {
    return;
  }
  auto notify_sysinfo = [this](const std::string &line) {
    this->ble_sysinfo_char_->set_value(line);
    this->ble_sysinfo_char_->notify();
  };
  const char* thr_mode = (this->threshold_mode_ == ThresholdMode::MANUAL) ? "manual" :
                         (this->threshold_mode_ == ThresholdMode::MIN) ? "min" : "auto";
  char line[96];
  notify_sysinfo("proto_version=1");
  snprintf(line, sizeof(line), "chip=%s", CONFIG_IDF_TARGET);
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "threshold=%.2f (%s)", this->segmentation_threshold_, thr_mode);
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "window=%d", this->segmentation_window_size_);
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "detector=%s", this->detector_ ? this->detector_->get_name() : "unknown");
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "subcarriers=%s", this->user_specified_subcarriers_ ? "yaml" : "auto");
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "lowpass=%s", this->lowpass_enabled_ ? "on" : "off");
  notify_sysinfo(line);
  if (this->lowpass_enabled_) {
    snprintf(line, sizeof(line), "lowpass_cutoff=%.1f", this->lowpass_cutoff_);
    notify_sysinfo(line);
  }
  snprintf(line, sizeof(line), "hampel=%s", this->hampel_enabled_ ? "on" : "off");
  notify_sysinfo(line);
  if (this->hampel_enabled_) {
    snprintf(line, sizeof(line), "hampel_window=%d", this->hampel_window_);
    notify_sysinfo(line);
    snprintf(line, sizeof(line), "hampel_threshold=%.1f", this->hampel_threshold_);
    notify_sysinfo(line);
  }
  snprintf(line, sizeof(line), "traffic_rate=%u", this->traffic_generator_rate_);
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "publish_interval=%u", this->publish_interval_);
  notify_sysinfo(line);
  snprintf(line, sizeof(line), "best_pxx=%.4f", this->best_pxx_);
  notify_sysinfo(line);
  notify_sysinfo("END");
#endif
}

void ESpectreComponent::on_ble_client_connected_(uint16_t conn_id) {
  (void) conn_id;
  this->ble_client_connected_ = true;
  this->last_ble_telemetry_ms_ = 0;
  this->send_system_info_ble_();
}

void ESpectreComponent::on_ble_client_disconnected_(uint16_t conn_id) {
  (void) conn_id;
#ifdef USE_ESP32_BLE_SERVER
  this->ble_client_connected_ = this->ble_server_ != nullptr && this->ble_server_->get_client_count() > 0;
#else
  this->ble_client_connected_ = false;
#endif
}

void ESpectreComponent::handle_ble_control_command_(const std::string &command) {
  if (!this->ble_channel_enabled_) {
    return;
  }
  if (command == "REQ_SYSINFO") {
    this->send_system_info_ble_();
    return;
  }
  if (command.rfind("SET_THRESHOLD:", 0) == 0) {
    const char *value_str = command.c_str() + 14;
    char *end_ptr = nullptr;
    errno = 0;
    float threshold = strtof(value_str, &end_ptr);
    bool parse_ok = (end_ptr != value_str) && (end_ptr != nullptr) && (*end_ptr == '\0') &&
                    (errno != ERANGE) && std::isfinite(threshold);
    if (!parse_ok || threshold < 0.0f || threshold > 10.0f) {
      ESP_LOGW(TAG, "Invalid BLE threshold command: %s", command.c_str());
      return;
    }
    this->set_threshold_runtime(threshold);
    this->send_system_info_ble_();
    return;
  }
  ESP_LOGW(TAG, "Unknown BLE control command: %s", command.c_str());
}

void ESpectreComponent::dump_config() {
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, "  _____ ____  ____           __            ");
  ESP_LOGCONFIG(TAG, " | ____/ ___||  _ \\ ___  ___| |_ _ __ ___ ");
  ESP_LOGCONFIG(TAG, " |  _| \\___ \\| |_) / _ \\/ __| __| '__/ _ \\");
  ESP_LOGCONFIG(TAG, " | |___ ___) |  __/  __/ (__| |_| | |  __/");
  ESP_LOGCONFIG(TAG, " |_____|____/|_|   \\___|\\___|\\__|_|  \\___|");
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, "      Wi-Fi CSI Motion Detection System");
  ESP_LOGCONFIG(TAG, "");
  const char* thr_mode_str = (this->threshold_mode_ == ThresholdMode::MANUAL) ? "Manual" :
                             (this->threshold_mode_ == ThresholdMode::MIN) ? "Min (P100)" : "Auto (P95x1.1)";
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Detector ........... %s", this->detector_ ? this->detector_->get_name() : "unknown");
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.2f (%s)", this->segmentation_threshold_, thr_mode_str);
  ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", this->segmentation_window_size_);
  ESP_LOGCONFIG(TAG, " └─ Baseline Pxx ....... %.4f", this->best_pxx_);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                this->selected_subcarriers_[0], this->selected_subcarriers_[1],
                this->selected_subcarriers_[2], this->selected_subcarriers_[3],
                this->selected_subcarriers_[4], this->selected_subcarriers_[5],
                this->selected_subcarriers_[6], this->selected_subcarriers_[7],
                this->selected_subcarriers_[8], this->selected_subcarriers_[9],
                this->selected_subcarriers_[10], this->selected_subcarriers_[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", 
                this->user_specified_subcarriers_ ? "YAML" : "NBVI");
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  if (this->traffic_generator_rate_ > 0) {
    const char* mode_str = (this->traffic_generator_mode_ == TrafficGeneratorMode::PING) ? "ping" : "dns";
    ESP_LOGCONFIG(TAG, " ├─ Mode ............... %s", mode_str);
    ESP_LOGCONFIG(TAG, " ├─ Rate ............... %u pps", this->traffic_generator_rate_);
    ESP_LOGCONFIG(TAG, " └─ Status ............. %s", 
                  this->traffic_generator_.is_running() ? "[RUNNING]" : "[STOPPED]");
  } else {
    ESP_LOGCONFIG(TAG, " └─ Mode ............... External Traffic");
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " PUBLISH INTERVAL");
  ESP_LOGCONFIG(TAG, " └─ Packets ............ %u", this->publish_interval_);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " LOW-PASS FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", this->lowpass_enabled_ ? "[ENABLED]" : "[DISABLED]");
  if (this->lowpass_enabled_) {
    ESP_LOGCONFIG(TAG, " └─ Cutoff ............. %.1f Hz", this->lowpass_cutoff_);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", this->hampel_enabled_ ? "[ENABLED]" : "[DISABLED]");
  if (this->hampel_enabled_) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", this->hampel_window_);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", this->hampel_threshold_);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " GAIN LOCK");
  const char* gain_mode_str = "auto";
  if (this->gain_lock_mode_ == GainLockMode::ENABLED) {
    gain_mode_str = "enabled";
  } else if (this->gain_lock_mode_ == GainLockMode::DISABLED) {
    gain_mode_str = "disabled";
  }
  ESP_LOGCONFIG(TAG, " └─ Mode ............... %s", gain_mode_str);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SENSORS");
  ESP_LOGCONFIG(TAG, " ├─ Movement ........... %s", 
                this->sensor_publisher_.has_movement_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " └─ Motion Binary ...... %s", 
                this->sensor_publisher_.has_motion_binary_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, "");
}

}  // namespace espectre
}  // namespace esphome
