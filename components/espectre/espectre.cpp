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
#include "esp_wifi.h"
#include "esp_err.h"
#include <cstring>

namespace esphome {
namespace espectre {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  
  // 0. Initialize WiFi for optimal CSI capture
  this->wifi_lifecycle_.init();
  
  // 1. Select and configure the motion detector based on detection_algorithm_
  if (this->detection_algorithm_ == DetectionAlgorithm::PCA) {
    this->detector_ = &this->pca_detector_;
    ESP_LOGI(TAG, "Using PCA detector");
  } else {
    // MVS is default - configure with parameters
    this->mvs_detector_ = MVSDetector(this->segmentation_window_size_, this->segmentation_threshold_);
    this->mvs_detector_.configure_lowpass(this->lowpass_enabled_, this->lowpass_cutoff_);
    this->mvs_detector_.configure_hampel(this->hampel_enabled_, this->hampel_window_, this->hampel_threshold_);
    this->detector_ = &this->mvs_detector_;
    ESP_LOGI(TAG, "Using MVS detector (window=%d, threshold=%.2f)", 
             this->segmentation_window_size_, this->segmentation_threshold_);
  }
  
  // 2. Initialize managers (each manager handles its own internal initialization)
  // Select and initialize the active calibrator based on configuration
  // PCA uses its own calibrator, MVS uses NBVI or P95
  if (this->detection_algorithm_ == DetectionAlgorithm::PCA) {
    this->active_calibrator_ = &this->pca_calibrator_;
    ESP_LOGI(TAG, "Using PCA calibrator");
  } else if (this->segmentation_calibration_ == CalibrationAlgorithm::NBVI) {
    this->active_calibrator_ = &this->nbvi_calibrator_;
  } else {
    this->active_calibrator_ = &this->p95_calibrator_;
  }
  this->active_calibrator_->init(&this->csi_manager_);
  this->traffic_generator_.init(this->traffic_generator_rate_, this->traffic_generator_mode_);
  this->udp_listener_.init(5555);  // UDP listener for external traffic mode
  this->serial_streamer_.init();
  this->serial_streamer_.set_threshold_callback([this](float threshold) {
    this->set_threshold_runtime(threshold);
  });
  this->serial_streamer_.set_start_callback([this]() {
    this->send_system_info_();
  });
  
  // 3. Initialize CSI manager with detector
  this->csi_manager_.init(
    this->detector_,
    this->selected_subcarriers_,
    this->publish_interval_,
    this->gain_lock_mode_
  );
  
  // 4. Register WiFi lifecycle handlers
  this->wifi_lifecycle_.register_handlers(
      [this]() { this->on_wifi_connected_(); },
      [this]() { this->on_wifi_disconnected_(); }
  );
  
  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  // Detector cleanup is handled by destructor of member objects
}

void ESpectreComponent::on_wifi_connected_() {
  
  // Enable CSI using CSI Manager with periodic callback
  if (!this->csi_manager_.is_enabled()) {
    ESP_ERROR_CHECK(this->csi_manager_.enable(
      [this](csi_motion_state_t state, uint32_t packets_received) {

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
    
    // Set up game mode callback (called every CSI packet when active)
    this->csi_manager_.set_game_mode_callback(
      [this](float movement, float threshold) {
        if (this->serial_streamer_.is_active()) {
          this->serial_streamer_.send_data(movement, threshold);
        }
      }
    );
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
  // 2. Baseline Calibration (~7 seconds, 700 packets) - calculates normalization scale
  this->csi_manager_.set_gain_lock_callback([this]() {
    ESP_LOGI(TAG, "Gain locked");
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
  // Check for game mode Serial commands
  this->serial_streamer_.check_commands();
  
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
  
  ESP_LOGI(TAG, "Threshold updated to %.2f (session-only, recalculated at boot)", threshold);
}

void ESpectreComponent::start_calibration_() {
  const char* algo_name;
  if (this->detection_algorithm_ == DetectionAlgorithm::PCA) {
    algo_name = "PCA";
  } else {
    algo_name = (this->segmentation_calibration_ == CalibrationAlgorithm::NBVI) ? "NBVI" : "P95";
  }
  
  if (this->user_specified_subcarriers_) {
    ESP_LOGI(TAG, "Starting baseline calibration (fixed subcarriers)...");
  } else {
    ESP_LOGI(TAG, "Starting band calibration (%s algorithm)...", algo_name);
  }
  
  // Update switch state to ON (calibrating)
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
  
  // Determine threshold mode for calculation
  espectre::ThresholdMode calc_mode = (this->threshold_mode_ == ThresholdMode::MIN) 
    ? espectre::ThresholdMode::MIN 
    : espectre::ThresholdMode::AUTO;
  
  if (this->threshold_mode_ == ThresholdMode::MIN) {
    ESP_LOGW(TAG, "Threshold mode: min - maximum sensitivity, may cause false positives");
  }
  
  // Common callback for all calibrators
  bool is_pca = (this->detection_algorithm_ == DetectionAlgorithm::PCA);
  auto calibration_callback = [this, calc_mode, is_pca](const uint8_t* band, uint8_t size, 
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
      float factor;
      float pxx;
      calculate_adaptive_threshold(cal_values, calc_mode, is_pca, adaptive_threshold, percentile, factor, pxx);
      
      this->best_pxx_ = pxx;
      
      if (this->threshold_mode_ != ThresholdMode::MANUAL) {
        this->set_threshold_runtime(adaptive_threshold);
        if (is_pca) {
          ESP_LOGI(TAG, "Adaptive threshold: %.4f (min_corr=%.4f)", 
                   adaptive_threshold, pxx);
        } else {
          ESP_LOGI(TAG, "Adaptive threshold: %.4f (P%d=%.4f x %.1f)", 
                   adaptive_threshold, percentile, pxx, factor);
        }
      } else {
        ESP_LOGI(TAG, "Using manual threshold: %.2f (adaptive would be: %.2f)", 
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
    
    ESP_LOGI(TAG, "Calibration %s", success ? "completed successfully" : "failed");
  };
  
  // P95-specific initialization
  if (this->segmentation_calibration_ == CalibrationAlgorithm::P95) {
    this->p95_calibrator_.init_subcarrier_config();
    this->p95_calibrator_.set_skip_subcarrier_selection(this->user_specified_subcarriers_);
  }
  
  // Start calibration using the active calibrator (polymorphic)
  this->active_calibrator_->set_collection_complete_callback([this]() {
    this->traffic_generator_.pause();
  });
  
  this->active_calibrator_->start_calibration(
    this->selected_subcarriers_,
    12,
    calibration_callback
  );
}

void ESpectreComponent::trigger_recalibration() {
  // Check if calibration already in progress
  if (this->is_calibrating()) {
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

void ESpectreComponent::send_system_info_() {
  ESP_LOGI(TAG, "[sysinfo] chip=" CONFIG_IDF_TARGET);
  const char* thr_mode = (this->threshold_mode_ == ThresholdMode::MANUAL) ? "manual" :
                         (this->threshold_mode_ == ThresholdMode::MIN) ? "min" : "auto";
  ESP_LOGI(TAG, "[sysinfo] threshold=%.2f (%s)", this->segmentation_threshold_, thr_mode);
  ESP_LOGI(TAG, "[sysinfo] window=%d", this->segmentation_window_size_);
  ESP_LOGI(TAG, "[sysinfo] detector=%s", this->detector_ ? this->detector_->get_name() : "unknown");
  ESP_LOGI(TAG, "[sysinfo] subcarriers=%s", this->user_specified_subcarriers_ ? "yaml" : "auto");
  ESP_LOGI(TAG, "[sysinfo] lowpass=%s", this->lowpass_enabled_ ? "on" : "off");
  if (this->lowpass_enabled_) {
    ESP_LOGI(TAG, "[sysinfo] lowpass_cutoff=%.1f", this->lowpass_cutoff_);
  }
  ESP_LOGI(TAG, "[sysinfo] hampel=%s", this->hampel_enabled_ ? "on" : "off");
  if (this->hampel_enabled_) {
    ESP_LOGI(TAG, "[sysinfo] hampel_window=%d", this->hampel_window_);
    ESP_LOGI(TAG, "[sysinfo] hampel_threshold=%.1f", this->hampel_threshold_);
  }
  ESP_LOGI(TAG, "[sysinfo] traffic_rate=%u", this->traffic_generator_rate_);
  ESP_LOGI(TAG, "[sysinfo] publish_interval=%u", this->publish_interval_);
  ESP_LOGI(TAG, "[sysinfo] best_pxx=%.4f", this->best_pxx_);
  ESP_LOGI(TAG, "[sysinfo] END");
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
                             (this->threshold_mode_ == ThresholdMode::MIN) ? "Min (P100×1.0)" : "Auto (P95×1.4)";
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
  const char* algo_str;
  if (this->detection_algorithm_ == DetectionAlgorithm::PCA) {
    algo_str = "PCA";
  } else {
    algo_str = (this->segmentation_calibration_ == CalibrationAlgorithm::NBVI) ? "NBVI" : "P95";
  }
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", 
                this->user_specified_subcarriers_ ? "YAML" : algo_str);
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
