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
#include "utils.h"
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
  
  // 1. Initialize configuration manager (load config before initializing managers)
  this->config_manager_.init(
      global_preferences->make_preference<ESpectreConfig>(fnv1_hash("espectre_cfg"))
  );
  
  // 2. Load configuration from preferences
  ESpectreConfig config;
  if (this->config_manager_.load(config)) {
    this->segmentation_threshold_ = config.segmentation_threshold;
    this->segmentation_window_size_ = config.segmentation_window_size;
    this->traffic_generator_rate_ = config.traffic_generator_rate;
    this->hampel_enabled_ = config.hampel_enabled;
    this->hampel_window_ = config.hampel_window;
    this->hampel_threshold_ = config.hampel_threshold;
  }
  
  // 3. Initialize CSI processor (allocates buffer internally)
  if (!csi_processor_init(&this->csi_processor_, 
                          this->segmentation_window_size_, 
                          this->segmentation_threshold_)) {
    ESP_LOGE(TAG, "Failed to initialize CSI processor");
    return;  // Component initialization failed
  }
  
  // 4. Initialize managers (each manager handles its own internal initialization)
  this->calibration_manager_.init(&this->csi_manager_);
  this->traffic_generator_.init(this->traffic_generator_rate_);
  this->csi_manager_.init(
    &this->csi_processor_,
    this->selected_subcarriers_,
    this->segmentation_threshold_,
    this->segmentation_window_size_,
    this->traffic_generator_rate_,
    this->hampel_enabled_,
    this->hampel_window_,
    this->hampel_threshold_
  );
  
  // 5. Register WiFi lifecycle handlers
  this->wifi_lifecycle_.register_handlers(
      [this]() { this->on_wifi_connected_(); },
      [this]() { this->on_wifi_disconnected_(); }
  );
  
  ESP_LOGI(TAG, "🛜 ESPectre 👻 - initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  // Cleanup CSI processor (deallocates turbulence buffer)
  csi_processor_cleanup(&this->csi_processor_);
}

void ESpectreComponent::on_wifi_connected_() {
  
  // Enable CSI using CSI Manager with periodic callback
  if (!this->csi_manager_.is_enabled()) {
    ESP_ERROR_CHECK(this->csi_manager_.enable(
      [this](csi_motion_state_t state) {

        // Don't publish until ready
        if (!this->ready_to_publish_) return;
        
        // Log status with progress bar and CSI rate
        this->sensor_publisher_.log_status(TAG, &this->csi_processor_, state, this->traffic_generator_rate_);
        
        // Publish all sensors
        this->sensor_publisher_.publish_all(&this->csi_processor_, state);
      }
    ));
  }
  
  // Start traffic generator
  ESP_LOGD(TAG, "Startingtart traffic generator (rate: %u pps)...", this->traffic_generator_rate_);
  if (!this->traffic_generator_.is_running()) {
    if (!this->traffic_generator_.start()) {
      ESP_LOGW(TAG, "Failed to start traffic generator");
      return;
    }
    ESP_LOGI(TAG, "Traffic generator started successfully");
  } else {
    ESP_LOGI(TAG, "Traffic generator already running");
  }
  
  // Run auto-calibration if user didn't specify subcarriers
  if (this->traffic_generator_.is_running() && !this->user_specified_subcarriers_) {
    this->calibration_manager_.start_auto_calibration(
      this->selected_subcarriers_,
      12,  // Always 12 subcarriers
      [this](const uint8_t* band, uint8_t size, bool success) {
        if (success) {
          memcpy(this->selected_subcarriers_, band, size);
          this->csi_manager_.update_subcarrier_selection(band);
        }
      }
    );
  }
  
  // Ready to publish sensors
  if (this->traffic_generator_.is_running()) {
    this->ready_to_publish_ = true;
  }
}

void ESpectreComponent::on_wifi_disconnected_() {
  // Disable CSI using CSI Manager
  this->csi_manager_.disable();
  
  // Stop traffic generator
  if (this->traffic_generator_.is_running()) {
    this->traffic_generator_.stop();
  }
  
  // Reset flags
  this->ready_to_publish_ = false;
}

void ESpectreComponent::loop() {
  // Event-driven component: all processing handled by managers via callbacks
}

void ESpectreComponent::set_threshold_runtime(float threshold) {
  // Update internal state
  this->segmentation_threshold_ = threshold;
  
  // Update CSI processor
  csi_processor_set_threshold(&this->csi_processor_, threshold);
  
  // Update CSI manager
  this->csi_manager_.set_threshold(threshold);
  
  // Save to preferences
  ESpectreConfig config;
  config.segmentation_threshold = threshold;
  config.segmentation_window_size = this->segmentation_window_size_;
  config.traffic_generator_rate = this->traffic_generator_rate_;
  config.hampel_enabled = this->hampel_enabled_;
  config.hampel_window = this->hampel_window_;
  config.hampel_threshold = this->hampel_threshold_;
  this->config_manager_.save(config);
  
  ESP_LOGI(TAG, "Threshold updated to %.2f (saved to flash)", threshold);
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
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.2f", this->segmentation_threshold_);
  ESP_LOGCONFIG(TAG, " └─ Window ............. %d pkts", this->segmentation_window_size_);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                this->selected_subcarriers_[0], this->selected_subcarriers_[1],
                this->selected_subcarriers_[2], this->selected_subcarriers_[3],
                this->selected_subcarriers_[4], this->selected_subcarriers_[5],
                this->selected_subcarriers_[6], this->selected_subcarriers_[7],
                this->selected_subcarriers_[8], this->selected_subcarriers_[9],
                this->selected_subcarriers_[10], this->selected_subcarriers_[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", 
                this->user_specified_subcarriers_ ? "YAML" : "Auto (NBVI)");
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  ESP_LOGCONFIG(TAG, " ├─ Rate ............... %u pps", this->traffic_generator_rate_);
  ESP_LOGCONFIG(TAG, " └─ Status ............. %s", 
                this->traffic_generator_.is_running() ? "[RUNNING]" : "[STOPPED]");
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", this->hampel_enabled_ ? "[ENABLED]" : "[DISABLED]");
  if (this->hampel_enabled_) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", this->hampel_window_);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", this->hampel_threshold_);
  }
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
