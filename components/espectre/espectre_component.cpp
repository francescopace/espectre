#include "espectre_component.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esp_wifi.h"
#include "esp_err.h"
#include <cstring>

namespace esphome {
namespace espectre {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  
  // 1. Initialize C modules
  csi_processor_init(&this->csi_processor_);
  filter_buffer_init(&this->filter_buffer_);
  butterworth_init(&this->butterworth_);
  wavelet_init(&this->wavelet_, this->wavelet_level_, 
               this->wavelet_threshold_, WAVELET_THRESH_SOFT);
  
  // 2. Initialize preferences object
  this->pref_ = global_preferences->make_preference<ESpectrePreferences>(
      fnv1_hash("espectre_cfg"));
  
  // 3. Load other configuration from preferences (not subcarriers)
  this->load_preferences_();
  
  // 4. Set subcarriers:
  //    - If user specified in YAML: use those (already set via set_selected_subcarriers)
  //    - Otherwise: use DEFAULT_SUBCARRIERS temporarily (NBVI will calibrate at WiFi connect)
  
  if (!this->user_specified_subcarriers_) {
    memcpy(this->selected_subcarriers_, DEFAULT_SUBCARRIERS, sizeof(DEFAULT_SUBCARRIERS));
    this->num_selected_subcarriers_ = sizeof(DEFAULT_SUBCARRIERS) / sizeof(DEFAULT_SUBCARRIERS[0]);
  }
  
  // 5. Apply configuration
  csi_processor_set_threshold(&this->csi_processor_, this->segmentation_threshold_);
  csi_processor_set_window_size(&this->csi_processor_, this->segmentation_window_size_);
  csi_set_subcarrier_selection(this->selected_subcarriers_, 
                                this->num_selected_subcarriers_);
  
  // 6. Initialize traffic generator (will be started when WiFi connects)
  traffic_generator_init();
  
  // 7. Register WiFi event handlers using ESP-IDF
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT,
      IP_EVENT_STA_GOT_IP,
      &ESpectreComponent::wifi_event_handler_,
      this,
      &this->wifi_event_instance_connected_
  ));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT,
      WIFI_EVENT_STA_DISCONNECTED,
      &ESpectreComponent::wifi_event_handler_,
      this,
      &this->wifi_event_instance_disconnected_
  ));
  ESP_LOGI(TAG, "WiFi event handlers registered");
  
  // Log sensor configuration
  ESP_LOGI(TAG, "Sensor configuration:");
  ESP_LOGI(TAG, "  Movement sensor: %s", this->movement_sensor_ != nullptr ? "CONFIGURED" : "NOT CONFIGURED");
  ESP_LOGI(TAG, "  Threshold sensor: %s", this->threshold_sensor_ != nullptr ? "CONFIGURED" : "NOT CONFIGURED");
  ESP_LOGI(TAG, "  Turbulence sensor: %s", this->turbulence_sensor_ != nullptr ? "CONFIGURED" : "NOT CONFIGURED");
  ESP_LOGI(TAG, "  Motion binary sensor: %s", this->motion_binary_sensor_ != nullptr ? "CONFIGURED" : "NOT CONFIGURED");
  
  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

void ESpectreComponent::setup_csi_() {
  
#if CONFIG_IDF_TARGET_ESP32C6
  wifi_csi_config_t csi_config = {
    .enable = 1,
    .acquire_csi_legacy = 1,
    .acquire_csi_ht20 = 1,
    .acquire_csi_ht40 = 0,
    .acquire_csi_su = 1,
    .acquire_csi_mu = 0,
    .acquire_csi_dcm = 0,
    .acquire_csi_beamformed = 0,
    .acquire_csi_he_stbc = 0,
    .val_scale_cfg = 0,
    .dump_ack_en = 0,
  };
  ESP_LOGI(TAG, "CSI config: ESP32-C6 mode");
#else
  wifi_csi_config_t csi_config = {
    .lltf_en = true,
    .htltf_en = true,
    .stbc_htltf2_en = true,
    .ltf_merge_en = true,
    .channel_filter_en = false,
    .manu_scale = false,
  };
  ESP_LOGI(TAG, "CSI config: ESP32-S3 mode");
#endif
  
  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&ESpectreComponent::csi_callback_wrapper_, this));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));
  
  ESP_LOGI(TAG, "CSI initialized and enabled");
}

void ESpectreComponent::csi_callback_wrapper_(void *ctx, wifi_csi_info_t *data) {
  ESpectreComponent *component = static_cast<ESpectreComponent *>(ctx);
  component->handle_csi_data_(data);
}

void ESpectreComponent::wifi_event_handler_(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
  ESpectreComponent *component = static_cast<ESpectreComponent *>(arg);
  component->handle_wifi_event_(event_base, event_id, event_data);
}

void ESpectreComponent::handle_wifi_event_(esp_event_base_t event_base, int32_t event_id, void* event_data) {
  // Handle WiFi connected event
  if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ESP_LOGI(TAG, "WiFi connected");
    
    // Configure CSI (only once)
    if (!this->csi_enabled_) {
      this->setup_csi_();
      this->csi_enabled_ = true;
    }

    // Start traffic generator when WiFi is connected (only once)
    if (!this->traffic_gen_started_ && this->traffic_generator_rate_ > 0) {
      if (traffic_generator_start(this->traffic_generator_rate_)) {
        ESP_LOGI(TAG, "Traffic generator started (%u pps)", this->traffic_generator_rate_);
        this->traffic_gen_started_ = true;
      } else {
        ESP_LOGW(TAG, "Failed to start traffic generator");
        return;
      }
    }
    
    // Run NBVI auto-calibration if user didn't specify subcarriers in YAML
    if (this->traffic_gen_started_ && !this->user_specified_subcarriers_) {
      ESP_LOGI(TAG, "Running NBVI auto-calibration...");
      this->run_nbvi_calibration_();
    }
    
    // Ready to publish sensors
    if (this->traffic_gen_started_) {
      this->ready_to_publish_ = true;
    }
  }
  
  // Handle WiFi disconnected event
  else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
    ESP_LOGW(TAG, "WiFi disconnected");
    
    // Disable CSI to save resources
    if (this->csi_enabled_) {
      esp_wifi_set_csi(false);
      ESP_LOGI(TAG, "CSI disabled");
    }
    
    // Stop traffic generator if running
    if (traffic_generator_is_running()) {
      traffic_generator_stop();
      ESP_LOGI(TAG, "Traffic generator stopped");
    }
    
    // Reset flags to allow restart on reconnection
    this->traffic_gen_started_ = false;
    this->csi_enabled_ = false;
    this->ready_to_publish_ = false;
  }
}

void ESpectreComponent::handle_csi_data_(wifi_csi_info_t *data) {
  int8_t *csi_data = data->buf;
  size_t csi_len = data->len;
  
  if (csi_len < 10) {
    ESP_LOGW(TAG, "CSI data too short: %d bytes", csi_len);
    return;
  }
  
  // If NBVI calibration is in progress, add packet to calibrator buffer
  if (this->nbvi_calibrator_ != nullptr) {
    nbvi_calibrator_add_packet(this->nbvi_calibrator_, csi_data, csi_len);
    return;  // Skip normal processing during calibration
  }
  
  // Process CSI packet
  csi_process_packet(&this->csi_processor_,
                     csi_data, csi_len,
                     this->selected_subcarriers_,
                     this->num_selected_subcarriers_,
                     this->features_enabled_ ? &this->current_features_ : nullptr);
  
  // Update motion state
  this->motion_state_ = csi_processor_get_state(&this->csi_processor_);
  
  this->packets_processed_++;
  
  // Publish sensors every N packets (N = traffic_generator_rate)
  // At 100 pps, this means publishing every 1 second
  if (this->packets_processed_ >= this->traffic_generator_rate_) {
    this->update_sensors_();
    this->packets_processed_ = 0;  // Reset counter
  }
}

void ESpectreComponent::loop() {
  // Event-driven component: sensor updates triggered by CSI packet processing
  // (see handle_csi_data_ for update logic)
}

void ESpectreComponent::update_sensors_() {
  // Don't publish sensors until CSI is ready and calibration is complete
  if (!this->ready_to_publish_) {
    return;
  }
  
  // Get current values for logging
  float moving_variance = csi_processor_get_moving_variance(&this->csi_processor_);
  float threshold = csi_processor_get_threshold(&this->csi_processor_);
  float turbulence = csi_processor_get_last_turbulence(&this->csi_processor_);
  bool is_motion = (this->motion_state_ == CSI_STATE_MOTION);
  
  // Create progress bar (like micro-espectre)
  float progress = (threshold > 0) ? (moving_variance / threshold) : 0.0f;
  const int width = 20;
  const int threshold_pos = 15;
  int filled = (int)(progress * threshold_pos);
  filled = (filled < 0) ? 0 : (filled > width ? width : filled);
  
  char bar[23];  // '[' + 20 chars + '|' + ']' = 23
  bar[0] = '[';
  for (int i = 0; i < width; i++) {
    if (i == threshold_pos) {
      bar[i + 1] = '|';
    } else if (i < filled) {
      bar[i + 1] = '#';
    } else {
      bar[i + 1] = '-';
    }
  }
  bar[21] = ']';
  bar[22] = '\0';
  
  int percent = (int)(progress * 100);
  
  // Log publication (like micro-espectre)
  ESP_LOGI(TAG, "📊 %s %d%% | mvmt:%.4f thr:%.4f turb:%.2f | %s",
           bar, percent, moving_variance, threshold, turbulence,
           is_motion ? "MOTION" : "IDLE");
  
  // Update binary sensor (motion)
  if (this->motion_binary_sensor_ != nullptr) {
    this->motion_binary_sensor_->publish_state(is_motion);
  }
  
  // Update numeric sensors
  if (this->movement_sensor_ != nullptr) {
    this->movement_sensor_->publish_state(moving_variance);
  }
  
  if (this->threshold_sensor_ != nullptr) {
    this->threshold_sensor_->publish_state(threshold);
  }
  
  if (this->turbulence_sensor_ != nullptr) {
    this->turbulence_sensor_->publish_state(turbulence);
  }
  
  // Update feature sensors (if enabled)
  if (this->features_enabled_) {
    if (this->variance_sensor_ != nullptr) {
      this->variance_sensor_->publish_state(this->current_features_.variance);
    }
    if (this->skewness_sensor_ != nullptr) {
      this->skewness_sensor_->publish_state(this->current_features_.skewness);
    }
    if (this->kurtosis_sensor_ != nullptr) {
      this->kurtosis_sensor_->publish_state(this->current_features_.kurtosis);
    }
    if (this->entropy_sensor_ != nullptr) {
      this->entropy_sensor_->publish_state(this->current_features_.entropy);
    }
    if (this->iqr_sensor_ != nullptr) {
      this->iqr_sensor_->publish_state(this->current_features_.iqr);
    }
    if (this->spatial_variance_sensor_ != nullptr) {
      this->spatial_variance_sensor_->publish_state(this->current_features_.spatial_variance);
    }
    if (this->spatial_correlation_sensor_ != nullptr) {
      this->spatial_correlation_sensor_->publish_state(this->current_features_.spatial_correlation);
    }
    if (this->spatial_gradient_sensor_ != nullptr) {
      this->spatial_gradient_sensor_->publish_state(this->current_features_.spatial_gradient);
    }
    if (this->temporal_delta_mean_sensor_ != nullptr) {
      this->temporal_delta_mean_sensor_->publish_state(this->current_features_.temporal_delta_mean);
    }
    if (this->temporal_delta_variance_sensor_ != nullptr) {
      this->temporal_delta_variance_sensor_->publish_state(this->current_features_.temporal_delta_variance);
    }
  }
}

bool ESpectreComponent::load_preferences_() {
  ESpectrePreferences pref;
  if (!this->pref_.load(&pref)) {
    ESP_LOGI(TAG, "No saved configuration found");
    return false;
  }
  
  // Load configuration
  this->segmentation_threshold_ = pref.segmentation_threshold;
  this->segmentation_window_size_ = pref.segmentation_window_size;
  this->traffic_generator_rate_ = pref.traffic_generator_rate;
  this->features_enabled_ = pref.features_enabled;
  this->butterworth_enabled_ = pref.butterworth_enabled;
  this->wavelet_enabled_ = pref.wavelet_enabled;
  this->wavelet_level_ = pref.wavelet_level;
  this->wavelet_threshold_ = pref.wavelet_threshold;
  this->hampel_enabled_ = pref.hampel_enabled;
  this->hampel_threshold_ = pref.hampel_threshold;
  this->savgol_enabled_ = pref.savgol_enabled;
  
  ESP_LOGI(TAG, "Loaded configuration from preferences");
  return true;
}

void ESpectreComponent::save_preferences_() {
  ESpectrePreferences pref;
  
  pref.segmentation_threshold = this->segmentation_threshold_;
  pref.segmentation_window_size = this->segmentation_window_size_;
  pref.traffic_generator_rate = this->traffic_generator_rate_;
  pref.features_enabled = this->features_enabled_;
  pref.butterworth_enabled = this->butterworth_enabled_;
  pref.wavelet_enabled = this->wavelet_enabled_;
  pref.wavelet_level = this->wavelet_level_;
  pref.wavelet_threshold = this->wavelet_threshold_;
  pref.hampel_enabled = this->hampel_enabled_;
  pref.hampel_threshold = this->hampel_threshold_;
  pref.savgol_enabled = this->savgol_enabled_;
  
  this->pref_.save(&pref);
  ESP_LOGI(TAG, "Saved configuration to preferences");
}

void ESpectreComponent::run_nbvi_calibration_() {
  // NBVI calibration implementation (adapted from espectre.c)
  
  ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
  ESP_LOGI(TAG, "🧬 NBVI Auto-Calibration Starting");
  ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
  ESP_LOGI(TAG, "Please remain still for 5 seconds...");
  ESP_LOGI(TAG, "Collecting CSI data for automatic subcarrier selection");
  
  // Initialize NBVI calibrator (uses default parameters from nbvi_calibrator_init)
  nbvi_calibrator_t calibrator;
  esp_err_t err = nbvi_calibrator_init(&calibrator);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize NBVI calibrator: %s", esp_err_to_name(err));
    return;
  }
  
  // Set global pointer for CSI callback to populate calibrator buffer
  this->nbvi_calibrator_ = &calibrator;
  
  // Wait for CSI callback to collect packets
  ESP_LOGI(TAG, "NBVI: Collecting packets via CSI callback...");
  uint16_t timeout_counter = 0;
  const uint16_t max_timeout_ms = 15000;  // 15 seconds timeout
  uint16_t last_count = 0;
  uint8_t last_progress = 0;
  
  while (calibrator.buffer_count < calibrator.buffer_size) {
    delay(100);  // ESPHome delay function
    timeout_counter += 100;
    
    // Print progress bar when count changes
    if (calibrator.buffer_count != last_count) {
      last_count = calibrator.buffer_count;
      timeout_counter = 0;  // Reset timeout on progress
      
      // Calculate progress percentage
      uint8_t progress = (calibrator.buffer_count * 100) / calibrator.buffer_size;
      
      // Print progress bar every 10%
      if (progress >= last_progress + 10 || calibrator.buffer_count == calibrator.buffer_size) {
        // Create progress bar: [████████░░] 80%
        char bar[21];  // 20 chars + null terminator
        uint8_t filled = progress / 5;  // 0-20 filled chars
        for (uint8_t i = 0; i < 20; i++) {
          bar[i] = (i < filled) ? '#' : '-';
        }
        bar[20] = '\0';
        
        ESP_LOGI(TAG, "NBVI: [%s] %d%% (%d/%d)",
                 bar, progress, calibrator.buffer_count, calibrator.buffer_size);
        last_progress = progress;
      }
    }
    
    // Check for timeout
    if (timeout_counter >= max_timeout_ms) {
      ESP_LOGE(TAG, "NBVI: Timeout waiting for CSI packets (collected %d/%d)",
               calibrator.buffer_count, calibrator.buffer_size);
      ESP_LOGE(TAG, "NBVI: Calibration aborted - using default band");
      this->nbvi_calibrator_ = nullptr;
      nbvi_calibrator_free(&calibrator);
      return;
    }
  }
  
  // Clear global pointer
  this->nbvi_calibrator_ = nullptr;
  
  ESP_LOGI(TAG, "NBVI: Collection complete (%d packets)", calibrator.buffer_size);
  
  // Run calibration
  uint8_t selected_band[12];
  uint8_t band_size = 0;
  
  err = nbvi_calibrator_calibrate(&calibrator,
                                  this->selected_subcarriers_,
                                  this->num_selected_subcarriers_,
                                  selected_band,
                                  &band_size);
  
  bool success = (err == ESP_OK && band_size == 12);
  
  if (success) {
    // Update configuration
    memcpy(this->selected_subcarriers_, selected_band, 12);
    this->num_selected_subcarriers_ = 12;
    
    // Update CSI processor
    csi_set_subcarrier_selection(selected_band, 12);
    
    ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
    ESP_LOGI(TAG, "✅ NBVI Calibration Successful!");
    ESP_LOGI(TAG, "   Selected band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
             selected_band[0], selected_band[1], selected_band[2], selected_band[3],
             selected_band[4], selected_band[5], selected_band[6], selected_band[7],
             selected_band[8], selected_band[9], selected_band[10], selected_band[11]);
    ESP_LOGI(TAG, "═══════════════════════════════════════════════════════════");
  } else {
    ESP_LOGW(TAG, "═══════════════════════════════════════════════════════════");
    ESP_LOGW(TAG, "⚠️  NBVI Calibration Failed");
    ESP_LOGW(TAG, "   Using default band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
             this->selected_subcarriers_[0], this->selected_subcarriers_[1],
             this->selected_subcarriers_[2], this->selected_subcarriers_[3],
             this->selected_subcarriers_[4], this->selected_subcarriers_[5],
             this->selected_subcarriers_[6], this->selected_subcarriers_[7],
             this->selected_subcarriers_[8], this->selected_subcarriers_[9],
             this->selected_subcarriers_[10], this->selected_subcarriers_[11]);
    ESP_LOGW(TAG, "═══════════════════════════════════════════════════════════");
  }
  
  // Cleanup
  nbvi_calibrator_free(&calibrator);
}

void ESpectreComponent::dump_config() {
  ESP_LOGCONFIG(TAG, "ESPectre:");
  ESP_LOGCONFIG(TAG, "  Segmentation Threshold: %.2f", this->segmentation_threshold_);
  ESP_LOGCONFIG(TAG, "  Window Size: %d", this->segmentation_window_size_);
  ESP_LOGCONFIG(TAG, "  Traffic Generator Rate: %u pps", this->traffic_generator_rate_);
  ESP_LOGCONFIG(TAG, "  Features Enabled: %s", this->features_enabled_ ? "YES" : "NO");
  ESP_LOGCONFIG(TAG, "  Filters:");
  ESP_LOGCONFIG(TAG, "    Butterworth: %s", this->butterworth_enabled_ ? "ON" : "OFF");
  ESP_LOGCONFIG(TAG, "    Wavelet: %s (level=%d, threshold=%.1f)", 
                this->wavelet_enabled_ ? "ON" : "OFF", 
                this->wavelet_level_, this->wavelet_threshold_);
  ESP_LOGCONFIG(TAG, "    Hampel: %s (threshold=%.1f)", 
                this->hampel_enabled_ ? "ON" : "OFF", this->hampel_threshold_);
  ESP_LOGCONFIG(TAG, "    Savitzky-Golay: %s", this->savgol_enabled_ ? "ON" : "OFF");
}

}  // namespace espectre
}  // namespace esphome
