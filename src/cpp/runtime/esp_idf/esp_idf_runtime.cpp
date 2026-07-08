#include "esp_idf_runtime.h"

#include <algorithm>
#include <cstring>
#include "espectre_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "runtime_config_utils.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const RUNTIME_TAG = "espectre.runtime";

TrafficGeneratorMode to_traffic_mode(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? TrafficGeneratorMode::PING : TrafficGeneratorMode::DNS;
}

StimulusServiceConfig to_stimulus_config(const RuntimeConfig &config) {
  StimulusServiceConfig stimulus_config;
  stimulus_config.mode =
      (config.stimulus_mode == StimulusMode::INTERNAL && config.traffic_generator_rate == 0U)
          ? StimulusMode::EXTERNAL
          : config.stimulus_mode;
  stimulus_config.rate_pps = config.traffic_generator_rate;
  stimulus_config.traffic_mode = to_traffic_mode(config.traffic_generator_mode);
  stimulus_config.udp_port = config.stimulus_udp_port;
  stimulus_config.multicast_group = config.stimulus_multicast_group;
  return stimulus_config;
}

}  // namespace

void EspIdfRuntime::update_live_telemetry_callback_() {
  if (live_telemetry_enabled_) {
    csi_manager_.set_live_telemetry_callback([this](float movement, float threshold) {
      if (listener_ != nullptr) {
        listener_->on_live_telemetry(movement, threshold);
      }
    });
  } else {
    csi_manager_.set_live_telemetry_callback({});
  }
}

EspIdfRuntime::EspIdfRuntime(const RuntimeConfig &config) : config_(config) {
  snapshot_.threshold = config_.segmentation_threshold;
  snapshot_.subcarrier_source = RuntimeSubcarrierSource::FIXED_DEFAULT;
}

bool EspIdfRuntime::setup() {
  if (setup_complete_) {
    return true;
  }

  ESP_LOGI(RUNTIME_TAG, "Initializing ESPectre runtime...");

  if (!configure_detector_()) {
    return false;
  }

  stimulus_service_.init(to_stimulus_config(config_));

  csi_manager_.init(detector_, config_.publish_interval);
  csi_manager_.set_evaluation_interval(config_.evaluation_interval);
  csi_manager_.set_motion_on_hits(config_.motion_on_hits);
  csi_manager_.set_motion_off_hits(config_.motion_off_hits);
  update_live_telemetry_callback_();

  if (wifi_lifecycle_.register_handlers([this]() { on_wifi_connected_(); },
                                        [this]() { on_wifi_disconnected_(); }) != ESP_OK) {
    notify_fault_("Failed to register WiFi handlers");
    return false;
  }

  wifi_ready_ = has_wifi_ip_();
  setup_complete_ = true;
  if (wifi_ready_) {
    if (services_armed_) {
      on_wifi_connected_();
    } else {
      ESP_LOGI(RUNTIME_TAG, "WiFi is ready, deferring CSI services until commissioning completes");
    }
  }
  ESP_LOGD(RUNTIME_TAG, "[resources] Free heap: %lu bytes, largest block: %lu bytes",
           static_cast<unsigned long>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)),
           static_cast<unsigned long>(heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT)));
  return true;
}

void EspIdfRuntime::shutdown() {
  if (!setup_complete_) {
    return;
  }

  on_wifi_disconnected_();
  wifi_lifecycle_.unregister_handlers();
  setup_complete_ = false;
}

void EspIdfRuntime::loop() {
  stimulus_service_.loop();
}

void EspIdfRuntime::set_services_armed(bool armed) {
  if (services_armed_ == armed) {
    return;
  }

  services_armed_ = armed;
  if (!setup_complete_) {
    return;
  }

  if (!services_armed_) {
    ESP_LOGI(RUNTIME_TAG, "CSI services disarmed until Matter commissioning is complete");
    on_wifi_disconnected_();
    return;
  }

  wifi_ready_ = has_wifi_ip_();
  if (wifi_ready_) {
    ESP_LOGI(RUNTIME_TAG, "Matter commissioning complete, starting CSI services");
    on_wifi_connected_();
  } else {
    ESP_LOGI(RUNTIME_TAG, "Matter commissioning complete, waiting for WiFi IP");
  }
}

void EspIdfRuntime::set_live_telemetry_enabled(bool enabled) {
  live_telemetry_enabled_ = enabled;
  update_live_telemetry_callback_();
}

bool EspIdfRuntime::set_threshold_runtime(float threshold) {
  if (!validate_runtime_threshold(threshold)) {
    ESP_LOGW(RUNTIME_TAG, "Rejected invalid runtime threshold: %.6f", threshold);
    return false;
  }
  if (!csi_manager_.set_threshold(threshold)) {
    return false;
  }
  config_.segmentation_threshold = threshold;
  snapshot_.threshold = threshold;
  if (listener_ != nullptr) {
    listener_->on_threshold_changed(snapshot_);
  }
  ESP_LOGD(RUNTIME_TAG, "Threshold updated to %.6f (session-only, recalculated at boot)", threshold);
  return true;
}

bool EspIdfRuntime::trigger_recalibration() {
  if (snapshot_.calibrating) {
    ESP_LOGW(RUNTIME_TAG, "Calibration already in progress");
    return false;
  }

  ESP_LOGI(RUNTIME_TAG, "Manual recalibration triggered");
  return start_calibration_();
}

bool EspIdfRuntime::is_calibrating() const { return snapshot_.calibrating; }

RuntimeSnapshot EspIdfRuntime::get_snapshot() const { return snapshot_; }

RuntimeCapabilities EspIdfRuntime::get_capabilities() const { return capabilities_; }

void EspIdfRuntime::set_listener(IRuntimeListener *listener) { listener_ = listener; }

bool EspIdfRuntime::configure_detector_() {
  if (config_.threshold_mode == ThresholdMode::MANUAL && !validate_runtime_threshold(config_.segmentation_threshold)) {
    notify_fault_("Invalid manual threshold");
    return false;
  }

  if (config_.detection_algorithm == DetectionAlgorithm::ML) {
    const float ml_threshold = (config_.threshold_mode == ThresholdMode::MANUAL) ? config_.segmentation_threshold
                                                                                 : ML_DEFAULT_THRESHOLD;
    config_.segmentation_threshold = ml_threshold;
    snapshot_.threshold = ml_threshold;
    ml_detector_ = MLDetector(config_.segmentation_window_size, ml_threshold);
    detector_ = &ml_detector_;
  } else {
    classic_detector_ = ClassicDetector(config_.segmentation_window_size, config_.segmentation_threshold);
    detector_ = &classic_detector_;
  }
  detector_->configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
  detector_->configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);

  if (detector_ == nullptr) {
    notify_fault_("Failed to configure detector");
    return false;
  }

  snapshot_.detector_name = detector_->get_name();
  return true;
}

void EspIdfRuntime::on_wifi_connected_() {
  wifi_ready_ = true;
  if (!services_armed_) {
    ESP_LOGI(RUNTIME_TAG, "WiFi connected, waiting for Matter commissioning before starting CSI services");
    return;
  }

  if (!csi_wifi_lifecycle_ready_) {
    const esp_err_t err = wifi_lifecycle_.init();
    if (err != ESP_OK) {
      notify_fault_("WiFi lifecycle init failed");
      return;
    }
    csi_wifi_lifecycle_ready_ = true;
    ESP_LOGI(RUNTIME_TAG, "WiFi CSI lifecycle initialized after connect");
  }

  snapshot_.motion_state = MotionState::IDLE;
  snapshot_.ready_to_publish = false;

  csi_manager_.set_motion_state_callback([this](MotionState state) {
    snapshot_.motion_state = state;
    if (snapshot_.ready_to_publish && listener_ != nullptr) {
      listener_->on_motion_state_changed(snapshot_);
    }
  });
  refresh_csi_local_identity_();

  if (!csi_manager_.is_enabled()) {
    const esp_err_t err = csi_manager_.enable([this](MotionState state, uint32_t packets_received) {
      snapshot_.motion_state = state;
      snapshot_.movement_metric = detector_ != nullptr ? detector_->get_motion_metric() : 0.0f;
      snapshot_.threshold = detector_ != nullptr ? detector_->get_threshold() : snapshot_.threshold;

      if (snapshot_.ready_to_publish && listener_ != nullptr) {
        listener_->on_periodic_update(snapshot_, packets_received);
      }
    });
    if (err != ESP_OK) {
      notify_fault_("Failed to enable CSI");
      return;
    }
  }

  if (!stimulus_service_.is_running() && !stimulus_service_.start()) {
    notify_fault_("Failed to start stimulus service");
    return;
  }

  start_calibration_();
  snapshot_.ready_to_publish = true;
}

void EspIdfRuntime::on_wifi_disconnected_() {
  wifi_ready_ = false;
  csi_wifi_lifecycle_ready_ = false;
  threshold_calibration_active_ = false;
  csi_manager_.set_packet_interceptor({});
  csi_manager_.set_local_identity(0U, nullptr);
  csi_manager_.disable();
  stimulus_service_.stop();
  snapshot_.ready_to_publish = false;
  snapshot_.motion_state = MotionState::IDLE;
  if (listener_ != nullptr) {
    listener_->on_motion_state_changed(snapshot_);
  }
}

bool EspIdfRuntime::start_calibration_() {
  snapshot_.subcarrier_source = RuntimeSubcarrierSource::FIXED_DEFAULT;

  if (config_.detection_algorithm == DetectionAlgorithm::ML) {
    snapshot_.calibrating = false;
    if (listener_ != nullptr) {
      listener_->on_calibration_finished(snapshot_, true);
    }
    return true;
  }

  snapshot_.calibrating = true;
  if (listener_ != nullptr) {
    listener_->on_calibration_started(snapshot_);
  }

  // Calibrate on the runtime detector itself (cold-cleared first), so the
  // observed metric matches the configured algorithm. Mirrors the Python
  // runtime calibration flow.
  threshold_calibrator_.begin(config_.segmentation_window_size * CALIBRATION_NUM_WINDOWS,
                              detector_ != nullptr && detector_->startup_gate_enabled());
  threshold_calibration_extending_logged_ = false;
  threshold_calibration_active_ = true;
  csi_manager_.clear_detector_buffer();
  csi_manager_.set_packet_interceptor(
      [this](const int8_t *csi_data, size_t csi_len) { return handle_threshold_calibration_packet_(csi_data, csi_len); });
  ESP_LOGI(RUNTIME_TAG, "Starting %s threshold calibration with fixed subcarriers",
           detector_ != nullptr ? detector_->get_name() : "detector");
  return true;
}

bool EspIdfRuntime::handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len) {
  if (!threshold_calibration_active_ || detector_ == nullptr) {
    return false;
  }

  detector_->process_packet(csi_data, csi_len, snapshot_.fixed_subcarriers.data(),
                            HT20_SELECTED_BAND_SIZE);
  detector_->update_state();
  threshold_calibrator_.observe(detector_->is_ready(), detector_->get_motion_metric());

  if (!threshold_calibration_extending_logged_ && threshold_calibrator_.is_extending()) {
    ESP_LOGI(RUNTIME_TAG,
             "Calibration window not consistent (movement during calibration?); extending");
    threshold_calibration_extending_logged_ = true;
  }

  if (threshold_calibrator_.is_complete()) {
    finish_threshold_calibration_(threshold_calibrator_.is_successful());
  }
  return true;
}

void EspIdfRuntime::finish_threshold_calibration_(bool success) {
  threshold_calibration_active_ = false;
  csi_manager_.set_packet_interceptor({});
  snapshot_.calibrating = false;

  if (success) {
    float adaptive_threshold = 0.0f;
    const ThresholdMode adaptive_mode =
        (config_.threshold_mode == ThresholdMode::MANUAL) ? ThresholdMode::AUTO : config_.threshold_mode;
    const float auto_factor =
        detector_ != nullptr ? detector_->get_startup_threshold_factor() : DEFAULT_ADAPTIVE_FACTOR;
    const float factor = get_threshold_factor(adaptive_mode, auto_factor);
    adaptive_threshold = threshold_calibrator_.threshold_metric() * factor;
    snapshot_.startup_threshold = adaptive_threshold;
    if (detector_ != nullptr) {
      detector_->on_startup_calibration_complete();
    }

    if (config_.threshold_mode != ThresholdMode::MANUAL) {
      set_threshold_runtime(adaptive_threshold);
      ESP_LOGD(RUNTIME_TAG, "Adaptive threshold: %.6f (%s x %.1f)", adaptive_threshold,
               threshold_calibrator_.statistic_name(), factor);
    }
    csi_manager_.clear_detector_buffer();
  }

  if (listener_ != nullptr) {
    listener_->on_calibration_finished(snapshot_, success);
  }
  ESP_LOGD(RUNTIME_TAG, "Calibration %s", success ? "completed successfully" : "failed");
}

void EspIdfRuntime::notify_fault_(const char *message) {
  last_fault_ = message != nullptr ? message : "Unknown runtime fault";
  ESP_LOGE(RUNTIME_TAG, "%s", last_fault_.c_str());
  if (listener_ != nullptr) {
    listener_->on_runtime_fault(last_fault_.c_str());
  }
}

bool EspIdfRuntime::has_wifi_ip_() const {
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  if (netif == nullptr) {
    return false;
  }

  esp_netif_ip_info_t ip_info{};
  if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
    return false;
  }

  return ip_info.ip.addr != 0;
}

uint32_t EspIdfRuntime::local_wifi_ip_addr_() const {
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  if (netif == nullptr) {
    return 0U;
  }

  esp_netif_ip_info_t ip_info{};
  if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
    return 0U;
  }
  return ip_info.ip.addr;
}

void EspIdfRuntime::refresh_csi_local_identity_() {
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_wifi_get_mac(WIFI_IF_STA, mac) != ESP_OK) {
    csi_manager_.set_local_identity(local_wifi_ip_addr_(), nullptr);
    return;
  }
  csi_manager_.set_local_identity(local_wifi_ip_addr_(), mac);
}

}  // namespace espectre
}  // namespace esphome
