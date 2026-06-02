#include "esp_idf_runtime.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "espectre_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const RUNTIME_TAG = "espectre.runtime";

TrafficGeneratorMode to_traffic_mode(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? TrafficGeneratorMode::PING : TrafficGeneratorMode::DNS;
}

GainLockMode to_gain_lock_mode(RuntimeGainLockMode mode) {
  switch (mode) {
    case RuntimeGainLockMode::ENABLED:
      return GainLockMode::ENABLED;
    case RuntimeGainLockMode::DISABLED:
      return GainLockMode::DISABLED;
    case RuntimeGainLockMode::AUTO:
    default:
      return GainLockMode::AUTO;
  }
}

}  // namespace

EspIdfRuntime::EspIdfRuntime(const RuntimeConfig &config) : config_(config) {
  snapshot_.threshold = config_.segmentation_threshold;
  snapshot_.selected_subcarriers = config_.selected_subcarriers;
  snapshot_.subcarrier_source =
      config_.user_specified_subcarriers ? RuntimeSubcarrierSource::USER_CONFIGURED
                                         : RuntimeSubcarrierSource::AUTO_CALIBRATED;
}

bool EspIdfRuntime::setup() {
  if (setup_complete_) {
    return true;
  }

  ESP_LOGI(RUNTIME_TAG, "Initializing ESPectre runtime...");

  if (wifi_lifecycle_.init() != ESP_OK) {
    notify_fault_("WiFi lifecycle init failed");
    return false;
  }

  if (!configure_detector_()) {
    return false;
  }

  nbvi_calibrator_.init(&csi_manager_);
  nbvi_calibrator_.set_mvs_window_size(config_.segmentation_window_size);
  nbvi_calibrator_.configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
  nbvi_calibrator_.configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);
  nbvi_calibrator_.set_buffer_size(config_.segmentation_window_size * CALIBRATION_NUM_WINDOWS);

  traffic_generator_.init(config_.traffic_generator_rate, to_traffic_mode(config_.traffic_generator_mode));
  udp_listener_.init(5555);

  csi_manager_.init(detector_, snapshot_.selected_subcarriers.data(), config_.publish_interval,
                    to_gain_lock_mode(config_.gain_lock_mode));
  csi_manager_.set_evaluation_interval(config_.evaluation_interval);
  csi_manager_.set_motion_on_hits(config_.motion_on_hits);
  csi_manager_.set_motion_off_hits(config_.motion_off_hits);
  csi_manager_.set_game_mode_callback([this](float movement, float threshold) {
    if (listener_ != nullptr) {
      listener_->on_live_telemetry(movement, threshold);
    }
  });

  if (wifi_lifecycle_.register_handlers([this]() { on_wifi_connected_(); },
                                        [this]() { on_wifi_disconnected_(); }) != ESP_OK) {
    notify_fault_("Failed to register WiFi handlers");
    return false;
  }

  setup_complete_ = true;
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
  if (udp_listener_.is_running()) {
    udp_listener_.loop();
  }
}

bool EspIdfRuntime::set_threshold_runtime(float threshold) {
  config_.segmentation_threshold = threshold;
  snapshot_.threshold = threshold;
  csi_manager_.set_threshold(threshold);
  if (listener_ != nullptr) {
    listener_->on_threshold_changed(snapshot_);
  }
  ESP_LOGD(RUNTIME_TAG, "Threshold updated to %.2f (session-only, recalculated at boot)", threshold);
  return true;
}

bool EspIdfRuntime::trigger_recalibration() {
  if (nbvi_calibrator_.is_calibrating()) {
    ESP_LOGW(RUNTIME_TAG, "Calibration already in progress");
    return false;
  }

  if (!csi_manager_.is_gain_locked()) {
    ESP_LOGW(RUNTIME_TAG, "Cannot recalibrate: gain not yet locked");
    return false;
  }

  ESP_LOGI(RUNTIME_TAG, "Manual recalibration triggered");
  return start_calibration_();
}

bool EspIdfRuntime::is_calibrating() const { return nbvi_calibrator_.is_calibrating(); }

RuntimeSnapshot EspIdfRuntime::get_snapshot() const { return snapshot_; }

RuntimeCapabilities EspIdfRuntime::get_capabilities() const { return capabilities_; }

void EspIdfRuntime::set_listener(IRuntimeListener *listener) { listener_ = listener; }

bool EspIdfRuntime::configure_detector_() {
  if (config_.detection_algorithm == DetectionAlgorithm::ML) {
    const float ml_threshold = (config_.threshold_mode == ThresholdMode::MANUAL) ? config_.segmentation_threshold
                                                                                 : ML_DEFAULT_THRESHOLD;
    config_.segmentation_threshold = ml_threshold;
    snapshot_.threshold = ml_threshold;
    ml_detector_ = MLDetector(config_.segmentation_window_size, ml_threshold);
    ml_detector_.configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
    ml_detector_.configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);
    detector_ = &ml_detector_;
  } else {
    mvs_detector_ = MVSDetector(config_.segmentation_window_size, config_.segmentation_threshold);
    mvs_detector_.configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
    mvs_detector_.configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);
    detector_ = &mvs_detector_;
  }

  if (detector_ == nullptr) {
    notify_fault_("Failed to configure detector");
    return false;
  }

  snapshot_.detector_name = detector_->get_name();
  return true;
}

void EspIdfRuntime::on_wifi_connected_() {
  snapshot_.motion_state = MotionState::IDLE;
  snapshot_.ready_to_publish = false;

  csi_manager_.set_motion_state_callback([this](MotionState state) {
    snapshot_.motion_state = state;
    if (snapshot_.ready_to_publish && listener_ != nullptr) {
      listener_->on_motion_state_changed(snapshot_);
    }
  });

  if (!csi_manager_.is_enabled()) {
    const esp_err_t err = csi_manager_.enable([this](MotionState state, uint32_t packets_received) {
      snapshot_.motion_state = state;
      snapshot_.movement_metric = detector_ != nullptr ? detector_->get_motion_metric() : 0.0f;
      snapshot_.threshold = detector_ != nullptr ? detector_->get_threshold() : snapshot_.threshold;
      snapshot_.gain_locked = csi_manager_.is_gain_locked();

      if (snapshot_.ready_to_publish && listener_ != nullptr) {
        listener_->on_periodic_update(snapshot_, packets_received);
      }
    });
    if (err != ESP_OK) {
      notify_fault_("Failed to enable CSI");
      return;
    }
  }

  if (config_.traffic_generator_rate > 0) {
    if (!traffic_generator_.is_running() && !traffic_generator_.start()) {
      notify_fault_("Failed to start traffic generator");
      return;
    }
  } else if (!udp_listener_.is_running() && !udp_listener_.start()) {
    notify_fault_("Failed to start UDP listener");
    return;
  }

  csi_manager_.set_gain_lock_callback([this]() {
    const GainController &gc = csi_manager_.get_gain_controller();
    const bool need_cv = gc.needs_cv_normalization();
    if (detector_ != nullptr) {
      detector_->set_cv_normalization(need_cv);
    }
    nbvi_calibrator_.set_cv_normalization(need_cv);
    snapshot_.gain_locked = csi_manager_.is_gain_locked();
    start_calibration_();
  });

  snapshot_.ready_to_publish = true;
}

void EspIdfRuntime::on_wifi_disconnected_() {
  csi_manager_.disable();
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
  snapshot_.ready_to_publish = false;
  snapshot_.motion_state = MotionState::IDLE;
  if (listener_ != nullptr) {
    listener_->on_motion_state_changed(snapshot_);
  }
}

bool EspIdfRuntime::start_calibration_() {
  if (config_.detection_algorithm == DetectionAlgorithm::ML) {
    std::copy(DEFAULT_SUBCARRIERS, DEFAULT_SUBCARRIERS + HT20_SELECTED_BAND_SIZE,
              snapshot_.selected_subcarriers.begin());
    snapshot_.subcarrier_source = RuntimeSubcarrierSource::MODEL_DEFAULT;
    snapshot_.calibrating = false;
    csi_manager_.update_subcarrier_selection(snapshot_.selected_subcarriers.data());
    if (listener_ != nullptr) {
      listener_->on_calibration_finished(snapshot_, true);
    }
    return true;
  }

  snapshot_.calibrating = true;
  if (listener_ != nullptr) {
    listener_->on_calibration_started(snapshot_);
  }

  nbvi_calibrator_.set_collection_complete_callback([this]() { traffic_generator_.pause(); });
  const esp_err_t err = nbvi_calibrator_.start_calibration(
      snapshot_.selected_subcarriers.data(), HT20_SELECTED_BAND_SIZE,
      [this](const uint8_t *band, uint8_t size, const std::vector<float> &cal_values, bool success) {
        if (success && band != nullptr && !config_.user_specified_subcarriers) {
          std::memcpy(snapshot_.selected_subcarriers.data(), band, size);
          csi_manager_.update_subcarrier_selection(band);
          snapshot_.subcarrier_source = RuntimeSubcarrierSource::AUTO_CALIBRATED;
        }

        if (band != nullptr && !cal_values.empty()) {
          float adaptive_threshold = 0.0f;
          uint8_t percentile = 0;
          calculate_adaptive_threshold(cal_values, config_.threshold_mode, adaptive_threshold, percentile);
          snapshot_.best_pxx = adaptive_threshold;

          if (config_.threshold_mode != ThresholdMode::MANUAL) {
            set_threshold_runtime(adaptive_threshold);
            ESP_LOGD(RUNTIME_TAG, "Adaptive threshold: %.4f (P%d)", adaptive_threshold, percentile);
          }

          csi_manager_.clear_detector_buffer();
        }

        traffic_generator_.resume();
        snapshot_.calibrating = false;

        if (listener_ != nullptr) {
          listener_->on_calibration_finished(snapshot_, success);
        }

        ESP_LOGD(RUNTIME_TAG, "Calibration %s", success ? "completed successfully" : "failed");
      });
  if (err != ESP_OK) {
    snapshot_.calibrating = false;
    notify_fault_("Failed to start calibration");
    return false;
  }

  return true;
}

void EspIdfRuntime::notify_fault_(const char *message) {
  last_fault_ = message != nullptr ? message : "Unknown runtime fault";
  ESP_LOGE(RUNTIME_TAG, "%s", last_fault_.c_str());
  if (listener_ != nullptr) {
    listener_->on_runtime_fault(last_fault_.c_str());
  }
}

}  // namespace espectre
}  // namespace esphome
