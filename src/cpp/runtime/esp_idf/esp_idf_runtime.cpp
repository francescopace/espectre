#include "esp_idf_runtime.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <new>
#include "espectre_log.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "classic_detector.h"
#include "csi_format.h"
#include "ml_detector.h"
#include "runtime_config_utils.h"

namespace espectre {

namespace {

static const char *const RUNTIME_TAG = "espectre.runtime";

TrafficGeneratorMode to_traffic_mode(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? TrafficGeneratorMode::PING : TrafficGeneratorMode::DNS;
}

CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config) {
  CsiTrafficServiceConfig csi_traffic_config;
  csi_traffic_config.mode =
      (config.csi_traffic_mode == CsiTrafficMode::INTERNAL && config.traffic_generator_rate == 0U)
          ? CsiTrafficMode::EXTERNAL
          : config.csi_traffic_mode;
  csi_traffic_config.rate_pps = config.traffic_generator_rate;
  csi_traffic_config.traffic_mode = to_traffic_mode(config.traffic_generator_mode);
  csi_traffic_config.udp_port = config.csi_traffic_udp_port;
  csi_traffic_config.multicast_group = config.csi_traffic_multicast_group;
  csi_traffic_config.expected_payload = config.csi_traffic_expected_payload;
  return csi_traffic_config;
}

}  // namespace

void EspIdfRuntime::update_live_telemetry_callback_() {
  if (live_telemetry_enabled_) {
    csi_pipeline_.set_live_telemetry_callback([this](float movement, float threshold) {
      if (listener_ != nullptr) {
        listener_->on_live_telemetry(movement, threshold);
      }
    });
  } else {
    csi_pipeline_.set_live_telemetry_callback({});
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

  csi_traffic_service_.init(to_csi_traffic_config(config_));

  csi_pipeline_.init(detector_.get(), config_.publish_interval);
  csi_pipeline_.set_evaluation_interval(config_.evaluation_interval);
  csi_pipeline_.set_motion_on_hits(config_.motion_on_hits);
  csi_pipeline_.set_motion_off_hits(config_.motion_off_hits);
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
  debug_telemetry_.reset();
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
  RuntimeDebugLoopScope debug_scope(debug_telemetry_, RUNTIME_TAG);
  wifi_lifecycle_.process_pending_events();
  uint8_t calibration_percent = 0U;
  uint32_t calibration_packets = 0U;
  uint16_t calibration_target_packets = 0U;
  if (calibration_progress_event_.take(calibration_percent, calibration_packets, calibration_target_packets)) {
    log_calibration_progress_(calibration_percent, calibration_packets, calibration_target_packets);
  }
  bool calibration_success = false;
  if (calibration_finished_event_.take(calibration_success)) {
    finish_threshold_calibration_(calibration_success);
  }
  csi_pipeline_.loop();
  DetectionTimingStats detection_timing;
  if (csi_pipeline_.take_detection_timing(&detection_timing)) {
    debug_telemetry_.record_detection_timing(detection_timing.duration_sum_us,
                                             detection_timing.samples,
                                             detection_timing.minimum_us,
                                             detection_timing.maximum_us);
  }
  csi_traffic_service_.loop();
}

void EspIdfRuntime::log_calibration_progress_(uint8_t percent, uint32_t packets, uint16_t target_packets) {
  if (target_packets == 0U) {
    return;
  }
  const float progress = static_cast<float>(packets) / static_cast<float>(target_packets);
  log_progress_bar(RUNTIME_TAG, progress, 20, -1,
                   "calibration %3u%% | %" PRIu32 "/%u packets",
                   static_cast<unsigned>(percent),
                   static_cast<uint32_t>(packets),
                   static_cast<unsigned>(target_packets));
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
  if (!validate_runtime_threshold_for_algorithm(threshold, config_.detection_algorithm)) {
    ESP_LOGW(RUNTIME_TAG,
             "Rejected invalid runtime threshold: %.6f (detector=%s max=%.3f)",
             threshold,
             detection_algorithm_name(config_.detection_algorithm),
             static_cast<double>(runtime_threshold_max(config_.detection_algorithm)));
    return false;
  }
  if (!csi_pipeline_.set_threshold(threshold)) {
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
  if (config_.threshold_mode == ThresholdMode::MANUAL &&
      !validate_runtime_threshold_for_algorithm(config_.segmentation_threshold, config_.detection_algorithm)) {
    notify_fault_("Invalid manual threshold");
    return false;
  }

  if (config_.detection_algorithm == DetectionAlgorithm::ML) {
    const float ml_threshold = (config_.threshold_mode == ThresholdMode::MANUAL) ? config_.segmentation_threshold
                                                                                 : ML_DEFAULT_THRESHOLD;
    config_.segmentation_threshold = ml_threshold;
    snapshot_.threshold = ml_threshold;
    detector_ = std::make_unique<MLDetector>(config_.segmentation_window_size, ml_threshold);
  } else {
    detector_ = std::make_unique<ClassicDetector>(config_.segmentation_window_size,
                                                  config_.segmentation_threshold,
                                                  config_.classic_recovery_vote_enabled);
  }

  if (detector_ == nullptr) {
    notify_fault_("Failed to configure detector");
    return false;
  }

  detector_->configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
  detector_->configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);

  snapshot_.detector_name = detector_->get_name();
  return true;
}

void EspIdfRuntime::on_wifi_connected_() {
  // Connect events are processed from the loop task; the connection may have
  // dropped again in the meantime. The next IP_EVENT_STA_GOT_IP retriggers.
  if (!has_wifi_ip_()) {
    return;
  }

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

  csi_pipeline_.set_motion_state_callback([this](MotionState state) {
    snapshot_.motion_state = state;
    if (snapshot_.ready_to_publish && listener_ != nullptr) {
      listener_->on_motion_state_changed(snapshot_);
    }
  });
  refresh_csi_local_identity_();

  if (!csi_pipeline_.is_enabled()) {
    const esp_err_t err = csi_pipeline_.enable([this](MotionState state, uint32_t packets_received) {
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

  if (!csi_traffic_service_.is_running() && !csi_traffic_service_.start()) {
    notify_fault_("Failed to start CSI traffic service");
    return;
  }

  start_calibration_();
  snapshot_.ready_to_publish = true;
}

void EspIdfRuntime::on_wifi_disconnected_() {
  wifi_ready_ = false;
  csi_wifi_lifecycle_ready_ = false;
  threshold_calibration_active_.store(false, std::memory_order_relaxed);
  next_calibration_progress_percent_.store(25U, std::memory_order_relaxed);
  calibration_progress_event_.clear();
  calibration_finished_event_.clear();
  csi_pipeline_.set_packet_interceptor(nullptr, nullptr);
  csi_pipeline_.set_local_identity(0U, nullptr);
  csi_pipeline_.disable();
  csi_traffic_service_.stop();
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
  threshold_calibrator_.reset(new (std::nothrow) StartupThresholdCalibrator());
  if (!threshold_calibrator_) {
    snapshot_.calibrating = false;
    notify_fault_("Failed to allocate startup calibrator");
    return false;
  }
  threshold_calibrator_->begin(config_.segmentation_window_size * CALIBRATION_NUM_WINDOWS,
                               detector_ != nullptr && detector_->startup_gate_enabled());
  calibration_finished_event_.clear();
  calibration_progress_event_.clear();
  next_calibration_progress_percent_.store(25U, std::memory_order_relaxed);
  threshold_calibration_active_.store(true, std::memory_order_relaxed);
  csi_pipeline_.clear_detector_buffer();
  csi_pipeline_.set_packet_interceptor(&EspIdfRuntime::threshold_calibration_packet_callback_, this);
  ESP_LOGI(RUNTIME_TAG, "Starting %s threshold calibration with fixed subcarriers",
           detector_ != nullptr ? detector_->get_name() : "detector");
  return true;
}

bool EspIdfRuntime::handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len) {
  if (!threshold_calibration_active_.load(std::memory_order_relaxed) || detector_ == nullptr ||
      !threshold_calibrator_) {
    return false;
  }

  detector_->process_packet(csi_data, csi_len, snapshot_.fixed_subcarriers.data(),
                            HT20_SELECTED_BAND_SIZE);
  detector_->update_state();
  threshold_calibrator_->observe(detector_->is_ready(), detector_->get_motion_metric(),
                                 detector_->get_startup_floor_metric());

  const uint32_t packet_count = threshold_calibrator_->packet_count();
  const uint16_t target_packets = threshold_calibrator_->target_packets();
  uint8_t next_progress = next_calibration_progress_percent_.load(std::memory_order_relaxed);
  while (next_progress <= 100U &&
         (static_cast<uint64_t>(packet_count) * 100U) >=
             (static_cast<uint64_t>(target_packets) * next_progress)) {
    calibration_progress_event_.post(next_progress, std::min<uint32_t>(packet_count, target_packets), target_packets);
    next_progress = static_cast<uint8_t>(next_progress + 25U);
  }
  next_calibration_progress_percent_.store(next_progress, std::memory_order_relaxed);

  if (threshold_calibrator_->is_complete()) {
    threshold_calibration_active_.store(false, std::memory_order_relaxed);
    calibration_finished_event_.post(threshold_calibrator_->is_successful());
  }
  return true;
}

bool EspIdfRuntime::threshold_calibration_packet_callback_(void *context,
                                                           const int8_t *csi_data,
                                                           size_t csi_len) {
  auto *runtime = static_cast<EspIdfRuntime *>(context);
  return runtime != nullptr && runtime->handle_threshold_calibration_packet_(csi_data, csi_len);
}

void EspIdfRuntime::finish_threshold_calibration_(bool success) {
  threshold_calibration_active_.store(false, std::memory_order_relaxed);
  next_calibration_progress_percent_.store(25U, std::memory_order_relaxed);
  calibration_progress_event_.clear();
  csi_pipeline_.set_packet_interceptor(nullptr, nullptr);
  snapshot_.calibrating = false;

  if (success && threshold_calibrator_) {
    float adaptive_threshold = 0.0f;
    const ThresholdMode adaptive_mode =
        (config_.threshold_mode == ThresholdMode::MANUAL) ? ThresholdMode::AUTO : config_.threshold_mode;
    const float auto_factor =
        detector_ != nullptr ? detector_->get_startup_threshold_factor() : DEFAULT_ADAPTIVE_FACTOR;
    const float factor = get_threshold_factor(adaptive_mode, auto_factor);
    adaptive_threshold = threshold_calibrator_->threshold_metric() * factor;
    snapshot_.startup_threshold = adaptive_threshold;
    if (detector_ != nullptr) {
      float variance_floor = 0.0f;
      bool vote_enabled = false;
      uint16_t floor_count = 0;
      threshold_calibrator_->floor_snapshot(variance_floor, vote_enabled, floor_count);
      detector_->apply_startup_floor(variance_floor, vote_enabled, floor_count);
      detector_->on_startup_calibration_complete();
    }

    if (config_.threshold_mode != ThresholdMode::MANUAL) {
      set_threshold_runtime(adaptive_threshold);
      ESP_LOGD(RUNTIME_TAG, "Adaptive threshold: %.6f (%s x %.1f)", adaptive_threshold,
               threshold_calibrator_->statistic_name(), factor);
    }
    csi_pipeline_.clear_detector_buffer();
  }

  if (listener_ != nullptr) {
    listener_->on_calibration_finished(snapshot_, success);
  }
  ESP_LOGD(RUNTIME_TAG, "Calibration %s", success ? "completed successfully" : "failed");
  threshold_calibrator_.reset();
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
    csi_pipeline_.set_local_identity(local_wifi_ip_addr_(), nullptr);
    return;
  }
  csi_pipeline_.set_local_identity(local_wifi_ip_addr_(), mac);
}

}  // namespace espectre
