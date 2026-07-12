#include "stream_esp_idf_runtime.h"

#include <algorithm>
#include <cinttypes>

#include "espectre_log.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

// Wi-Fi credentials come from the streamer/native Kconfig surface. Builds
// without it (e.g. ESPHome/PlatformIO) still compile this runtime but never
// select the STREAM profile, so fall back to empty values.
#ifndef CONFIG_ESPECTRE_WIFI_SSID
#define CONFIG_ESPECTRE_WIFI_SSID ""
#endif
#ifndef CONFIG_ESPECTRE_WIFI_PASSWORD
#define CONFIG_ESPECTRE_WIFI_PASSWORD ""
#endif
#ifndef CONFIG_ESPECTRE_WIFI_BSSID
#define CONFIG_ESPECTRE_WIFI_BSSID ""
#endif
#ifndef CONFIG_ESPECTRE_WIFI_CHANNEL
#define CONFIG_ESPECTRE_WIFI_CHANNEL 0
#endif

namespace espectre {

namespace {

static const char *const TAG = "espectre.stream.runtime";
constexpr int kWifiConnectMaxRetry = 8;
constexpr uint8_t kDefaultPacingPayload[] = {'E', 'S', 'P', 'E'};

TrafficGeneratorMode to_traffic_mode(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? TrafficGeneratorMode::PING : TrafficGeneratorMode::DNS;
}

CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config) {
  CsiTrafficServiceConfig csi_traffic_config;
  csi_traffic_config.mode =
      (config.csi_traffic_mode == CsiTrafficMode::INTERNAL && config.traffic_generator_rate == 0U)
          ? CsiTrafficMode::PACING
          : config.csi_traffic_mode;
  csi_traffic_config.rate_pps = config.traffic_generator_rate;
  csi_traffic_config.traffic_mode = to_traffic_mode(config.traffic_generator_mode);
  csi_traffic_config.udp_port = config.csi_traffic_udp_port;
  csi_traffic_config.multicast_group = config.csi_traffic_multicast_group;
  csi_traffic_config.expected_payload = config.csi_traffic_expected_payload;
  return csi_traffic_config;
}

bool check_esp(esp_err_t err, const char *what) {
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "%s failed: %s", what, esp_err_to_name(err));
    return false;
  }
  return true;
}

}  // namespace

StreamEspIdfRuntime::StreamEspIdfRuntime(const RuntimeConfig &config) : config_(config) {
  snapshot_.threshold = 0.0f;
  snapshot_.detector_name = "stream";
  capabilities_.supports_runtime_threshold_updates = false;
  capabilities_.supports_manual_recalibration = false;
  capabilities_.supports_ble_telemetry = false;
  capabilities_.supports_extended_diagnostics = true;
  capabilities_.supports_traffic_control = true;
}

const char *StreamEspIdfRuntime::workflow_state_name_(WorkflowState state) const {
  switch (state) {
    case WorkflowState::WAIT_WIFI:
      return "WAIT_WIFI";
    case WorkflowState::WIFI_READY:
      return "WIFI_READY";
    case WorkflowState::CSI_READY:
      return "CSI_READY";
    case WorkflowState::STREAMING:
      return "STREAMING";
    default:
      return "UNKNOWN";
  }
}

bool StreamEspIdfRuntime::setup() {
  if (setup_complete_) {
    return true;
  }

  if (!init_nvs_()) {
    return false;
  }

  if (config_.csi_traffic_expected_payload.empty()) {
    config_.csi_traffic_expected_payload.assign(reinterpret_cast<const char *>(kDefaultPacingPayload),
                                               sizeof(kDefaultPacingPayload));
  }
  if (config_.csi_traffic_mode == CsiTrafficMode::EXTERNAL) {
    config_.csi_traffic_mode = CsiTrafficMode::PACING;
  }

  capture_service_.init();
  capture_service_.set_raw_packet_interceptor({});
  capture_service_.set_packet_callback(
      [this](const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) { this->handle_csi_packet_(info, normalized); });

  csi_traffic_service_.init(to_csi_traffic_config(config_));
  stream_transport_.configure(config_.device_id,
                              config_.collector_port,
                              config_.stream_log_interval_ms,
                              config_.stream_tx_batch_records);
  stream_transport_.reset_session();

  if (!init_wifi_station_()) {
    return false;
  }

  setup_complete_ = true;
  ESP_LOGI(TAG,
           "Stream runtime ready: collector=learned:%u traffic_rx_port=%u",
           static_cast<unsigned>(config_.collector_port),
           static_cast<unsigned>(config_.csi_traffic_udp_port));
  return true;
}

void StreamEspIdfRuntime::shutdown() {
  if (!setup_complete_) {
    return;
  }

  on_wifi_disconnected_();
  wifi_manager_.shutdown();
  setup_complete_ = false;
}

void StreamEspIdfRuntime::loop() {
  if (!setup_complete_) {
    return;
  }

  wifi_manager_.loop();
  capture_service_.loop();
  csi_traffic_service_.loop();

  if (!wifi_connected_.load(std::memory_order_relaxed)) {
    stream_transport_.log_runtime_telemetry(capture_service_, csi_traffic_service_, false,
                                            workflow_state_name_(state_.load(std::memory_order_relaxed)));
    return;
  }

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  if (state == WorkflowState::WAIT_WIFI) {
    transition_to_(WorkflowState::WIFI_READY, "wifi connected");
  } else if (state == WorkflowState::WIFI_READY) {
    if (start_capture_()) {
      if (!csi_traffic_service_.is_running() && !csi_traffic_service_.start()) {
        notify_fault_("Failed to start UDP pacing listener");
        return;
      }
      transition_to_(WorkflowState::CSI_READY, "csi enabled");
    }
  } else if (state == WorkflowState::CSI_READY) {
    transition_to_(WorkflowState::STREAMING, "pipeline ready");
  }

  const bool streaming_ready = state_.load(std::memory_order_relaxed) == WorkflowState::STREAMING;
  stream_transport_.update_from_traffic(csi_traffic_service_, streaming_ready);
  stream_transport_.log_runtime_telemetry(capture_service_, csi_traffic_service_, streaming_ready,
                                          workflow_state_name_(state_.load(std::memory_order_relaxed)));
  snapshot_.ready_to_publish = streaming_ready;
}

void StreamEspIdfRuntime::set_services_armed(bool armed) {
  if (services_armed_ == armed) {
    return;
  }
  services_armed_ = armed;
  if (!setup_complete_) {
    return;
  }
  if (!services_armed_) {
    on_wifi_disconnected_();
    return;
  }
  if (wifi_connected_.load(std::memory_order_relaxed)) {
    on_wifi_connected_();
  }
}

void StreamEspIdfRuntime::set_live_telemetry_enabled(bool enabled) { live_telemetry_enabled_ = enabled; }

bool StreamEspIdfRuntime::set_threshold_runtime(float threshold) {
  (void)threshold;
  return false;
}

bool StreamEspIdfRuntime::trigger_recalibration() { return false; }

bool StreamEspIdfRuntime::is_calibrating() const { return false; }

RuntimeSnapshot StreamEspIdfRuntime::get_snapshot() const { return snapshot_; }

RuntimeCapabilities StreamEspIdfRuntime::get_capabilities() const { return capabilities_; }

void StreamEspIdfRuntime::set_listener(IRuntimeListener *listener) { listener_ = listener; }

bool StreamEspIdfRuntime::init_nvs_() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    if (!check_esp(nvs_flash_erase(), "nvs_flash_erase")) {
      return false;
    }
    err = nvs_flash_init();
  }
  return check_esp(err, "nvs_flash_init");
}

bool StreamEspIdfRuntime::init_wifi_station_() {
  const esp_err_t setup_err = wifi_manager_.setup(
      StandaloneWifiConfig{CONFIG_ESPECTRE_WIFI_SSID,
                           CONFIG_ESPECTRE_WIFI_PASSWORD,
                           CONFIG_ESPECTRE_WIFI_BSSID,
                           static_cast<uint8_t>(CONFIG_ESPECTRE_WIFI_CHANNEL),
                           kWifiConnectMaxRetry,
                           true},
      [this]() { this->on_wifi_connected_(); },
      [this]() { this->on_wifi_disconnected_(); });
  if (!check_esp(setup_err, "wifi_manager_.setup")) {
    return false;
  }
  if (!check_esp(wifi_manager_.start(), "wifi_manager_.start")) {
    return false;
  }
  return true;
}

bool StreamEspIdfRuntime::start_capture_() {
  if (capture_service_.is_enabled()) {
    return true;
  }

  ESP_LOGI(TAG, "Starting CSI capture: state=%s", workflow_state_name_(state_.load(std::memory_order_relaxed)));
  return check_esp(capture_service_.enable(), "capture_service_.enable");
}

void StreamEspIdfRuntime::stop_capture_() {
  if (!capture_service_.is_enabled()) {
    return;
  }

  (void)capture_service_.disable();
}

void StreamEspIdfRuntime::on_wifi_connected_() {
  // The callback is deferred to the runtime loop, so a disconnect may have
  // arrived before this connect event was processed.
  StandaloneWifiInfo wifi_info;
  if (!wifi_manager_.get_info(&wifi_info) || !wifi_info.connected || wifi_info.ip_address[0] == '\0') {
    return;
  }

  wifi_connected_.store(true, std::memory_order_relaxed);
  ap_bssid_.fill(0U);
  stream_transport_.reset_session();

  snapshot_.ready_to_publish = false;
  wifi_ap_record_t ap_info{};
  if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
    std::copy(std::begin(ap_info.bssid), std::end(ap_info.bssid), ap_bssid_.begin());
    stream_transport_.set_ap_bssid(ap_bssid_.data(), ap_bssid_.size());
  } else {
    stream_transport_.clear_ap_bssid();
    ESP_LOGW(TAG, "Failed to read AP BSSID; accepting CSI from all sources");
  }
  transition_to_(WorkflowState::WAIT_WIFI, "wifi connected");
}

void StreamEspIdfRuntime::on_wifi_disconnected_() {
  wifi_connected_.store(false, std::memory_order_relaxed);
  stop_capture_();
  csi_traffic_service_.stop();
  stream_transport_.reset_session();
  stream_transport_.clear_ap_bssid();
  ap_bssid_.fill(0U);
  transition_to_(WorkflowState::WAIT_WIFI, "wifi disconnected");
  snapshot_.ready_to_publish = false;
}

void StreamEspIdfRuntime::transition_to_(WorkflowState next, const char *reason) {
  const WorkflowState prev = state_.exchange(next, std::memory_order_relaxed);
  if (prev != next) {
    ESP_LOGI(TAG, "[STATE] %s -> %s (%s)", workflow_state_name_(prev), workflow_state_name_(next),
             reason != nullptr ? reason : "n/a");
  }
}

void StreamEspIdfRuntime::notify_fault_(const char *message) {
  last_fault_ = message != nullptr ? message : "Unknown stream runtime fault";
  ESP_LOGE(TAG, "%s", last_fault_.c_str());
  if (listener_ != nullptr) {
    listener_->on_runtime_fault(last_fault_.c_str());
  }
}

void StreamEspIdfRuntime::handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) {
  stream_transport_.handle_csi_packet(info, normalized, state_.load(std::memory_order_relaxed) == WorkflowState::STREAMING);
}

}  // namespace espectre
