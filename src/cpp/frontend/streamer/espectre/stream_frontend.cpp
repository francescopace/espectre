/*
 * ESPectre - Streamer Frontend
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "stream_frontend.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "ble_protocol.h"
#include "csi_stream_protocol.h"
#include "device_config_store.h"
#include "device_identity.h"
#include "espectre_log.h"
#include "firmware_version.h"
#include "frontend_bootstrap_helpers.h"
#include "frontend_control_helpers.h"
#include "frontend_mqtt_helpers.h"
#include "frontend_sysinfo_helpers.h"
#include "runtime_time.h"
#include "stimulus_protocol.h"
#include "utils.h"
#include "esp_attr.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.stream";
constexpr int kWifiConnectMaxRetry = 8;

#ifdef CONFIG_ESPECTRE_STREAM_OUTPUT_ENABLED
constexpr bool kStreamOutputEnabled = true;
#else
constexpr bool kStreamOutputEnabled = false;
#endif

float current_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

const char *workflow_state_name(StreamFrontend::WorkflowState state) {
  switch (state) {
    case StreamFrontend::WorkflowState::WAIT_WIFI:
      return "WAIT_WIFI";
    case StreamFrontend::WorkflowState::WIFI_READY:
      return "WIFI_READY";
    case StreamFrontend::WorkflowState::CSI_READY:
      return "CSI_READY";
    case StreamFrontend::WorkflowState::STREAMING:
      return "STREAMING";
    case StreamFrontend::WorkflowState::OTA_IN_PROGRESS:
      return "OTA_IN_PROGRESS";
    default:
      return "UNKNOWN";
  }
}

bool check_esp(esp_err_t err, const char *what) {
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "%s failed: %s", what, esp_err_to_name(err));
    return false;
  }
  return true;
}

StreamChipType detect_chip_code() {
#if CONFIG_IDF_TARGET_ESP32C6
  return StreamChipType::C6;
#elif CONFIG_IDF_TARGET_ESP32C5
  return StreamChipType::C5;
#elif CONFIG_IDF_TARGET_ESP32C3
  return StreamChipType::C3;
#elif CONFIG_IDF_TARGET_ESP32S3
  return StreamChipType::S3;
#elif CONFIG_IDF_TARGET_ESP32S2
  return StreamChipType::S2;
#elif CONFIG_IDF_TARGET_ESP32
  return StreamChipType::ESP32;
#else
  return StreamChipType::UNKNOWN;
#endif
}

void format_ipv4_addr(uint32_t network_addr, char *buffer, size_t buffer_len) {
  if (buffer == nullptr || buffer_len == 0U) {
    return;
  }
  if (network_addr == 0U) {
    std::snprintf(buffer, buffer_len, "0.0.0.0");
    return;
  }

  const uint32_t host_addr = ntohl(network_addr);
  std::snprintf(buffer,
                buffer_len,
                "%u.%u.%u.%u",
                static_cast<unsigned>((host_addr >> 24U) & 0xFFU),
                static_cast<unsigned>((host_addr >> 16U) & 0xFFU),
                static_cast<unsigned>((host_addr >> 8U) & 0xFFU),
                static_cast<unsigned>(host_addr & 0xFFU));
}

#if CONFIG_IDF_TARGET_ESP32C3
typedef struct {
  signed rssi : 8;
  unsigned rate : 5;
  unsigned : 1;
  unsigned sig_mode : 2;
  unsigned : 16;
  unsigned mcs : 7;
  unsigned cwb : 1;
  unsigned : 16;
  unsigned smoothing : 1;
  unsigned not_sounding : 1;
  unsigned : 1;
  unsigned aggregation : 1;
  unsigned stbc : 2;
  unsigned fec_coding : 1;
  unsigned sgi : 1;
  unsigned : 8;
  unsigned ampdu_cnt : 8;
  unsigned channel : 4;
  unsigned secondary_channel : 4;
  unsigned rxstart_time_cyc : 7;
  unsigned : 1;
  unsigned timestamp : 32;
  unsigned : 32;
  signed noise_floor : 8;
  unsigned : 24;
  unsigned : 32;
  unsigned : 20;
  unsigned rxstart_time_cyc_dec : 11;
  unsigned ant : 1;
  unsigned : 32;
  unsigned : 32;
  unsigned : 32;
  unsigned sig_len : 12;
  unsigned : 12;
  unsigned rx_state : 8;
} wifi_pkt_rx_ctrl_time_t;

static_assert(sizeof(wifi_pkt_rx_ctrl_time_t) == sizeof(wifi_pkt_rx_ctrl_t),
              "timestamp overlay must match wifi_pkt_rx_ctrl_t");
#endif

bool fill_rx_timestamp_metadata(const wifi_csi_info_t *info, CsiStreamHeaderV3 *header) {
  if (info == nullptr || header == nullptr) {
    return false;
  }

  header->wifi_rx_start_ts_ns = 0U;
#if CONFIG_IDF_TARGET_ESP32C3
  const auto *time_info = reinterpret_cast<const wifi_pkt_rx_ctrl_time_t *>(&info->rx_ctrl);
  if (time_info->timestamp == 0U) {
    return false;
  }

  const uint16_t cyc_dec = (time_info->rxstart_time_cyc_dec >= 1024U)
                               ? static_cast<uint16_t>(2048U - time_info->rxstart_time_cyc_dec)
                               : static_cast<uint16_t>(time_info->rxstart_time_cyc_dec);
  const uint64_t coarse_ns = static_cast<uint64_t>(time_info->timestamp) * 1000ULL;
  const uint64_t cyc_ns = (static_cast<uint64_t>(time_info->rxstart_time_cyc) * 12500ULL) / 1000ULL;
  const uint64_t cyc_dec_ns = (static_cast<uint64_t>(cyc_dec) * 1562ULL) / 1000ULL;
  constexpr uint64_t kAlignmentNs = 20800ULL;
  if (coarse_ns + cyc_ns + cyc_dec_ns <= kAlignmentNs) {
    return false;
  }

  header->wifi_rx_start_ts_ns = coarse_ns + cyc_ns + cyc_dec_ns - kAlignmentNs;
  return true;
#else
  return false;
#endif
}

}  // namespace

StreamFrontend::StreamFrontend(IMqttTransport *mqtt_transport, IOtaService *ota_service)
    : mqtt_transport_(mqtt_transport), ota_service_(ota_service) {}

bool StreamFrontend::setup() {
  if (setup_complete_) {
    return true;
  }

  if (!init_nvs_()) {
    return false;
  }

  stream_seq_ = 0U;
  FrontendDeviceConfigDefaults device_config_defaults;
  device_config_defaults.runtime_device_id = derive_runtime_device_id();
#if defined(CONFIG_ESPECTRE_MQTT_HOST)
  device_config_defaults.mqtt_host = CONFIG_ESPECTRE_MQTT_HOST;
  device_config_defaults.mqtt_port = CONFIG_ESPECTRE_MQTT_PORT;
  device_config_defaults.topic_prefix = CONFIG_ESPECTRE_TOPIC_PREFIX;
#endif
  device_config_ = load_frontend_device_config(device_config_defaults,
                                               TAG,
                                               "Using device config provisioned over BLE",
                                               "Failed to load BLE-provisioned device config");
  device_info_.frontend = "streamer";
  device_info_.firmware_version = espectre_firmware_version();
  device_info_.chip = CONFIG_IDF_TARGET;

  if (!udp_sender_.setup()) {
    return false;
  }

  reset_collector_endpoint_();
  StimulusServiceConfig stimulus_config;
  stimulus_config.mode = StimulusMode::EXTERNAL;
  stimulus_config.udp_port = static_cast<uint16_t>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT);
  stimulus_config.multicast_group = CONFIG_ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP;
  stimulus_service_.init(stimulus_config);
  capture_service_.init();
  capture_service_.set_packet_callback(
      [this](const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) { this->handle_csi_packet_(info, normalized); });

  ESP_LOGI(TAG, "ESPectre streamer smoke marker: transport configured, starting streamer frontend");
  if (!init_wifi_station_()) {
    return false;
  }

  if (!setup_ble_provisioning_()) {
    ESP_LOGW(TAG, "BLE Wi-Fi provisioning is unavailable");
  }

  if (ota_service_ != nullptr) {
    ota_service_->set_prepare_for_update_callback([this]() { this->prepare_for_ota_(); });
    ota_service_->set_status_callback([this](const EspectreOtaStatus &status) { this->publish_mqtt_ota_status_(status); });
  }
  setup_mqtt_();

  setup_complete_ = true;
  ESP_LOGI(TAG,
           "Streamer frontend ready: collector=learned:%u traffic_rx_port=%u device_id=0x%016" PRIx64,
           static_cast<unsigned>(CONFIG_ESPECTRE_COLLECTOR_PORT),
           static_cast<unsigned>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT),
           device_config_.device_id);
  return true;
}

void StreamFrontend::loop() {
  const uint64_t loop_start_us = monotonic_now_us();
  const auto finish_loop = [this, loop_start_us]() {
    last_loop_time_ms_ = static_cast<float>(monotonic_now_us() - loop_start_us) / 1000.0F;
  };
  if (!setup_complete_) {
    finish_loop();
    return;
  }

  ble_bindings_.loop();

  if (stimulus_service_.is_running()) {
    stimulus_service_.loop();
    if (kStreamOutputEnabled) {
      sockaddr_in sender_addr{};
      if (stimulus_service_.get_last_sender(&sender_addr) && sender_addr.sin_addr.s_addr != collector_ip_addr_) {
        collector_ip_addr_ = sender_addr.sin_addr.s_addr;
        sender_addr.sin_family = AF_INET;
        sender_addr.sin_port = htons(static_cast<uint16_t>(CONFIG_ESPECTRE_COLLECTOR_PORT));
        udp_sender_.set_collector(sender_addr, true);

        char addr_text[16];
        format_ipv4_addr(collector_ip_addr_, addr_text, sizeof(addr_text));
        ESP_LOGI(TAG, "Learned collector address from stimulus sender: %s:%u", addr_text,
                 static_cast<unsigned>(CONFIG_ESPECTRE_COLLECTOR_PORT));
      }
    }
  }

  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
  }
  if (ota_service_ != nullptr) {
    ota_service_->loop();
  }

  if (!wifi_connected_.load(std::memory_order_relaxed)) {
    log_runtime_telemetry_();
    finish_loop();
    return;
  }

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  if (state == WorkflowState::OTA_IN_PROGRESS) {
    log_runtime_telemetry_();
    finish_loop();
    return;
  }
  if (state == WorkflowState::WAIT_WIFI) {
    transition_to_(WorkflowState::WIFI_READY, "wifi connected");
  } else if (state == WorkflowState::WIFI_READY) {
    if (start_capture_()) {
      if (!stimulus_service_.is_running() && !stimulus_service_.start()) {
        ESP_LOGW(TAG, "Failed to start stimulus service");
      }
      transition_to_(WorkflowState::CSI_READY, "csi enabled");
    }
  } else if (state == WorkflowState::CSI_READY) {
    transition_to_(WorkflowState::STREAMING, "pipeline ready");
  }

  log_runtime_telemetry_();
  finish_loop();
}

void StreamFrontend::shutdown() {
  if (!setup_complete_) {
    return;
  }

  stop_capture_();
  stimulus_service_.stop();
  if (ble_ready_) {
    ble_bindings_.shutdown();
    ble_ready_ = false;
  }
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  if (ota_service_ != nullptr) {
    ota_service_->shutdown();
  }
  wifi_manager_.shutdown();
  udp_sender_.shutdown();
  setup_complete_ = false;
}

StreamFrontend::~StreamFrontend() { shutdown(); }

bool StreamFrontend::init_nvs_() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    if (!check_esp(nvs_flash_erase(), "nvs_flash_erase")) {
      return false;
    }
    err = nvs_flash_init();
  }
  return check_esp(err, "nvs_flash_init");
}

bool StreamFrontend::init_wifi_station_() {
  const esp_err_t setup_err = setup_frontend_wifi_station(
      &wifi_provisioning_,
      &wifi_manager_,
      FrontendWifiStationOptions{CONFIG_ESPECTRE_WIFI_SSID,
                                 CONFIG_ESPECTRE_WIFI_PASSWORD,
                                 CONFIG_ESPECTRE_WIFI_BSSID,
                                 CONFIG_ESPECTRE_WIFI_CHANNEL,
                                 kWifiConnectMaxRetry,
                                 true,
                                 true,
                                 [this]() { this->publish_ble_sysinfo_(); },
                                 [this]() { this->on_wifi_connected_(); },
                                 [this]() { this->on_wifi_disconnected_(); }},
      TAG,
      "Using Wi-Fi credentials provisioned over BLE");
  if (!check_esp(setup_err, "wifi_provisioning_.setup_station")) {
    return false;
  }
  return true;
}

bool StreamFrontend::setup_ble_provisioning_() {
  const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                       device_info_.chip.empty() ? nullptr
                                                                                 : device_info_.chip.c_str());
  ble_bindings_.set_device_name(device_name.c_str());
  ble_bindings_.set_connection_state_callback([this](bool connected) {
    this->ble_client_connected_ = connected;
    if (connected) {
      this->publish_ble_sysinfo_();
    }
  });
  ble_bindings_.set_control_write_callback([this](const std::string &command) { this->handle_ble_control_(command); });
  ble_bindings_.set_telemetry_subscription_callback([](bool) {});
  if (!ble_bindings_.setup()) {
    return false;
  }
  ble_ready_ = true;
  ESP_LOGI(TAG, "BLE Wi-Fi provisioning ready");
  return true;
}

void StreamFrontend::setup_mqtt_() {
  (void) setup_frontend_mqtt_transport(mqtt_transport_,
                                       device_config_,
                                       [this](const std::string &payload) { this->handle_mqtt_command_(payload); },
                                       [this]() {
                                         this->publish_mqtt_info_();
                                         this->publish_mqtt_status_(true);
                                         if (this->ota_service_ != nullptr) {
                                           this->publish_mqtt_ota_status_(this->ota_service_->status());
                                         }
                                       },
                                       TAG);
}

bool StreamFrontend::start_capture_() {
  if (capture_service_.is_enabled()) {
    return true;
  }

  return check_esp(capture_service_.enable(), "capture_service_.enable");
}

void StreamFrontend::stop_capture_() {
  if (!capture_service_.is_enabled()) {
    return;
  }

  (void)capture_service_.disable();
}

void StreamFrontend::reset_collector_endpoint_() {
  sockaddr_in collector_addr{};
  collector_addr.sin_family = AF_INET;
  collector_addr.sin_port = htons(static_cast<uint16_t>(CONFIG_ESPECTRE_COLLECTOR_PORT));
  udp_sender_.set_collector(collector_addr, false);
}

void StreamFrontend::prepare_for_ota_() {
  stop_capture_();
  stimulus_service_.stop();
  reset_collector_endpoint_();
  transition_to_(WorkflowState::OTA_IN_PROGRESS, "ota requested");
}

void StreamFrontend::on_wifi_connected_() {
  wifi_connected_.store(true, std::memory_order_relaxed);
  collector_ip_addr_ = 0U;
  local_ip_addr_ = 0U;
  local_mac_addr_.fill(0U);
  reset_runtime_telemetry_baseline_();
  reset_collector_endpoint_();
  capture_service_.reset_session();
  StandaloneWifiInfo wifi_info;
  if (wifi_manager_.get_info(&wifi_info) && wifi_info.ip_address[0] != '\0') {
    local_ip_addr_ = inet_addr(wifi_info.ip_address);
    device_info_.network.ip_address = wifi_info.ip_address;
    device_info_.network.mac_address = wifi_info.mac_address;
    device_info_.network.channel = wifi_info.channel;
  }
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) == ESP_OK) {
    std::copy(std::begin(mac), std::end(mac), local_mac_addr_.begin());
  }
  publish_ble_sysinfo_();
}

void StreamFrontend::on_wifi_disconnected_() {
  wifi_connected_.store(false, std::memory_order_relaxed);
  stop_capture_();
  stimulus_service_.stop();
  collector_ip_addr_ = 0U;
  local_ip_addr_ = 0U;
  local_mac_addr_.fill(0U);
  reset_collector_endpoint_();
  transition_to_(WorkflowState::WAIT_WIFI, "wifi disconnected");
  device_info_.network = EspectreNetworkInfo{};
  publish_ble_sysinfo_();
}

void StreamFrontend::handle_ble_control_(const std::string &command) {
  if (command == "REQ_SYSINFO") {
    publish_ble_sysinfo_();
    return;
  }

  std::string message;
  bool accepted = false;
  bool refresh_sysinfo_now = true;
  DeviceConfigBleCommandResult device_config_result = handle_ble_device_config_command(
      command,
      device_config_,
      [](EspectreDeviceConfig *cleared_config, std::string *message) {
        const esp_err_t err = clear_stored_device_config();
        if (err != ESP_OK) {
          if (message != nullptr) {
            *message = esp_err_to_name(err);
          }
          return false;
        }
        if (cleared_config != nullptr) {
          *cleared_config = EspectreDeviceConfig{};
          cleared_config->device_id = derive_runtime_device_id();
        }
        if (message != nullptr) {
          *message = "device settings cleared";
        }
        return true;
      },
      [](EspectreDeviceConfig *updated_config, std::string *message) {
        if (updated_config == nullptr) {
          return false;
        }
        updated_config->device_id = derive_runtime_device_id();
        const esp_err_t err = save_stored_device_config(*updated_config);
        if (err != ESP_OK) {
          if (message != nullptr) {
            *message = esp_err_to_name(err);
          }
          return false;
        }
        if (message != nullptr) {
          *message = "device settings saved";
        }
        return true;
      });
  if (device_config_result.handled) {
    accepted = device_config_result.accepted;
    message = device_config_result.message;
    if (accepted && device_config_result.config_changed) {
      publish_mqtt_status_(false);
      device_config_ = device_config_result.config;
      const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                           device_info_.chip.empty() ? nullptr
                                                                                     : device_info_.chip.c_str());
      ble_bindings_.set_device_name(device_name.c_str());
      setup_mqtt_();
    }
  } else {
    accepted = wifi_provisioning_.handle_command(command, &message);
    if (accepted) {
      refresh_sysinfo_now = false;
    }
  }
  char line[192];
  std::snprintf(line,
                sizeof(line),
                "last_command=%s:%s",
                accepted ? "ok" : "error",
                message.empty() ? command.c_str() : message.c_str());
  publish_ble_line_(line);
  if (refresh_sysinfo_now) {
    publish_ble_sysinfo_();
  }
}

void StreamFrontend::handle_mqtt_command_(const std::string &payload) {
  const FrontendMqttCommandResult result =
      handle_frontend_mqtt_command(payload,
                                   ota_service_,
                                   device_info_.firmware_version.c_str(),
                                   FrontendMqttCommandCapabilities{true, true, false, ota_service_ != nullptr},
                                   [this]() { this->publish_mqtt_info_(); },
                                   [this]() { this->publish_mqtt_stats_(); },
                                   {},
                                   [this](const EspectreOtaStatus &status) { this->publish_mqtt_ota_status_(status); });
  if (result.handled) {
    publish_mqtt_command_result_(result.command, result.accepted, result.message.c_str());
  }
}

void StreamFrontend::publish_ble_sysinfo_() {
  if (!ble_ready_ || !ble_client_connected_) {
    return;
  }

  std::vector<std::string> lines;
  EspectreDeviceInfo sysinfo_device_info = device_info_;
  if (sysinfo_device_info.chip.empty()) {
    sysinfo_device_info.chip = CONFIG_IDF_TARGET;
  }
  const StoredWifiConfig &wifi_config = wifi_provisioning_.config();
  lines = build_frontend_sysinfo_lines(FrontendSysinfoBase{
      "streamer",
      SysinfoCapabilities{true, true, true, false, false, false, true},
      device_config_,
      sysinfo_device_info,
      false,
      true,
      mqtt_transport_ != nullptr && mqtt_transport_->connected(),
      SysinfoWifiState{
          wifi_config.ssid,
          wifi_config.bssid,
          wifi_config.channel,
          wifi_provisioning_.password_set(),
          wifi_connected_.load(std::memory_order_relaxed),
      },
  });

  StandaloneWifiInfo wifi_info;
  if (wifi_manager_.get_info(&wifi_info)) {
    append_sysinfo_network_lines(&lines, wifi_info.ip_address, wifi_info.mac_address);
  }
  append_sysinfo_end_line(&lines);
  ble_bindings_.replace_sysinfo_lines(std::move(lines));
}

void StreamFrontend::publish_ble_line_(const char *line) {
  if (ble_ready_ && ble_client_connected_ && line != nullptr) {
    ble_bindings_.publish_sysinfo_line(line);
  }
}

void StreamFrontend::publish_mqtt_info_() {
  const EspectreDeviceInfo info =
      normalize_protocol_device_info(device_info_, nullptr, ota_service_ != nullptr, "streamer", CONFIG_IDF_TARGET);
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "info", espectre_info_payload(device_config_, info), false);
}

void StreamFrontend::publish_mqtt_status_(bool online) {
  (void) publish_frontend_mqtt_status(mqtt_transport_, device_config_, online, monotonic_now_ms());
}

void StreamFrontend::publish_mqtt_stats_() {
  const uint32_t now = monotonic_now_ms();
  (void) publish_frontend_mqtt_message(mqtt_transport_,
                                       device_config_,
                                       "stats",
                                       espectre_stats_payload(
                                           device_config_, RuntimeSnapshot{}, now, now / 1000U, current_free_memory_kb(), last_loop_time_ms_),
                                       false);
}

void StreamFrontend::publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message) {
  (void) publish_frontend_mqtt_command_result(mqtt_transport_, device_config_, command, accepted, message);
}

void StreamFrontend::publish_mqtt_ota_status_(const EspectreOtaStatus &status) {
  (void) publish_frontend_mqtt_ota_status(mqtt_transport_, device_config_, status, monotonic_now_ms());
}

void StreamFrontend::handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) {
  csi_callback_total_++;
  if (info == nullptr || !normalized.valid()) {
    return;
  }
  last_csi_len_ = info->len;
  last_csi_payload_len_ = info->payload_len;
  csi_nonempty_total_++;
  if (info->payload != nullptr && info->payload_len > 0U) {
    csi_payload_present_total_++;
  }

  StimulusMetadata stimulus{};
  const bool has_stimulus =
      extract_stimulus_metadata_from_csi(info, collector_ip_addr_, local_ip_addr_, local_mac_addr_.data(), &stimulus);
  if (!has_stimulus) {
    stimulus_parse_fail_total_++;
    filtered_total_++;
    return;
  }

  if (state_.load(std::memory_order_relaxed) != WorkflowState::STREAMING) {
    return;
  }

  csi_rx_total_++;
  last_csi_ms_ = monotonic_now_ms();
  last_csi_channel_ = info->rx_ctrl.channel;

  std::array<uint8_t, CsiUdpSender::MAX_PACKET_BYTES> packet{};
  auto *header = reinterpret_cast<CsiStreamHeaderV3 *>(packet.data());
  header->magic = STREAM_MAGIC;
  header->version = STREAM_VERSION;
  header->header_len = static_cast<uint8_t>(sizeof(*header));
  header->chip = static_cast<uint8_t>(detect_chip_code());
  header->flags = 0U;
  header->seq_num = stream_seq_++;
  header->num_subcarriers = static_cast<uint16_t>(normalized.len / 2U);
  header->csi_len_bytes = static_cast<uint16_t>(normalized.len);
  header->device_id = espectre_effective_device_id_u64(device_config_);
  header->device_ticks_us = static_cast<uint64_t>(esp_timer_get_time());
  header->wifi_rx_ts_us = info->rx_ctrl.timestamp;
  header->wifi_rx_start_ts_ns = 0U;
  header->stimulus_id = 0U;
  header->channel = info->rx_ctrl.channel;
  header->rssi_dbm = info->rx_ctrl.rssi;
#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2 || CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || \
    CONFIG_IDF_TARGET_ESP32C2
  header->noise_floor_dbm = info->rx_ctrl.noise_floor;
#else
  header->noise_floor_dbm = -128;
#endif

  if (info->first_word_invalid) {
    header->flags |= STREAM_FLAG_FIRST_WORD_INVALID;
  }
  if (header->wifi_rx_ts_us != 0U) {
    header->flags |= STREAM_FLAG_WIFI_RX_TS_VALID;
  }
  if (fill_rx_timestamp_metadata(info, header)) {
    header->flags |= STREAM_FLAG_WIFI_RX_START_TS_NS_VALID;
  }
  if (has_stimulus) {
    stream_set_stimulus_id(header, stimulus.stimulus_id);
    stimulus_valid_total_++;
    header->flags |= STREAM_FLAG_STIMULUS_ID_VALID;
    if (stimulus.is_reference) {
      reference_frame_total_++;
      header->flags |= STREAM_FLAG_REFERENCE_FRAME;
    }
  }

  std::memcpy(packet.data() + sizeof(*header), normalized.data, normalized.len);
  const size_t packet_len = sizeof(*header) + normalized.len;
  (void)udp_sender_.queue_packet(packet.data(), packet_len);
}

void StreamFrontend::transition_to_(WorkflowState next, const char *reason) {
  const WorkflowState prev = state_.exchange(next, std::memory_order_relaxed);
  if (prev != next) {
    ESP_LOGI(TAG, "[STATE] %s -> %s (%s)", workflow_state_name(prev), workflow_state_name(next),
             reason != nullptr ? reason : "n/a");
  }
}

void StreamFrontend::log_runtime_telemetry_() {
  const uint64_t now_ms = static_cast<uint64_t>(monotonic_now_ms());
  if (last_log_ms_ != 0U && now_ms - last_log_ms_ < CONFIG_ESPECTRE_STREAM_LOG_INTERVAL_MS) {
    return;
  }

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  if (state == WorkflowState::WAIT_WIFI || state == WorkflowState::WIFI_READY || state == WorkflowState::CSI_READY) {
    last_log_ms_ = now_ms;
    return;
  }

  if (prev_log_sample_ms_ == 0U) {
    reset_runtime_telemetry_baseline_();
    prev_log_sample_ms_ = now_ms;
  }

  const uint64_t dt_ms = std::max<uint64_t>(1U, now_ms - prev_log_sample_ms_);
  const float csi_callback_pps =
      static_cast<float>(csi_callback_total_ - prev_csi_callback_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float stimulus_pps =
      static_cast<float>(stimulus_valid_total_ - prev_stimulus_valid_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float traffic_rx_pps =
      static_cast<float>(stimulus_service_.get_packets_received() - prev_traffic_rx_total_) * 1000.0F /
      static_cast<float>(dt_ms);
  const float tx_pps = static_cast<float>(udp_sender_.tx_total() - prev_tx_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float drop_pps =
      static_cast<float>(udp_sender_.drop_total() - prev_drop_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float fail_pps =
      static_cast<float>(udp_sender_.send_fail_total() - prev_fail_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float parse_fail_pps =
      static_cast<float>(stimulus_parse_fail_total_ - prev_parse_fail_total_) * 1000.0F / static_cast<float>(dt_ms);
  const uint32_t csi_age_ms = (last_csi_ms_ > 0U && now_ms >= last_csi_ms_) ? static_cast<uint32_t>(now_ms - last_csi_ms_)
                                                                              : 0U;
  const unsigned queue_ready = udp_sender_.ready_queue_depth();
  const unsigned queue_peak = udp_sender_.take_ready_queue_high_watermark();
  const unsigned queue_capacity = CsiUdpSender::QUEUE_CAPACITY;

  if (state == WorkflowState::STREAMING) {
    const bool stream_active =
        csi_callback_pps > 1.0F || stimulus_pps > 1.0F || tx_pps > 1.0F || traffic_rx_pps > 1.0F;
    if (stream_active) {
      ESP_LOGI(TAG,
               "state=STREAMING csi=%.2f stim=%.2f traffic=%.2f tx=%.2f queue=%u peak=%u/%u channel=%u age_ms=%" PRIu32,
               csi_callback_pps,
               stimulus_pps,
               traffic_rx_pps,
               tx_pps,
               queue_ready,
               queue_peak,
               queue_capacity,
               static_cast<unsigned>(last_csi_channel_),
               csi_age_ms);
      if (parse_fail_pps > 0.0F || drop_pps > 0.0F || fail_pps > 0.0F) {
        ESP_LOGW(TAG,
                 "stream anomalies: parse_fail=%.2f drop=%.2f fail=%.2f queue=%u peak=%u/%u payload_len=%u",
                 parse_fail_pps,
                 drop_pps,
                 fail_pps,
                 queue_ready,
                 queue_peak,
                 queue_capacity,
                 static_cast<unsigned>(last_csi_payload_len_));
      }
    } else if (stream_active_last_tick_) {
      ESP_LOGW(TAG,
               "stream idle: no stimulus/csi activity for %" PRIu32 " ms",
               csi_age_ms);
    }
    stream_active_last_tick_ = stream_active;
  } else {
    stream_active_last_tick_ = true;
  }

  prev_csi_callback_total_ = csi_callback_total_;
  prev_stimulus_valid_total_ = stimulus_valid_total_;
  prev_traffic_rx_total_ = stimulus_service_.get_packets_received();
  prev_tx_total_ = udp_sender_.tx_total();
  prev_drop_total_ = udp_sender_.drop_total();
  prev_fail_total_ = udp_sender_.send_fail_total();
  prev_parse_fail_total_ = stimulus_parse_fail_total_;
  prev_log_sample_ms_ = now_ms;
  last_log_ms_ = now_ms;
}

void StreamFrontend::reset_runtime_telemetry_baseline_() {
  prev_csi_callback_total_ = csi_callback_total_;
  prev_stimulus_valid_total_ = stimulus_valid_total_;
  prev_traffic_rx_total_ = stimulus_service_.get_packets_received();
  prev_tx_total_ = udp_sender_.tx_total();
  prev_drop_total_ = udp_sender_.drop_total();
  prev_fail_total_ = udp_sender_.send_fail_total();
  prev_parse_fail_total_ = stimulus_parse_fail_total_;
  prev_log_sample_ms_ = static_cast<uint64_t>(monotonic_now_ms());
  stream_active_last_tick_ = true;
}

}  // namespace espectre
}  // namespace esphome
