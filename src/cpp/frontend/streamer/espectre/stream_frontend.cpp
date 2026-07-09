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
#include <cstdlib>
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
constexpr float kBleSuspendStimulusPps = 10.0F;
constexpr float kBleResumeStimulusPps = 1.0F;
constexpr uint64_t kBleSuspendStableMs = 2000U;
constexpr uint64_t kBleResumeStableMs = 60000U;
constexpr uint16_t kWifiFrameControlRetryMask = 0x0800U;

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

float minimum_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

bool wifi_frame_has_retry_flag_(const wifi_csi_info_t *info) {
  if (info == nullptr || info->hdr == nullptr) {
    return false;
  }
  const uint8_t *hdr = reinterpret_cast<const uint8_t *>(info->hdr);
  const uint16_t frame_control = static_cast<uint16_t>(hdr[0]) | (static_cast<uint16_t>(hdr[1]) << 8U);
  return (frame_control & kWifiFrameControlRetryMask) != 0U;
}

template<typename T>
T *allocate_stream_storage(const char *label, size_t count) {
  const size_t bytes = sizeof(T) * count;
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
#ifdef MALLOC_CAP_SPIRAM
  if (T *external = static_cast<T *>(heap_caps_calloc(count, sizeof(T), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT))) {
    ESP_LOGI(TAG, "Allocated %s in external RAM (%u bytes)", label, static_cast<unsigned>(bytes));
    return external;
  }
#endif
  if (T *internal = static_cast<T *>(heap_caps_calloc(count, sizeof(T), MALLOC_CAP_8BIT))) {
    ESP_LOGI(TAG, "Allocated %s in internal RAM (%u bytes)", label, static_cast<unsigned>(bytes));
    return internal;
  }
#else
  if (T *storage = static_cast<T *>(std::calloc(count, sizeof(T)))) {
    return storage;
  }
#endif
  ESP_LOGE(TAG, "Failed to allocate %s (%u bytes)", label, static_cast<unsigned>(bytes));
  return nullptr;
}

void free_stream_storage(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  heap_caps_free(ptr);
#else
  std::free(ptr);
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
  if (!setup_deferred_csi_queue_()) {
    udp_sender_.shutdown();
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
  ble_sysinfo_refresh_.flush_if([this]() { return this->ble_ready_ && this->ble_client_connected_; },
                                [this]() { this->publish_ble_sysinfo_(); });

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
  process_deferred_csi_packets_();

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
  shutdown_deferred_csi_queue_();
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
                                 [this]() { this->ble_sysinfo_refresh_.request(); },
                                 [this]() { this->on_wifi_connected_(); },
                                 [this]() { this->on_wifi_disconnected_(); }},
      TAG,
      "Using Wi-Fi credentials provisioned over BLE");
  if (!check_esp(setup_err, "wifi_provisioning_.setup_station")) {
    return false;
  }
  return true;
}

bool StreamFrontend::setup_deferred_csi_queue_() {
  deferred_csi_slots_ = allocate_stream_storage<DeferredCsiPacket>("deferred CSI slots", CSI_DEFERRED_QUEUE_SLOTS);
  if (deferred_csi_slots_ == nullptr) {
    return false;
  }

  deferred_csi_free_slots_ = xQueueCreate(CSI_DEFERRED_QUEUE_SLOTS, sizeof(uint8_t));
  deferred_csi_ready_slots_ = xQueueCreate(CSI_DEFERRED_QUEUE_SLOTS, sizeof(uint8_t));
  if (deferred_csi_free_slots_ == nullptr || deferred_csi_ready_slots_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create deferred CSI queues");
    shutdown_deferred_csi_queue_();
    return false;
  }

  for (uint8_t idx = 0; idx < CSI_DEFERRED_QUEUE_SLOTS; idx++) {
    if (xQueueSend(deferred_csi_free_slots_, &idx, 0) != pdTRUE) {
      ESP_LOGE(TAG, "Failed to initialize deferred CSI free queue");
      shutdown_deferred_csi_queue_();
      return false;
    }
  }
  csi_deferred_drop_total_ = 0U;
  return true;
}

void StreamFrontend::shutdown_deferred_csi_queue_() {
  if (deferred_csi_free_slots_ != nullptr) {
    vQueueDelete(deferred_csi_free_slots_);
    deferred_csi_free_slots_ = nullptr;
  }
  if (deferred_csi_ready_slots_ != nullptr) {
    vQueueDelete(deferred_csi_ready_slots_);
    deferred_csi_ready_slots_ = nullptr;
  }
  free_stream_storage(deferred_csi_slots_);
  deferred_csi_slots_ = nullptr;
}

bool StreamFrontend::setup_ble_provisioning_() {
  const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                       device_info_.chip.empty() ? nullptr
                                                                                 : device_info_.chip.c_str());
  ble_bindings_.set_device_name(device_name.c_str());
  ble_bindings_.set_connection_state_callback([this](bool connected) {
    this->ble_client_connected_ = connected;
    if (connected) {
      this->ble_sysinfo_refresh_.request();
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

void StreamFrontend::suspend_ble_for_streaming_() {
  if (!ble_ready_) {
    return;
  }
  ESP_LOGI(TAG, "Disabling BLE during active streaming to free memory and reduce radio coexistence pressure");
  ble_bindings_.shutdown();
  ble_ready_ = false;
  ble_client_connected_ = false;
  ble_suspended_for_streaming_ = true;
  ble_high_stimulus_ms_ = 0U;
  ble_idle_stimulus_ms_ = 0U;
}

void StreamFrontend::resume_ble_after_streaming_() {
  if (!ble_suspended_for_streaming_) {
    return;
  }
  ESP_LOGI(TAG, "Re-enabling BLE after streaming returned idle");
  if (setup_ble_provisioning_()) {
    ble_suspended_for_streaming_ = false;
    ble_high_stimulus_ms_ = 0U;
    ble_idle_stimulus_ms_ = 0U;
  } else {
    ESP_LOGW(TAG, "Failed to re-enable BLE provisioning after streaming idle");
  }
}

void StreamFrontend::update_streaming_ble_policy_(float stimulus_pps, uint64_t dt_ms) {
  if (dt_ms == 0U) {
    return;
  }

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  const bool streaming = state == WorkflowState::STREAMING;

  if (!ble_suspended_for_streaming_) {
    if (streaming && ble_ready_ && stimulus_pps > kBleSuspendStimulusPps) {
      ble_high_stimulus_ms_ += dt_ms;
      if (ble_high_stimulus_ms_ >= kBleSuspendStableMs) {
        suspend_ble_for_streaming_();
      }
    } else {
      ble_high_stimulus_ms_ = 0U;
    }
    ble_idle_stimulus_ms_ = 0U;
    return;
  }

  if (!streaming || stimulus_pps < kBleResumeStimulusPps) {
    ble_idle_stimulus_ms_ += dt_ms;
    if (ble_idle_stimulus_ms_ >= kBleResumeStableMs) {
      resume_ble_after_streaming_();
    }
  } else {
    ble_idle_stimulus_ms_ = 0U;
  }
  ble_high_stimulus_ms_ = 0U;
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
  for (RecentWifiRxFrame &frame : recent_wifi_frames_) {
    frame.valid = false;
    frame.rx_seq = 0U;
    frame.src_mac.fill(0U);
  }
  recent_wifi_frame_idx_ = 0U;
  recent_stimulus_ids_.fill(0xFFFFFFFFU);
  recent_stimulus_idx_ = 0U;
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
  ble_sysinfo_refresh_.request();
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
  ble_sysinfo_refresh_.request();
}

void StreamFrontend::handle_ble_control_(const std::string &command) {
  if (command == "REQ_SYSINFO") {
    ble_sysinfo_refresh_.request();
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
    ble_sysinfo_refresh_.request();
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

bool StreamFrontend::enqueue_deferred_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized) {
  if (info == nullptr || !normalized.valid() || normalized.len > HT20_CSI_LEN || deferred_csi_free_slots_ == nullptr ||
      deferred_csi_ready_slots_ == nullptr || deferred_csi_slots_ == nullptr) {
    csi_deferred_drop_total_++;
    return false;
  }

  uint8_t slot_idx = 0U;
  if (xQueueReceive(deferred_csi_free_slots_, &slot_idx, 0) != pdTRUE) {
    csi_deferred_drop_total_++;
    return false;
  }

  DeferredCsiPacket &slot = deferred_csi_slots_[slot_idx];
  slot.rx_ctrl = info->rx_ctrl;
  slot.enqueued_at_ms = monotonic_now_ms();
  std::memcpy(slot.mac.data(), info->mac, slot.mac.size());
  std::memcpy(slot.dmac.data(), info->dmac, slot.dmac.size());
  std::memcpy(slot.normalized_csi.data(), normalized.data, normalized.len);
  slot.normalized_len = static_cast<uint16_t>(normalized.len);
  slot.payload_len = info->payload_len;
  slot.captured_payload_len = 0U;
  slot.rx_seq = info->rx_seq;
  slot.first_word_invalid = info->first_word_invalid;
  slot.payload_present = info->payload != nullptr && info->payload_len > 0U;
  if (slot.payload_present) {
    slot.captured_payload_len =
        static_cast<uint16_t>(std::min<size_t>(info->payload_len, slot.payload_prefix.size()));
    std::memcpy(slot.payload_prefix.data(), info->payload, slot.captured_payload_len);
  }

  if (xQueueSend(deferred_csi_ready_slots_, &slot_idx, 0) != pdTRUE) {
    csi_deferred_drop_total_++;
    (void)xQueueSend(deferred_csi_free_slots_, &slot_idx, 0);
    return false;
  }
  return true;
}

void StreamFrontend::process_deferred_csi_packets_() {
  if (deferred_csi_ready_slots_ == nullptr || deferred_csi_free_slots_ == nullptr || deferred_csi_slots_ == nullptr) {
    return;
  }

  for (uint8_t processed = 0U; processed < CSI_DEFERRED_QUEUE_SLOTS; processed++) {
    uint8_t slot_idx = 0U;
    if (xQueueReceive(deferred_csi_ready_slots_, &slot_idx, 0) != pdTRUE) {
      break;
    }

    process_deferred_csi_packet_(deferred_csi_slots_[slot_idx]);
    (void)xQueueSend(deferred_csi_free_slots_, &slot_idx, 0);
  }
}

void StreamFrontend::process_deferred_csi_packet_(const DeferredCsiPacket &packet) {
  if (packet.normalized_len == 0U || packet.normalized_len > packet.normalized_csi.size()) {
    return;
  }

  if (state_.load(std::memory_order_relaxed) != WorkflowState::STREAMING) {
    return;
  }

  const uint32_t now_ms = monotonic_now_ms();
  if (packet.enqueued_at_ms != 0U && now_ms >= packet.enqueued_at_ms) {
    const uint32_t deferred_age_ms = now_ms - packet.enqueued_at_ms;
    deferred_max_age_ms_since_log_ = std::max(deferred_max_age_ms_since_log_, deferred_age_ms);
  }

  wifi_csi_info_t info{};
  info.rx_ctrl = packet.rx_ctrl;
  std::memcpy(info.mac, packet.mac.data(), packet.mac.size());
  info.first_word_invalid = packet.first_word_invalid;
  info.payload_len = packet.captured_payload_len;
  info.payload = packet.captured_payload_len > 0U ? const_cast<uint8_t *>(packet.payload_prefix.data()) : nullptr;
  std::memcpy(info.dmac, packet.dmac.data(), packet.dmac.size());

  StimulusMetadata stimulus{};
  const bool has_stimulus =
      extract_stimulus_metadata_from_csi(&info, collector_ip_addr_, local_ip_addr_, local_mac_addr_.data(), &stimulus);
  if (!has_stimulus) {
    stimulus_parse_fail_total_++;
    filtered_total_++;
    return;
  }

  // Drop MAC-layer retransmissions of an already-forwarded stimulus frame. The
  // CSI callback fires once per received PHY frame, so a retransmitted stimulus
  // yields a duplicate CSI record carrying the same stimulus_id. Keeping only the
  // first copy holds the forwarded stream at the true stimulus rate (one record
  // per stimulus) regardless of how often the AP retransmits it.
  if (stimulus_recently_seen_(stimulus.stimulus_id)) {
    stimulus_dup_drop_total_++;
    return;
  }

  csi_rx_total_++;
  stimulus_valid_total_++;
  last_csi_ms_ = monotonic_now_ms();
  last_csi_channel_ = packet.rx_ctrl.channel;

  std::array<uint8_t, CsiUdpSender::MAX_PACKET_BYTES> packet_bytes{};
  auto *header = reinterpret_cast<CsiStreamHeaderV3 *>(packet_bytes.data());
  header->magic = STREAM_MAGIC;
  header->version = STREAM_VERSION;
  header->header_len = static_cast<uint8_t>(sizeof(*header));
  header->chip = static_cast<uint8_t>(detect_chip_code());
  header->flags = 0U;
  header->seq_num = stream_seq_++;
  header->num_subcarriers = static_cast<uint16_t>(packet.normalized_len / 2U);
  header->csi_len_bytes = packet.normalized_len;
  header->device_id = espectre_effective_device_id_u64(device_config_);
  header->device_ticks_us = static_cast<uint64_t>(esp_timer_get_time());
  header->wifi_rx_ts_us = packet.rx_ctrl.timestamp;
  header->wifi_rx_start_ts_ns = 0U;
  header->stimulus_id = 0U;
  header->channel = packet.rx_ctrl.channel;
  header->rssi_dbm = packet.rx_ctrl.rssi;
#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2 || CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || \
    CONFIG_IDF_TARGET_ESP32C2
  header->noise_floor_dbm = packet.rx_ctrl.noise_floor;
#else
  header->noise_floor_dbm = -128;
#endif

  if (packet.first_word_invalid) {
    header->flags |= STREAM_FLAG_FIRST_WORD_INVALID;
  }
  if (header->wifi_rx_ts_us != 0U) {
    header->flags |= STREAM_FLAG_WIFI_RX_TS_VALID;
  }
  if (fill_rx_timestamp_metadata(&info, header)) {
    header->flags |= STREAM_FLAG_WIFI_RX_START_TS_NS_VALID;
  }
  stream_set_stimulus_id(header, stimulus.stimulus_id);
  header->flags |= STREAM_FLAG_STIMULUS_ID_VALID;
  if (stimulus.is_reference) {
    reference_frame_total_++;
    header->flags |= STREAM_FLAG_REFERENCE_FRAME;
  }

  std::memcpy(packet_bytes.data() + sizeof(*header), packet.normalized_csi.data(), packet.normalized_len);
  const size_t packet_len = sizeof(*header) + packet.normalized_len;
  (void)udp_sender_.queue_packet(packet_bytes.data(), packet_len);
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

  if (wifi_frame_has_retry_flag_(info)) {
    wifi_retry_marked_total_++;
  }

  std::array<uint8_t, 6> src_mac{};
  std::memcpy(src_mac.data(), info->mac, src_mac.size());
  if (wifi_frame_recently_seen_(src_mac, info->rx_seq)) {
    wifi_seq_dup_drop_total_++;
    return;
  }

  (void)enqueue_deferred_csi_packet_(info, normalized);
}

bool StreamFrontend::wifi_frame_recently_seen_(const std::array<uint8_t, 6> &src_mac, uint16_t rx_seq) {
  for (const RecentWifiRxFrame &frame : recent_wifi_frames_) {
    if (!frame.valid || frame.rx_seq != rx_seq) {
      continue;
    }
    if (std::memcmp(frame.src_mac.data(), src_mac.data(), src_mac.size()) == 0) {
      return true;
    }
  }

  RecentWifiRxFrame &slot = recent_wifi_frames_[recent_wifi_frame_idx_];
  slot.src_mac = src_mac;
  slot.rx_seq = rx_seq;
  slot.valid = true;
  recent_wifi_frame_idx_ = static_cast<uint8_t>((recent_wifi_frame_idx_ + 1U) % STIMULUS_DEDUP_WINDOW);
  return false;
}

bool StreamFrontend::stimulus_recently_seen_(uint32_t stimulus_id) {
  for (uint32_t seen : recent_stimulus_ids_) {
    if (seen == stimulus_id) {
      return true;
    }
  }
  recent_stimulus_ids_[recent_stimulus_idx_] = stimulus_id;
  recent_stimulus_idx_ = static_cast<uint8_t>((recent_stimulus_idx_ + 1U) % STIMULUS_DEDUP_WINDOW);
  return false;
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
  const uint64_t traffic_raw_total = stimulus_service_.get_raw_packets_received();
  const uint64_t traffic_valid_total = stimulus_service_.get_packets_received();
  const uint64_t queued_total = udp_sender_.queued_total();
  const uint64_t tx_total = udp_sender_.tx_total();
  const uint64_t fail_total = udp_sender_.send_fail_total();
  const float csi_callback_pps =
      static_cast<float>(csi_callback_total_ - prev_csi_callback_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float stimulus_pps =
      static_cast<float>(stimulus_valid_total_ - prev_stimulus_valid_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float traffic_raw_pps =
      static_cast<float>(traffic_raw_total - prev_traffic_raw_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float traffic_rx_pps =
      static_cast<float>(traffic_valid_total - prev_traffic_rx_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float csi_deferred_drop_pps =
      static_cast<float>(csi_deferred_drop_total_ - prev_csi_deferred_drop_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float wifi_retry_marked_pps =
      static_cast<float>(wifi_retry_marked_total_ - prev_wifi_retry_marked_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float wifi_seq_dup_drop_pps =
      static_cast<float>(wifi_seq_dup_drop_total_ - prev_wifi_seq_dup_drop_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float stimulus_dup_drop_pps =
      static_cast<float>(stimulus_dup_drop_total_ - prev_stimulus_dup_drop_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float dup_total_pps = wifi_seq_dup_drop_pps + stimulus_dup_drop_pps;
  const float queued_pps = static_cast<float>(queued_total - prev_queued_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float tx_pps = static_cast<float>(tx_total - prev_tx_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float drop_pps =
      static_cast<float>(udp_sender_.drop_total() - prev_drop_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float fail_pps =
      static_cast<float>(fail_total - prev_fail_total_) * 1000.0F / static_cast<float>(dt_ms);
  const float parse_fail_pps =
      static_cast<float>(stimulus_parse_fail_total_ - prev_parse_fail_total_) * 1000.0F / static_cast<float>(dt_ms);
  const uint32_t csi_age_ms = (last_csi_ms_ > 0U && now_ms >= last_csi_ms_) ? static_cast<uint32_t>(now_ms - last_csi_ms_)
                                                                              : 0U;
  const uint32_t deferred_age_ms = deferred_max_age_ms_since_log_;
  const uint32_t sender_queue_age_ms = udp_sender_.oldest_ready_age_ms();
  const uint32_t sender_last_tx_age_ms = udp_sender_.last_tx_batch_age_ms();
  const uint32_t sender_last_fail_age_ms = udp_sender_.last_fail_batch_age_ms();
  const unsigned queue_ready = udp_sender_.ready_queue_depth();
  const unsigned queue_peak = udp_sender_.take_ready_queue_high_watermark();
  const unsigned queue_capacity = CsiUdpSender::QUEUE_CAPACITY;
  const unsigned csi_queue_ready =
      deferred_csi_ready_slots_ != nullptr ? static_cast<unsigned>(uxQueueMessagesWaiting(deferred_csi_ready_slots_)) : 0U;
  const uint64_t accepted_gap = queued_total >= tx_total ? (queued_total - tx_total) : 0U;
  const uint64_t queue_backlog = accepted_gap >= fail_total ? (accepted_gap - fail_total) : 0U;

  update_streaming_ble_policy_(stimulus_pps, dt_ms);
  if (state == WorkflowState::STREAMING) {
    const bool stream_active =
        csi_callback_pps > 1.0F || stimulus_pps > 1.0F || tx_pps > 1.0F || traffic_raw_pps > 1.0F ||
        traffic_rx_pps > 1.0F;
    if (stream_active) {
      ESP_LOGI(TAG,
               "state=STREAMING raw=%.2f traffic=%.2f csi=%.2f stim=%.2f queued=%.2f tx=%.2f backlog=%" PRIu64
               " csi_q=%u csi_drop=%.2f dup=%.2f wifi_dup=%.2f stim_dup=%.2f retry=%.2f defer_age=%" PRIu32
               " txq_age=%" PRIu32
               " tx_age=%" PRIu32 " fail_age=%" PRIu32
               " queue=%u peak=%u/%u loop=%.2fms heap=%.1f min=%.1f channel=%u age_ms=%" PRIu32,
               traffic_raw_pps,
               traffic_rx_pps,
               csi_callback_pps,
               stimulus_pps,
               queued_pps,
               tx_pps,
               queue_backlog,
               csi_queue_ready,
               csi_deferred_drop_pps,
               dup_total_pps,
               wifi_seq_dup_drop_pps,
               stimulus_dup_drop_pps,
               wifi_retry_marked_pps,
               deferred_age_ms,
               sender_queue_age_ms,
               sender_last_tx_age_ms,
               sender_last_fail_age_ms,
               queue_ready,
               queue_peak,
               queue_capacity,
               static_cast<double>(last_loop_time_ms_),
               static_cast<double>(current_free_memory_kb()),
               static_cast<double>(minimum_free_memory_kb()),
               static_cast<unsigned>(last_csi_channel_),
               csi_age_ms);
      if (parse_fail_pps > 0.0F || drop_pps > 0.0F || fail_pps > 0.0F || csi_deferred_drop_pps > 0.0F) {
        ESP_LOGW(TAG,
                 "stream anomalies: parse_fail=%.2f drop=%.2f fail=%.2f backlog=%" PRIu64
                 " raw=%.2f traffic=%.2f queued=%.2f tx=%.2f csi_drop=%.2f csi_q=%u"
                 " dup=%.2f wifi_dup=%.2f stim_dup=%.2f retry=%.2f"
                 " defer_age=%" PRIu32 " txq_age=%" PRIu32 " tx_age=%" PRIu32 " fail_age=%" PRIu32
                 " queue=%u peak=%u/%u payload_len=%u loop=%.2fms",
                 parse_fail_pps,
                 drop_pps,
                 fail_pps,
                 queue_backlog,
                 traffic_raw_pps,
                 traffic_rx_pps,
                 queued_pps,
                 tx_pps,
                 csi_deferred_drop_pps,
                 csi_queue_ready,
                 dup_total_pps,
                 wifi_seq_dup_drop_pps,
                 stimulus_dup_drop_pps,
                 wifi_retry_marked_pps,
                 deferred_age_ms,
                 sender_queue_age_ms,
                 sender_last_tx_age_ms,
                 sender_last_fail_age_ms,
                 queue_ready,
                 queue_peak,
                 queue_capacity,
                 static_cast<unsigned>(last_csi_payload_len_),
                 static_cast<double>(last_loop_time_ms_));
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
  prev_traffic_raw_total_ = traffic_raw_total;
  prev_traffic_rx_total_ = traffic_valid_total;
  prev_csi_deferred_drop_total_ = csi_deferred_drop_total_;
  prev_wifi_retry_marked_total_ = wifi_retry_marked_total_;
  prev_wifi_seq_dup_drop_total_ = wifi_seq_dup_drop_total_;
  prev_stimulus_dup_drop_total_ = stimulus_dup_drop_total_;
  prev_queued_total_ = queued_total;
  prev_tx_total_ = tx_total;
  prev_drop_total_ = udp_sender_.drop_total();
  prev_fail_total_ = fail_total;
  prev_parse_fail_total_ = stimulus_parse_fail_total_;
  prev_log_sample_ms_ = now_ms;
  deferred_max_age_ms_since_log_ = 0U;
  last_log_ms_ = now_ms;
}

void StreamFrontend::reset_runtime_telemetry_baseline_() {
  prev_csi_callback_total_ = csi_callback_total_;
  prev_stimulus_valid_total_ = stimulus_valid_total_;
  prev_traffic_raw_total_ = stimulus_service_.get_raw_packets_received();
  prev_traffic_rx_total_ = stimulus_service_.get_packets_received();
  prev_csi_deferred_drop_total_ = csi_deferred_drop_total_;
  prev_wifi_retry_marked_total_ = wifi_retry_marked_total_;
  prev_wifi_seq_dup_drop_total_ = wifi_seq_dup_drop_total_;
  prev_stimulus_dup_drop_total_ = stimulus_dup_drop_total_;
  prev_queued_total_ = udp_sender_.queued_total();
  prev_tx_total_ = udp_sender_.tx_total();
  prev_drop_total_ = udp_sender_.drop_total();
  prev_fail_total_ = udp_sender_.send_fail_total();
  prev_parse_fail_total_ = stimulus_parse_fail_total_;
  prev_log_sample_ms_ = static_cast<uint64_t>(monotonic_now_ms());
  deferred_max_age_ms_since_log_ = 0U;
  ble_high_stimulus_ms_ = 0U;
  ble_idle_stimulus_ms_ = 0U;
  stream_active_last_tick_ = true;
}

}  // namespace espectre
}  // namespace esphome
