/*
 * ESPectre - Streamer Frontend
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "stream_frontend.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "csi_payload_normalizer.h"
#include "csi_platform_config.h"
#include "csi_stream_protocol.h"
#include "espectre_log.h"
#include "utils.h"
#include "esp_attr.h"
#include "esp_event.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/inet.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

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

#ifdef CONFIG_ESPECTRE_GAIN_LOCK_ENABLED
constexpr bool kGainLockEnabled = true;
#else
constexpr bool kGainLockEnabled = false;
#endif

constexpr GainLockMode kGainLockMode =
#if CONFIG_ESPECTRE_GAIN_LOCK_MODE_ENABLED
    GainLockMode::ENABLED;
#elif CONFIG_ESPECTRE_GAIN_LOCK_MODE_DISABLED
    GainLockMode::DISABLED;
#else
    GainLockMode::AUTO;
#endif

constexpr TrafficGeneratorMode kTrafficGeneratorMode =
#if CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS
    TrafficGeneratorMode::DNS;
#else
    TrafficGeneratorMode::PING;
#endif

constexpr uint8_t kStimulusMagic[4] = {'E', 'S', 'T', 'M'};
constexpr uint8_t kStimulusVersion = 1U;
constexpr uint8_t kStimulusRoleMeasurement = 0U;
constexpr uint8_t kStimulusRoleReference = 1U;
constexpr size_t kStimulusHeaderBytes = 10U;
constexpr size_t kLlcSnapHeaderBytes = 8U;
constexpr uint16_t kEtherTypeIpv4 = 0x0800U;
constexpr uint8_t kIpProtoUdp = 17U;

const char *workflow_state_name(StreamFrontend::WorkflowState state) {
  switch (state) {
    case StreamFrontend::WorkflowState::WAIT_WIFI:
      return "WAIT_WIFI";
    case StreamFrontend::WorkflowState::WIFI_READY:
      return "WIFI_READY";
    case StreamFrontend::WorkflowState::CSI_READY:
      return "CSI_READY";
    case StreamFrontend::WorkflowState::GAIN_LOCK:
      return "GAIN_LOCK";
    case StreamFrontend::WorkflowState::STREAMING:
      return "STREAMING";
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

bool parse_bssid(const char *text, uint8_t out[6]) {
  if (text == nullptr || out == nullptr || text[0] == '\0') {
    return false;
  }

  unsigned int bytes[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (std::sscanf(text, "%2x:%2x:%2x:%2x:%2x:%2x", &bytes[0], &bytes[1], &bytes[2], &bytes[3], &bytes[4],
                  &bytes[5]) != 6) {
    return false;
  }

  for (size_t i = 0; i < 6; i++) {
    out[i] = static_cast<uint8_t>(bytes[i]);
  }
  return true;
}

uint16_t read_be16(const uint8_t *data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8U) | static_cast<uint16_t>(data[1]));
}

uint32_t read_be32(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24U) | (static_cast<uint32_t>(data[1]) << 16U) |
         (static_cast<uint32_t>(data[2]) << 8U) | static_cast<uint32_t>(data[3]);
}

bool parse_stimulus_datagram(const uint8_t *payload, size_t payload_len, uint32_t *stimulus_id, bool *is_reference) {
  if (payload == nullptr || stimulus_id == nullptr || is_reference == nullptr || payload_len < kStimulusHeaderBytes) {
    return false;
  }
  if (std::memcmp(payload, kStimulusMagic, sizeof(kStimulusMagic)) != 0 || payload[4] != kStimulusVersion) {
    return false;
  }
  if (payload[5] != kStimulusRoleMeasurement && payload[5] != kStimulusRoleReference) {
    return false;
  }

  *stimulus_id = read_be32(payload + 6U);
  *is_reference = (payload[5] == kStimulusRoleReference);
  return true;
}

bool parse_stimulus_from_llc_snap(const uint8_t *payload,
                                  size_t payload_len,
                                  uint32_t *stimulus_id,
                                  bool *is_reference) {
  if (payload == nullptr || stimulus_id == nullptr || is_reference == nullptr ||
      payload_len < kLlcSnapHeaderBytes + 20U + 8U + kStimulusHeaderBytes) {
    return false;
  }
  if (payload[0] != 0xAAU || payload[1] != 0xAAU || payload[2] != 0x03U || payload[3] != 0x00U ||
      payload[4] != 0x00U || payload[5] != 0x00U || read_be16(payload + 6U) != kEtherTypeIpv4) {
    return false;
  }

  const uint8_t *ip = payload + kLlcSnapHeaderBytes;
  const size_t ip_len = payload_len - kLlcSnapHeaderBytes;
  if (ip_len < 20U || (ip[0] >> 4U) != 4U || ip[9] != kIpProtoUdp) {
    return false;
  }

  const size_t ip_header_len = static_cast<size_t>(ip[0] & 0x0FU) * 4U;
  if (ip_header_len < 20U || ip_len < ip_header_len + 8U + kStimulusHeaderBytes) {
    return false;
  }

  const uint16_t fragment_field = read_be16(ip + 6U);
  if ((fragment_field & 0x3FFFU) != 0U) {
    return false;
  }

  const uint8_t *udp = ip + ip_header_len;
  const uint16_t dst_port = read_be16(udp + 2U);
  if (dst_port != static_cast<uint16_t>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT)) {
    return false;
  }

  return parse_stimulus_datagram(udp + 8U, ip_len - ip_header_len - 8U, stimulus_id, is_reference);
}

bool extract_stimulus_metadata(const wifi_csi_info_t *info, uint32_t *stimulus_id, bool *is_reference) {
  if (info == nullptr || stimulus_id == nullptr || is_reference == nullptr || info->payload == nullptr ||
      info->payload_len == 0U) {
    return false;
  }

  const auto *payload = reinterpret_cast<const uint8_t *>(info->payload);
  const size_t payload_len = info->payload_len;
  if (parse_stimulus_datagram(payload, payload_len, stimulus_id, is_reference)) {
    return true;
  }
  if (parse_stimulus_from_llc_snap(payload, payload_len, stimulus_id, is_reference)) {
    return true;
  }

  constexpr uint8_t kLlcSnapPrefix[6] = {0xAAU, 0xAAU, 0x03U, 0x00U, 0x00U, 0x00U};
  const size_t scan_limit = std::min<size_t>(payload_len, 32U);
  for (size_t offset = 1U; offset + sizeof(kLlcSnapPrefix) <= scan_limit; offset++) {
    if (std::memcmp(payload + offset, kLlcSnapPrefix, sizeof(kLlcSnapPrefix)) != 0) {
      continue;
    }
    if (parse_stimulus_from_llc_snap(payload + offset, payload_len - offset, stimulus_id, is_reference)) {
      return true;
    }
  }

  return false;
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

uint64_t derive_device_id() {
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) != ESP_OK) {
    return 0U;
  }

  uint64_t device_id = 0U;
  for (uint8_t byte : mac) {
    device_id = (device_id << 8U) | static_cast<uint64_t>(byte);
  }
  return device_id;
}

bool parse_collector_addr(sockaddr_in *out_addr) {
  if (out_addr == nullptr) {
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(CONFIG_ESPECTRE_COLLECTOR_PORT));
  if (inet_aton(CONFIG_ESPECTRE_COLLECTOR_IP, &addr.sin_addr) == 0) {
    return false;
  }

  *out_addr = addr;
  return true;
}

bool fill_gain_metadata(const wifi_csi_info_t *info, CsiStreamHeaderV2 *header) {
  if (info == nullptr || header == nullptr) {
    return false;
  }
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  const auto *phy_info = reinterpret_cast<const wifi_pkt_rx_ctrl_phy_t *>(info);
  header->agc_gain = static_cast<uint8_t>(phy_info->agc_gain);
  header->fft_gain = static_cast<int8_t>(phy_info->fft_gain);
  return true;
#else
  header->agc_gain = 0U;
  header->fft_gain = 0;
  return false;
#endif
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

bool fill_rx_timestamp_metadata(const wifi_csi_info_t *info, CsiStreamHeaderV2 *header) {
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

bool StreamFrontend::setup() {
  if (setup_complete_) {
    return true;
  }

  if (!init_nvs_()) {
    return false;
  }

  device_id_ = derive_device_id();
  stream_seq_ = 0U;

  if (!udp_sender_.setup()) {
    return false;
  }

  sockaddr_in collector_addr{};
  if (kStreamOutputEnabled) {
    if (!parse_collector_addr(&collector_addr)) {
      ESP_LOGE(TAG, "Invalid collector address: %s", CONFIG_ESPECTRE_COLLECTOR_IP);
      return false;
    }
    udp_sender_.set_collector(collector_addr, true);
  } else {
    udp_sender_.set_collector(collector_addr, false);
  }

  traffic_generator_.init(CONFIG_ESPECTRE_TRAFFIC_GENERATOR_RATE, kTrafficGeneratorMode);
  udp_listener_.init(static_cast<uint16_t>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT));
  if (CONFIG_ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP[0] != '\0') {
    udp_listener_.set_multicast_group(CONFIG_ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP);
  }
  gain_controller_.init(kGainLockMode);
  gain_controller_.set_lock_complete_callback([this]() { gain_lock_complete_.store(true, std::memory_order_relaxed); });

  if (!init_wifi_station_()) {
    return false;
  }

  setup_complete_ = true;
  ESP_LOGI(TAG,
           "Streamer frontend ready: collector=%s:%u gain_lock=%s traffic_rx_port=%u device_id=0x%016" PRIx64,
           kStreamOutputEnabled ? CONFIG_ESPECTRE_COLLECTOR_IP : "(disabled)",
           static_cast<unsigned>(CONFIG_ESPECTRE_COLLECTOR_PORT),
           kGainLockEnabled ? "on" : "off",
           static_cast<unsigned>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT),
           device_id_);
  return true;
}

void StreamFrontend::loop() {
  if (!setup_complete_) {
    return;
  }

  if (udp_listener_.is_running()) {
    udp_listener_.loop();
  }

  if (!wifi_connected_.load(std::memory_order_relaxed)) {
    log_runtime_telemetry_();
    return;
  }

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  if (state == WorkflowState::WAIT_WIFI) {
    transition_to_(WorkflowState::WIFI_READY, "wifi connected");
  } else if (state == WorkflowState::WIFI_READY) {
    if (start_csi_()) {
      transition_to_(WorkflowState::CSI_READY, "csi enabled");
    }
  } else if (state == WorkflowState::CSI_READY) {
    if (!kGainLockEnabled || gain_controller_.is_locked()) {
      transition_to_(WorkflowState::STREAMING, "gain lock skipped");
    } else {
      transition_to_(WorkflowState::GAIN_LOCK, "collecting gain baseline");
    }
  } else if (state == WorkflowState::GAIN_LOCK && gain_lock_complete_.exchange(false, std::memory_order_relaxed)) {
    ESP_LOGI(TAG,
             "Gain lock completed: agc=%u fft=%d needs_cv=%s",
             static_cast<unsigned>(gain_controller_.get_agc_gain()),
             static_cast<int>(gain_controller_.get_fft_gain()),
             gain_controller_.needs_cv_normalization() ? "yes" : "no");
    transition_to_(WorkflowState::STREAMING, "gain lock complete");
  }

  log_runtime_telemetry_();
}

void StreamFrontend::shutdown() {
  if (!setup_complete_) {
    return;
  }

  stop_csi_();
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
  wifi_lifecycle_.unregister_handlers();
  if (wifi_event_instance_ != nullptr) {
    esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_instance_);
    wifi_event_instance_ = nullptr;
  }
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
  if (!check_esp(esp_netif_init(), "esp_netif_init")) {
    return false;
  }
  const esp_err_t loop_err = esp_event_loop_create_default();
  if (loop_err != ESP_OK && loop_err != ESP_ERR_INVALID_STATE) {
    return check_esp(loop_err, "esp_event_loop_create_default");
  }
  if (esp_netif_create_default_wifi_sta() == nullptr) {
    ESP_LOGE(TAG, "esp_netif_create_default_wifi_sta failed");
    return false;
  }

  wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
  if (!check_esp(esp_wifi_init(&wifi_cfg), "esp_wifi_init") ||
      !check_esp(esp_wifi_set_storage(WIFI_STORAGE_RAM), "esp_wifi_set_storage") ||
      !check_esp(esp_wifi_set_mode(WIFI_MODE_STA), "esp_wifi_set_mode") ||
      !check_esp(esp_wifi_set_ps(WIFI_PS_NONE), "esp_wifi_set_ps")) {
    return false;
  }

  if (wifi_lifecycle_.init() != ESP_OK ||
      wifi_lifecycle_.register_handlers([this]() { on_wifi_connected_(); },
                                        [this]() { on_wifi_disconnected_(); }) != ESP_OK) {
    return false;
  }

  if (!check_esp(esp_event_handler_instance_register(WIFI_EVENT,
                                                     ESP_EVENT_ANY_ID,
                                                     &StreamFrontend::wifi_event_handler_,
                                                     this,
                                                     &wifi_event_instance_),
                 "esp_event_handler_instance_register(WIFI_EVENT)")) {
    return false;
  }

  wifi_config_t sta_cfg{};
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.ssid), sizeof(sta_cfg.sta.ssid), "%s", CONFIG_ESPECTRE_WIFI_SSID);
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.password), sizeof(sta_cfg.sta.password), "%s",
                CONFIG_ESPECTRE_WIFI_PASSWORD);
  sta_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  sta_cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
  sta_cfg.sta.pmf_cfg.capable = true;
  sta_cfg.sta.pmf_cfg.required = false;

  if (CONFIG_ESPECTRE_WIFI_BSSID[0] != '\0') {
    if (!parse_bssid(CONFIG_ESPECTRE_WIFI_BSSID, sta_cfg.sta.bssid)) {
      ESP_LOGE(TAG, "Invalid BSSID format: %s", CONFIG_ESPECTRE_WIFI_BSSID);
      return false;
    }
    sta_cfg.sta.bssid_set = true;
  }

  if (!check_esp(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg), "esp_wifi_set_config")) {
    return false;
  }
  return check_esp(esp_wifi_start(), "esp_wifi_start");
}

bool StreamFrontend::start_csi_() {
  if (csi_enabled_.load(std::memory_order_relaxed)) {
    return true;
  }

  if (!check_esp(configure_ht20_csi(&wifi_csi_), "configure_ht20_csi") ||
      !check_esp(wifi_csi_.set_csi_rx_cb(&StreamFrontend::csi_rx_callback_wrapper_, this), "esp_wifi_set_csi_rx_cb") ||
      !check_esp(wifi_csi_.set_csi(true), "esp_wifi_set_csi")) {
    return false;
  }

  csi_enabled_.store(true, std::memory_order_relaxed);
  return true;
}

void StreamFrontend::stop_csi_() {
  if (!csi_enabled_.load(std::memory_order_relaxed)) {
    return;
  }

  (void)wifi_csi_.set_csi(false);
  (void)wifi_csi_.set_csi_rx_cb(nullptr, nullptr);
  csi_enabled_.store(false, std::memory_order_relaxed);
}

void StreamFrontend::on_wifi_connected_() {
  wifi_connected_.store(true, std::memory_order_relaxed);
  wifi_retry_count_ = 0;
  gain_controller_.init(kGainLockMode);
  gain_controller_.set_lock_complete_callback([this]() { gain_lock_complete_.store(true, std::memory_order_relaxed); });
  gain_lock_complete_.store(false, std::memory_order_relaxed);

  if (CONFIG_ESPECTRE_TRAFFIC_GENERATOR_RATE > 0) {
    if (!traffic_generator_.is_running() && !traffic_generator_.start()) {
      ESP_LOGW(TAG, "Failed to start traffic generator");
    }
  } else if (!udp_listener_.is_running() && !udp_listener_.start()) {
    ESP_LOGW(TAG, "Failed to start UDP listener");
  }
}

void StreamFrontend::on_wifi_disconnected_() {
  wifi_connected_.store(false, std::memory_order_relaxed);
  stop_csi_();
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
  transition_to_(WorkflowState::WAIT_WIFI, "wifi disconnected");
}

void StreamFrontend::handle_csi_packet_(wifi_csi_info_t *info) {
  if (info == nullptr || info->buf == nullptr || info->len == 0U) {
    return;
  }

  csi_rx_total_++;
  last_csi_ms_ = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
  last_csi_channel_ = info->rx_ctrl.channel;

  const WorkflowState state = state_.load(std::memory_order_relaxed);
  if (state == WorkflowState::GAIN_LOCK) {
    gain_controller_.process_packet(info);
    return;
  }
  if (state != WorkflowState::STREAMING) {
    return;
  }

  int8_t remap_buffer[HT20_CSI_LEN];
  const NormalizedCSIPayload normalized =
      normalize_ht20_csi_payload(info->buf, info->len, remap_buffer, sizeof(remap_buffer));
  if (!normalized.valid()) {
    filtered_total_++;
    if (filtered_total_ % 100 == 1) {
      ESP_LOGW(TAG, "Filtered %llu packets with unsupported CSI length %u",
               static_cast<unsigned long long>(filtered_total_), static_cast<unsigned>(info->len));
    }
    return;
  }

  if (!collapse_logged_ &&
      (normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT20 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI double-length collapse active: 256->128 and/or 228->114");
    collapse_logged_ = true;
  }
  if (!remap_logged_ &&
      (normalized.tag == NormalizedCSIPayloadTag::HT57_TO_64 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI remap active: 57->64 SC (left_pad=4, right_pad=3)");
    remap_logged_ = true;
  }

  uint32_t stimulus_id = 0U;
  bool is_reference = false;
  const bool has_stimulus = extract_stimulus_metadata(info, &stimulus_id, &is_reference);

  std::array<uint8_t, CsiUdpSender::MAX_PACKET_BYTES> packet{};
  auto *header = reinterpret_cast<CsiStreamHeaderV2 *>(packet.data());
  header->magic = STREAM_MAGIC;
  header->version = STREAM_VERSION;
  header->header_len = static_cast<uint8_t>(sizeof(*header));
  header->chip = static_cast<uint8_t>(detect_chip_code());
  header->flags = 0U;
  header->seq_num = stream_seq_++;
  header->num_subcarriers = static_cast<uint16_t>(normalized.len / 2U);
  header->csi_len_bytes = static_cast<uint16_t>(normalized.len);
  header->device_id = device_id_;
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
  header->agc_gain = 0U;
  header->fft_gain = 0;
  header->reserved0 = 0U;

  if (!gain_controller_.needs_cv_normalization()) {
    header->flags |= STREAM_FLAG_GAIN_LOCKED;
  }
  if (info->first_word_invalid) {
    header->flags |= STREAM_FLAG_FIRST_WORD_INVALID;
  }
  if (header->wifi_rx_ts_us != 0U) {
    header->flags |= STREAM_FLAG_WIFI_RX_TS_VALID;
  }
  if (fill_gain_metadata(info, header)) {
    header->flags |= STREAM_FLAG_GAIN_INFO_VALID;
  }
  if (fill_rx_timestamp_metadata(info, header)) {
    header->flags |= STREAM_FLAG_WIFI_RX_START_TS_NS_VALID;
  }
  if (has_stimulus) {
    stream_set_stimulus_id(header, stimulus_id);
    stimulus_valid_total_++;
    header->flags |= STREAM_FLAG_STIMULUS_ID_VALID;
    if (is_reference) {
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
  const uint64_t now_ms = static_cast<uint64_t>(esp_timer_get_time() / 1000ULL);
  if (last_log_ms_ != 0U && now_ms - last_log_ms_ < CONFIG_ESPECTRE_STREAM_LOG_INTERVAL_MS) {
    return;
  }

  static uint64_t prev_csi_rx = 0U;
  static uint64_t prev_stimulus_valid = 0U;
  static uint64_t prev_reference = 0U;
  static uint64_t prev_traffic_rx = 0U;
  static uint64_t prev_queued = 0U;
  static uint64_t prev_tx = 0U;
  static uint64_t prev_drop = 0U;
  static uint64_t prev_fail = 0U;
  static uint64_t prev_ms = now_ms;

  const uint64_t dt_ms = std::max<uint64_t>(1U, now_ms - prev_ms);
  const float csi_rx_pps = static_cast<float>(csi_rx_total_ - prev_csi_rx) * 1000.0F / static_cast<float>(dt_ms);
  const float stimulus_pps =
      static_cast<float>(stimulus_valid_total_ - prev_stimulus_valid) * 1000.0F / static_cast<float>(dt_ms);
  const float reference_pps =
      static_cast<float>(reference_frame_total_ - prev_reference) * 1000.0F / static_cast<float>(dt_ms);
  const float queued_pps =
      static_cast<float>(udp_sender_.queued_total() - prev_queued) * 1000.0F / static_cast<float>(dt_ms);
  const float traffic_rx_pps =
      static_cast<float>(udp_listener_.get_packets_received() - prev_traffic_rx) * 1000.0F / static_cast<float>(dt_ms);
  const float tx_pps = static_cast<float>(udp_sender_.tx_total() - prev_tx) * 1000.0F / static_cast<float>(dt_ms);
  const float drop_pps =
      static_cast<float>(udp_sender_.drop_total() - prev_drop) * 1000.0F / static_cast<float>(dt_ms);
  const float fail_pps =
      static_cast<float>(udp_sender_.send_fail_total() - prev_fail) * 1000.0F / static_cast<float>(dt_ms);
  const uint32_t csi_age_ms = (last_csi_ms_ > 0U && now_ms >= last_csi_ms_) ? static_cast<uint32_t>(now_ms - last_csi_ms_)
                                                                              : 0U;

  ESP_LOGI(TAG,
           "state=%s pps[csi=%.2f stim=%.2f ref=%.2f traffic_rx=%.2f queued=%.2f tx=%.2f] drop=%.2f fail=%.2f channel=%u age_ms=%" PRIu32,
           workflow_state_name(state_.load(std::memory_order_relaxed)),
           csi_rx_pps,
           stimulus_pps,
           reference_pps,
           traffic_rx_pps,
           queued_pps,
           tx_pps,
           drop_pps,
           fail_pps,
           static_cast<unsigned>(last_csi_channel_),
           csi_age_ms);

  prev_csi_rx = csi_rx_total_;
  prev_stimulus_valid = stimulus_valid_total_;
  prev_reference = reference_frame_total_;
  prev_traffic_rx = udp_listener_.get_packets_received();
  prev_queued = udp_sender_.queued_total();
  prev_tx = udp_sender_.tx_total();
  prev_drop = udp_sender_.drop_total();
  prev_fail = udp_sender_.send_fail_total();
  prev_ms = now_ms;
  last_log_ms_ = now_ms;
}

void IRAM_ATTR StreamFrontend::csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *info) {
  StreamFrontend *frontend = static_cast<StreamFrontend *>(ctx);
  if (frontend != nullptr) {
    frontend->handle_csi_packet_(info);
  }
}

void StreamFrontend::wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
  StreamFrontend *frontend = static_cast<StreamFrontend *>(arg);
  if (frontend == nullptr || event_base != WIFI_EVENT) {
    return;
  }

  if (event_id == WIFI_EVENT_STA_START) {
    (void)esp_wifi_connect();
    return;
  }

  if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    const auto *event = static_cast<const wifi_event_sta_disconnected_t *>(event_data);
    ESP_LOGW(TAG, "Wi-Fi disconnected: reason=%u", event != nullptr ? static_cast<unsigned>(event->reason) : 0U);
    if (frontend->wifi_retry_count_ < kWifiConnectMaxRetry) {
      frontend->wifi_retry_count_++;
      (void)esp_wifi_connect();
    }
    return;
  }

  (void)event_data;
}

}  // namespace espectre
}  // namespace esphome
