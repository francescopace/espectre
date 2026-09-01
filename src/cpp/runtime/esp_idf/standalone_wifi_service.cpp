/*
 * ESPectre - Standalone Wi-Fi Service
 *
 * Starts and monitors the standalone station connection used by sensing
 * frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "standalone_wifi_service.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <utility>

#include "espectre_log.h"
#include "esp_netif.h"
#include "esp_netif_ip_addr.h"
#include "esp_wifi.h"
#include "runtime_config_utils.h"
#include "runtime_time.h"
#include "wifi_band_helpers.h"

namespace espectre {

namespace {

static const char *const TAG = "StandaloneWiFi";
constexpr uint64_t DEFERRED_CONNECT_FALLBACK_DELAY_US = 1500000ULL;
constexpr uint16_t MAX_SCAN_ACCESS_POINTS = 32U;

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

bool has_text(const char *text) { return text != nullptr && text[0] != '\0'; }

std::string format_bssid(const uint8_t bssid[6]) {
  char formatted[18]{};
  std::snprintf(formatted,
                sizeof(formatted),
                "%02X:%02X:%02X:%02X:%02X:%02X",
                bssid[0],
                bssid[1],
                bssid[2],
                bssid[3],
                bssid[4],
                bssid[5]);
  return formatted;
}

void format_ip_address(const esp_ip4_addr_t &ip, char *out, size_t out_size) {
  if (out == nullptr || out_size == 0U) {
    return;
  }
  out[0] = '\0';
  if (ip.addr != 0U) {
    esp_ip4addr_ntoa(&ip, out, static_cast<int>(out_size));
  }
}

const char *wifi_disconnect_reason_to_str(uint8_t reason) {
  switch (reason) {
    case WIFI_REASON_BEACON_TIMEOUT:
      return "beacon-timeout";
    case WIFI_REASON_NO_AP_FOUND:
      return "no-ap-found";
    case WIFI_REASON_AUTH_FAIL:
      return "auth-fail";
    case WIFI_REASON_ASSOC_FAIL:
      return "assoc-fail";
    case WIFI_REASON_HANDSHAKE_TIMEOUT:
      return "handshake-timeout";
    case WIFI_REASON_CONNECTION_FAIL:
      return "connection-fail";
    case WIFI_REASON_NO_AP_FOUND_W_COMPATIBLE_SECURITY:
      return "no-ap-compatible-security";
    case WIFI_REASON_NO_AP_FOUND_IN_AUTHMODE_THRESHOLD:
      return "no-ap-authmode-threshold";
    case WIFI_REASON_NO_AP_FOUND_IN_RSSI_THRESHOLD:
      return "no-ap-rssi-threshold";
    default:
      return "unknown";
  }
}

}  // namespace

esp_err_t StandaloneWifiService::setup(const StandaloneWifiConfig &config,
                                       standalone_wifi_callback_t connected_cb,
                                       standalone_wifi_callback_t disconnected_cb) {
  config_ = config;
  if (!wifi_band_policy_is_supported(config_.band_policy)) {
    ESPECTRE_LOGE(TAG, "Wi-Fi band policy is not supported by this target: %s",
             wifi_band_policy_name(config_.band_policy));
    return ESP_ERR_NOT_SUPPORTED;
  }
  if (!wifi_channel_is_supported(config_.channel) ||
      !wifi_channel_matches_band_policy(config_.channel, config_.band_policy)) {
    ESPECTRE_LOGE(TAG, "Invalid Wi-Fi channel: %u (expected %s)",
             static_cast<unsigned>(config_.channel),
             wifi_channel_supported_description(config_.band_policy));
    return ESP_ERR_INVALID_ARG;
  }
  connected_cb_ = connected_cb;
  disconnected_cb_ = disconnected_cb;

  esp_err_t err = esp_netif_init();
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESPECTRE_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(err));
    return err;
  }

  if (esp_netif_create_default_wifi_sta() == nullptr) {
    ESPECTRE_LOGE(TAG, "esp_netif_create_default_wifi_sta failed");
    return ESP_FAIL;
  }

  wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&wifi_cfg);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_wifi_set_storage(WIFI_STORAGE_RAM);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_wifi_set_storage failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_wifi_set_mode(WIFI_MODE_STA);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_wifi_set_mode failed: %s", esp_err_to_name(err));
    return err;
  }

  // Keep the CSI bootstrap deterministic: initialize
  // the internal Wi-Fi CSI structures before the station starts associating.
  err = esp_wifi_set_promiscuous(false);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_wifi_set_promiscuous failed: %s", esp_err_to_name(err));
    return err;
  }

  if (config_.manage_csi_lifecycle) {
    err = wifi_lifecycle_.register_handlers([this](const esp_netif_ip_info_t &) {
                                              handle_lifecycle_connected_();
                                            },
                                            [this]() { handle_lifecycle_disconnected_(); },
                                            config_.band_policy);
    if (err != ESP_OK) {
      ESPECTRE_LOGE(TAG, "Wi-Fi lifecycle handler registration failed: %s", esp_err_to_name(err));
      return err;
    }
  }

  err = esp_event_handler_instance_register(WIFI_EVENT,
                                            ESP_EVENT_ANY_ID,
                                            &StandaloneWifiService::wifi_event_handler_,
                                            this,
                                            &wifi_event_instance_);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "Wi-Fi event handler registration failed: %s", esp_err_to_name(err));
    if (config_.manage_csi_lifecycle) {
      wifi_lifecycle_.unregister_handlers();
    }
    return err;
  }

  err = esp_event_handler_instance_register(IP_EVENT,
                                            IP_EVENT_STA_GOT_IP,
                                            &StandaloneWifiService::wifi_event_handler_,
                                            this,
                                            &ip_event_instance_);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "IP event handler registration failed: %s", esp_err_to_name(err));
    esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_instance_);
    wifi_event_instance_ = nullptr;
    if (config_.manage_csi_lifecycle) {
      wifi_lifecycle_.unregister_handlers();
    }
    return err;
  }

  err = configure_station_();
  if (err != ESP_OK) {
    if (ip_event_instance_ != nullptr) {
      esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, ip_event_instance_);
      ip_event_instance_ = nullptr;
    }
    if (wifi_event_instance_ != nullptr) {
      esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_instance_);
      wifi_event_instance_ = nullptr;
    }
    if (config_.manage_csi_lifecycle) {
      wifi_lifecycle_.unregister_handlers();
    }
    return err;
  }

  setup_complete_ = true;
  return ESP_OK;
}

bool StandaloneWifiService::get_info(StandaloneWifiInfo *info) const {
  if (info == nullptr) {
    return false;
  }

  *info = StandaloneWifiInfo{};

  uint8_t mac[6] = {0};
  if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK) {
    std::snprintf(info->mac_address,
                  sizeof(info->mac_address),
                  "%02X:%02X:%02X:%02X:%02X:%02X",
                  mac[0],
                  mac[1],
                  mac[2],
                  mac[3],
                  mac[4],
                  mac[5]);
  }

  if (cached_ip_info_.ip.addr != 0U) {
    // esp_wifi_sta_get_ap_info() logs a warning whenever the station is not
    // associated. Improv polls this accessor from the 10 ms runtime loop, so
    // querying before GOT_IP floods the provisioning console and can starve
    // its binary protocol. The cached address is cleared on disconnect and is
    // therefore a quiet, reliable gate for the AP query.
    wifi_ap_record_t ap_info{};
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
      info->connected = true;
      info->channel = ap_info.primary;
    }
    format_ip_address(cached_ip_info_.ip, info->ip_address, sizeof(info->ip_address));
  }

  return info->connected || info->ip_address[0] != '\0' || info->mac_address[0] != '\0';
}

esp_err_t StandaloneWifiService::configure_station_() {
  wifi_config_t sta_cfg{};
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.ssid), sizeof(sta_cfg.sta.ssid), "%s",
                config_.ssid != nullptr ? config_.ssid : "");
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.password), sizeof(sta_cfg.sta.password), "%s",
                config_.password != nullptr ? config_.password : "");
  sta_cfg.sta.scan_method = WIFI_ALL_CHANNEL_SCAN;
  sta_cfg.sta.sort_method = WIFI_CONNECT_AP_BY_SIGNAL;
  sta_cfg.sta.threshold.authmode = WIFI_AUTH_OPEN;
  sta_cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
  sta_cfg.sta.pmf_cfg.capable = true;
  sta_cfg.sta.pmf_cfg.required = false;

  if (config_.channel > 0U) {
    sta_cfg.sta.channel = config_.channel;
  }

  if (has_text(config_.bssid)) {
    if (!parse_bssid(config_.bssid, sta_cfg.sta.bssid)) {
      ESPECTRE_LOGE(TAG, "Invalid BSSID format: %s", config_.bssid);
      return ESP_ERR_INVALID_ARG;
    }
    sta_cfg.sta.bssid_set = true;
    sta_cfg.sta.scan_method = WIFI_FAST_SCAN;
    if (sta_cfg.sta.channel != 0U) {
      ESPECTRE_LOGI(TAG,
               "Wi-Fi fast scan enabled: BSSID=%s channel=%u",
               config_.bssid,
               static_cast<unsigned>(sta_cfg.sta.channel));
    } else {
      ESPECTRE_LOGI(TAG, "Wi-Fi fast scan enabled: BSSID=%s channel=auto", config_.bssid);
    }
  } else if (sta_cfg.sta.channel != 0U) {
    ESPECTRE_LOGI(TAG, "Wi-Fi channel hint enabled: channel=%u", static_cast<unsigned>(sta_cfg.sta.channel));
  } else {
    ESPECTRE_LOGI(TAG, "Wi-Fi full scan enabled: selecting strongest matching AP");
  }

  const esp_err_t err = esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(err));
  }
  return err;
}

esp_err_t StandaloneWifiService::start() {
  clear_cached_ip_info_();
  wifi_connect_requested_ = false;
  defer_connect_once_after_start_ = true;
  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;
  wifi_retry_count_ = 0;
  const esp_err_t err = esp_wifi_start();
  wifi_started_ = err == ESP_OK;
  return err;
}

void StandaloneWifiService::loop() {
  PendingWifiEvent event;
  while (pending_events_.take(event)) {
    switch (event.type) {
      case PendingWifiEventType::STARTED:
        wifi_started_ = true;
        handle_wifi_started_();
        break;
      case PendingWifiEventType::STOPPED:
        wifi_started_ = false;
        handle_wifi_stopped_();
        break;
      case PendingWifiEventType::DISCONNECTED:
        handle_wifi_disconnected_(event.disconnect_reason);
        if (!config_.manage_csi_lifecycle && disconnected_cb_) {
          disconnected_cb_();
        }
        break;
      case PendingWifiEventType::GOT_IP:
        cached_ip_info_ = event.ip_info;
        deferred_connect_fallback_pending_ = false;
        deferred_connect_fallback_deadline_us_ = 0U;
        wifi_retry_count_ = 0;
        if (!config_.manage_csi_lifecycle && connected_cb_) {
          connected_cb_();
        }
        break;
      case PendingWifiEventType::SCAN_DONE:
        handle_scan_done_(event.scan_status);
        break;
    }
  }
  maybe_run_deferred_connect_fallback_();
  if (config_.manage_csi_lifecycle) {
    (void)wifi_lifecycle_.process_pending_events();
  }
}

esp_err_t StandaloneWifiService::request_scan(standalone_wifi_scan_callback_t callback) {
  if (!setup_complete_ || !wifi_started_) {
    return ESP_ERR_INVALID_STATE;
  }
  if (scan_pending_) {
    return ESP_ERR_INVALID_STATE;
  }

  if (!has_text(config_.ssid)) {
    return ESP_ERR_INVALID_STATE;
  }
  const auto *ssid_begin = reinterpret_cast<const uint8_t *>(config_.ssid);
  std::vector<uint8_t> scan_ssid(ssid_begin, ssid_begin + std::strlen(config_.ssid));
  scan_ssid.push_back(0U);
  wifi_scan_config_t scan{};
  scan.ssid = scan_ssid.data();
  // Keep channel zero so the driver searches every channel and band for BSSIDs
  // that advertise the configured SSID.

  scan_callback_ = std::move(callback);
  scan_pending_ = true;
  const esp_err_t err = esp_wifi_scan_start(&scan, false);
  if (err != ESP_OK) {
    scan_pending_ = false;
    scan_callback_ = {};
  }
  return err;
}

esp_err_t StandaloneWifiService::update_station_config(const StandaloneWifiConfig &config) {
  if (!setup_complete_) {
    ESPECTRE_LOGE(TAG, "Cannot update Wi-Fi station config before setup");
    return ESP_ERR_INVALID_STATE;
  }

  if (!wifi_band_policy_is_supported(config.band_policy)) {
    return ESP_ERR_NOT_SUPPORTED;
  }
  if (!wifi_channel_is_supported(config.channel) ||
      !wifi_channel_matches_band_policy(config.channel, config.band_policy)) {
    return ESP_ERR_INVALID_ARG;
  }
  // The lifecycle handler captures this policy during setup. A live band
  // change would require re-registering the handler before reconnecting.
  if (config.band_policy != config_.band_policy) {
    ESPECTRE_LOGE(TAG, "Cannot change the Wi-Fi band policy without restarting the Wi-Fi service");
    return ESP_ERR_INVALID_STATE;
  }

  config_ = config;
  clear_cached_ip_info_();
  wifi_retry_count_ = 0;
  wifi_connect_requested_ = false;
  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;

  if (wifi_started_) {
    const esp_err_t disconnect_err = esp_wifi_disconnect();
    if (disconnect_err != ESP_OK && disconnect_err != ESP_ERR_WIFI_NOT_CONNECT) {
      ESPECTRE_LOGW(TAG, "esp_wifi_disconnect before reconfigure failed: %s", esp_err_to_name(disconnect_err));
    }
  }

  esp_err_t err = configure_station_();
  if (err != ESP_OK) {
    return err;
  }

  if (!wifi_started_) {
    return ESP_OK;
  }

  if (!has_text(config_.ssid)) {
    ESPECTRE_LOGW(TAG, "Wi-Fi SSID is empty; station config updated without reconnecting");
    return ESP_OK;
  }

  wifi_connect_requested_ = true;
  err = esp_wifi_connect();
  if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {
    ESPECTRE_LOGE(TAG, "esp_wifi_connect after reconfigure failed: %s", esp_err_to_name(err));
    wifi_connect_requested_ = false;
    return err;
  }
  ESPECTRE_LOGI(TAG, "Wi-Fi station config updated; reconnecting");
  return ESP_OK;
}

void StandaloneWifiService::shutdown() {
  if (wifi_started_) {
    const esp_err_t err = esp_wifi_stop();
    if (err != ESP_OK) {
      ESPECTRE_LOGW(TAG, "esp_wifi_stop failed during shutdown: %s", esp_err_to_name(err));
    }
    wifi_started_ = false;
  }
  if (wifi_event_instance_ != nullptr) {
    esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_instance_);
    wifi_event_instance_ = nullptr;
  }
  if (ip_event_instance_ != nullptr) {
    esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, ip_event_instance_);
    ip_event_instance_ = nullptr;
  }
  if (config_.manage_csi_lifecycle) {
    wifi_lifecycle_.unregister_handlers();
  }
  setup_complete_ = false;
  scan_pending_ = false;
  scan_callback_ = {};
  pending_events_.clear();
  wifi_connect_requested_ = false;
  defer_connect_once_after_start_ = false;
  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;
  clear_cached_ip_info_();
}

void StandaloneWifiService::handle_wifi_started_() {
  if (!has_text(config_.ssid)) {
    ESPECTRE_LOGW(TAG, "Wi-Fi SSID is empty; configure credentials in sdkconfig.wifi or at build time");
    return;
  }

  // When this service does not own the CSI lifecycle handlers, EspIdfRuntime may
  // register its STA_START policy handler after ours. Apply the radio policy
  // here before associating so set_protocol cannot abort an in-flight connect.
  if (!config_.manage_csi_lifecycle) {
    const esp_err_t policy_err = WiFiLifecycleManager::apply_started_csi_policy(config_.band_policy);
    if (policy_err != ESP_OK) {
      ESPECTRE_LOGW(TAG, "Failed to apply CSI Wi-Fi policy before connect: %s", esp_err_to_name(policy_err));
    }
  }

  if (!wifi_connect_requested_) {
    wifi_connect_requested_ = true;
    if (defer_connect_once_after_start_) {
      defer_connect_once_after_start_ = false;
      deferred_connect_fallback_pending_ = true;
      deferred_connect_fallback_deadline_us_ = monotonic_now_us() + DEFERRED_CONNECT_FALLBACK_DELAY_US;
      ESPECTRE_LOGI(TAG, "STA start observed; deferring first explicit connect request");
      return;
    }
    deferred_connect_fallback_pending_ = false;
    deferred_connect_fallback_deadline_us_ = 0U;
    (void)esp_wifi_connect();
  }
}

void StandaloneWifiService::handle_wifi_stopped_() {
  // Protocol or bandwidth changes can stop and restart STA
  // after an earlier connect request. Clear the latch so the next STA_START
  // associates again instead of leaving the radio idle.
  wifi_connect_requested_ = false;
  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;
  clear_cached_ip_info_();
}

void StandaloneWifiService::handle_wifi_disconnected_(uint8_t reason) {
  ESPECTRE_LOGW(TAG,
           "Wi-Fi disconnected: reason=%u (%s)",
           static_cast<unsigned>(reason),
           wifi_disconnect_reason_to_str(reason));
  clear_cached_ip_info_();
  wifi_connect_requested_ = false;
  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;
  if (has_text(config_.ssid) && wifi_retry_count_ < config_.max_retry) {
    wifi_retry_count_++;
    wifi_connect_requested_ = true;
    (void)esp_wifi_connect();
  }
}

void StandaloneWifiService::handle_lifecycle_connected_() {
  wifi_retry_count_ = 0;
  if (connected_cb_) {
    connected_cb_();
  }
}

void StandaloneWifiService::handle_lifecycle_disconnected_() {
  if (disconnected_cb_) {
    disconnected_cb_();
  }
}

void StandaloneWifiService::handle_scan_done_(uint8_t status) {
  if (!scan_pending_) {
    return;
  }

  std::vector<StandaloneWifiAccessPoint> access_points;
  esp_err_t result = status == 0U ? ESP_OK : ESP_FAIL;
  uint16_t count = 0U;
  if (result == ESP_OK) {
    result = esp_wifi_scan_get_ap_num(&count);
  }
  count = std::min(count, MAX_SCAN_ACCESS_POINTS);
  if (result == ESP_OK && count > 0U) {
    std::vector<wifi_ap_record_t> records(count);
    result = esp_wifi_scan_get_ap_records(&count, records.data());
    if (result == ESP_OK) {
      access_points.reserve(count);
      for (uint16_t index = 0U; index < count; ++index) {
        const wifi_ap_record_t &record = records[index];
        access_points.push_back(StandaloneWifiAccessPoint{
            reinterpret_cast<const char *>(record.ssid),
            format_bssid(record.bssid),
            record.rssi,
            record.primary,
        });
      }
      std::sort(access_points.begin(),
                access_points.end(),
                [](const StandaloneWifiAccessPoint &left, const StandaloneWifiAccessPoint &right) {
                  if (left.rssi_dbm != right.rssi_dbm) return left.rssi_dbm > right.rssi_dbm;
                  return left.bssid < right.bssid;
                });
    }
  }

  scan_pending_ = false;
  standalone_wifi_scan_callback_t callback = std::move(scan_callback_);
  scan_callback_ = {};
  if (callback) {
    callback(result, access_points);
  }
}

void StandaloneWifiService::maybe_run_deferred_connect_fallback_() {
  if (!deferred_connect_fallback_pending_ || !wifi_started_ || !has_text(config_.ssid)) {
    return;
  }
  if (cached_ip_info_.ip.addr != 0U) {
    deferred_connect_fallback_pending_ = false;
    deferred_connect_fallback_deadline_us_ = 0U;
    return;
  }

  const uint64_t now_us = monotonic_now_us();
  if (now_us < deferred_connect_fallback_deadline_us_) {
    return;
  }

  deferred_connect_fallback_pending_ = false;
  deferred_connect_fallback_deadline_us_ = 0U;
  ESPECTRE_LOGI(TAG, "Deferred STA-start connect fallback expired; issuing one explicit connect");
  const esp_err_t err = esp_wifi_connect();
  if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {
    ESPECTRE_LOGE(TAG, "Deferred esp_wifi_connect fallback failed: %s", esp_err_to_name(err));
    wifi_connect_requested_ = false;
  }
}

void StandaloneWifiService::clear_cached_ip_info_() { cached_ip_info_ = {}; }

void StandaloneWifiService::wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id,
                                                void *event_data) {
  auto *service = static_cast<StandaloneWifiService *>(arg);
  if (service == nullptr || event_base == nullptr) {
    return;
  }

  PendingWifiEvent pending;
  if (std::strcmp(event_base, WIFI_EVENT) == 0) {
    if (event_id == WIFI_EVENT_STA_START) {
      pending.type = PendingWifiEventType::STARTED;
    } else if (event_id == WIFI_EVENT_STA_STOP) {
      pending.type = PendingWifiEventType::STOPPED;
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
      pending.type = PendingWifiEventType::DISCONNECTED;
      const auto *event = static_cast<const wifi_event_sta_disconnected_t *>(event_data);
      pending.disconnect_reason = event != nullptr ? event->reason : 0U;
    } else if (event_id == WIFI_EVENT_SCAN_DONE) {
      pending.type = PendingWifiEventType::SCAN_DONE;
      const auto *event = static_cast<const wifi_event_sta_scan_done_t *>(event_data);
      pending.scan_status = event != nullptr ? event->status : 0U;
    } else {
      return;
    }
  } else if (std::strcmp(event_base, IP_EVENT) == 0 && event_id == IP_EVENT_STA_GOT_IP) {
    pending.type = PendingWifiEventType::GOT_IP;
    const auto *event = static_cast<const ip_event_got_ip_t *>(event_data);
    if (event != nullptr) {
      pending.ip_info = event->ip_info;
    }
  } else {
    return;
  }

  if (!service->pending_events_.post(pending)) {
    service->dropped_events_.fetch_add(1U, std::memory_order_relaxed);
  }
}

}  // namespace espectre
