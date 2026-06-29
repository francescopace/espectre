#include "standalone_wifi_manager.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

#include "espectre_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "StandaloneWiFi";

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

}  // namespace

esp_err_t StandaloneWifiManager::setup(const StandaloneWifiConfig &config,
                                       standalone_wifi_callback_t connected_cb,
                                       standalone_wifi_callback_t disconnected_cb) {
  config_ = config;
  connected_cb_ = connected_cb;
  disconnected_cb_ = disconnected_cb;

  esp_err_t err = esp_netif_init();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(err));
    return err;
  }

  if (esp_netif_create_default_wifi_sta() == nullptr) {
    ESP_LOGE(TAG, "esp_netif_create_default_wifi_sta failed");
    return ESP_FAIL;
  }

  wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&wifi_cfg);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_wifi_set_storage(WIFI_STORAGE_RAM);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_storage failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_wifi_set_mode(WIFI_MODE_STA);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_mode failed: %s", esp_err_to_name(err));
    return err;
  }

  err = esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_ps failed: %s", esp_err_to_name(err));
    return err;
  }

  if (config_.manage_csi_lifecycle) {
    err = wifi_lifecycle_.register_handlers([this]() { handle_lifecycle_connected_(); },
                                            [this]() { handle_lifecycle_disconnected_(); });
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "WiFi lifecycle handler registration failed: %s", esp_err_to_name(err));
      return err;
    }
  }

  err = esp_event_handler_instance_register(WIFI_EVENT,
                                            ESP_EVENT_ANY_ID,
                                            &StandaloneWifiManager::wifi_event_handler_,
                                            this,
                                            &wifi_event_instance_);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "WiFi event handler registration failed: %s", esp_err_to_name(err));
    if (config_.manage_csi_lifecycle) {
      wifi_lifecycle_.unregister_handlers();
    }
    return err;
  }

  err = configure_station_();
  if (err != ESP_OK) {
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

esp_err_t StandaloneWifiManager::configure_station_() {
  wifi_config_t sta_cfg{};
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.ssid), sizeof(sta_cfg.sta.ssid), "%s",
                config_.ssid != nullptr ? config_.ssid : "");
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.password), sizeof(sta_cfg.sta.password), "%s",
                config_.password != nullptr ? config_.password : "");
  sta_cfg.sta.scan_method = WIFI_ALL_CHANNEL_SCAN;
  sta_cfg.sta.sort_method = WIFI_CONNECT_AP_BY_SIGNAL;
  sta_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_WPA3_PSK;
  sta_cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
  sta_cfg.sta.pmf_cfg.capable = true;
  sta_cfg.sta.pmf_cfg.required = false;

  if (config_.channel > 0U) {
    sta_cfg.sta.channel = config_.channel;
  }

  if (has_text(config_.bssid)) {
    if (!parse_bssid(config_.bssid, sta_cfg.sta.bssid)) {
      ESP_LOGE(TAG, "Invalid BSSID format: %s", config_.bssid);
      return ESP_ERR_INVALID_ARG;
    }
    sta_cfg.sta.bssid_set = true;
    sta_cfg.sta.scan_method = WIFI_FAST_SCAN;
    if (sta_cfg.sta.channel != 0U) {
      ESP_LOGI(TAG,
               "Wi-Fi fast scan enabled: BSSID=%s channel=%u",
               config_.bssid,
               static_cast<unsigned>(sta_cfg.sta.channel));
    } else {
      ESP_LOGI(TAG, "Wi-Fi fast scan enabled: BSSID=%s channel=auto", config_.bssid);
    }
  } else if (sta_cfg.sta.channel != 0U) {
    ESP_LOGI(TAG, "Wi-Fi channel hint enabled: channel=%u", static_cast<unsigned>(sta_cfg.sta.channel));
  } else {
    ESP_LOGI(TAG, "Wi-Fi full scan enabled: selecting strongest matching AP");
  }

  const esp_err_t err = esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(err));
  }
  return err;
}

esp_err_t StandaloneWifiManager::start() {
  wifi_start_policy_applied_ = false;
  wifi_connect_requested_ = false;
  csi_wifi_lifecycle_ready_ = false;
  wifi_retry_count_ = 0;
  const esp_err_t err = esp_wifi_start();
  wifi_started_ = err == ESP_OK;
  return err;
}

void StandaloneWifiManager::shutdown() {
  if (wifi_started_) {
    const esp_err_t err = esp_wifi_stop();
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "esp_wifi_stop failed during shutdown: %s", esp_err_to_name(err));
    }
    wifi_started_ = false;
  }
  if (wifi_event_instance_ != nullptr) {
    esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_instance_);
    wifi_event_instance_ = nullptr;
  }
  if (config_.manage_csi_lifecycle) {
    wifi_lifecycle_.unregister_handlers();
  }
  setup_complete_ = false;
  csi_wifi_lifecycle_ready_ = false;
  wifi_connect_requested_ = false;
  wifi_start_policy_applied_ = false;
}

esp_err_t StandaloneWifiManager::apply_started_csi_policy() {
  esp_err_t ps_err = esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
  if (ps_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to set Wi-Fi power save for CSI: %s", esp_err_to_name(ps_err));
  }

  const esp_err_t policy_err = WiFiLifecycleManager::apply_csi_wifi_policy();
  if (policy_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to apply started Wi-Fi CSI policy: %s", esp_err_to_name(policy_err));
    return policy_err;
  }

  return ps_err == ESP_OK ? ESP_OK : ps_err;
}

void StandaloneWifiManager::handle_wifi_started_() {
  if (!has_text(config_.ssid)) {
    ESP_LOGW(TAG, "Wi-Fi SSID is empty; configure sdkconfig credentials before connecting");
    return;
  }

  if (!wifi_start_policy_applied_) {
    wifi_start_policy_applied_ = true;
    (void)apply_started_csi_policy();
  }

  if (!wifi_connect_requested_) {
    wifi_connect_requested_ = true;
    (void)esp_wifi_connect();
  }
}

void StandaloneWifiManager::handle_wifi_disconnected_(void *event_data) {
  const auto *event = static_cast<const wifi_event_sta_disconnected_t *>(event_data);
  ESP_LOGW(TAG, "Wi-Fi disconnected: reason=%u", event != nullptr ? static_cast<unsigned>(event->reason) : 0U);
  wifi_connect_requested_ = false;
  if (has_text(config_.ssid) && wifi_retry_count_ < config_.max_retry) {
    wifi_retry_count_++;
    wifi_connect_requested_ = true;
    (void)esp_wifi_connect();
  }
}

bool StandaloneWifiManager::ensure_csi_lifecycle_ready_() {
  if (!config_.manage_csi_lifecycle || csi_wifi_lifecycle_ready_) {
    return true;
  }

  const esp_err_t err = wifi_lifecycle_.init();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Wi-Fi CSI lifecycle init failed after connect: %s", esp_err_to_name(err));
    return false;
  }

  csi_wifi_lifecycle_ready_ = true;
  ESP_LOGI(TAG, "Wi-Fi CSI lifecycle initialized after connect");
  return true;
}

void StandaloneWifiManager::handle_lifecycle_connected_() {
  if (!ensure_csi_lifecycle_ready_()) {
    return;
  }
  wifi_retry_count_ = 0;
  if (connected_cb_) {
    connected_cb_();
  }
}

void StandaloneWifiManager::handle_lifecycle_disconnected_() {
  csi_wifi_lifecycle_ready_ = false;
  if (disconnected_cb_) {
    disconnected_cb_();
  }
}

void StandaloneWifiManager::wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id,
                                                void *event_data) {
  auto *manager = static_cast<StandaloneWifiManager *>(arg);
  if (manager == nullptr || event_base == nullptr || std::strcmp(event_base, WIFI_EVENT) != 0) {
    return;
  }

  if (event_id == WIFI_EVENT_STA_START) {
    manager->handle_wifi_started_();
  } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    manager->handle_wifi_disconnected_(event_data);
  }
}

}  // namespace espectre
}  // namespace esphome
