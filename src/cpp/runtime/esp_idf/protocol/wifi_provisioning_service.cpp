/*
 * ESPectre - ESP-IDF Wi-Fi Provisioning Service
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "wifi_provisioning_service.h"

#include <cstdlib>
#include <cstring>
#include <utility>

#include "espectre_log.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.wifi_prov";

}  // namespace

WifiProvisioningService::WifiProvisioningService(StandaloneWifiManager *wifi_manager) : wifi_manager_(wifi_manager) {}

void WifiProvisioningService::set_change_callback(ChangeCallback callback) { change_callback_ = std::move(callback); }

esp_err_t WifiProvisioningService::load_or_set_defaults(const WifiProvisioningDefaults &defaults) {
  defaults_ = defaults;
  StoredWifiConfig stored_config;
  const esp_err_t load_err = load_stored_wifi_config(&stored_config);
  last_load_result_ = load_err;
  if (load_err == ESP_OK && stored_config.has_saved_config) {
    wifi_config_ = stored_config;
  } else {
    if (load_err != ESP_OK) {
      ESP_LOGW(TAG, "Failed to load stored Wi-Fi config: %s; using build defaults", esp_err_to_name(load_err));
    }
    wifi_config_.ssid = defaults.ssid != nullptr ? defaults.ssid : "";
    wifi_config_.password = defaults.password != nullptr ? defaults.password : "";
    wifi_config_.bssid = defaults.bssid != nullptr ? defaults.bssid : "";
    wifi_config_.channel = defaults.channel;
    wifi_config_.has_saved_config = false;
  }
  refresh_cached_strings_();
  notify_changed_();
  return ESP_OK;
}

esp_err_t WifiProvisioningService::setup_station(const WifiProvisioningDefaults &defaults,
                                                 standalone_wifi_callback_t connected_cb,
                                                 standalone_wifi_callback_t disconnected_cb) {
  if (wifi_manager_ == nullptr) {
    return ESP_ERR_INVALID_STATE;
  }
  const esp_err_t load_err = load_or_set_defaults(defaults);
  if (load_err != ESP_OK) {
    return load_err;
  }

  StandaloneWifiConfig wifi_config;
  wifi_config.ssid = wifi_ssid_.c_str();
  wifi_config.password = wifi_password_.c_str();
  wifi_config.bssid = wifi_bssid_.c_str();
  wifi_config.channel = wifi_config_.channel;
  wifi_config.max_retry = defaults_.max_retry;
  wifi_config.manage_csi_lifecycle = defaults_.manage_csi_lifecycle;
  auto on_connected = [this, connected_cb]() {
    if (connected_cb) {
      connected_cb();
    }
    this->notify_changed_();
  };
  auto on_disconnected = [this, disconnected_cb]() {
    if (disconnected_cb) {
      disconnected_cb();
    }
    this->notify_changed_();
  };
  return wifi_manager_->setup(wifi_config, on_connected, on_disconnected);
}

bool WifiProvisioningService::handle_command(const std::string &command, std::string *message) {
  auto set_message = [message](const char *value) {
    if (message != nullptr) {
      *message = value;
    }
  };

  constexpr const char *kSsidPrefix = "SET_WIFI_SSID:";
  constexpr const char *kPasswordPrefix = "SET_WIFI_PASSWORD:";
  constexpr const char *kBssidPrefix = "SET_WIFI_BSSID:";
  constexpr const char *kChannelPrefix = "SET_WIFI_CHANNEL:";

  if (command.rfind(kSsidPrefix, 0) == 0) {
    const std::string ssid = command.substr(std::strlen(kSsidPrefix));
    if (ssid.empty() || ssid.size() > 32) {
      set_message("SSID must be 1..32 bytes");
      return false;
    }
    wifi_config_.ssid = ssid;
    wifi_config_.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(wifi_config_);
    refresh_cached_strings_();
    notify_changed_();
    set_message(err == ESP_OK ? "SSID saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kPasswordPrefix, 0) == 0) {
    const std::string password = command.substr(std::strlen(kPasswordPrefix));
    if (password.size() > 63) {
      set_message("password must be 0..63 bytes");
      return false;
    }
    wifi_config_.password = password;
    wifi_config_.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(wifi_config_);
    refresh_cached_strings_();
    notify_changed_();
    set_message(err == ESP_OK ? "password saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kBssidPrefix, 0) == 0) {
    const std::string bssid = command.substr(std::strlen(kBssidPrefix));
    if (!bssid.empty() && bssid.size() != 17) {
      set_message("BSSID must be empty or 17 chars");
      return false;
    }
    wifi_config_.bssid = bssid;
    wifi_config_.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(wifi_config_);
    refresh_cached_strings_();
    notify_changed_();
    set_message(err == ESP_OK ? "BSSID saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kChannelPrefix, 0) == 0) {
    uint8_t channel = 0;
    if (!parse_wifi_channel_(command.substr(std::strlen(kChannelPrefix)), &channel)) {
      set_message("channel must be 0..14");
      return false;
    }
    wifi_config_.channel = channel;
    wifi_config_.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(wifi_config_);
    notify_changed_();
    set_message(err == ESP_OK ? "channel saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command == "CLEAR_WIFI") {
    const esp_err_t err = clear_stored_wifi_config();
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    wifi_config_ = StoredWifiConfig{};
    refresh_cached_strings_();
    notify_changed_();
    return apply_live(message);
  }

  if (command == "APPLY_WIFI") {
    if (wifi_config_.ssid.empty()) {
      set_message("SSID is empty");
      return false;
    }
    wifi_config_.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(wifi_config_);
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    refresh_cached_strings_();
    notify_changed_();
    return apply_live(message);
  }

  set_message("unknown provisioning command");
  return false;
}

bool WifiProvisioningService::apply_live(std::string *message) {
  if (wifi_manager_ == nullptr) {
    if (message != nullptr) {
      *message = "Wi-Fi manager is not configured";
    }
    return false;
  }

  refresh_cached_strings_();
  StandaloneWifiConfig wifi_config;
  wifi_config.ssid = wifi_ssid_.c_str();
  wifi_config.password = wifi_password_.c_str();
  wifi_config.bssid = wifi_bssid_.c_str();
  wifi_config.channel = wifi_config_.channel;
  wifi_config.max_retry = defaults_.max_retry;
  wifi_config.manage_csi_lifecycle = defaults_.manage_csi_lifecycle;

  const esp_err_t err = wifi_manager_->update_station_config(wifi_config);
  notify_changed_();
  if (message != nullptr) {
    *message = err == ESP_OK ? "Wi-Fi config applied" : esp_err_to_name(err);
  }
  return err == ESP_OK;
}

bool WifiProvisioningService::parse_wifi_channel_(const std::string &value, uint8_t *channel) const {
  if (channel == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  const long parsed = std::strtol(value.c_str(), &end_ptr, 10);
  if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' || parsed < 0 || parsed > 14) {
    return false;
  }
  *channel = static_cast<uint8_t>(parsed);
  return true;
}

void WifiProvisioningService::refresh_cached_strings_() {
  wifi_ssid_ = wifi_config_.ssid;
  wifi_password_ = wifi_config_.password;
  wifi_bssid_ = wifi_config_.bssid;
}

void WifiProvisioningService::notify_changed_() {
  if (change_callback_) {
    change_callback_();
  }
}

}  // namespace espectre
}  // namespace esphome
