/*
 * ESPectre - Wi-Fi Provisioning Service
 *
 * Stores Wi-Fi credentials and applies live station provisioning changes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "wifi_provisioning_service.h"

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "espectre_log.h"
#include "protocol_json.h"
#include "runtime_config_utils.h"
#include "wifi_band_helpers.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.wifi_prov";

bool assign_wifi_config_field(const std::string &field,
                              const std::string &value,
                              StoredWifiConfig *config,
                              std::string *error) {
  if (config == nullptr) {
    if (error != nullptr) {
      *error = "wifi config output is required";
    }
    return false;
  }
  if (field == "ssid") {
    if (value.empty() || value.size() > 32) {
      if (error != nullptr) {
        *error = "SSID must be 1..32 bytes";
      }
      return false;
    }
    config->ssid = value;
    return true;
  }
  if (field == "password") {
    if (value.size() > 63) {
      if (error != nullptr) {
        *error = "password must be 0..63 bytes";
      }
      return false;
    }
    config->password = value;
    return true;
  }
  if (field == "bssid") {
    if (!value.empty() && value.size() != 17) {
      if (error != nullptr) {
        *error = "BSSID must be empty or 17 chars";
      }
      return false;
    }
    config->bssid = value;
    return true;
  }
  if (field == "channel") {
    char *end_ptr = nullptr;
    const long parsed = std::strtol(value.c_str(), &end_ptr, 10);
    // The range check bounds the narrowing cast; the channel plan itself is
    // what actually decides which of those values is a channel.
    if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' ||
        parsed < 0 || parsed > 255 ||
        !wifi_channel_is_supported(static_cast<int>(parsed))) {
      if (error != nullptr) {
        *error = std::string("channel must be ") + wifi_channel_supported_description();
      }
      return false;
    }
    config->channel = static_cast<uint8_t>(parsed);
    return true;
  }
  if (field == "band_policy") {
    if (value != "2g" && value != "5g" && value != "auto") {
      if (error != nullptr) {
        *error = "band_policy must be 2g, 5g, or auto";
      }
      return false;
    }
    const WifiBandPolicy policy = parse_wifi_band_policy(value.c_str());
    if (!wifi_band_policy_is_supported(policy)) {
      if (error != nullptr) {
        *error = "5 GHz Wi-Fi is not supported by this device";
      }
      return false;
    }
    config->band_policy = policy;
    config->has_saved_band_policy = true;
    return true;
  }
  if (error != nullptr) {
    *error = "unsupported wifi config field";
  }
  return false;
}

}  // namespace

WifiProvisioningService::WifiProvisioningService(StandaloneWifiService *wifi_manager) : wifi_manager_(wifi_manager) {}

void WifiProvisioningService::set_change_callback(ChangeCallback callback) { change_callback_ = std::move(callback); }

esp_err_t WifiProvisioningService::load_or_set_defaults(const WifiProvisioningDefaults &defaults) {
  defaults_ = defaults;
  StoredWifiConfig stored_config;
  const esp_err_t load_err = load_stored_wifi_config(&stored_config);
  last_load_result_ = load_err;
  if (load_err == ESP_OK && stored_config.has_saved_config) {
    wifi_config_ = stored_config;
    if (!wifi_config_.has_saved_band_policy) {
      wifi_config_.band_policy = defaults.band_policy;
    }
  } else {
    if (load_err != ESP_OK) {
      ESP_LOGW(TAG, "Failed to load stored Wi-Fi config: %s; using build defaults", esp_err_to_name(load_err));
    }
    wifi_config_.ssid = defaults.ssid != nullptr ? defaults.ssid : "";
    wifi_config_.password = defaults.password != nullptr ? defaults.password : "";
    wifi_config_.bssid = defaults.bssid != nullptr ? defaults.bssid : "";
    wifi_config_.channel = defaults.channel;
    wifi_config_.band_policy = defaults.band_policy;
    wifi_config_.has_saved_band_policy = false;
    wifi_config_.has_saved_config = false;
  }
  if (!wifi_band_policy_is_supported(wifi_config_.band_policy)) {
    ESP_LOGW(TAG, "Stored Wi-Fi band policy is unsupported; restoring build default");
    wifi_config_.band_policy = defaults_.band_policy;
    wifi_config_.has_saved_band_policy = false;
  }
  if (!wifi_channel_is_supported(wifi_config_.channel) ||
      !wifi_channel_matches_band_policy(wifi_config_.channel, wifi_config_.band_policy)) {
    ESP_LOGW(TAG, "Stored Wi-Fi channel %u does not match band policy; restoring automatic scan",
             static_cast<unsigned>(wifi_config_.channel));
    wifi_config_.channel = WIFI_CHANNEL_AUTO;
    if (wifi_config_.has_saved_config) {
      const esp_err_t save_err = save_stored_wifi_config(wifi_config_);
      if (save_err != ESP_OK) {
        ESP_LOGW(TAG, "Failed to persist the normalized Wi-Fi channel: %s", esp_err_to_name(save_err));
      }
    }
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
  wifi_config.band_policy = wifi_config_.band_policy;
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

  constexpr const char *kBatchPrefix = "SET_WIFI_CONFIG:";

  if (command.rfind(kBatchPrefix, 0) == 0) {
    std::vector<std::pair<std::string, std::string>> pairs;
    std::string error;
    if (!parse_urlencoded_key_value_pairs(command.substr(std::strlen(kBatchPrefix)), &pairs, &error)) {
      set_message(error.c_str());
      return false;
    }
    StoredWifiConfig updated = wifi_config_;
    const WifiBandPolicy previous_band_policy = wifi_config_.band_policy;
    bool has_ssid = false;
    for (const auto &pair : pairs) {
      if (!assign_wifi_config_field(pair.first, pair.second, &updated, &error)) {
        set_message(error.c_str());
        return false;
      }
      if (pair.first == "ssid") {
        has_ssid = true;
      }
    }
    if (!has_ssid || updated.ssid.empty()) {
      set_message("SSID must be 1..32 bytes");
      return false;
    }
    if (!wifi_channel_matches_band_policy(updated.channel, updated.band_policy)) {
      set_message((std::string("channel must be ") +
                   wifi_channel_supported_description(updated.band_policy)).c_str());
      return false;
    }
    updated.has_saved_config = true;
    const esp_err_t err = save_stored_wifi_config(updated);
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    wifi_config_ = std::move(updated);
    refresh_cached_strings_();
    notify_changed_();
    if (wifi_config_.band_policy != previous_band_policy) {
      set_message("Wi-Fi config saved; restart required to apply the band policy");
      return true;
    }
    return apply_live(message);
  }

  if (command == "CLEAR_WIFI") {
    const esp_err_t err = clear_stored_wifi_config();
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    const WifiBandPolicy previous_band_policy = wifi_config_.band_policy;
    wifi_config_ = StoredWifiConfig{};
    wifi_config_.band_policy = defaults_.band_policy;
    refresh_cached_strings_();
    notify_changed_();
    if (wifi_config_.band_policy != previous_band_policy) {
      const WifiBandPolicy restored_band_policy = wifi_config_.band_policy;
      wifi_config_.band_policy = previous_band_policy;
      const bool applied = apply_live(message);
      wifi_config_.band_policy = restored_band_policy;
      notify_changed_();
      if (!applied) {
        return false;
      }
      set_message("Wi-Fi config cleared; restart required to restore the default band policy");
      return true;
    }
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
  wifi_config.band_policy = wifi_config_.band_policy;

  const esp_err_t err = wifi_manager_->update_station_config(wifi_config);
  notify_changed_();
  if (message != nullptr) {
    *message = err == ESP_OK ? "Wi-Fi config applied" : esp_err_to_name(err);
  }
  return err == ESP_OK;
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
