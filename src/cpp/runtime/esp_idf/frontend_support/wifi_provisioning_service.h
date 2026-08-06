/*
 * ESPectre - Wi-Fi Provisioning Service
 *
 * Stores Wi-Fi credentials and applies live station provisioning changes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "device_config_store.h"
#include "esp_err.h"
#include "standalone_wifi_service.h"

namespace espectre {

struct WifiProvisioningDefaults {
  const char *ssid{nullptr};
  const char *password{nullptr};
  const char *bssid{nullptr};
  uint8_t channel{0U};
  int max_retry{8};
  bool manage_csi_lifecycle{false};
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
};

class WifiProvisioningService {
 public:
  using ChangeCallback = std::function<void()>;

  explicit WifiProvisioningService(StandaloneWifiService *wifi_manager);

  void set_change_callback(ChangeCallback callback);
  esp_err_t load_or_set_defaults(const WifiProvisioningDefaults &defaults);
  esp_err_t setup_station(const WifiProvisioningDefaults &defaults,
                          standalone_wifi_callback_t connected_cb = {},
                          standalone_wifi_callback_t disconnected_cb = {});
  bool handle_command(const std::string &command, std::string *message);
  bool apply_live(std::string *message);

  const StoredWifiConfig &config() const { return wifi_config_; }
  bool password_set() const { return !wifi_config_.password.empty(); }
  esp_err_t last_load_result() const { return last_load_result_; }

 private:
  void refresh_cached_strings_();
  void notify_changed_();

  StandaloneWifiService *wifi_manager_;
  ChangeCallback change_callback_;
  StoredWifiConfig wifi_config_;
  WifiProvisioningDefaults defaults_;
  esp_err_t last_load_result_{ESP_OK};
  std::string wifi_ssid_;
  std::string wifi_password_;
  std::string wifi_bssid_;
};

}  // namespace espectre
