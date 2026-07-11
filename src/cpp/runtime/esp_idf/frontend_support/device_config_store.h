#pragma once

#include <cstdint>
#include <string>

#include "esp_err.h"
#include "espectre_protocol.h"

namespace esphome {
namespace espectre {

struct StoredWifiConfig {
  std::string ssid;
  std::string password;
  std::string bssid;
  uint8_t channel{0U};
  bool has_saved_config{false};
};

esp_err_t load_stored_wifi_config(StoredWifiConfig *config);
esp_err_t save_stored_wifi_config(const StoredWifiConfig &config);
esp_err_t clear_stored_wifi_config();

esp_err_t load_stored_device_config(EspectreDeviceConfig *config, bool *has_saved_config);
esp_err_t save_stored_device_config(const EspectreDeviceConfig &config);
esp_err_t clear_stored_device_config();

}  // namespace espectre
}  // namespace esphome
