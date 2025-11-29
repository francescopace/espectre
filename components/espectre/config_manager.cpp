/*
 * ESPectre - Configuration Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "config_manager.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "ConfigManager";

bool ConfigurationManager::load(ESpectreConfig& config) {
  if (!pref_.load(&config)) {
    ESP_LOGI(TAG, "No saved configuration found. Using defaults");
    return false;
  }
  
  ESP_LOGI(TAG, "Configuration loaded from preferences");
  return true;
}

void ConfigurationManager::save(const ESpectreConfig& config) {
  pref_.save(&config);
  ESP_LOGI(TAG, "Configuration saved to preferences");
}

}  // namespace espectre
}  // namespace esphome
