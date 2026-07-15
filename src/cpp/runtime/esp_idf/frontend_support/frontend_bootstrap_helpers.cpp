/*
 * ESPectre - Frontend Bootstrap Helpers
 *
 * Loads persisted frontend config and initializes shared Wi-Fi station
 * setup.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "frontend_bootstrap_helpers.h"

#include "device_config_store.h"
#include "espectre_log.h"

namespace espectre {

EspectreDeviceConfig load_frontend_device_config(const FrontendDeviceConfigDefaults &defaults,
                                                 const char *log_tag,
                                                 const char *stored_config_message,
                                                 const char *load_error_prefix) {
  EspectreDeviceConfig config;
  config.device_id = defaults.runtime_device_id != 0U ? defaults.runtime_device_id : derive_runtime_device_id();
  config.device_label = defaults.device_label != nullptr ? defaults.device_label : ESPECTRE_DEFAULT_DEVICE_LABEL;
  config.mqtt_host = defaults.mqtt_host != nullptr ? defaults.mqtt_host : "";
  config.mqtt_port = defaults.mqtt_port;
  config.topic_prefix = defaults.topic_prefix != nullptr ? defaults.topic_prefix : ESPECTRE_TOPIC_PREFIX;

  EspectreDeviceConfig stored_config;
  bool has_stored_config = false;
  const esp_err_t load_err = load_stored_device_config(&stored_config, &has_stored_config);
  if (load_err == ESP_OK && has_stored_config) {
    config = stored_config;
    if (stored_config_message != nullptr && stored_config_message[0] != '\0') {
      ESP_LOGI(log_tag, "%s", stored_config_message);
    }
  } else if (load_err != ESP_OK) {
    ESP_LOGW(log_tag, "%s: %s", load_error_prefix != nullptr ? load_error_prefix : "Failed to load device config",
             esp_err_to_name(load_err));
  }

  config.device_id = defaults.runtime_device_id != 0U ? defaults.runtime_device_id : derive_runtime_device_id();
  return config;
}

esp_err_t setup_frontend_wifi_station(WifiProvisioningService *provisioning,
                                      StandaloneWifiService *wifi_manager,
                                      const FrontendWifiStationOptions &options,
                                      const char *log_tag,
                                      const char *stored_config_message) {
  if (provisioning == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  if (options.start_manager && wifi_manager == nullptr) {
    return ESP_ERR_INVALID_STATE;
  }
  if (options.configured_channel < 0 || options.configured_channel > 14) {
    ESP_LOGW(log_tag, "Invalid Wi-Fi channel: %d", options.configured_channel);
    return ESP_ERR_INVALID_ARG;
  }

  WifiProvisioningDefaults defaults;
  defaults.ssid = options.ssid;
  defaults.password = options.password;
  defaults.bssid = options.bssid;
  defaults.channel = static_cast<uint8_t>(options.configured_channel);
  defaults.max_retry = options.max_retry;
  defaults.manage_csi_lifecycle = options.manage_csi_lifecycle;

  provisioning->set_change_callback(options.change_callback);
  const esp_err_t setup_err =
      provisioning->setup_station(defaults, options.connected_callback, options.disconnected_callback);
  if (setup_err != ESP_OK) {
    return setup_err;
  }
  if (provisioning->config().has_saved_config && stored_config_message != nullptr && stored_config_message[0] != '\0') {
    ESP_LOGI(log_tag, "%s", stored_config_message);
  }
  if (options.start_manager && wifi_manager != nullptr) {
    return wifi_manager->start();
  }
  return ESP_OK;
}

}  // namespace espectre
