/*
 * ESPectre - Frontend Bootstrap Helpers
 *
 * Loads persisted frontend config and initializes shared Wi-Fi station
 * setup.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "device_identity.h"
#include "esp_err.h"
#include "runtime/espectre_protocol.h"
#include "standalone_wifi_service.h"
#include "wifi_provisioning_service.h"

namespace espectre {

/**
 * Build-time device settings used when nothing is saved in NVS.
 *
 * The fields mirror `EspectreDeviceConfig`; `nullptr` strings mean empty.
 */
struct FrontendDeviceConfigDefaults {
  const char *device_label{ESPECTRE_DEFAULT_DEVICE_LABEL};
  const char *mqtt_scheme{""};
  const char *mqtt_host{""};
  uint16_t mqtt_port{0U};
  const char *mqtt_username{""};
  const char *mqtt_password{""};
  const char *topic_prefix{ESPECTRE_TOPIC_PREFIX};
  /** Device identity to use; zero derives it with derive_runtime_device_id(). */
  uint64_t runtime_device_id{0U};
};

/**
 * Station settings for setup_frontend_wifi_station().
 *
 * The credential and channel fields are the build-time defaults of
 * WifiProvisioningDefaults; saved provisioning takes precedence.
 */
struct FrontendWifiStationOptions {
  const char *ssid{nullptr};
  const char *password{nullptr};
  const char *bssid{nullptr};
  /** Channel hint; `WIFI_CHANNEL_AUTO` for none. */
  int configured_channel{0};
  int max_retry{8};
  /** Forwarded to `StandaloneWifiConfig::manage_csi_lifecycle`. */
  bool manage_csi_lifecycle{false};
  /** Start the station after setting it up. */
  bool start_manager{false};
  /** Forwarded to WifiProvisioningService::set_change_callback(). */
  WifiProvisioningService::ChangeCallback change_callback{};
  standalone_wifi_callback_t connected_callback{};
  standalone_wifi_callback_t disconnected_callback{};
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
};

/**
 * Load the saved device settings, or build them from `defaults`.
 *
 * The device id always comes from `defaults.runtime_device_id` or
 * derive_runtime_device_id(), never from NVS. `stored_config_message` is
 * logged when saved settings are used, and `load_error_prefix` prefixes a
 * load failure; both use `log_tag`.
 */
EspectreDeviceConfig load_frontend_device_config(const FrontendDeviceConfigDefaults &defaults,
                                                 const char *log_tag,
                                                 const char *stored_config_message,
                                                 const char *load_error_prefix);

/**
 * Set up the station through `provisioning`, which must already be bound to
 * `wifi_manager`, and optionally start it.
 *
 * @return `ESP_ERR_INVALID_ARG` for a null `provisioning` or an unusable
 *         channel, `ESP_ERR_NOT_SUPPORTED` for an unsupported band policy,
 *         `ESP_ERR_INVALID_STATE` when starting without `wifi_manager`, or the
 *         setup or start error.
 */
esp_err_t setup_frontend_wifi_station(WifiProvisioningService *provisioning,
                                      StandaloneWifiService *wifi_manager,
                                      const FrontendWifiStationOptions &options,
                                      const char *log_tag,
                                      const char *stored_config_message);

}  // namespace espectre
