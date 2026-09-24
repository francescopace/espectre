/*
 * ESPectre - Device Config Store
 *
 * Persists Wi-Fi and device configuration in ESP-IDF non-volatile storage.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "esp_err.h"
#include "runtime/espectre_protocol.h"
#include "runtime/runtime_config.h"

namespace espectre {

/** Wi-Fi station settings as persisted in NVS. */
struct StoredWifiConfig {
  std::string ssid;
  std::string password;
  /** Optional pinned access point, as `AA:BB:CC:DD:EE:FF`; empty when unpinned. */
  std::string bssid;
  /** Optional channel hint; zero means automatic. */
  uint8_t channel{0U};
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
  /** Whether `band_policy` came from NVS rather than a build default. */
  bool has_saved_band_policy{false};
  /** Whether these settings came from NVS rather than build defaults. */
  bool has_saved_config{false};
};

/**
 * @name Persistent device settings
 * Read and write the `espectre` NVS namespace. NVS must be initialized first.
 * A missing namespace or key is not an error: loads then return `ESP_OK` with
 * default values. Other NVS errors are returned unchanged.
 * @{
 */

/** Load the saved station configuration; `ESP_ERR_INVALID_ARG` for a null `config`. */
esp_err_t load_stored_wifi_config(StoredWifiConfig *config);
/** Save and commit the station configuration. */
esp_err_t save_stored_wifi_config(const StoredWifiConfig &config);
/** Erase the saved station configuration. */
esp_err_t clear_stored_wifi_config();
/**
 * Load a staged candidate that has not been verified yet.
 *
 * `has_pending` reports whether one exists. Used by WifiProvisioningService
 * to resume verification after a reboot.
 */
esp_err_t load_pending_wifi_config(StoredWifiConfig *config, bool *has_pending);
/** Stage a candidate configuration before applying it. */
esp_err_t save_pending_wifi_config(const StoredWifiConfig &config);
/** Discard the staged candidate. */
esp_err_t clear_pending_wifi_config();

/**
 * Load the saved device label and broker settings into `config`.
 *
 * `has_saved_config` may be `nullptr`; otherwise it reports whether saved
 * settings exist. When they do, `config` is replaced and every field that is
 * not stored takes its default, including `device_id`. Otherwise `config` is
 * left unchanged.
 */
esp_err_t load_stored_device_config(EspectreDeviceConfig *config, bool *has_saved_config);
/** Save and commit the device label and broker settings. */
esp_err_t save_stored_device_config(const EspectreDeviceConfig &config);
/** Erase the saved device label and broker settings. */
esp_err_t clear_stored_device_config();

/** @} */

}  // namespace espectre
