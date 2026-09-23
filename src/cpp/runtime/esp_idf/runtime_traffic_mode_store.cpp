/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "runtime_traffic_mode_store.h"

#include <cstring>

#include "nvs.h"
#include "core/espectre_log.h"
#include "runtime/runtime_config_utils.h"

namespace espectre {

namespace {

constexpr const char *kNamespace = "espectre";
// Legacy key from when the traffic source was a separate setting.
constexpr const char *kLegacyCsiTrafficKey = "csi_traffic";
constexpr const char *kTrafficGeneratorModeKey = "traffic_gen";
constexpr const char *kTag = "espectre.traffic";

esp_err_t load_string_key(const char *key, char *value, size_t value_size, bool *has_saved_value) {
  if (key == nullptr || value == nullptr || value_size == 0U || has_saved_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  *has_saved_value = false;
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READONLY, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }
  size_t length = value_size;
  err = nvs_get_str(handle, key, value, &length);
  nvs_close(handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }
  *has_saved_value = true;
  return ESP_OK;
}

esp_err_t save_string_key(const char *key, const char *value) {
  if (key == nullptr || value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }
  err = nvs_set_str(handle, key, value);
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

esp_err_t erase_key(const char *key) {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }
  err = nvs_erase_key(handle, key);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    err = ESP_OK;
  } else if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

esp_err_t migrate_legacy_csi_traffic_key() {
  char value[16]{};
  bool has_legacy_value = false;
  esp_err_t err = load_string_key(kLegacyCsiTrafficKey, value, sizeof(value), &has_legacy_value);
  if (err != ESP_OK || !has_legacy_value) {
    return err;
  }
  if (std::strcmp(value, RUNTIME_TRAFFIC_GENERATOR_MODE_EXTERNAL_NAME) == 0) {
    ESPECTRE_LOGI(kTag, "Migrating saved external CSI traffic to traffic generator mode");
    err = save_string_key(kTrafficGeneratorModeKey, RUNTIME_TRAFFIC_GENERATOR_MODE_EXTERNAL_NAME);
    if (err != ESP_OK) {
      return err;
    }
  }
  return erase_key(kLegacyCsiTrafficKey);
}

}  // namespace

esp_err_t load_runtime_traffic_generator_mode(TrafficGeneratorMode *mode, bool *has_saved_value) {
  if (mode == nullptr || has_saved_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  esp_err_t err = migrate_legacy_csi_traffic_key();
  if (err != ESP_OK) {
    return err;
  }
  char value[16]{};
  err = load_string_key(kTrafficGeneratorModeKey, value, sizeof(value), has_saved_value);
  if (err != ESP_OK || !*has_saved_value) {
    return err;
  }
  *mode = parse_traffic_generator_mode(value);
  if (!runtime_traffic_generator_mode_valid(*mode) || std::strcmp(value, traffic_generator_mode_name(*mode)) != 0) {
    return ESP_ERR_INVALID_STATE;
  }
  return ESP_OK;
}

esp_err_t save_runtime_traffic_generator_mode(TrafficGeneratorMode mode) {
  if (!runtime_traffic_generator_mode_valid(mode)) {
    return ESP_ERR_INVALID_ARG;
  }
  return save_string_key(kTrafficGeneratorModeKey, traffic_generator_mode_name(mode));
}

}  // namespace espectre
