/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "runtime_traffic_mode_store.h"

#include <cstring>

#include "nvs.h"
#include "runtime_config_utils.h"

namespace espectre {

namespace {

constexpr const char *kNamespace = "espectre";
constexpr const char *kCsiTrafficModeKey = "csi_traffic";
constexpr const char *kTrafficGeneratorModeKey = "traffic_gen";

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

}  // namespace

esp_err_t load_runtime_csi_traffic_mode(CsiTrafficMode *mode, bool *has_saved_value) {
  if (mode == nullptr || has_saved_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  char value[16]{};
  esp_err_t err = load_string_key(kCsiTrafficModeKey, value, sizeof(value), has_saved_value);
  if (err != ESP_OK || !*has_saved_value) {
    return err;
  }
  *mode = parse_csi_traffic_mode(value);
  if (!runtime_csi_traffic_mode_valid(*mode) || std::strcmp(value, csi_traffic_mode_name(*mode)) != 0) {
    return ESP_ERR_INVALID_STATE;
  }
  return ESP_OK;
}

esp_err_t load_runtime_traffic_generator_mode(RuntimeTrafficMode *mode, bool *has_saved_value) {
  if (mode == nullptr || has_saved_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  char value[16]{};
  esp_err_t err = load_string_key(kTrafficGeneratorModeKey, value, sizeof(value), has_saved_value);
  if (err != ESP_OK || !*has_saved_value) {
    return err;
  }
  *mode = parse_traffic_mode(value);
  if (!runtime_traffic_mode_valid(*mode) || std::strcmp(value, traffic_mode_name(*mode)) != 0) {
    return ESP_ERR_INVALID_STATE;
  }
  return ESP_OK;
}

esp_err_t save_runtime_csi_traffic_mode(CsiTrafficMode mode) {
  if (!runtime_csi_traffic_mode_valid(mode)) {
    return ESP_ERR_INVALID_ARG;
  }
  return save_string_key(kCsiTrafficModeKey, csi_traffic_mode_name(mode));
}

esp_err_t save_runtime_traffic_generator_mode(RuntimeTrafficMode mode) {
  if (!runtime_traffic_mode_valid(mode)) {
    return ESP_ERR_INVALID_ARG;
  }
  return save_string_key(kTrafficGeneratorModeKey, traffic_mode_name(mode));
}

}  // namespace espectre
