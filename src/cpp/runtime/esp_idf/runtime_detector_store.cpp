/*
 * ESPectre - Runtime Detector Store
 *
 * Persists the selected runtime detector algorithm across reboots.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_detector_store.h"

#include <cstring>

#include "nvs.h"
#include "runtime_config_utils.h"

namespace espectre {

namespace {

constexpr const char *kNamespace = "espectre";
constexpr const char *kDetectorKey = "detector";

}  // namespace

esp_err_t load_runtime_detection_algorithm(DetectionAlgorithm *algorithm, bool *has_saved_value) {
  if (algorithm == nullptr || has_saved_value == nullptr) {
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
  size_t length = 0U;
  err = nvs_get_str(handle, kDetectorKey, nullptr, &length);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    nvs_close(handle);
    return ESP_OK;
  }
  if (err != ESP_OK || length == 0U || length > 16U) {
    nvs_close(handle);
    return err == ESP_OK ? ESP_ERR_INVALID_SIZE : err;
  }
  char value[16]{};
  err = nvs_get_str(handle, kDetectorKey, value, &length);
  nvs_close(handle);
  if (err != ESP_OK) {
    return err;
  }
  if (std::strcmp(value, RUNTIME_DETECTION_ALGORITHM_CLASSIC_NAME) != 0 &&
      std::strcmp(value, RUNTIME_DETECTION_ALGORITHM_ML_NAME) != 0) {
    return ESP_ERR_INVALID_STATE;
  }
  *algorithm = parse_detection_algorithm(value);
  *has_saved_value = true;
  return ESP_OK;
}

esp_err_t save_runtime_detection_algorithm(DetectionAlgorithm algorithm) {
  if (!runtime_detection_algorithm_valid(algorithm)) {
    return ESP_ERR_INVALID_ARG;
  }
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }
  err = nvs_set_str(handle, kDetectorKey, detection_algorithm_name(algorithm));
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

}  // namespace espectre
