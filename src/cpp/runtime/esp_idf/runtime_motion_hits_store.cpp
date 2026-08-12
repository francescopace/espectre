/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "runtime_motion_hits_store.h"

#include "nvs.h"
#include "runtime_sensing_schema.h"

namespace espectre {

namespace {

constexpr const char *kNamespace = "espectre";
constexpr const char *kMotionOnHitsKey = "motion_on";
constexpr const char *kMotionOffHitsKey = "motion_off";

bool motion_hits_valid(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  return motion_on_hits >= RUNTIME_MOTION_HITS_MIN && motion_on_hits <= RUNTIME_MOTION_HITS_MAX &&
         motion_off_hits >= RUNTIME_MOTION_HITS_MIN && motion_off_hits <= RUNTIME_MOTION_HITS_MAX;
}

}  // namespace

esp_err_t load_runtime_motion_hits(uint8_t *motion_on_hits, uint8_t *motion_off_hits, bool *has_saved_value) {
  if (motion_on_hits == nullptr || motion_off_hits == nullptr || has_saved_value == nullptr) {
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

  uint8_t loaded_motion_on_hits = 0U;
  uint8_t loaded_motion_off_hits = 0U;
  const esp_err_t on_err = nvs_get_u8(handle, kMotionOnHitsKey, &loaded_motion_on_hits);
  const esp_err_t off_err = nvs_get_u8(handle, kMotionOffHitsKey, &loaded_motion_off_hits);
  nvs_close(handle);

  if (on_err == ESP_ERR_NVS_NOT_FOUND && off_err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (on_err == ESP_ERR_NVS_NOT_FOUND || off_err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_ERR_INVALID_STATE;
  }
  if (on_err != ESP_OK) {
    return on_err;
  }
  if (off_err != ESP_OK) {
    return off_err;
  }
  if (!motion_hits_valid(loaded_motion_on_hits, loaded_motion_off_hits)) {
    return ESP_ERR_INVALID_ARG;
  }

  *motion_on_hits = loaded_motion_on_hits;
  *motion_off_hits = loaded_motion_off_hits;
  *has_saved_value = true;
  return ESP_OK;
}

esp_err_t save_runtime_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (!motion_hits_valid(motion_on_hits, motion_off_hits)) {
    return ESP_ERR_INVALID_ARG;
  }

  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }
  err = nvs_set_u8(handle, kMotionOnHitsKey, motion_on_hits);
  if (err == ESP_OK) {
    err = nvs_set_u8(handle, kMotionOffHitsKey, motion_off_hits);
  }
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

esp_err_t clear_runtime_motion_hits() {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  const esp_err_t erase_on_err = nvs_erase_key(handle, kMotionOnHitsKey);
  if (erase_on_err != ESP_OK && erase_on_err != ESP_ERR_NVS_NOT_FOUND) {
    nvs_close(handle);
    return erase_on_err;
  }
  const esp_err_t erase_off_err = nvs_erase_key(handle, kMotionOffHitsKey);
  if (erase_off_err != ESP_OK && erase_off_err != ESP_ERR_NVS_NOT_FOUND) {
    nvs_close(handle);
    return erase_off_err;
  }

  err = nvs_commit(handle);
  nvs_close(handle);
  return err;
}

}  // namespace espectre
