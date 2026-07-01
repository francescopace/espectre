#include "nvs.h"

#include <cstring>
#include <map>
#include <string>

namespace {

struct NvsMockState {
  esp_err_t open_result{ESP_OK};
  std::map<std::string, std::string> strings;
  std::map<std::string, uint8_t> u8_values;
  std::map<std::string, uint16_t> u16_values;
};

NvsMockState g_nvs_mock;

}  // namespace

extern "C" {

void nvs_mock_reset(void) { g_nvs_mock = {}; }

void nvs_mock_set_open_result(esp_err_t result) { g_nvs_mock.open_result = result; }

void nvs_mock_put_str(const char *key, const char *value) {
  if (key != nullptr && value != nullptr) {
    g_nvs_mock.strings[key] = value;
  }
}

void nvs_mock_put_u8(const char *key, uint8_t value) {
  if (key != nullptr) {
    g_nvs_mock.u8_values[key] = value;
  }
}

void nvs_mock_put_u16(const char *key, uint16_t value) {
  if (key != nullptr) {
    g_nvs_mock.u16_values[key] = value;
  }
}

esp_err_t nvs_open(const char *name, nvs_open_mode_t open_mode, nvs_handle_t *out_handle) {
  (void) name;
  (void) open_mode;
  if (out_handle == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  if (g_nvs_mock.open_result != ESP_OK) {
    return g_nvs_mock.open_result;
  }
  *out_handle = 1;
  return ESP_OK;
}

esp_err_t nvs_get_str(nvs_handle_t handle, const char *key, char *out_value, size_t *length) {
  (void) handle;
  if (key == nullptr || length == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  const auto it = g_nvs_mock.strings.find(key);
  if (it == g_nvs_mock.strings.end()) {
    return ESP_ERR_NVS_NOT_FOUND;
  }

  const size_t required = it->second.size() + 1U;
  if (out_value == nullptr) {
    *length = required;
    return ESP_OK;
  }
  if (*length < required) {
    *length = required;
    return ESP_ERR_INVALID_SIZE;
  }

  std::memcpy(out_value, it->second.c_str(), required);
  *length = required;
  return ESP_OK;
}

esp_err_t nvs_set_str(nvs_handle_t handle, const char *key, const char *value) {
  (void) handle;
  if (key == nullptr || value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  g_nvs_mock.strings[key] = value;
  return ESP_OK;
}

esp_err_t nvs_get_u8(nvs_handle_t handle, const char *key, uint8_t *out_value) {
  (void) handle;
  if (key == nullptr || out_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  const auto it = g_nvs_mock.u8_values.find(key);
  if (it == g_nvs_mock.u8_values.end()) {
    return ESP_ERR_NVS_NOT_FOUND;
  }
  *out_value = it->second;
  return ESP_OK;
}

esp_err_t nvs_set_u8(nvs_handle_t handle, const char *key, uint8_t value) {
  (void) handle;
  if (key == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  g_nvs_mock.u8_values[key] = value;
  return ESP_OK;
}

esp_err_t nvs_get_u16(nvs_handle_t handle, const char *key, uint16_t *out_value) {
  (void) handle;
  if (key == nullptr || out_value == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  const auto it = g_nvs_mock.u16_values.find(key);
  if (it == g_nvs_mock.u16_values.end()) {
    return ESP_ERR_NVS_NOT_FOUND;
  }
  *out_value = it->second;
  return ESP_OK;
}

esp_err_t nvs_set_u16(nvs_handle_t handle, const char *key, uint16_t value) {
  (void) handle;
  if (key == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  g_nvs_mock.u16_values[key] = value;
  return ESP_OK;
}

esp_err_t nvs_erase_key(nvs_handle_t handle, const char *key) {
  (void) handle;
  if (key == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  const bool erased_string = g_nvs_mock.strings.erase(key) > 0U;
  const bool erased_u8 = g_nvs_mock.u8_values.erase(key) > 0U;
  const bool erased_u16 = g_nvs_mock.u16_values.erase(key) > 0U;
  return erased_string || erased_u8 || erased_u16 ? ESP_OK : ESP_ERR_NVS_NOT_FOUND;
}

esp_err_t nvs_commit(nvs_handle_t handle) {
  (void) handle;
  return ESP_OK;
}

void nvs_close(nvs_handle_t handle) { (void) handle; }

}  // extern "C"

namespace {
struct NvsMockResetInitializer {
  NvsMockResetInitializer() { nvs_mock_reset(); }
} g_nvs_mock_reset_initializer;
}  // namespace
