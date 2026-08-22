/*
 * ESPectre - Native Firmware Entrypoint
 *
 * Native firmware application entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include <cstdio>
#include <string>

#include <esp_err.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <driver/gpio.h>

#if CONFIG_BT_ENABLED
#include "ble_bindings_nimble.h"
#else
#include "ble_bindings_noop.h"
#endif
#include "native_frontend.h"
#include "ble_recovery_button_service.h"
#include "device_config_store.h"
#include "nvs_helpers.h"
#include "device_identity.h"
#include "espectre_banner.h"
#include "firmware_version.h"
#include "frontend_bootstrap_helpers.h"
#include "ota_service_https.h"
#include "mqtt_transport_esp_idf.h"
#include "runtime_motion_hits_store.h"
#include "runtime_sensing_kconfig.h"
#include "standalone_wifi_service.h"
#include "debug_telemetry_log_helpers.h"
#include "wifi_provisioning_service.h"

static const char *TAG = "espectre.native.app";

namespace {

#ifdef ESPECTRE_OTA_DEVELOP_BUILD
constexpr espectre::OtaReleaseChannel kOtaReleaseChannel = espectre::OtaReleaseChannel::DEVELOP;
#elif defined(ESPECTRE_OTA_PREVIEW_BUILD)
constexpr espectre::OtaReleaseChannel kOtaReleaseChannel = espectre::OtaReleaseChannel::PREVIEW;
#else
constexpr espectre::OtaReleaseChannel kOtaReleaseChannel = espectre::OtaReleaseChannel::RELEASE;
#endif

constexpr int kWifiConnectMaxRetry = 8;

espectre::NativeFrontend *g_frontend = nullptr;
espectre::BleRecoveryButtonService *g_ble_recovery_button = nullptr;
espectre::StandaloneWifiService g_wifi_manager;
espectre::WifiProvisioningService g_wifi_provisioning(&g_wifi_manager);

void sync_frontend_wifi_info() {
  if (g_frontend == nullptr) {
    return;
  }
  espectre::NativeFrontend::WifiProvisioningInfo info;
  const espectre::StoredWifiConfig &wifi_config = g_wifi_provisioning.config();
  info.ssid = wifi_config.ssid;
  info.bssid = wifi_config.bssid;
  info.channel = wifi_config.channel;
  info.has_saved_config = wifi_config.has_saved_config;
  info.band_policy = wifi_config.band_policy;
  g_frontend->set_wifi_provisioning_info(info);

  espectre::EspectreDeviceInfo device_info;
  device_info.frontend = "native";
  device_info.firmware_version = espectre::espectre_firmware_version();
  device_info.chip = CONFIG_IDF_TARGET;

  espectre::StandaloneWifiInfo wifi_info;
  if (g_wifi_manager.get_info(&wifi_info)) {
    device_info.network.ip_address = wifi_info.ip_address;
    device_info.network.mac_address = wifi_info.mac_address;
    device_info.network.channel = wifi_info.channel;
  }
  g_frontend->set_device_info(device_info);
}

espectre::RuntimeConfig make_runtime_config() {
  espectre::RuntimeConfig config = espectre::make_runtime_sensing_config_from_kconfig();
  config.wifi_band_policy = g_wifi_provisioning.config().band_policy;
  uint8_t saved_motion_on_hits = 0U;
  uint8_t saved_motion_off_hits = 0U;
  bool has_saved_motion_hits = false;
  const esp_err_t err =
      espectre::load_runtime_motion_hits(&saved_motion_on_hits, &saved_motion_off_hits, &has_saved_motion_hits);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to load persisted motion hits: %s", esp_err_to_name(err));
  } else if (has_saved_motion_hits) {
    config.motion_on_hits = saved_motion_on_hits;
    config.motion_off_hits = saved_motion_off_hits;
  }
  return config;
}

espectre::EspectreDeviceConfig make_device_config() {
  return espectre::load_frontend_device_config(espectre::FrontendDeviceConfigDefaults{
                                                            CONFIG_ESPECTRE_DEVICE_LABEL,
                                                            CONFIG_ESPECTRE_MQTT_HOST,
                                                            CONFIG_ESPECTRE_MQTT_PORT,
                                                            CONFIG_ESPECTRE_MQTT_USERNAME,
                                                            CONFIG_ESPECTRE_MQTT_PASSWORD,
                                                            CONFIG_ESPECTRE_TOPIC_PREFIX,
                                                            espectre::derive_runtime_device_id(),
                                                        },
                                                        TAG,
                                                        "Using ESPectre Protocol config provisioned over BLE",
                                                        "Failed to load BLE-provisioned device config");
}

void espectre_loop_task(void *arg) {
  (void) arg;
  while (true) {
    g_wifi_manager.loop();
    if (g_frontend != nullptr) {
      g_frontend->loop();
    }
#if CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_ENABLED
    if (g_ble_recovery_button != nullptr) {
      const bool pressed = gpio_get_level(static_cast<gpio_num_t>(CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_GPIO)) == 0;
      g_ble_recovery_button->update(pressed, static_cast<uint32_t>(esp_timer_get_time() / 1000));
    }
#endif
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

bool init_wifi_station() {
  const esp_err_t setup_err = espectre::setup_frontend_wifi_station(
      &g_wifi_provisioning,
      &g_wifi_manager,
      espectre::FrontendWifiStationOptions{CONFIG_ESPECTRE_WIFI_SSID,
                                                    CONFIG_ESPECTRE_WIFI_PASSWORD,
                                                    CONFIG_ESPECTRE_WIFI_BSSID,
                                                    CONFIG_ESPECTRE_WIFI_CHANNEL,
                                                    kWifiConnectMaxRetry,
                                                    false,
                                                    false,
                                                    sync_frontend_wifi_info,
                                                    sync_frontend_wifi_info,
                                                    sync_frontend_wifi_info,
                                                    espectre::make_runtime_sensing_config_from_kconfig().wifi_band_policy},
      TAG,
      "Using Wi-Fi credentials provisioned over BLE");
  if (setup_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to initialize Wi-Fi provisioning service: %s", esp_err_to_name(setup_err));
    return false;
  }
  sync_frontend_wifi_info();
  return true;
}

bool handle_wifi_provisioning_command(const std::string &command, std::string *message) {
  return g_wifi_provisioning.handle_command(command, message);
}

bool handle_device_config_change(const espectre::EspectreDeviceConfig &config, bool clear, std::string *message) {
  if (clear) {
    const esp_err_t err = espectre::clear_stored_device_config();
    if (message != nullptr) {
      *message = err == ESP_OK ? "device config cleared" : esp_err_to_name(err);
    }
    return err == ESP_OK;
  }

  const esp_err_t err = espectre::save_stored_device_config(config);
  if (message != nullptr) {
    *message = err == ESP_OK ? "device config saved" : esp_err_to_name(err);
  }
  return err == ESP_OK;
}

void request_ble_recovery() {
  if (g_frontend != nullptr) {
    g_frontend->request_ble_recovery();
  }
}

}  // namespace

extern "C" void app_main() {
  espectre::configure_debug_telemetry_log_levels();
  ESP_ERROR_CHECK(espectre::nvs_init_with_erase_fallback());

  espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });

  if (!init_wifi_station()) {
    return;
  }

#if CONFIG_BT_ENABLED
  static espectre::NimbleBleBindings bindings;
#else
  static espectre::NoopBleBindings bindings;
#endif
  static espectre::EspIdfMqttTransport mqtt_transport;
  static espectre::HttpsOtaService ota_service("native", CONFIG_IDF_TARGET, kOtaReleaseChannel);
  static espectre::NativeFrontend frontend(&bindings, &mqtt_transport, &ota_service);
  frontend.set_runtime_config(make_runtime_config());
  frontend.set_device_config(make_device_config());
  g_frontend = &frontend;
  sync_frontend_wifi_info();
  frontend.set_provisioning_command_callback(handle_wifi_provisioning_command);
  frontend.set_device_config_change_callback(handle_device_config_change);
  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre native frontend");
    return;
  }

#if CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_ENABLED
  gpio_config_t recovery_button_config{};
  recovery_button_config.pin_bit_mask = 1ULL << CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_GPIO;
  recovery_button_config.mode = GPIO_MODE_INPUT;
  recovery_button_config.pull_up_en = GPIO_PULLUP_ENABLE;
  recovery_button_config.pull_down_en = GPIO_PULLDOWN_DISABLE;
  recovery_button_config.intr_type = GPIO_INTR_DISABLE;
  ESP_ERROR_CHECK(gpio_config(&recovery_button_config));
  static espectre::BleRecoveryButtonService recovery_button(
      CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_HOLD_MS, request_ble_recovery);
  g_ble_recovery_button = &recovery_button;
  ESP_LOGI(TAG,
           "Hold BOOT on GPIO%d for %d ms to start BLE recovery",
           CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_GPIO,
           CONFIG_ESPECTRE_BLE_RECOVERY_BUTTON_HOLD_MS);
#endif

  ESP_ERROR_CHECK(g_wifi_manager.start());
  xTaskCreate(espectre_loop_task, "espectre_native_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre native firmware started");
}
