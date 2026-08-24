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
#include <esp_netif.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <driver/gpio.h>

#include "native_frontend.h"
#include "native_shared_mdns_alias.h"
#include "recovery_button_service.h"
#include "device_config_store.h"
#include "direct_websocket_protocol.h"
#include "direct_websocket_service_esp_idf.h"
#include "nvs_helpers.h"
#include "device_identity.h"
#include "espectre_banner.h"
#include "espectre_protocol.h"
#include "firmware_version.h"
#include "frontend_bootstrap_helpers.h"
#include "improv_serial_service.h"
#include "mdns_discovery_service.h"
#include "ota_service_https.h"
#include "peer_discovery_service_esp_idf.h"
#include "mqtt_transport_esp_idf.h"
#include "runtime_motion_hits_store.h"
#include "runtime_sensing_kconfig.h"
#include "standalone_wifi_service.h"
#include "runtime_log_helpers.h"
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
espectre::RecoveryButtonService *g_recovery_button = nullptr;
espectre::ImprovSerialService *g_improv_serial = nullptr;
espectre::MdnsDiscoveryService *g_mdns_discovery = nullptr;
espectre::NativeSharedMdnsAlias *g_shared_mdns_alias = nullptr;
bool g_restart_after_wifi_apply = false;
espectre::StandaloneWifiService g_wifi_manager;
espectre::WifiProvisioningService g_wifi_provisioning(&g_wifi_manager);

const char *native_capabilities() {
  return "config,monitor,ota,peer_discovery";
}

espectre::MdnsTxtRecords native_mdns_txt(const espectre::EspectreDeviceConfig &config) {
  return {
      {"device_id", espectre::format_espectre_device_id(config.device_id)},
      {"name", config.device_label},
      {"frontend", "native"},
      {"txtvers", "1"},
      {"protovers", "1"},
      {"path", espectre::ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT},
      {"firmware", espectre::espectre_firmware_version()},
      {"chip", CONFIG_IDF_TARGET},
      {"tls", "0"},
      {"capabilities", native_capabilities()},
  };
}

espectre::PeerDiscoveryCandidate native_peer_candidate(
    const espectre::EspectreDeviceConfig &config,
    const std::string &instance_name) {
  const std::string device_id = espectre::format_espectre_device_id(config.device_id);
  espectre::PeerDiscoveryCandidate candidate;
  candidate.instance = instance_name;
  candidate.hostname = "espectre-" + device_id;
  candidate.device_id = device_id;
  candidate.name = instance_name;
  candidate.frontend = "native";
  candidate.txt_version = "1";
  candidate.protocol_version = "1";
  candidate.path = espectre::ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT;
  candidate.firmware = espectre::espectre_firmware_version();
  candidate.chip = CONFIG_IDF_TARGET;
  candidate.tls = "0";
  candidate.capabilities = native_capabilities();
  candidate.port = 80U;
  return candidate;
}

std::string improv_device_url() {
  espectre::StandaloneWifiInfo wifi_info;
  if (!g_wifi_manager.get_info(&wifi_info) || !wifi_info.connected || wifi_info.ip_address[0] == '\0') {
    return {};
  }
  return std::string("https://espectre.dev/tools/configure/?target=") + wifi_info.ip_address;
}

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
  info.apply_state = espectre::wifi_provisioning_apply_state_name(g_wifi_provisioning.apply_state());
  info.apply_message = g_wifi_provisioning.apply_message();
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
  if (g_mdns_discovery != nullptr) {
    if (wifi_info.connected) {
      g_mdns_discovery->on_wifi_connected();
      if (g_shared_mdns_alias != nullptr) {
        esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
        esp_netif_ip_info_t ip_info{};
        if (netif != nullptr && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
          if (!g_shared_mdns_alias->published()) {
            (void) g_shared_mdns_alias->setup("espectre-devices");
          }
          (void) g_shared_mdns_alias->update(ip_info.ip.addr);
        }
      }
    } else {
      g_mdns_discovery->on_wifi_disconnected();
      if (g_shared_mdns_alias != nullptr) {
        g_shared_mdns_alias->shutdown();
      }
    }
  }
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
                                                        "Using stored ESPectre Protocol device config",
                                                        "Failed to load stored device config");
}

void espectre_loop_task(void *arg) {
  (void) arg;
  while (true) {
    g_wifi_manager.loop();
    g_wifi_provisioning.loop();
    if (g_restart_after_wifi_apply) {
      ESP_LOGI(TAG, "Restarting after verified Wi-Fi configuration change");
      esp_restart();
    }
    if (g_improv_serial != nullptr) {
      g_improv_serial->loop();
    }
    if (g_frontend != nullptr) {
      g_frontend->loop();
    }
#if CONFIG_ESPECTRE_RECOVERY_BUTTON_ENABLED
    if (g_recovery_button != nullptr) {
      const bool pressed = gpio_get_level(static_cast<gpio_num_t>(CONFIG_ESPECTRE_RECOVERY_BUTTON_GPIO)) == 0;
      g_recovery_button->update(pressed, static_cast<uint32_t>(esp_timer_get_time() / 1000));
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
      "Using stored Wi-Fi credentials");
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
  if (err == ESP_OK && g_mdns_discovery != nullptr) {
    (void) g_mdns_discovery->update_txt(native_mdns_txt(config));
  }
  if (message != nullptr) {
    *message = err == ESP_OK ? "device config saved" : esp_err_to_name(err);
  }
  return err == ESP_OK;
}

void request_wifi_recovery() {
  std::string message;
  if (g_wifi_provisioning.handle_command("CLEAR_WIFI", &message)) {
    ESP_LOGW(TAG, "Physical recovery cleared saved Wi-Fi configuration; use Improv Serial to provision again");
  } else {
    ESP_LOGE(TAG, "Physical Wi-Fi recovery failed: %s", message.c_str());
  }
}

}  // namespace

extern "C" void app_main() {
  espectre::configure_runtime_log_levels();
  ESP_ERROR_CHECK(espectre::nvs_init_with_erase_fallback());

  espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });

  if (!init_wifi_station()) {
    return;
  }

  static espectre::EspIdfMqttTransport mqtt_transport;
  static espectre::EspIdfDirectWebSocketService direct_service;
  static espectre::MdnsDiscoveryService mdns_discovery;
  static espectre::NativeSharedMdnsAlias shared_mdns_alias;
  static espectre::EspIdfPeerDiscoveryService peer_discovery;
  static espectre::HttpsOtaService ota_service("native", CONFIG_IDF_TARGET, kOtaReleaseChannel);
  static espectre::NativeFrontend frontend(&mqtt_transport, &ota_service, &direct_service);
  const espectre::EspectreDeviceConfig device_config = make_device_config();
  const std::string device_id = espectre::format_espectre_device_id(device_config.device_id);
  const std::string mdns_name = device_config.device_label.empty() ? "ESPectre " + device_id : device_config.device_label;
  if (!mdns_discovery.setup(espectre::MdnsDiscoveryServiceConfig{
          "espectre-" + device_id,
          mdns_name,
          "_espectre",
          "_tcp",
          80U,
          native_mdns_txt(device_config),
      })) {
    ESP_LOGE(TAG, "Failed to initialize Native mDNS discovery");
    return;
  }
  g_mdns_discovery = &mdns_discovery;
  if (!shared_mdns_alias.setup("espectre-devices")) {
    ESP_LOGE(TAG, "Failed to initialize shared mDNS bootstrap alias");
    return;
  }
  g_shared_mdns_alias = &shared_mdns_alias;
  peer_discovery.set_local_candidate(native_peer_candidate(device_config, mdns_name));
  frontend.set_peer_discovery_service(&peer_discovery);
  frontend.set_runtime_config(make_runtime_config());
  frontend.set_device_config(device_config);
  g_frontend = &frontend;
  sync_frontend_wifi_info();
  frontend.set_provisioning_command_callback(handle_wifi_provisioning_command);
  frontend.set_device_config_change_callback(handle_device_config_change);
  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre native frontend");
    return;
  }
  g_wifi_provisioning.set_reconfigure_callbacks(
      []() {
        // Send the shared-record goodbye while the station interface is still
        // usable. The later disconnect callback is too late for mDNS to write.
        if (g_shared_mdns_alias != nullptr) {
          g_shared_mdns_alias->shutdown();
        }
        if (g_frontend != nullptr) {
          g_frontend->prepare_for_wifi_reconfigure();
        }
      },
      []() {
        if (g_frontend != nullptr) {
          g_frontend->resume_after_wifi_reconfigure();
        }
      });
  g_wifi_provisioning.set_apply_completed_callback([]() { g_restart_after_wifi_apply = true; });

  static espectre::ImprovSerialService improv_serial(&g_wifi_provisioning, &g_wifi_manager);
  const std::string improv_device_name = device_config.device_label.empty()
                                             ? espectre::espectre_device_name(device_config.device_id, CONFIG_IDF_TARGET)
                                             : device_config.device_label;
  if (!improv_serial.setup(espectre::ImprovSerialServiceConfig{
          "ESPectre Native",
          espectre::espectre_firmware_version(),
          CONFIG_IDF_TARGET,
          improv_device_name,
          improv_device_url,
      })) {
    ESP_LOGE(TAG, "Failed to initialize Improv Serial");
    return;
  }
  g_improv_serial = &improv_serial;

#if CONFIG_ESPECTRE_RECOVERY_BUTTON_ENABLED
  gpio_config_t recovery_button_config{};
  recovery_button_config.pin_bit_mask = 1ULL << CONFIG_ESPECTRE_RECOVERY_BUTTON_GPIO;
  recovery_button_config.mode = GPIO_MODE_INPUT;
  recovery_button_config.pull_up_en = GPIO_PULLUP_ENABLE;
  recovery_button_config.pull_down_en = GPIO_PULLDOWN_DISABLE;
  recovery_button_config.intr_type = GPIO_INTR_DISABLE;
  ESP_ERROR_CHECK(gpio_config(&recovery_button_config));
  static espectre::RecoveryButtonService recovery_button(
      CONFIG_ESPECTRE_RECOVERY_BUTTON_HOLD_MS, request_wifi_recovery);
  g_recovery_button = &recovery_button;
  ESP_LOGI(TAG,
           "Hold BOOT on GPIO%d for %d ms to clear Wi-Fi and return to Improv Serial setup",
           CONFIG_ESPECTRE_RECOVERY_BUTTON_GPIO,
           CONFIG_ESPECTRE_RECOVERY_BUTTON_HOLD_MS);
#endif

  ESP_ERROR_CHECK(g_wifi_manager.start());
  xTaskCreate(espectre_loop_task, "espectre_native_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre native firmware started");
}
