/*
 * ESPectre - Matter Firmware Entrypoint
 *
 * Matter firmware application entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include <esp_err.h>
#include <esp_event.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <freertos/FreeRTOS.h>
#include <sdkconfig.h>

#include <cstdio>

#include <app/server/CommissioningWindowManager.h>
#include <app/server/Server.h>
#include <esp_matter.h>
#include <esp_matter_attribute.h>
#include <esp_matter_core.h>
#include <esp_matter_endpoint.h>
#include <esp_matter_providers.h>
#include <platform/CHIPDeviceLayer.h>
#include <setup_payload/OnboardingCodesUtil.h>

#include "espectre_banner.h"
#include "runtime_log_helpers.h"
#include "device_identity.h"
#include "direct_http_protocol.h"
#include "direct_http_service_esp_idf.h"
#include "espectre_protocol.h"
#include "firmware_version.h"
#include "matter_bindings_esp_matter.h"
#include "matter_commissioning_data.h"
#include "matter_frontend.h"
#include "mdns_discovery_service.h"
#include "mdns_bootstrap_responder.h"
#include "nvs_helpers.h"
#include "runtime_config_utils.h"
#include "runtime_sensing_kconfig.h"

static const char *TAG = "espectre.matter.app";

using namespace esp_matter;
using namespace esp_matter::attribute;
using namespace esp_matter::endpoint;
using namespace chip::app::Clusters;

namespace {

espectre::MatterEspBindings g_bindings;
espectre::MatterCommissioningDataProvider g_commissioning_data;
espectre::MatterFrontend *g_frontend = nullptr;
espectre::MdnsDiscoveryService *g_mdns_discovery = nullptr;
espectre::MdnsBootstrapResponder *g_mdns_bootstrap_responder = nullptr;
espectre::MdnsDiscoveryServiceConfig g_mdns_config;
uint16_t g_motion_endpoint_id = 0;

espectre::RuntimeConfig build_runtime_config() {
  espectre::RuntimeConfig config = espectre::make_runtime_sensing_config_from_kconfig();
  config.device_id = espectre::derive_runtime_device_id();
  config.runtime_detector_selection_enabled = true;
  return config;
}

espectre::MdnsTxtRecords matter_mdns_txt(uint64_t device_id,
                                         const std::string &device_label = CONFIG_ESPECTRE_MATTER_NODE_LABEL) {
  return {
      {"device_id", espectre::format_espectre_device_id(device_id)},
      {"name", device_label},
      {"frontend", "matter"},
      {"txtvers", espectre::ESPECTRE_DNS_SD_TXT_SCHEMA_VERSION},
      {"protovers", espectre::ESPECTRE_PROTOCOL_VERSION},
      {"transport", espectre::ESPECTRE_DIRECT_HTTP_TRANSPORT},
      {"path", espectre::ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT},
      {"events", espectre::ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT},
      {"firmware", espectre::espectre_firmware_version()},
      {"chip", CONFIG_IDF_TARGET},
      {"capabilities", "config,monitor,raw_csi"},
  };
}

bool has_commissioned_fabric() {
  lock::ScopedChipStackLock chip_stack_lock(portMAX_DELAY);
  return chip::Server::GetInstance().GetFabricTable().FabricCount() != 0;
}

void configure_log_levels() {
  espectre::configure_runtime_log_levels();
  // CHIP logs are reduced at build time; mute esp-matter attribute chatter at runtime.
  esp_log_level_set("esp_matter_attribute", ESP_LOG_WARN);
}

void open_commissioning_window_if_necessary() {
  lock::ScopedChipStackLock chip_stack_lock(portMAX_DELAY);
  if (chip::Server::GetInstance().GetFabricTable().FabricCount() != 0) {
    return;
  }

  chip::CommissioningWindowManager &commission_mgr = chip::Server::GetInstance().GetCommissioningWindowManager();
  if (commission_mgr.IsCommissioningWindowOpen()) {
    return;
  }

  CHIP_ERROR err = commission_mgr.OpenBasicCommissioningWindow(
      chip::System::Clock::Seconds16(300), chip::CommissioningWindowAdvertisement::kAllSupported);
  if (err != CHIP_NO_ERROR) {
    ESP_LOGE(TAG, "Failed to open commissioning window");
  }
}

void log_onboarding_codes() {
  constexpr auto rendezvous =
      chip::RendezvousInformationFlags(chip::RendezvousInformationFlag::kBLE);
  char qr_code[128] = {};
  char manual_code[32] = {};
  chip::MutableCharSpan qr_span(qr_code, sizeof(qr_code));
  chip::MutableCharSpan manual_span(manual_code, sizeof(manual_code));

  CHIP_ERROR qr_error = GetQRCode(qr_span, rendezvous);
  CHIP_ERROR manual_error = GetManualPairingCode(manual_span, rendezvous);
  if (qr_error != CHIP_NO_ERROR || manual_error != CHIP_NO_ERROR) {
    ESP_LOGE(TAG, "Failed to generate Matter onboarding codes");
    return;
  }

  ESP_LOGI(TAG, "MATTER_QR=%.*s", static_cast<int>(qr_span.size()), qr_span.data());
  ESP_LOGI(TAG, "MATTER_MANUAL_CODE=%.*s", static_cast<int>(manual_span.size()),
           manual_span.data());
  ESP_LOGI(TAG, "MATTER_DISCRIMINATOR=%u",
           static_cast<unsigned>(g_commissioning_data.setup_discriminator()));
}

void sync_post_start_state_on_chip_thread(intptr_t arg) {
  (void) arg;

  bool commissioned = false;
  {
    lock::ScopedChipStackLock chip_stack_lock(portMAX_DELAY);
    commissioned = has_commissioned_fabric();
    log_onboarding_codes();
  }

  if (g_frontend != nullptr) {
    g_frontend->set_runtime_services_armed(commissioned);
  }

  ESP_LOGI(TAG, "ESPectre Matter CSI services: %s", commissioned ? "armed" : "waiting for commissioning");
}

void app_event_cb(const ChipDeviceEvent *event, intptr_t arg) {
  switch (event->Type) {
    case chip::DeviceLayer::DeviceEventType::kDnssdInitialized:
      if (g_mdns_discovery != nullptr && !g_mdns_discovery->initialized()) {
        if (g_mdns_discovery->setup(g_mdns_config)) {
          ESP_LOGI(TAG, "ESPectre Direct discovery registered with Matter mDNS");
        } else {
          ESP_LOGE(TAG, "Failed to register ESPectre Direct discovery with Matter mDNS");
        }
      }
      break;
    case chip::DeviceLayer::DeviceEventType::kCommissioningComplete:
      ESP_LOGI(TAG, "Commissioning complete");
      if (g_frontend != nullptr) {
        g_frontend->set_runtime_services_armed(true);
      }
      break;
    case chip::DeviceLayer::DeviceEventType::kFailSafeTimerExpired:
      ESP_LOGW(TAG, "Commissioning failed, fail safe timer expired");
      break;
    case chip::DeviceLayer::DeviceEventType::kFabricRemoved:
      ESP_LOGI(TAG, "Fabric removed");
      if (g_frontend != nullptr && !has_commissioned_fabric()) {
        g_frontend->set_runtime_services_armed(false);
      }
      open_commissioning_window_if_necessary();
      break;
    default:
      break;
  }
}

esp_err_t app_identification_cb(identification::callback_type_t type, uint16_t endpoint_id, uint8_t effect_id,
                                uint8_t effect_variant, void *priv_data) {
  ESP_LOGI(TAG, "Identify endpoint %u", endpoint_id);
  return ESP_OK;
}

esp_err_t app_attribute_update_cb(attribute::callback_type_t type, uint16_t endpoint_id, uint32_t cluster_id,
                                  uint32_t attribute_id, esp_matter_attr_val_t *val, void *priv_data) {
  (void) priv_data;
  if (type == attribute::POST_UPDATE && endpoint_id == 0 &&
      cluster_id == BasicInformation::Id &&
      attribute_id == BasicInformation::Attributes::NodeLabel::Id && val != nullptr &&
      val->type == ESP_MATTER_VAL_TYPE_CHAR_STRING) {
    const std::string label(reinterpret_cast<const char *>(val->val.a.b), val->val.a.s);
    if (g_frontend != nullptr) g_frontend->sync_device_label();
    for (auto &record : g_mdns_config.txt_records) {
      if (record.first == "name") record.second = label;
    }
    if (g_mdns_discovery != nullptr && g_mdns_discovery->initialized()) {
      (void) g_mdns_discovery->update_txt(g_mdns_config.txt_records);
    }
  }
  return ESP_OK;
}

void espectre_loop_task(void *arg) {
  (void) arg;
  while (true) {
    if (g_frontend != nullptr) {
      g_frontend->loop();
    }
    if (g_mdns_bootstrap_responder != nullptr) {
      esp_netif_ip_info_t ip_info{};
      esp_netif_t *station = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
      (void) g_mdns_bootstrap_responder->update(
          station != nullptr && esp_netif_get_ip_info(station, &ip_info) == ESP_OK
              ? ip_info.ip.addr
              : 0U);
      g_mdns_bootstrap_responder->loop();
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

}  // namespace

extern "C" void app_main() {
  ESP_ERROR_CHECK(espectre::nvs_init_with_erase_fallback());
  configure_log_levels();
  espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });

  CHIP_ERROR commissioning_error = g_commissioning_data.initialize();
  if (commissioning_error != CHIP_NO_ERROR) {
    ESP_LOGE(TAG, "Failed to initialize per-device Matter commissioning data");
    return;
  }
  esp_matter::set_custom_commissionable_data_provider(&g_commissioning_data);
  ESP_LOGI(TAG, "Matter factory data: %s",
           g_commissioning_data.generated_on_boot() ? "generated" : "loaded");

  node::config_t node_config;
  std::snprintf(node_config.root_node.basic_information.node_label,
                sizeof(node_config.root_node.basic_information.node_label),
                "%s",
                CONFIG_ESPECTRE_MATTER_NODE_LABEL);
  node_t *node = node::create(&node_config, app_attribute_update_cb, app_identification_cb);
  if (node == nullptr) {
    ESP_LOGE(TAG, "Failed to create Matter node");
    return;
  }

  occupancy_sensor::config_t occupancy_config;
  occupancy_config.occupancy_sensing.feature_flags =
      chip::to_underlying(OccupancySensing::Feature::kOther);

  endpoint_t *motion_endpoint = occupancy_sensor::create(node, &occupancy_config, ENDPOINT_FLAG_NONE, nullptr);
  if (motion_endpoint == nullptr) {
    ESP_LOGE(TAG, "Failed to create occupancy endpoint");
    return;
  }

  g_motion_endpoint_id = endpoint::get_id(motion_endpoint);

  static espectre::EspIdfDirectHttpService direct_service;
  static espectre::MdnsDiscoveryService mdns_discovery;
  static espectre::MdnsBootstrapResponder mdns_bootstrap_responder;
  static espectre::MatterFrontend frontend(&g_bindings, g_motion_endpoint_id, &direct_service);
  const espectre::RuntimeConfig runtime_config = build_runtime_config();
  frontend.set_runtime_config(runtime_config);
  frontend.set_runtime_services_armed(false);
  g_frontend = &frontend;
  const std::string device_id = espectre::format_espectre_device_id(runtime_config.device_id);
  g_mdns_discovery = &mdns_discovery;
  if (!mdns_bootstrap_responder.setup()) {
    ESP_LOGE(TAG, "Failed to initialize the mDNS bootstrap responder");
    return;
  }
  g_mdns_bootstrap_responder = &mdns_bootstrap_responder;
  g_mdns_config = espectre::MdnsDiscoveryServiceConfig{
      "",
      std::string(CONFIG_ESPECTRE_MATTER_NODE_LABEL) + " " + device_id,
      "_espectre",
      "_tcp",
      espectre::ESPECTRE_DIRECT_HTTP_PORT,
      matter_mdns_txt(runtime_config.device_id),
      espectre::MdnsResponderMode::USE_EXISTING_RESPONDER,
  };
  esp_err_t err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "Failed to create default event loop (%d)", err);
    return;
  }
  err = esp_matter::start(app_event_cb);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to start Matter (%d)", err);
    return;
  }
  chip::DeviceLayer::PlatformMgr().ScheduleWork(sync_post_start_state_on_chip_thread, 0);

  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre Matter frontend");
    return;
  }

  ESP_LOGI(TAG, "ESPectre Matter detector: %s", espectre::detection_algorithm_name(frontend.runtime_config().detection_algorithm));

  xTaskCreate(espectre_loop_task, "espectre_loop", 8192, nullptr, 5, nullptr);

  ESP_LOGI(TAG, "ESPectre Matter firmware started on endpoint %u", g_motion_endpoint_id);
}
