/*
 * ESPectre - Matter Firmware Entrypoint
 *
 * Matter firmware application entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include <esp_err.h>
#include <esp_event.h>
#include <esp_log.h>
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
#include "matter_bindings_esp_matter.h"
#include "matter_commissioning_data.h"
#include "matter_frontend.h"
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
uint16_t g_motion_endpoint_id = 0;

espectre::RuntimeConfig build_runtime_config() { return espectre::make_runtime_sensing_config_from_kconfig(); }

bool has_commissioned_fabric() {
  lock::ScopedChipStackLock chip_stack_lock(portMAX_DELAY);
  return chip::Server::GetInstance().GetFabricTable().FabricCount() != 0;
}

void configure_log_levels() {
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
    PrintOnboardingCodes(chip::RendezvousInformationFlags(chip::RendezvousInformationFlag::kBLE));
  }

  if (g_frontend != nullptr) {
    g_frontend->set_runtime_services_armed(commissioned);
  }

  ESP_LOGI(TAG, "ESPectre Matter CSI services: %s", commissioned ? "armed" : "waiting for commissioning");
}

void app_event_cb(const ChipDeviceEvent *event, intptr_t arg) {
  switch (event->Type) {
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
  (void) type;
  (void) endpoint_id;
  (void) cluster_id;
  (void) attribute_id;
  (void) val;
  (void) priv_data;
  return ESP_OK;
}

void espectre_loop_task(void *arg) {
  (void) arg;
  while (true) {
    if (g_frontend != nullptr) {
      g_frontend->loop();
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

  static espectre::MatterFrontend frontend(&g_bindings, g_motion_endpoint_id);
  frontend.set_runtime_config(build_runtime_config());
  frontend.set_runtime_services_armed(false);
  g_frontend = &frontend;
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
