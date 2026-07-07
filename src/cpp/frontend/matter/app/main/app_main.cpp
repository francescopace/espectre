/*
 * ESPectre Matter firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <esp_err.h>
#include <esp_event.h>
#include <esp_log.h>
#include <nvs_flash.h>
#include <sdkconfig.h>

#include <cstdio>

#include <app/server/CommissioningWindowManager.h>
#include <app/server/Server.h>
#include <esp_matter.h>
#include <esp_matter_attribute.h>
#include <esp_matter_cluster.h>
#include <esp_matter_endpoint.h>
#include <esp_matter_ota.h>
#include <setup_payload/OnboardingCodesUtil.h>

#include "matter_bindings_esp_matter.h"
#include "matter_frontend.h"
#include "matter_surface.h"

static const char *TAG = "espectre.matter.app";

using namespace esp_matter;
using namespace esp_matter::attribute;
using namespace esp_matter::cluster;
using namespace esp_matter::endpoint;
using namespace chip::app::Clusters;

namespace {

esphome::espectre::MatterEspBindings g_bindings;
esphome::espectre::MatterFrontend *g_frontend = nullptr;
uint16_t g_motion_endpoint_id = 0;

esphome::espectre::RuntimeConfig build_runtime_config() {
  esphome::espectre::RuntimeConfig config;
#if CONFIG_ESPECTRE_MATTER_DETECTION_ALGORITHM_ML
  config.detection_algorithm = esphome::espectre::DetectionAlgorithm::ML;
#elif CONFIG_ESPECTRE_MATTER_DETECTION_ALGORITHM_L1_DELTA
  config.detection_algorithm = esphome::espectre::DetectionAlgorithm::L1_DELTA;
#else
  config.detection_algorithm = esphome::espectre::DetectionAlgorithm::MVS;
#endif
  return config;
}

const char *detector_name(const esphome::espectre::RuntimeConfig &config) {
  switch (config.detection_algorithm) {
    case esphome::espectre::DetectionAlgorithm::ML:
      return "ML";
    case esphome::espectre::DetectionAlgorithm::L1_DELTA:
      return "L1D";
    case esphome::espectre::DetectionAlgorithm::MVS:
    default:
      return "MVS";
  }
}

bool has_commissioned_fabric() { return chip::Server::GetInstance().GetFabricTable().FabricCount() != 0; }

void configure_log_levels() {
  // CHIP logs are reduced at build time; mute esp-matter attribute chatter at runtime.
  esp_log_level_set("esp_matter_attribute", ESP_LOG_WARN);
}

cluster_t *create_espectre_vendor_cluster(endpoint_t *endpoint) {
  cluster_t *vendor_cluster = cluster::create(endpoint, esphome::espectre::ESPECTRE_MATTER_VENDOR_CLUSTER_ID,
                                            CLUSTER_FLAG_SERVER);
  if (vendor_cluster == nullptr) {
    return nullptr;
  }

  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_MOVEMENT_METRIC, ATTRIBUTE_FLAG_NONE,
                    esp_matter_nullable_float(0.0f));
  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_THRESHOLD,
                    ATTRIBUTE_FLAG_WRITABLE | ATTRIBUTE_FLAG_NONVOLATILE, esp_matter_nullable_float(1.0f));
  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_CALIBRATING, ATTRIBUTE_FLAG_NONE,
                    esp_matter_bool(false));
  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_READY_TO_PUBLISH, ATTRIBUTE_FLAG_NONE,
                    esp_matter_bool(false));
  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_STARTUP_THRESHOLD, ATTRIBUTE_FLAG_NONE,
                    esp_matter_nullable_float(0.0f));
  attribute::create(vendor_cluster, esphome::espectre::ESPECTRE_MATTER_ATTR_REQUEST_RECALIBRATE,
                    ATTRIBUTE_FLAG_WRITABLE, esp_matter_bool(false));
  return vendor_cluster;
}

void open_commissioning_window_if_necessary() {
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
  if (g_frontend == nullptr || endpoint_id != g_motion_endpoint_id ||
      cluster_id != esphome::espectre::ESPECTRE_MATTER_VENDOR_CLUSTER_ID) {
    return ESP_OK;
  }

  if (attribute_id == esphome::espectre::ESPECTRE_MATTER_ATTR_THRESHOLD && val != nullptr &&
      val->type == ESP_MATTER_VAL_TYPE_NULLABLE_FLOAT) {
    if (!g_frontend->handle_threshold_write(val->val.f)) {
      return ESP_FAIL;
    }
    return ESP_OK;
  }

  if (attribute_id == esphome::espectre::ESPECTRE_MATTER_ATTR_REQUEST_RECALIBRATE && val != nullptr &&
      val->type == ESP_MATTER_VAL_TYPE_BOOLEAN && val->val.b) {
    if (!g_frontend->handle_recalibrate_request()) {
      return ESP_FAIL;
    }
    esp_matter_attr_val_t cleared = esp_matter_bool(false);
    attribute::update(endpoint_id, cluster_id, attribute_id, &cleared);
    return ESP_OK;
  }

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
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    err = nvs_flash_init();
  }
  ESP_ERROR_CHECK(err);
  configure_log_levels();

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

  if (create_espectre_vendor_cluster(motion_endpoint) == nullptr) {
    ESP_LOGE(TAG, "Failed to create ESPectre vendor cluster");
    return;
  }

  g_motion_endpoint_id = endpoint::get_id(motion_endpoint);

  err = esp_matter_ota_requestor_init();
  if (err != ESP_OK && err != ESP_ERR_NOT_SUPPORTED) {
    ESP_LOGE(TAG, "Failed to initialize Matter OTA requestor (%d)", err);
    return;
  }

  static esphome::espectre::MatterFrontend frontend(&g_bindings, g_motion_endpoint_id);
  frontend.set_runtime_config(build_runtime_config());
  g_frontend = &frontend;
  ESP_LOGI(TAG, "ESPectre Matter smoke marker: endpoint %u configured, starting Matter stack",
           g_motion_endpoint_id);
  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "Failed to create default event loop (%d)", err);
    return;
  }
  err = esp_matter::start(app_event_cb);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to start Matter (%d)", err);
    return;
  }
  esp_matter_ota_requestor_start();

  frontend.set_runtime_services_armed(has_commissioned_fabric());
  PrintOnboardingCodes(chip::RendezvousInformationFlags(chip::RendezvousInformationFlag::kBLE));

  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre Matter frontend");
    return;
  }

  ESP_LOGI(TAG, "ESPectre Matter detector: %s", detector_name(frontend.runtime_config()));
  ESP_LOGI(TAG, "ESPectre Matter CSI services: %s", frontend.runtime_services_armed() ? "armed" : "waiting for commissioning");

  xTaskCreate(espectre_loop_task, "espectre_loop", 8192, nullptr, 5, nullptr);

  ESP_LOGI(TAG, "ESPectre Matter firmware started on endpoint %u", g_motion_endpoint_id);
}
