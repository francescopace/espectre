/*
 * ESPectre - esp-matter Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "matter_bindings_esp_matter.h"

#include <app-common/zap-generated/cluster-objects.h>
#include <esp_log.h>

#include <esp_matter.h>
#include <esp_matter_attribute.h>

namespace esphome {
namespace espectre {

namespace {

void update_nullable_float(uint16_t endpoint_id, uint32_t cluster_id, uint32_t attribute_id, float value) {
  esp_matter_attr_val_t val = esp_matter_nullable_float(value);
  esp_matter::attribute::update(endpoint_id, cluster_id, attribute_id, &val);
}

void update_bool(uint16_t endpoint_id, uint32_t cluster_id, uint32_t attribute_id, bool value) {
  esp_matter_attr_val_t val = esp_matter_bool(value);
  esp_matter::attribute::update(endpoint_id, cluster_id, attribute_id, &val);
}

}  // namespace

void MatterEspBindings::publish_motion(uint16_t endpoint_id, bool motion_detected) {
  using namespace chip::app::Clusters::OccupancySensing;
  const uint8_t occupancy = motion_detected ? chip::to_underlying(OccupancyBitmap::kOccupied) : 0;
  esp_matter_attr_val_t val = esp_matter_bitmap8(occupancy);
  esp_matter::attribute::update(endpoint_id, Id, Attributes::Occupancy::Id, &val);
}

void MatterEspBindings::publish_periodic_state(uint16_t endpoint_id, const MatterPeriodicState &state) {
  update_nullable_float(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_MOVEMENT_METRIC,
                        state.movement_metric);
  update_nullable_float(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_THRESHOLD,
                        state.threshold);
  update_nullable_float(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_BEST_PXX, state.best_pxx);
  update_bool(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_READY_TO_PUBLISH,
              state.ready_to_publish);
  update_bool(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_CALIBRATING, state.calibrating);
}

void MatterEspBindings::publish_threshold(uint16_t endpoint_id, float threshold) {
  update_nullable_float(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_THRESHOLD, threshold);
}

void MatterEspBindings::publish_calibrating(uint16_t endpoint_id, bool calibrating) {
  update_bool(endpoint_id, ESPECTRE_MATTER_VENDOR_CLUSTER_ID, ESPECTRE_MATTER_ATTR_CALIBRATING, calibrating);
}

void MatterEspBindings::report_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW("espectre.matter", "Runtime fault reported to Matter: %s", message);
  }
}

}  // namespace espectre
}  // namespace esphome
