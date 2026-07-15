/*
 * ESPectre - esp-matter Bindings
 *
 * ESP-Matter-backed bindings that publish ESPectre state to Matter
 * endpoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "matter_bindings_esp_matter.h"

#include <app-common/zap-generated/cluster-objects.h>
#include <esp_log.h>
#include <esp_matter.h>

namespace espectre {

void MatterEspBindings::publish_motion(uint16_t endpoint_id, bool motion_detected) {
  using namespace chip::app::Clusters::OccupancySensing;
  const uint8_t occupancy = motion_detected ? chip::to_underlying(OccupancyBitmap::kOccupied) : 0;
  esp_matter_attr_val_t val = esp_matter_bitmap8(occupancy);
  esp_matter::attribute::update(endpoint_id, Id, Attributes::Occupancy::Id, &val);
}

void MatterEspBindings::report_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW("espectre.matter", "Runtime fault reported to Matter: %s", message);
  }
}

}  // namespace espectre
