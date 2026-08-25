/*
 * ESPectre - esp-matter Bindings
 *
 * ESP-Matter-backed bindings that publish ESPectre state to Matter
 * endpoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
  (void)message;
}

bool MatterEspBindings::get_node_label(std::string *label) {
  if (label == nullptr) return false;
  using namespace chip::app::Clusters::BasicInformation;
  esp_matter_attr_val_t value = esp_matter_invalid(nullptr);
  if (esp_matter::attribute::get_val(0, Id, Attributes::NodeLabel::Id, &value) != ESP_OK ||
      value.type != ESP_MATTER_VAL_TYPE_CHAR_STRING) {
    return false;
  }
  label->assign(reinterpret_cast<const char *>(value.val.a.b), value.val.a.s);
  return true;
}

bool MatterEspBindings::set_node_label(const std::string &label) {
  using namespace chip::app::Clusters::BasicInformation;
  if (label.size() > esp_matter::cluster::basic_information::k_max_node_label_length) return false;
  esp_matter_attr_val_t value =
      esp_matter_char_str(const_cast<char *>(label.data()), static_cast<uint16_t>(label.size()));
  return esp_matter::attribute::update(0, Id, Attributes::NodeLabel::Id, &value) == ESP_OK;
}

}  // namespace espectre
