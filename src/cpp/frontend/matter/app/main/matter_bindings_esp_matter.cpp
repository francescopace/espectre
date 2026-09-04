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
#include <platform/CHIPDeviceLayer.h>

namespace espectre {

namespace {

const char *const TAG = "espectre.matter.bindings";

}  // namespace

void MatterEspBindings::publish_motion(uint16_t endpoint_id, bool motion_detected) {
  if (!pending_motion_.post(PendingMotionPublish{endpoint_id, motion_detected})) {
    (void) pending_motion_.post_overwrite_oldest(PendingMotionPublish{endpoint_id, motion_detected});
    ESP_LOGW(TAG, "Matter occupancy publish queue full; retained the newest state");
  }
  schedule_motion_publish_();
}

void MatterEspBindings::flush_pending() {
  if (pending_motion_.size() > 0U) {
    schedule_motion_publish_();
  }
}

void MatterEspBindings::schedule_motion_publish_() {
  bool expected = false;
  if (!motion_work_scheduled_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return;
  }
  const CHIP_ERROR err = chip::DeviceLayer::PlatformMgr().ScheduleWork(
      &MatterEspBindings::publish_motion_on_chip_thread_, reinterpret_cast<intptr_t>(this));
  if (err != CHIP_NO_ERROR) {
    motion_work_scheduled_.store(false, std::memory_order_release);
    ESP_LOGW(TAG, "Failed to schedule Matter occupancy publish: %s", err.AsString());
  }
}

void MatterEspBindings::publish_motion_on_chip_thread_(intptr_t context) {
  auto *bindings = reinterpret_cast<MatterEspBindings *>(context);
  if (bindings != nullptr) {
    bindings->drain_motion_queue_on_chip_thread_();
  }
}

void MatterEspBindings::drain_motion_queue_on_chip_thread_() {
  PendingMotionPublish pending;
  while (pending_motion_.take(pending)) {
    using namespace chip::app::Clusters::OccupancySensing;
    const uint8_t occupancy = pending.motion_detected
                                  ? chip::to_underlying(OccupancyBitmap::kOccupied)
                                  : 0U;
    esp_matter_attr_val_t val = esp_matter_bitmap8(occupancy);
    if (esp_matter::attribute::update(
            pending.endpoint_id, Id, Attributes::Occupancy::Id, &val) != ESP_OK) {
      ESP_LOGW(TAG, "Failed to update Matter occupancy attribute");
    }
  }
  motion_work_scheduled_.store(false, std::memory_order_release);
  if (pending_motion_.size() > 0U) {
    schedule_motion_publish_();
  }
}

/*
 * NodeLabel reads and writes originate from low-rate Direct commands. The
 * occupancy path above is scheduled separately because it originates in the
 * latency-sensitive sensing loop.
 */
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
