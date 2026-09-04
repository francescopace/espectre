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
#include <app/clusters/occupancy-sensor-server/OccupancySensingCluster.h>
#include <data_model_provider/esp_matter_data_model_provider.h>
#include <esp_log.h>
#include <esp_matter.h>
#include <esp_matter_mem.h>
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
    auto *cluster = esp_matter::data_model::provider::get_instance().registry().Get(
        chip::app::ConcreteClusterPath(pending.endpoint_id, Id));
    if (cluster == nullptr) {
      ESP_LOGW(TAG, "Matter occupancy cluster is unavailable");
      continue;
    }
    static_cast<chip::app::Clusters::OccupancySensingCluster *>(cluster)->SetOccupancy(
        pending.motion_detected);
  }
  motion_work_scheduled_.store(false, std::memory_order_release);
  if (pending_motion_.size() > 0U) {
    schedule_motion_publish_();
  }
}

/*
 * Direct reads a NodeLabel snapshot maintained by CHIP callbacks. Writes use
 * attribute::update(), which acquires the CHIP lock internally. Occupancy work
 * is scheduled separately because it originates in the sensing loop.
 */
void MatterEspBindings::report_fault(const char *message) {
  (void)message;
}

bool MatterEspBindings::get_node_label(std::string *label) {
  if (label == nullptr) return false;
  std::lock_guard<std::mutex> lock(node_label_mutex_);
  if (!node_label_ready_) return false;
  *label = node_label_;
  return true;
}

void MatterEspBindings::cache_node_label(const std::string &label) {
  std::lock_guard<std::mutex> lock(node_label_mutex_);
  node_label_ = label;
  node_label_ready_ = true;
}

void MatterEspBindings::refresh_node_label_on_chip_thread() {
  // Read the data model only on CHIP; Direct consumes the cached snapshot.
  using namespace chip::app::Clusters::BasicInformation;
  esp_matter_attr_val_t value = esp_matter_invalid(nullptr);
  if (esp_matter::attribute::get_val(0, Id, Attributes::NodeLabel::Id, &value) != ESP_OK ||
      value.type != ESP_MATTER_VAL_TYPE_CHAR_STRING) {
    ESP_LOGW(TAG, "Failed to read Matter NodeLabel");
    return;
  }
  cache_node_label(value.val.a.s == 0U
                      ? std::string{}
                      : std::string(reinterpret_cast<const char *>(value.val.a.b), value.val.a.s));
  // esp-matter 1.6 transfers ownership of string reads to the caller.
  esp_matter_mem_free(value.val.a.b);
}

bool MatterEspBindings::set_node_label(const std::string &label) {
  using namespace chip::app::Clusters::BasicInformation;
  if (label.size() > esp_matter::cluster::basic_information::k_max_node_label_length) return false;
  esp_matter_attr_val_t value =
      esp_matter_char_str(const_cast<char *>(label.data()), static_cast<uint16_t>(label.size()));
  return esp_matter::attribute::update(0, Id, Attributes::NodeLabel::Id, &value) == ESP_OK;
}

}  // namespace espectre
