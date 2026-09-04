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
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "matter_bindings.h"
#include "pending_queue.h"

namespace espectre {

class MatterEspBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void flush_pending() override;
  void report_fault(const char *message) override;
  bool get_node_label(std::string *label) override;
  bool set_node_label(const std::string &label) override;
  /** Load the persisted label while running on the CHIP task. */
  void refresh_node_label_on_chip_thread();
  /** Copy a successful NodeLabel update for readers outside the CHIP task. */
  void cache_node_label(const std::string &label);

 private:
  struct PendingMotionPublish {
    uint16_t endpoint_id{0U};
    bool motion_detected{false};
  };

  static void publish_motion_on_chip_thread_(intptr_t context);
  void drain_motion_queue_on_chip_thread_();
  void schedule_motion_publish_();

  PendingQueue<PendingMotionPublish, 8U> pending_motion_{};
  std::atomic<bool> motion_work_scheduled_{false};
  std::mutex node_label_mutex_;
  std::string node_label_;
  bool node_label_ready_{false};
};

}  // namespace espectre
