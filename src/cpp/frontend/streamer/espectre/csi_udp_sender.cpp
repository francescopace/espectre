/*
 * ESPectre - CSI UDP Sender
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_udp_sender.h"

#include <cstring>

#include "espectre_log.h"

namespace esphome {
namespace espectre {

namespace {
static const char *const TAG = "espectre.stream.tx";
}

bool CsiUdpSender::setup() {
  if (running_.load(std::memory_order_relaxed)) {
    return true;
  }

  free_slots_ = xQueueCreate(CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS, sizeof(uint8_t));
  ready_slots_ = xQueueCreate(CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS, sizeof(uint8_t));
  if (free_slots_ == nullptr || ready_slots_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create stream queues");
    shutdown();
    return false;
  }

  for (uint8_t idx = 0; idx < CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS; idx++) {
    if (xQueueSend(free_slots_, &idx, 0) != pdTRUE) {
      ESP_LOGE(TAG, "Failed to initialize free slot queue");
      shutdown();
      return false;
    }
  }

  running_.store(true, std::memory_order_relaxed);
  if (xTaskCreate(&CsiUdpSender::sender_task_entry_, "espectre_udp_tx", 4096, this, 5, &sender_task_handle_) !=
      pdPASS) {
    ESP_LOGE(TAG, "Failed to create sender task");
    shutdown();
    return false;
  }

  return true;
}

void CsiUdpSender::shutdown() {
  running_.store(false, std::memory_order_relaxed);
  collector_enabled_.store(false, std::memory_order_relaxed);

  if (sender_task_handle_ != nullptr) {
    vTaskDelete(sender_task_handle_);
    sender_task_handle_ = nullptr;
  }
  if (free_slots_ != nullptr) {
    vQueueDelete(free_slots_);
    free_slots_ = nullptr;
  }
  if (ready_slots_ != nullptr) {
    vQueueDelete(ready_slots_);
    ready_slots_ = nullptr;
  }
}

void CsiUdpSender::set_collector(const sockaddr_in &collector_addr, bool enabled) {
  collector_addr_ = collector_addr;
  collector_enabled_.store(enabled, std::memory_order_relaxed);
}

bool CsiUdpSender::queue_packet(const uint8_t *data, size_t data_len) {
  if (!running_.load(std::memory_order_relaxed) || data == nullptr || data_len == 0U || data_len > MAX_PACKET_BYTES ||
      free_slots_ == nullptr || ready_slots_ == nullptr) {
    drop_total_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  uint8_t slot_idx = 0U;
  if (xQueueReceive(free_slots_, &slot_idx, 0) != pdTRUE) {
    drop_total_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  PacketSlot &slot = slots_[slot_idx];
  std::memcpy(slot.packet.data(), data, data_len);
  slot.packet_len = data_len;
  if (xQueueSend(ready_slots_, &slot_idx, 0) != pdTRUE) {
    drop_total_.fetch_add(1U, std::memory_order_relaxed);
    recycle_slot_(slot_idx);
    return false;
  }

  queued_total_.fetch_add(1U, std::memory_order_relaxed);
  return true;
}

void CsiUdpSender::sender_task_entry_(void *ctx) {
  CsiUdpSender *sender = static_cast<CsiUdpSender *>(ctx);
  if (sender != nullptr) {
    sender->run_sender_task_();
  }
  vTaskDelete(nullptr);
}

void CsiUdpSender::run_sender_task_() {
  int sock = -1;
  while (running_.load(std::memory_order_relaxed)) {
    uint8_t slot_idx = 0U;
    if (xQueueReceive(ready_slots_, &slot_idx, pdMS_TO_TICKS(250)) != pdTRUE) {
      continue;
    }

    PacketSlot &slot = slots_[slot_idx];
    if (!collector_enabled_.load(std::memory_order_relaxed)) {
      send_fail_total_.fetch_add(1U, std::memory_order_relaxed);
      recycle_slot_(slot_idx);
      continue;
    }

    if (sock < 0) {
      sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    }

    if (sock >= 0) {
      const int rc = sendto(sock,
                            slot.packet.data(),
                            slot.packet_len,
                            0,
                            reinterpret_cast<const sockaddr *>(&collector_addr_),
                            sizeof(collector_addr_));
      if (rc == static_cast<int>(slot.packet_len)) {
        tx_total_.fetch_add(1U, std::memory_order_relaxed);
      } else {
        send_fail_total_.fetch_add(1U, std::memory_order_relaxed);
        close(sock);
        sock = -1;
      }
    } else {
      send_fail_total_.fetch_add(1U, std::memory_order_relaxed);
    }

    recycle_slot_(slot_idx);
  }

  if (sock >= 0) {
    close(sock);
  }
}

void CsiUdpSender::recycle_slot_(uint8_t slot_idx) {
  if (free_slots_ != nullptr) {
    (void)xQueueSend(free_slots_, &slot_idx, 0);
  }
}

}  // namespace espectre
}  // namespace esphome
