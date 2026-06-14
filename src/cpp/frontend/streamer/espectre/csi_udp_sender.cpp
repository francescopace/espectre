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
constexpr UBaseType_t kSenderTaskPriority = 7;

void update_high_watermark(std::atomic<uint8_t> &high_watermark, uint8_t candidate) {
  uint8_t observed = high_watermark.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !high_watermark.compare_exchange_weak(observed, candidate, std::memory_order_relaxed)) {
  }
}
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
  if (xTaskCreate(&CsiUdpSender::sender_task_entry_,
                  "espectre_udp_tx",
                  4096,
                  this,
                  kSenderTaskPriority,
                  &sender_task_handle_) !=
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
  collector_ip_addr_.store(collector_addr.sin_addr.s_addr, std::memory_order_relaxed);
  collector_port_.store(collector_addr.sin_port, std::memory_order_relaxed);
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
  update_high_watermark(ready_queue_high_watermark_,
                        ready_slots_ != nullptr ? static_cast<uint8_t>(uxQueueMessagesWaiting(ready_slots_)) : 0U);

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
  bool has_pending_slot = false;
  uint8_t pending_slot_idx = 0U;
  std::array<uint8_t, MAX_BATCH_BYTES> batch{};
  std::array<uint8_t, MAX_BATCH_RECORDS> batch_slots{};

  while (running_.load(std::memory_order_relaxed)) {
    uint8_t slot_idx = 0U;
    if (has_pending_slot) {
      slot_idx = pending_slot_idx;
      has_pending_slot = false;
    } else if (xQueueReceive(ready_slots_, &slot_idx, pdMS_TO_TICKS(250)) != pdTRUE) {
      continue;
    }

    size_t batch_len = 0U;
    uint8_t batch_count = 0U;
    bool batch_ready = false;
    while (!batch_ready) {
      PacketSlot &slot = slots_[slot_idx];
      if (batch_count > 0U &&
          (batch_count >= MAX_BATCH_RECORDS || batch_len + slot.packet_len > MAX_BATCH_BYTES)) {
        pending_slot_idx = slot_idx;
        has_pending_slot = true;
        break;
      }

      std::memcpy(batch.data() + batch_len, slot.packet.data(), slot.packet_len);
      batch_slots[batch_count++] = slot_idx;
      batch_len += slot.packet_len;

      if (batch_count >= MAX_BATCH_RECORDS || batch_len >= MAX_BATCH_BYTES ||
          xQueueReceive(ready_slots_, &slot_idx, 0) != pdTRUE) {
        batch_ready = true;
      }
    }

    const uint32_t collector_ip_addr = collector_ip_addr_.load(std::memory_order_relaxed);
    const uint16_t collector_port = collector_port_.load(std::memory_order_relaxed);
    if (!collector_enabled_.load(std::memory_order_relaxed) || collector_ip_addr == 0U || collector_port == 0U) {
      send_fail_total_.fetch_add(static_cast<uint64_t>(batch_count), std::memory_order_relaxed);
      for (uint8_t idx = 0U; idx < batch_count; idx++) {
        recycle_slot_(batch_slots[idx]);
      }
      continue;
    }

    if (sock < 0) {
      sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    }

    if (sock >= 0) {
      sockaddr_in collector_addr{};
      collector_addr.sin_family = AF_INET;
      collector_addr.sin_addr.s_addr = collector_ip_addr;
      collector_addr.sin_port = collector_port;
      const int rc = sendto(sock,
                            batch.data(),
                            batch_len,
                            0,
                            reinterpret_cast<const sockaddr *>(&collector_addr),
                            sizeof(collector_addr));
      if (rc == static_cast<int>(batch_len)) {
        tx_total_.fetch_add(static_cast<uint64_t>(batch_count), std::memory_order_relaxed);
      } else {
        send_fail_total_.fetch_add(static_cast<uint64_t>(batch_count), std::memory_order_relaxed);
        close(sock);
        sock = -1;
      }
    } else {
      send_fail_total_.fetch_add(static_cast<uint64_t>(batch_count), std::memory_order_relaxed);
    }

    for (uint8_t idx = 0U; idx < batch_count; idx++) {
      recycle_slot_(batch_slots[idx]);
    }
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
