/*
 * ESPectre - CSI UDP Sender
 *
 * Dedicated UDP sender with a preallocated queue for CSI records.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "csi_stream_protocol.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "lwip/sockets.h"

namespace esphome {
namespace espectre {

#ifndef CONFIG_ESPECTRE_STREAM_BATCH_MAX_RECORDS
#define CONFIG_ESPECTRE_STREAM_BATCH_MAX_RECORDS 4
#endif

#ifndef CONFIG_ESPECTRE_STREAM_BATCH_MAX_BYTES
#define CONFIG_ESPECTRE_STREAM_BATCH_MAX_BYTES 1200
#endif

class CsiUdpSender {
 public:
  static constexpr size_t MAX_CSI_LEN_BYTES = 512U;
  static constexpr size_t MAX_PACKET_BYTES = sizeof(CsiStreamHeaderV2) + MAX_CSI_LEN_BYTES;
  static constexpr uint8_t QUEUE_CAPACITY = CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS;
  static constexpr size_t MAX_BATCH_BYTES = CONFIG_ESPECTRE_STREAM_BATCH_MAX_BYTES;
  static constexpr uint8_t MAX_BATCH_RECORDS = CONFIG_ESPECTRE_STREAM_BATCH_MAX_RECORDS;

  static_assert(MAX_BATCH_BYTES >= MAX_PACKET_BYTES, "Batch payload must fit at least one CSI record");

  bool setup();
  void shutdown();

  void set_collector(const sockaddr_in &collector_addr, bool enabled);
  bool queue_packet(const uint8_t *data, size_t data_len);

  uint64_t queued_total() const { return queued_total_.load(std::memory_order_relaxed); }
  uint64_t tx_total() const { return tx_total_.load(std::memory_order_relaxed); }
  uint64_t drop_total() const { return drop_total_.load(std::memory_order_relaxed); }
  uint64_t send_fail_total() const { return send_fail_total_.load(std::memory_order_relaxed); }
  uint8_t ready_queue_depth() const {
    return ready_slots_ != nullptr ? static_cast<uint8_t>(uxQueueMessagesWaiting(ready_slots_)) : 0U;
  }
  uint8_t take_ready_queue_high_watermark() {
    const uint8_t current = ready_queue_depth();
    const uint8_t peak = ready_queue_high_watermark_.exchange(current, std::memory_order_relaxed);
    return peak > current ? peak : current;
  }
  uint8_t free_queue_depth() const {
    return free_slots_ != nullptr ? static_cast<uint8_t>(uxQueueMessagesWaiting(free_slots_)) : 0U;
  }

 private:
  struct PacketSlot final {
    size_t packet_len{0U};
    std::array<uint8_t, MAX_PACKET_BYTES> packet{};
  };

  static void sender_task_entry_(void *ctx);
  void run_sender_task_();
  void recycle_slot_(uint8_t slot_idx);

  std::array<PacketSlot, CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS> slots_{};
  QueueHandle_t free_slots_{nullptr};
  QueueHandle_t ready_slots_{nullptr};
  TaskHandle_t sender_task_handle_{nullptr};
  std::atomic<uint32_t> collector_ip_addr_{0U};
  std::atomic<uint16_t> collector_port_{0U};
  std::atomic<bool> collector_enabled_{false};
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> queued_total_{0U};
  std::atomic<uint64_t> tx_total_{0U};
  std::atomic<uint64_t> drop_total_{0U};
  std::atomic<uint64_t> send_fail_total_{0U};
  std::atomic<uint8_t> ready_queue_high_watermark_{0U};
};

}  // namespace espectre
}  // namespace esphome
