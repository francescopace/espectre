/*
 * ESPectre - CSI UDP Sender
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_udp_sender.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <inttypes.h>
#include <net/if.h>

#include "espectre_log.h"
#include "esp_netif.h"
#include "runtime_time.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace esphome {
namespace espectre {

namespace {
static const char *const TAG = "espectre.stream.tx";
constexpr UBaseType_t kSenderTaskPriority = 7;
constexpr int kSenderSocketTxBufferBytes = 16 * 1024;
constexpr size_t kAggressiveBatchSmallPacketBytes = 192U;
constexpr TickType_t kAggressiveBatchWaitTicks = pdMS_TO_TICKS(4);
constexpr TickType_t kSendFailureLogIntervalTicks = pdMS_TO_TICKS(1000);
constexpr TickType_t kSendBackpressureBackoffTicks = pdMS_TO_TICKS(3);

template<typename T>
T *allocate_stream_storage(const char *label, size_t count) {
  const size_t bytes = sizeof(T) * count;
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
#ifdef MALLOC_CAP_SPIRAM
  if (T *external = static_cast<T *>(heap_caps_calloc(count, sizeof(T), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT))) {
    ESP_LOGI(TAG, "Allocated %s in external RAM (%u bytes)", label, static_cast<unsigned>(bytes));
    return external;
  }
#endif
  if (T *internal = static_cast<T *>(heap_caps_calloc(count, sizeof(T), MALLOC_CAP_8BIT))) {
    ESP_LOGI(TAG, "Allocated %s in internal RAM (%u bytes)", label, static_cast<unsigned>(bytes));
    return internal;
  }
#else
  if (T *storage = static_cast<T *>(std::calloc(count, sizeof(T)))) {
    return storage;
  }
#endif
  ESP_LOGE(TAG, "Failed to allocate %s (%u bytes)", label, static_cast<unsigned>(bytes));
  return nullptr;
}

void free_stream_storage(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  heap_caps_free(ptr);
#else
  std::free(ptr);
#endif
}

void update_high_watermark(std::atomic<uint8_t> &high_watermark, uint8_t candidate) {
  uint8_t observed = high_watermark.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !high_watermark.compare_exchange_weak(observed, candidate, std::memory_order_relaxed)) {
  }
}

esp_netif_t *get_sta_netif() { return esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"); }

bool get_sta_netif_index(uint32_t *out_index) {
  if (out_index == nullptr) {
    return false;
  }
  esp_netif_t *netif = get_sta_netif();
  if (netif == nullptr) {
    ESP_LOGW(TAG, "Failed to get STA netif for UDP TX socket");
    return false;
  }

  const int if_index = esp_netif_get_netif_impl_index(netif);
  if (if_index <= 0) {
    ESP_LOGW(TAG, "Invalid STA netif index for UDP TX socket: %d", if_index);
    return false;
  }

  *out_index = static_cast<uint32_t>(if_index);
  return true;
}

bool bind_socket_to_sta_interface(int sock) {
  uint32_t if_index = 0U;
  if (!get_sta_netif_index(&if_index)) {
    return false;
  }

  struct ifreq iface = {};
  if (if_indextoname(static_cast<unsigned int>(if_index), iface.ifr_name) == nullptr) {
    ESP_LOGW(TAG, "Failed to resolve STA interface name for UDP TX socket index %" PRIu32, if_index);
    return false;
  }

  if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &iface, sizeof(iface)) != 0) {
    ESP_LOGW(TAG, "Failed to bind UDP TX socket to %s (errno=%d)", iface.ifr_name, errno);
    return false;
  }

  return true;
}

void configure_tx_socket(int sock) {
  if (!bind_socket_to_sta_interface(sock)) {
    ESP_LOGW(TAG, "Continuing without explicit UDP TX socket binding");
  }

  const int send_buffer_bytes = kSenderSocketTxBufferBytes;
  if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &send_buffer_bytes, sizeof(send_buffer_bytes)) != 0) {
    static bool logged_sndbuf_unsupported = false;
    if (errno == ENOPROTOOPT) {
      if (!logged_sndbuf_unsupported) {
        ESP_LOGI(TAG, "UDP TX socket SO_SNDBUF is not supported on this target; continuing with lwIP defaults");
        logged_sndbuf_unsupported = true;
      }
    } else {
      ESP_LOGW(TAG, "Failed to set UDP TX socket send buffer to %d bytes (errno=%d)", send_buffer_bytes, errno);
    }
  }
}

int create_tx_socket() {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock < 0) {
    ESP_LOGW(TAG, "Failed to create UDP TX socket (errno=%d)", errno);
    return -1;
  }

  configure_tx_socket(sock);
  return sock;
}
}

bool CsiUdpSender::setup() {
  if (running_.load(std::memory_order_relaxed)) {
    return true;
  }

  slots_ = allocate_stream_storage<PacketSlot>("UDP sender packet slots", QUEUE_CAPACITY);
  if (slots_ == nullptr) {
    return false;
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
  sender_task_running_.store(true, std::memory_order_relaxed);
  if (xTaskCreate(&CsiUdpSender::sender_task_entry_,
                  "espectre_udp_tx",
                  4096,
                  this,
                  kSenderTaskPriority,
                  &sender_task_handle_) !=
      pdPASS) {
    ESP_LOGE(TAG, "Failed to create sender task");
    sender_task_running_.store(false, std::memory_order_relaxed);
    shutdown();
    return false;
  }

  return true;
}

void CsiUdpSender::shutdown() {
  running_.store(false, std::memory_order_relaxed);
  collector_enabled_.store(false, std::memory_order_relaxed);

  if (sender_task_handle_ != nullptr) {
    for (uint8_t attempt = 0U; attempt < 12U && sender_task_running_.load(std::memory_order_relaxed); attempt++) {
      vTaskDelay(pdMS_TO_TICKS(25));
    }
    if (sender_task_running_.load(std::memory_order_relaxed)) {
      ESP_LOGW(TAG, "UDP sender task did not stop cooperatively; forcing delete");
      vTaskDelete(sender_task_handle_);
      sender_task_running_.store(false, std::memory_order_relaxed);
    }
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
  free_stream_storage(slots_);
  slots_ = nullptr;
}

void CsiUdpSender::set_collector(const sockaddr_in &collector_addr, bool enabled) {
  collector_ip_addr_.store(collector_addr.sin_addr.s_addr, std::memory_order_relaxed);
  collector_port_.store(collector_addr.sin_port, std::memory_order_relaxed);
  collector_enabled_.store(enabled, std::memory_order_relaxed);
}

bool CsiUdpSender::queue_packet(const uint8_t *data, size_t data_len) {
  if (!running_.load(std::memory_order_relaxed) || data == nullptr || data_len == 0U || data_len > MAX_PACKET_BYTES ||
      free_slots_ == nullptr || ready_slots_ == nullptr || slots_ == nullptr) {
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
  slot.queued_at_ms = monotonic_now_ms();
  slot.queued = true;
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

uint32_t CsiUdpSender::oldest_ready_age_ms() const {
  if (slots_ == nullptr) {
    return 0U;
  }
  const uint32_t now_ms = monotonic_now_ms();
  uint32_t oldest_age_ms = 0U;
  for (uint8_t idx = 0U; idx < QUEUE_CAPACITY; idx++) {
    const PacketSlot &slot = slots_[idx];
    if (!slot.queued || slot.queued_at_ms == 0U || now_ms < slot.queued_at_ms) {
      continue;
    }
    const uint32_t age_ms = now_ms - slot.queued_at_ms;
    if (age_ms > oldest_age_ms) {
      oldest_age_ms = age_ms;
    }
  }
  return oldest_age_ms;
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
  TickType_t last_send_failure_log_ticks = 0U;

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

      if (batch_count >= MAX_BATCH_RECORDS || batch_len >= MAX_BATCH_BYTES) {
        batch_ready = true;
        continue;
      }

      const TickType_t wait_ticks =
          (batch_count == 1U && slot.packet_len <= kAggressiveBatchSmallPacketBytes) ? kAggressiveBatchWaitTicks : 0U;
      if (xQueueReceive(ready_slots_, &slot_idx, wait_ticks) != pdTRUE) {
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
      sock = create_tx_socket();
    }

    if (sock >= 0) {
      uint32_t batch_oldest_age_ms = 0U;
      const uint32_t now_ms = monotonic_now_ms();
      for (uint8_t idx = 0U; idx < batch_count; idx++) {
        const PacketSlot &slot = slots_[batch_slots[idx]];
        if (!slot.queued || slot.queued_at_ms == 0U || now_ms < slot.queued_at_ms) {
          continue;
        }
        const uint32_t age_ms = now_ms - slot.queued_at_ms;
        if (age_ms > batch_oldest_age_ms) {
          batch_oldest_age_ms = age_ms;
        }
      }
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
        last_tx_batch_age_ms_.store(batch_oldest_age_ms, std::memory_order_relaxed);
      } else {
        const int send_errno = errno;
        send_fail_total_.fetch_add(static_cast<uint64_t>(batch_count), std::memory_order_relaxed);
        last_fail_batch_age_ms_.store(batch_oldest_age_ms, std::memory_order_relaxed);
        const TickType_t now_ticks = xTaskGetTickCount();
        if (last_send_failure_log_ticks == 0U || (now_ticks - last_send_failure_log_ticks) >= kSendFailureLogIntervalTicks) {
          ESP_LOGW(TAG,
                   "sendto() failed rc=%d errno=%d batch_len=%u batch_count=%u batch_age_ms=%u",
                   rc,
                   send_errno,
                   static_cast<unsigned>(batch_len),
                   static_cast<unsigned>(batch_count),
                   static_cast<unsigned>(batch_oldest_age_ms));
          last_send_failure_log_ticks = now_ticks;
        }
        if (send_errno == ENOMEM || send_errno == ENOBUFS) {
          // Transient Wi-Fi/lwIP TX backpressure, not a socket error. Closing and
          // recreating the socket would not free the Wi-Fi TX buffer pool and only
          // churns lwIP netconns. Keep the socket open and yield briefly so the TX
          // pool can drain instead of hammering sendto().
          vTaskDelay(kSendBackpressureBackoffTicks);
        } else {
          close(sock);
          sock = -1;
        }
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
  sender_task_running_.store(false, std::memory_order_relaxed);
}

void CsiUdpSender::recycle_slot_(uint8_t slot_idx) {
  if (slots_ != nullptr && slot_idx < QUEUE_CAPACITY) {
    slots_[slot_idx].queued = false;
    slots_[slot_idx].queued_at_ms = 0U;
    slots_[slot_idx].packet_len = 0U;
  }
  if (free_slots_ != nullptr) {
    (void)xQueueSend(free_slots_, &slot_idx, 0);
  }
}

}  // namespace espectre
}  // namespace esphome
