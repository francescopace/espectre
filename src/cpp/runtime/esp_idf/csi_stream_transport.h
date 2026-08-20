/*
 * ESPectre - CSI Stream Transport
 *
 * Packages accepted CSI samples into UDP stream datagrams for collectors.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "csi_capture_service.h"
#include "csi_format.h"
#include "csi_stream_protocol.h"
#include "csi_traffic_service.h"

#include "freertos/FreeRTOS.h"
#if defined(ESP_PLATFORM)
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#endif
#include "lwip/sockets.h"

#if !defined(ESP_PLATFORM)
using portMUX_TYPE = int;
#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(mux) (void)(mux)
#define portEXIT_CRITICAL(mux) (void)(mux)
#endif

namespace espectre {

class CsiStreamTransport {
 public:
  static constexpr size_t kStreamRecordMaxBytes = sizeof(CsiStreamHeaderV7) + HT20_CSI_LEN;

  ~CsiStreamTransport();
  void configure(uint64_t device_id, uint16_t collector_port, uint32_t log_interval_ms, uint8_t tx_batch_records);
  void reset_session();
  void shutdown();
  void clear_ap_bssid();
  void set_ap_bssid(const uint8_t *bssid, size_t len);

  void handle_csi_packet(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized, bool streaming_ready);
  void handle_pacing_packet(const sockaddr_in &sender_addr, bool streaming_ready, uint32_t pacing_total);
  void update_from_traffic(const CsiTrafficService &traffic_service, bool streaming_ready);
  void log_runtime_telemetry(const CsiCaptureService &capture_service,
                             const CsiTrafficService &traffic_service,
                             bool streaming_ready,
                             const char *state_name);

 private:
  struct LatestCsiSample final {
    wifi_pkt_rx_ctrl_t rx_ctrl{};
    std::array<int8_t, HT20_CSI_LEN> csi{};
    uint16_t len{0U};
    bool first_word_invalid{false};
    bool valid{false};
    uint64_t update_total{0U};
    uint64_t captured_at_us{0U};
  };

  struct StreamRecordView final {
    const wifi_pkt_rx_ctrl_t *rx_ctrl{nullptr};
    const int8_t *csi{nullptr};
    uint16_t csi_len{0U};
    bool first_word_invalid{false};
  };

  static constexpr uint64_t kFreshSampleMaxAgeUs = 10000U;

  size_t build_stream_packet_(uint8_t *buffer, size_t buffer_len);
  size_t build_stream_packet_from_live_csi_(const wifi_pkt_rx_ctrl_t &rx_ctrl,
                                            bool first_word_invalid,
                                            const int8_t *normalized_csi,
                                            uint16_t normalized_len,
                                            uint32_t pacing_rx_total,
                                            uint8_t *buffer,
                                            size_t buffer_len);
  size_t build_stream_packet_from_sample_(const LatestCsiSample &sample,
                                          uint32_t pacing_rx_total,
                                          uint8_t *buffer,
                                          size_t buffer_len);
  size_t build_stream_packet_from_view_(const StreamRecordView &view,
                                        uint32_t pacing_rx_total,
                                        uint8_t *buffer,
                                        size_t buffer_len);
  bool ensure_stream_socket_();
  void close_stream_socket_();
  bool send_stream_datagram_();
  bool send_datagram_(const void *payload, size_t payload_len);
  bool flush_stream_batch_();
  void drop_stream_batch_();
  bool direct_credit_streaming_enabled_() const;
  bool consume_pending_pacing_credit_();
  bool ensure_direct_tx_worker_();
  void stop_direct_tx_worker_();
  void destroy_direct_tx_resources_();
  void reset_direct_tx_queue_();
  static void direct_tx_task_entry_(void *context);
  void run_direct_tx_task_();
  void reset_runtime_telemetry_baseline_(const CsiCaptureService &capture_service,
                                         const CsiTrafficService &traffic_service);

#if defined(ESP_PLATFORM)
  struct DirectTxSlot final {
    size_t packet_len{0U};
    std::array<uint8_t, kStreamRecordMaxBytes> packet{};
  };
  static constexpr uint8_t kDirectTxQueueSlots = 32U;
#endif

  uint64_t device_id_{0U};
  uint16_t collector_port_{5001U};
  uint32_t log_interval_ms_{1000U};
  uint8_t tx_batch_records_{1U};
  std::atomic<uint32_t> stream_seq_{0U};
  std::atomic<uint32_t> collector_ip_addr_{0U};
  std::array<uint8_t, 6> ap_bssid_{};

  portMUX_TYPE latch_lock_ = portMUX_INITIALIZER_UNLOCKED;
  LatestCsiSample latest_csi_{};
  uint64_t latest_csi_sent_total_{0U};
  int stream_sock_{-1};

  std::unique_ptr<uint8_t[]> batch_buffer_;
  size_t batch_capacity_{0U};
  size_t batch_len_{0U};
  uint8_t batch_records_pending_{0U};
  uint64_t batch_first_ms_{0U};
#if defined(ESP_PLATFORM)
  std::array<DirectTxSlot, kDirectTxQueueSlots> direct_tx_slots_{};
  QueueHandle_t direct_tx_free_slots_{nullptr};
  QueueHandle_t direct_tx_ready_slots_{nullptr};
  TaskHandle_t direct_tx_task_handle_{nullptr};
  SemaphoreHandle_t direct_tx_stopped_{nullptr};
  std::atomic<bool> direct_tx_task_running_{false};
#endif

  std::atomic<uint32_t> last_csi_ms_{0U};
  std::atomic<uint64_t> csi_callback_total_{0U};
  std::atomic<uint64_t> csi_accepted_total_{0U};
  std::atomic<uint64_t> csi_filtered_total_{0U};
  std::atomic<uint64_t> stream_fresh_total_{0U};
  std::atomic<uint64_t> stream_repeat_total_{0U};
  std::atomic<uint64_t> stream_tx_total_{0U};
  std::atomic<uint64_t> stream_tx_error_total_{0U};
  std::atomic<uint64_t> stream_tx_backpressure_total_{0U};
  std::atomic<uint32_t> pending_pacing_credits_{0U};
  uint64_t last_pacing_streamed_total_{0U};
  uint32_t last_pacing_credit_total_{0U};
  std::atomic<uint32_t> latest_pacing_rx_total_{0U};

  uint64_t last_log_ms_{0U};
  uint64_t prev_log_sample_ms_{0U};
  uint64_t prev_capture_callback_total_{0U};
  uint64_t prev_capture_valid_total_{0U};
  uint64_t prev_capture_invalid_total_{0U};
  uint64_t prev_csi_callback_total_{0U};
  uint64_t prev_csi_accepted_total_{0U};
  uint64_t prev_csi_filtered_total_{0U};
  uint64_t prev_stream_fresh_total_{0U};
  uint64_t prev_stream_repeat_total_{0U};
  uint64_t prev_traffic_rx_total_{0U};
  uint64_t prev_tx_success_total_{0U};
  uint64_t prev_tx_error_total_{0U};
  uint64_t prev_tx_backpressure_total_{0U};
  bool last_tx_backpressure_{false};
  bool telemetry_paused_no_traffic_{false};
};

}  // namespace espectre
