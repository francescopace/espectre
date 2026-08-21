/*
 * ESPectre - CSI Stream Transport
 *
 * Packages accepted CSI samples into UDP stream datagrams for collectors.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "csi_stream_transport.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <fcntl.h>

#include "counter_helpers.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "mac_address_helpers.h"
#include "runtime_time.h"
#include "sdkconfig.h"
#include "sta_socket_helpers.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace espectre {

namespace {

static const char *const TAG = "espectre.stream";
// Upper bound on how long a partial batch may wait for more pacing slots
// before it is flushed, so low pacing rates keep bounded record latency.
constexpr uint64_t kStreamBatchFlushMs = 100U;

float current_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

float minimum_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

size_t build_transport_csi_payload_(const int8_t *normalized_csi,
                                    size_t normalized_len,
                                    uint8_t *out_payload,
                                    uint16_t *out_num_subcarriers) {
  if (normalized_csi == nullptr || out_payload == nullptr || out_num_subcarriers == nullptr || normalized_len == 0U) {
    return 0U;
  }

  std::memcpy(out_payload, normalized_csi, normalized_len);
  *out_num_subcarriers = static_cast<uint16_t>(normalized_len / 2U);
  return normalized_len;
}

StreamChipType detect_chip_code() {
#if CONFIG_IDF_TARGET_ESP32C6
  return StreamChipType::C6;
#elif CONFIG_IDF_TARGET_ESP32C5
  return StreamChipType::C5;
#elif CONFIG_IDF_TARGET_ESP32C3
  return StreamChipType::C3;
#elif CONFIG_IDF_TARGET_ESP32S3
  return StreamChipType::S3;
#elif CONFIG_IDF_TARGET_ESP32
  return StreamChipType::ESP32;
#else
  return StreamChipType::UNKNOWN;
#endif
}

void format_ipv4_addr(uint32_t network_addr, char *buffer, size_t buffer_len) {
  if (buffer == nullptr || buffer_len == 0U) {
    return;
  }
  if (network_addr == 0U) {
    std::snprintf(buffer, buffer_len, "0.0.0.0");
    return;
  }

  const uint32_t host_addr = ntohl(network_addr);
  std::snprintf(buffer,
                buffer_len,
                "%u.%u.%u.%u",
                static_cast<unsigned>((host_addr >> 24U) & 0xFFU),
                static_cast<unsigned>((host_addr >> 16U) & 0xFFU),
                static_cast<unsigned>((host_addr >> 8U) & 0xFFU),
                static_cast<unsigned>(host_addr & 0xFFU));
}

int create_stream_socket() {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock < 0) {
    ESP_LOGW(TAG, "Failed to create stream socket (errno=%d)", errno);
    return -1;
  }

  if (!bind_socket_to_sta_interface(sock, TAG, "stream")) {
    ESP_LOGW(TAG, "Continuing without explicit stream socket binding");
  }

  const int flags = fcntl(sock, F_GETFL, 0);
  if (flags >= 0) {
    if (fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
      ESP_LOGW(TAG, "Failed to set stream socket non-blocking (errno=%d)", errno);
    }
  } else {
    ESP_LOGW(TAG, "Failed to read stream socket flags (errno=%d)", errno);
  }

  return sock;
}

#if CONFIG_IDF_TARGET_ESP32C3
typedef struct {
  signed rssi : 8;
  unsigned rate : 5;
  unsigned : 1;
  unsigned sig_mode : 2;
  unsigned : 16;
  unsigned mcs : 7;
  unsigned cwb : 1;
  unsigned : 16;
  unsigned smoothing : 1;
  unsigned not_sounding : 1;
  unsigned : 1;
  unsigned aggregation : 1;
  unsigned stbc : 2;
  unsigned fec_coding : 1;
  unsigned sgi : 1;
  unsigned : 8;
  unsigned ampdu_cnt : 8;
  unsigned channel : 4;
  unsigned secondary_channel : 4;
  unsigned rxstart_time_cyc : 7;
  unsigned : 1;
  unsigned timestamp : 32;
  unsigned : 32;
  signed noise_floor : 8;
  unsigned : 24;
  unsigned : 32;
  unsigned : 20;
  unsigned rxstart_time_cyc_dec : 11;
  unsigned ant : 1;
  unsigned : 32;
  unsigned : 32;
  unsigned : 32;
  unsigned sig_len : 12;
  unsigned : 12;
  unsigned rx_state : 8;
} wifi_pkt_rx_ctrl_time_t;

static_assert(sizeof(wifi_pkt_rx_ctrl_time_t) == sizeof(wifi_pkt_rx_ctrl_t),
              "timestamp overlay must match wifi_pkt_rx_ctrl_t");
#endif

struct StreamPhyMetadata {
  StreamPhyMode mode{StreamPhyMode::UNKNOWN};
  StreamLtfType ltf_type{StreamLtfType::UNKNOWN};
  StreamChannelWidth channel_width{StreamChannelWidth::UNKNOWN};
};

StreamPhyMetadata extract_phy_metadata(const wifi_pkt_rx_ctrl_t &rx_ctrl) {
  StreamPhyMetadata metadata;

#if CONFIG_SOC_WIFI_HE_SUPPORT
  switch (rx_ctrl.cur_bb_format) {
    case RX_BB_FORMAT_HT:
      metadata.mode = StreamPhyMode::HT;
      metadata.ltf_type = StreamLtfType::HT_LTF;
      metadata.channel_width =
          rx_ctrl.second == 0U ? StreamChannelWidth::MHZ_20 : StreamChannelWidth::MHZ_40;
      break;
    case RX_BB_FORMAT_VHT:
    case RX_BB_FORMAT_VHT_MU:
      metadata.mode = StreamPhyMode::VHT;
      metadata.ltf_type = StreamLtfType::VHT_LTF;
      break;
    case RX_BB_FORMAT_HE_SU:
      metadata.mode = StreamPhyMode::HE_SU;
      metadata.ltf_type = StreamLtfType::HE_LTF;
      break;
    case RX_BB_FORMAT_HE_MU:
      metadata.mode = StreamPhyMode::HE_MU;
      metadata.ltf_type = StreamLtfType::HE_LTF;
      break;
    case RX_BB_FORMAT_HE_ERSU:
      metadata.mode = StreamPhyMode::HE_ERSU;
      metadata.ltf_type = StreamLtfType::HE_LTF;
      break;
    case RX_BB_FORMAT_HE_TB:
      metadata.mode = StreamPhyMode::HE_TB;
      metadata.ltf_type = StreamLtfType::HE_LTF;
      break;
    default:
      break;
  }
#else
  switch (rx_ctrl.sig_mode) {
    case 1U:
      metadata.mode = StreamPhyMode::HT;
      metadata.ltf_type = StreamLtfType::HT_LTF;
      metadata.channel_width =
          rx_ctrl.cwb == 0U ? StreamChannelWidth::MHZ_20 : StreamChannelWidth::MHZ_40;
      break;
    case 3U:
      metadata.mode = StreamPhyMode::VHT;
      metadata.ltf_type = StreamLtfType::VHT_LTF;
      break;
    default:
      break;
  }
#endif

  return metadata;
}

bool fill_rx_timestamp_metadata(const wifi_pkt_rx_ctrl_t &rx_ctrl, CsiStreamHeaderV7 *header) {
  if (header == nullptr) {
    return false;
  }

  header->wifi_rx_start_ts_ns = 0U;
#if CONFIG_IDF_TARGET_ESP32C3
  const auto *time_info = reinterpret_cast<const wifi_pkt_rx_ctrl_time_t *>(&rx_ctrl);
  if (time_info->timestamp == 0U) {
    return false;
  }

  const uint16_t cyc_dec = (time_info->rxstart_time_cyc_dec >= 1024U)
                               ? static_cast<uint16_t>(2048U - time_info->rxstart_time_cyc_dec)
                               : static_cast<uint16_t>(time_info->rxstart_time_cyc_dec);
  const uint64_t coarse_ns = static_cast<uint64_t>(time_info->timestamp) * 1000ULL;
  const uint64_t cyc_ns = (static_cast<uint64_t>(time_info->rxstart_time_cyc) * 12500ULL) / 1000ULL;
  const uint64_t cyc_dec_ns = (static_cast<uint64_t>(cyc_dec) * 1562ULL) / 1000ULL;
  constexpr uint64_t kAlignmentNs = 20800ULL;
  if (coarse_ns + cyc_ns + cyc_dec_ns <= kAlignmentNs) {
    return false;
  }

  header->wifi_rx_start_ts_ns = coarse_ns + cyc_ns + cyc_dec_ns - kAlignmentNs;
  return true;
#else
  (void)rx_ctrl;
  return false;
#endif
}

// Every ESP-IDF Wi-Fi target reports the RF noise floor: classic MACs expose it
// through `wifi_pkt_rx_ctrl_t`, Wi-Fi 6 parts through `esp_wifi_rxctrl_t`, which
// `wifi_pkt_rx_ctrl_t` aliases when `SOC_WIFI_HE_SUPPORT` is set. Selecting it
// from a hand-maintained target list silently reported the invalid sentinel on
// C5 and C6, so read it unconditionally instead.
int8_t rx_ctrl_noise_floor_dbm(const wifi_pkt_rx_ctrl_t &rx_ctrl) {
  return static_cast<int8_t>(rx_ctrl.noise_floor);
}

}  // namespace

CsiStreamTransport::~CsiStreamTransport() { shutdown(); }

void CsiStreamTransport::configure(uint64_t device_id,
                                   uint16_t collector_port,
                                   uint32_t log_interval_ms,
                                   uint8_t tx_batch_records) {
  device_id_ = device_id;
  collector_port_ = collector_port;
  log_interval_ms_ = log_interval_ms;
#if CONFIG_IDF_TARGET_ESP32
  tx_batch_records_ = 1U;
  if (tx_batch_records > 1U) {
    ESP_LOGI(TAG, "Original ESP32 forces 1-record stream datagrams to avoid stale CSI batching");
  }
#else
  tx_batch_records_ = std::clamp<uint8_t>(tx_batch_records,
                                          RUNTIME_STREAM_TX_BATCH_RECORDS_MIN,
                                          RUNTIME_STREAM_TX_BATCH_RECORDS_MAX);
#endif
  batch_capacity_ = static_cast<size_t>(tx_batch_records_) * kStreamRecordMaxBytes;
  batch_buffer_.reset(new (std::nothrow) uint8_t[batch_capacity_]);
  if (!batch_buffer_) {
    batch_capacity_ = 0U;
    ESP_LOGE(TAG, "Failed to allocate %u-record stream batch", static_cast<unsigned>(tx_batch_records_));
  }
  if (direct_credit_streaming_enabled_() && !ensure_direct_tx_worker_()) {
    ESP_LOGE(TAG, "Failed to start the direct CSI streaming worker");
  }
}

void CsiStreamTransport::reset_session() {
  close_stream_socket_();
  drop_stream_batch_();
  collector_ip_addr_.store(0U, std::memory_order_relaxed);
  stream_seq_.store(0U, std::memory_order_relaxed);
  last_pacing_streamed_total_ = 0U;
  last_csi_ms_.store(0U, std::memory_order_relaxed);
  csi_callback_total_.store(0U, std::memory_order_relaxed);
  csi_accepted_total_.store(0U, std::memory_order_relaxed);
  csi_filtered_total_.store(0U, std::memory_order_relaxed);
  stream_fresh_total_.store(0U, std::memory_order_relaxed);
  stream_repeat_total_.store(0U, std::memory_order_relaxed);
  stream_tx_total_.store(0U, std::memory_order_relaxed);
  stream_tx_error_total_.store(0U, std::memory_order_relaxed);
  stream_tx_backpressure_total_.store(0U, std::memory_order_relaxed);
  pending_pacing_credits_.store(0U, std::memory_order_relaxed);
  latest_pacing_rx_total_.store(0U, std::memory_order_relaxed);
  last_pacing_credit_total_ = 0U;
  last_log_ms_ = 0U;
  prev_log_sample_ms_ = 0U;
  prev_capture_callback_total_ = 0U;
  prev_capture_valid_total_ = 0U;
  prev_capture_invalid_total_ = 0U;
  prev_csi_callback_total_ = 0U;
  prev_csi_accepted_total_ = 0U;
  prev_csi_filtered_total_ = 0U;
  prev_stream_fresh_total_ = 0U;
  prev_stream_repeat_total_ = 0U;
  prev_traffic_rx_total_ = 0U;
  prev_tx_success_total_ = 0U;
  prev_tx_error_total_ = 0U;
  prev_tx_backpressure_total_ = 0U;
  last_tx_backpressure_ = false;
  telemetry_paused_no_traffic_ = false;

  portENTER_CRITICAL(&latch_lock_);
  latest_csi_ = LatestCsiSample{};
  latest_csi_sent_total_ = 0U;
  portEXIT_CRITICAL(&latch_lock_);

  reset_direct_tx_queue_();
}

void CsiStreamTransport::shutdown() {
  close_stream_socket_();
  drop_stream_batch_();
  stop_direct_tx_worker_();
}

void CsiStreamTransport::clear_ap_bssid() { ap_bssid_.fill(0U); }

void CsiStreamTransport::set_ap_bssid(const uint8_t *bssid, size_t len) {
  ap_bssid_.fill(0U);
  if (bssid == nullptr || len < ap_bssid_.size()) {
    return;
  }
  std::memcpy(ap_bssid_.data(), bssid, ap_bssid_.size());
}

void CsiStreamTransport::handle_csi_packet(const wifi_csi_info_t *info,
                                           const NormalizedCSIPayload &normalized,
                                           bool streaming_ready) {
  csi_callback_total_.fetch_add(1U, std::memory_order_relaxed);
  if (info == nullptr) {
    return;
  }
  if (!normalized.valid() || normalized.len == 0U || normalized.len > HT20_CSI_LEN) {
    return;
  }
  if (!streaming_ready) {
    return;
  }

  if (!is_zero_mac_address(ap_bssid_.data()) && std::memcmp(info->mac, ap_bssid_.data(), ap_bssid_.size()) != 0) {
    csi_filtered_total_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  csi_accepted_total_.fetch_add(1U, std::memory_order_relaxed);
  last_csi_ms_.store(monotonic_now_ms(), std::memory_order_relaxed);

  if (direct_credit_streaming_enabled_()) {
    if (!consume_pending_pacing_credit_()) {
      return;
    }

#if defined(ESP_PLATFORM)
    if (!direct_tx_task_running_.load(std::memory_order_acquire)) {
      stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }

    uint8_t slot_idx = 0U;
    if (xQueueReceive(direct_tx_free_slots_, &slot_idx, 0) != pdTRUE) {
      stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
      stream_tx_backpressure_total_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }

    DirectTxSlot &slot = direct_tx_slots_[slot_idx];
    const size_t record_len =
        build_stream_packet_from_live_csi_(info->rx_ctrl,
                                           info->first_word_invalid,
                                           normalized.data,
                                           static_cast<uint16_t>(normalized.len),
                                           latest_pacing_rx_total_.load(std::memory_order_relaxed),
                                           slot.packet.data(),
                                           slot.packet.size());
    if (record_len == 0U) {
      (void)xQueueSend(direct_tx_free_slots_, &slot_idx, 0);
      return;
    }
    slot.packet_len = record_len;
    if (xQueueSend(direct_tx_ready_slots_, &slot_idx, 0) != pdTRUE) {
      stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
      stream_tx_backpressure_total_.fetch_add(1U, std::memory_order_relaxed);
      (void)xQueueSend(direct_tx_free_slots_, &slot_idx, 0);
    }
    return;
#else
    if (!ensure_stream_socket_() || !batch_buffer_) {
      stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }

    const size_t record_len =
        build_stream_packet_from_live_csi_(info->rx_ctrl,
                                           info->first_word_invalid,
                                           normalized.data,
                                           static_cast<uint16_t>(normalized.len),
                                           latest_pacing_rx_total_.load(std::memory_order_relaxed),
                                           batch_buffer_.get(),
                                           batch_capacity_);
    if (record_len == 0U) {
      return;
    }
    (void)send_datagram_(batch_buffer_.get(), record_len);
    return;
#endif
  }

  portENTER_CRITICAL(&latch_lock_);
  latest_csi_.rx_ctrl = info->rx_ctrl;
  std::memcpy(latest_csi_.csi.data(), normalized.data, normalized.len);
  latest_csi_.len = static_cast<uint16_t>(normalized.len);
  latest_csi_.first_word_invalid = info->first_word_invalid;
  latest_csi_.valid = true;
  latest_csi_.update_total++;
  latest_csi_.captured_at_us = monotonic_now_us();
  portEXIT_CRITICAL(&latch_lock_);
}

void CsiStreamTransport::handle_pacing_packet(const sockaddr_in &sender_addr,
                                              bool streaming_ready,
                                              uint32_t pacing_total) {
  if (sender_addr.sin_addr.s_addr != 0U &&
      sender_addr.sin_addr.s_addr != collector_ip_addr_.load(std::memory_order_relaxed)) {
    collector_ip_addr_.store(sender_addr.sin_addr.s_addr, std::memory_order_relaxed);

    char addr_text[16];
    format_ipv4_addr(collector_ip_addr_.load(std::memory_order_relaxed), addr_text, sizeof(addr_text));
    ESP_LOGI(TAG,
             "Collector learned from UDP pacing: address=%s pacing_port=%u stream_port=%u",
             addr_text,
             static_cast<unsigned>(ntohs(sender_addr.sin_port)),
             static_cast<unsigned>(collector_port_));
  }

  latest_pacing_rx_total_.store(pacing_total, std::memory_order_relaxed);
  if (direct_credit_streaming_enabled_()) {
    const uint32_t credit_delta = pacing_total >= last_pacing_credit_total_
                                      ? (pacing_total - last_pacing_credit_total_)
                                      : pacing_total;
    pending_pacing_credits_.fetch_add(credit_delta, std::memory_order_relaxed);
    last_pacing_credit_total_ = pacing_total;
    if (!streaming_ready || collector_ip_addr_.load(std::memory_order_relaxed) == 0U) {
      pending_pacing_credits_.store(0U, std::memory_order_relaxed);
    }
    return;
  }

  const bool should_stream = collector_ip_addr_.load(std::memory_order_relaxed) != 0U && streaming_ready;
  if (!should_stream) {
    drop_stream_batch_();
    last_pacing_streamed_total_ = pacing_total;
    return;
  }
  if (!ensure_stream_socket_()) {
    stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
    last_pacing_streamed_total_ = pacing_total;
    return;
  }
  (void)send_stream_datagram_();
  last_pacing_streamed_total_ = pacing_total;
}

void CsiStreamTransport::update_from_traffic(const CsiTrafficService &traffic_service, bool streaming_ready) {
  if (direct_credit_streaming_enabled_()) {
    (void)traffic_service;
    (void)streaming_ready;
    if (batch_records_pending_ > 0U) {
      drop_stream_batch_();
    }
    return;
  }

  sockaddr_in sender_addr{};
  if (traffic_service.get_last_sender(&sender_addr) && sender_addr.sin_addr.s_addr != 0U &&
      sender_addr.sin_addr.s_addr != collector_ip_addr_.load(std::memory_order_relaxed)) {
    collector_ip_addr_.store(sender_addr.sin_addr.s_addr, std::memory_order_relaxed);

    char addr_text[16];
    format_ipv4_addr(collector_ip_addr_.load(std::memory_order_relaxed), addr_text, sizeof(addr_text));
    ESP_LOGI(TAG,
             "Collector learned from UDP pacing: address=%s pacing_port=%u stream_port=%u",
             addr_text,
             static_cast<unsigned>(ntohs(sender_addr.sin_port)),
             static_cast<unsigned>(collector_port_));
  }

  const uint64_t pacing_rx_total = traffic_service.get_packets_received();
  latest_pacing_rx_total_.store(static_cast<uint32_t>(pacing_rx_total), std::memory_order_relaxed);
  const bool should_stream = collector_ip_addr_.load(std::memory_order_relaxed) != 0U && streaming_ready;
  if (!should_stream) {
    drop_stream_batch_();
    last_pacing_streamed_total_ = pacing_rx_total;
    return;
  }

  const uint64_t pending_packets = pacing_rx_total >= last_pacing_streamed_total_
                                       ? pacing_rx_total - last_pacing_streamed_total_
                                       : pacing_rx_total;
  if (pending_packets > 0U) {
    if (!ensure_stream_socket_()) {
      stream_tx_error_total_.fetch_add(pending_packets, std::memory_order_relaxed);
    } else {
      for (uint64_t idx = 0U; idx < pending_packets; idx++) {
        if (send_stream_datagram_()) {
          continue;
        }
        if (last_tx_backpressure_) {
          const uint64_t unsent_packets = pending_packets - idx - 1U;
          stream_tx_error_total_.fetch_add(unsent_packets, std::memory_order_relaxed);
          stream_tx_backpressure_total_.fetch_add(unsent_packets, std::memory_order_relaxed);
          break;
        }
      }
    }
    last_pacing_streamed_total_ = pacing_rx_total;
  }

  if (batch_records_pending_ > 0U &&
      static_cast<uint64_t>(monotonic_now_ms()) - batch_first_ms_ >= kStreamBatchFlushMs) {
    flush_stream_batch_();
  }
}

void CsiStreamTransport::log_runtime_telemetry(const CsiCaptureService &capture_service,
                                               const CsiTrafficService &traffic_service,
                                               bool streaming_ready,
                                               const char *state_name) {
  (void)state_name;
  const uint64_t now_ms = static_cast<uint64_t>(monotonic_now_ms());
  if (last_log_ms_ != 0U && now_ms - last_log_ms_ < log_interval_ms_) {
    return;
  }
  if (!streaming_ready) {
    last_log_ms_ = now_ms;
    return;
  }
  if (prev_log_sample_ms_ == 0U) {
    reset_runtime_telemetry_baseline_(capture_service, traffic_service);
    prev_log_sample_ms_ = now_ms;
  }

  const uint64_t dt_ms = std::max<uint64_t>(1U, now_ms - prev_log_sample_ms_);
  const uint64_t capture_callback_total = capture_service.callback_invocations();
  const uint64_t capture_valid_total = capture_service.valid_packets();
  const uint64_t capture_invalid_total = capture_service.normalized_invalid_packets();
  const uint64_t csi_callback_total = csi_callback_total_.load(std::memory_order_relaxed);
  const uint64_t csi_accepted_total = csi_accepted_total_.load(std::memory_order_relaxed);
  const uint64_t csi_filtered_total = csi_filtered_total_.load(std::memory_order_relaxed);
  const uint64_t stream_fresh_total = stream_fresh_total_.load(std::memory_order_relaxed);
  const uint64_t stream_repeat_total = stream_repeat_total_.load(std::memory_order_relaxed);
  const uint64_t traffic_valid_total = traffic_service.get_packets_received();
  const uint64_t traffic_rx_delta = counter_delta(traffic_valid_total, prev_traffic_rx_total_);
  const uint64_t tx_success_total = stream_tx_total_.load(std::memory_order_relaxed);
  const uint64_t tx_error_total = stream_tx_error_total_.load(std::memory_order_relaxed);
  const uint64_t tx_backpressure_total = stream_tx_backpressure_total_.load(std::memory_order_relaxed);
  const uint64_t tx_error_delta = counter_delta(tx_error_total, prev_tx_error_total_);
  const uint64_t tx_backpressure_delta = counter_delta(tx_backpressure_total, prev_tx_backpressure_total_);

  if (traffic_rx_delta == 0U) {
    if (!telemetry_paused_no_traffic_) {
      ESP_LOGI(TAG, "UDP pacing idle, suspending periodic stream telemetry");
      telemetry_paused_no_traffic_ = true;
    }
    reset_runtime_telemetry_baseline_(capture_service, traffic_service);
    last_log_ms_ = now_ms;
    return;
  }
  if (telemetry_paused_no_traffic_) {
    ESP_LOGI(TAG, "UDP pacing resumed, stream telemetry re-enabled");
    telemetry_paused_no_traffic_ = false;
  }

  const auto to_pps = [dt_ms](uint64_t delta) { return static_cast<float>(delta) * 1000.0F / static_cast<float>(dt_ms); };
  const float capture_callback_pps = to_pps(counter_delta(capture_callback_total, prev_capture_callback_total_));
  const float capture_valid_pps = to_pps(counter_delta(capture_valid_total, prev_capture_valid_total_));
  const float capture_invalid_pps = to_pps(counter_delta(capture_invalid_total, prev_capture_invalid_total_));
  const float csi_callback_pps = to_pps(counter_delta(csi_callback_total, prev_csi_callback_total_));
  const float csi_accepted_pps = to_pps(counter_delta(csi_accepted_total, prev_csi_accepted_total_));
  const float csi_filtered_pps = to_pps(counter_delta(csi_filtered_total, prev_csi_filtered_total_));
  const float stream_fresh_pps = to_pps(counter_delta(stream_fresh_total, prev_stream_fresh_total_));
  const float stream_repeat_pps = to_pps(counter_delta(stream_repeat_total, prev_stream_repeat_total_));
  const float traffic_rx_pps = to_pps(counter_delta(traffic_valid_total, prev_traffic_rx_total_));
  const float tx_pps = to_pps(counter_delta(tx_success_total, prev_tx_success_total_));
  const float tx_error_pps = to_pps(counter_delta(tx_error_total, prev_tx_error_total_));
  const float tx_backpressure_pps = to_pps(counter_delta(tx_backpressure_total, prev_tx_backpressure_total_));

  const uint32_t last_csi_ms = last_csi_ms_.load(std::memory_order_relaxed);
  const uint32_t csi_age_ms =
      (last_csi_ms > 0U && now_ms >= last_csi_ms) ? static_cast<uint32_t>(now_ms - last_csi_ms) : 0U;
  const uint32_t csi_capture_bad_sc = static_cast<uint32_t>(capture_invalid_total);
  const uint32_t csi_capture_valid = static_cast<uint32_t>(capture_valid_total);

  const bool csi_enabled = capture_service.is_enabled();
  const char *csi_health = !csi_enabled                  ? "not_armed"
                          : capture_service.callback_invocations() == 0U ? "armed_no_callback"
                          : csi_capture_valid == 0U      ? "callback_no_valid_packets"
                          : csi_callback_pps < 1.0F      ? "callback_silent"
                                                         : "ok";

  if (std::strcmp(csi_health, "ok") == 0 && tx_error_delta == 0U && tx_backpressure_delta == 0U) {
    ESP_LOGI(TAG,
             "drv_cb=%.0f cap_valid=%.0f csi_ap=%.0f udp_rx=%.1f udp_tx=%.0f fresh=%.0f age_ms=%" PRIu32,
             static_cast<double>(capture_callback_pps),
             static_cast<double>(capture_valid_pps),
             static_cast<double>(csi_accepted_pps),
             static_cast<double>(traffic_rx_pps),
             static_cast<double>(tx_pps),
             static_cast<double>(stream_fresh_pps),
             csi_age_ms);
  } else {
    ESP_LOGI(TAG,
             "drv_cb=%.0f cap_valid=%.0f cap_bad=%.0f csi_ap=%.0f csi_filt=%.0f valid=%" PRIu32
             " bad_sc=%" PRIu32
             " udp_rx=%.1f udp_tx=%.0f fresh=%.0f repeat=%.0f"
             " tx_err=%.0f/%" PRIu64 " tx_bp=%.0f/%" PRIu64
             " age_ms=%" PRIu32 " heap=%.1f min=%.1f",
             static_cast<double>(capture_callback_pps),
             static_cast<double>(capture_valid_pps),
             static_cast<double>(capture_invalid_pps),
             static_cast<double>(csi_accepted_pps),
             static_cast<double>(csi_filtered_pps),
             csi_capture_valid,
             csi_capture_bad_sc,
             static_cast<double>(traffic_rx_pps),
             static_cast<double>(tx_pps),
             static_cast<double>(stream_fresh_pps),
             static_cast<double>(stream_repeat_pps),
             static_cast<double>(tx_error_pps),
             tx_error_total,
             static_cast<double>(tx_backpressure_pps),
             tx_backpressure_total,
             csi_age_ms,
             static_cast<double>(current_free_memory_kb()),
             static_cast<double>(minimum_free_memory_kb()));
  }

  prev_capture_callback_total_ = capture_callback_total;
  prev_capture_valid_total_ = capture_valid_total;
  prev_capture_invalid_total_ = capture_invalid_total;
  prev_csi_callback_total_ = csi_callback_total;
  prev_csi_accepted_total_ = csi_accepted_total;
  prev_csi_filtered_total_ = csi_filtered_total;
  prev_stream_fresh_total_ = stream_fresh_total;
  prev_stream_repeat_total_ = stream_repeat_total;
  prev_traffic_rx_total_ = traffic_valid_total;
  prev_tx_success_total_ = tx_success_total;
  prev_tx_error_total_ = tx_error_total;
  prev_tx_backpressure_total_ = tx_backpressure_total;
  prev_log_sample_ms_ = now_ms;
  last_log_ms_ = now_ms;
}

size_t CsiStreamTransport::build_stream_packet_(uint8_t *buffer, size_t buffer_len) {
  if (buffer == nullptr || buffer_len < kStreamRecordMaxBytes) {
    return 0U;
  }

  LatestCsiSample sample;
  bool fresh = false;
  const uint64_t now_us = monotonic_now_us();
  portENTER_CRITICAL(&latch_lock_);
  const bool valid = latest_csi_.valid;
  if (valid) {
    sample = latest_csi_;
    fresh = latest_csi_.update_total != latest_csi_sent_total_;
    latest_csi_sent_total_ = latest_csi_.update_total;
  }
  portEXIT_CRITICAL(&latch_lock_);

  if (!valid) {
    return 0U;
  }
  if (!fresh) {
    if (csi_accepted_total_.load(std::memory_order_relaxed) != 0U) {
      stream_repeat_total_.fetch_add(1U, std::memory_order_relaxed);
    }
    return 0U;
  }
  if (sample.captured_at_us == 0U || now_us < sample.captured_at_us ||
      (now_us - sample.captured_at_us) > kFreshSampleMaxAgeUs) {
    if (csi_accepted_total_.load(std::memory_order_relaxed) != 0U) {
      stream_repeat_total_.fetch_add(1U, std::memory_order_relaxed);
    }
    return 0U;
  }

  return build_stream_packet_from_sample_(sample, latest_pacing_rx_total_.load(std::memory_order_relaxed), buffer,
                                          buffer_len);
}

size_t CsiStreamTransport::build_stream_packet_from_live_csi_(const wifi_pkt_rx_ctrl_t &rx_ctrl,
                                                              bool first_word_invalid,
                                                              const int8_t *normalized_csi,
                                                              uint16_t normalized_len,
                                                              uint32_t pacing_rx_total,
                                                              uint8_t *buffer,
                                                              size_t buffer_len) {
  return build_stream_packet_from_view_(
      StreamRecordView{&rx_ctrl, normalized_csi, normalized_len, first_word_invalid},
      pacing_rx_total, buffer, buffer_len);
}

size_t CsiStreamTransport::build_stream_packet_from_sample_(const LatestCsiSample &sample,
                                                            uint32_t pacing_rx_total,
                                                            uint8_t *buffer,
                                                            size_t buffer_len) {
  if (!sample.valid) {
    return 0U;
  }
  return build_stream_packet_from_view_(
      StreamRecordView{&sample.rx_ctrl, sample.csi.data(), sample.len, sample.first_word_invalid},
      pacing_rx_total, buffer, buffer_len);
}

size_t CsiStreamTransport::build_stream_packet_from_view_(const StreamRecordView &view,
                                                          uint32_t pacing_rx_total,
                                                          uint8_t *buffer,
                                                          size_t buffer_len) {
  if (buffer == nullptr || buffer_len < kStreamRecordMaxBytes || view.rx_ctrl == nullptr ||
      view.csi == nullptr || view.csi_len == 0U || view.csi_len > HT20_CSI_LEN) {
    return 0U;
  }
  const wifi_pkt_rx_ctrl_t &rx_ctrl = *view.rx_ctrl;

  auto *header = reinterpret_cast<CsiStreamHeaderV7 *>(buffer);
  *header = CsiStreamHeaderV7{};
  header->magic = STREAM_MAGIC;
  header->version = STREAM_VERSION;
  header->header_len = static_cast<uint8_t>(sizeof(*header));
  header->chip = static_cast<uint8_t>(detect_chip_code());
  header->flags = 0U;
  header->seq_num = stream_seq_.fetch_add(1U, std::memory_order_relaxed);
  header->device_id = device_id_;
  header->device_ticks_us = monotonic_now_us();
  header->wifi_rx_ts_us = rx_ctrl.timestamp;
  header->wifi_rx_start_ts_ns = 0U;
  header->channel = rx_ctrl.channel;
  header->rssi_dbm = rx_ctrl.rssi;
  header->noise_floor_dbm = rx_ctrl_noise_floor_dbm(rx_ctrl);
  header->tx_backpressure_total = static_cast<uint32_t>(stream_tx_backpressure_total_.load(std::memory_order_relaxed));
  header->stream_fresh_total =
      static_cast<uint32_t>(stream_fresh_total_.load(std::memory_order_relaxed) + 1U);
  header->pacing_rx_total = pacing_rx_total;
  const StreamPhyMetadata phy = extract_phy_metadata(rx_ctrl);
  header->phy_mode = static_cast<uint8_t>(phy.mode);
  header->ltf_type = static_cast<uint8_t>(phy.ltf_type);
  header->channel_width = static_cast<uint8_t>(phy.channel_width);

  if (view.first_word_invalid) {
    header->flags |= STREAM_FLAG_FIRST_WORD_INVALID;
  }
  if (header->wifi_rx_ts_us != 0U) {
    header->flags |= STREAM_FLAG_WIFI_RX_TS_VALID;
  }
  if (fill_rx_timestamp_metadata(rx_ctrl, header)) {
    header->flags |= STREAM_FLAG_WIFI_RX_START_TS_NS_VALID;
  }
  header->flags |= STREAM_FLAG_CSI_FRESH;
  stream_fresh_total_.fetch_add(1U, std::memory_order_relaxed);

  header->csi_len_bytes = static_cast<uint16_t>(
      build_transport_csi_payload_(view.csi, view.csi_len, buffer + sizeof(*header), &header->num_subcarriers));
  if (header->csi_len_bytes == 0U || header->num_subcarriers == 0U) {
    return 0U;
  }
  return sizeof(*header) + header->csi_len_bytes;
}

bool CsiStreamTransport::ensure_stream_socket_() {
  if (stream_sock_ >= 0) {
    return true;
  }

  stream_sock_ = create_stream_socket();
  return stream_sock_ >= 0;
}

void CsiStreamTransport::close_stream_socket_() {
  if (stream_sock_ >= 0) {
    close(stream_sock_);
    stream_sock_ = -1;
  }
}

bool CsiStreamTransport::send_stream_datagram_() {
  last_tx_backpressure_ = false;
  if (collector_ip_addr_.load(std::memory_order_relaxed) == 0U || !batch_buffer_) {
    return false;
  }
  if (!ensure_stream_socket_()) {
    stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  const size_t record_len = build_stream_packet_(batch_buffer_.get() + batch_len_, batch_capacity_ - batch_len_);
  if (record_len == 0U) {
    return true;
  }

  if (batch_records_pending_ == 0U) {
    batch_first_ms_ = static_cast<uint64_t>(monotonic_now_ms());
  }
  batch_len_ += record_len;
  batch_records_pending_++;
  if (batch_records_pending_ < tx_batch_records_) {
    return true;
  }
  return flush_stream_batch_();
}

bool CsiStreamTransport::flush_stream_batch_() {
  if (batch_records_pending_ == 0U) {
    return true;
  }
  const bool sent = send_datagram_(batch_buffer_.get(), batch_len_);
  drop_stream_batch_();
  return sent;
}

void CsiStreamTransport::drop_stream_batch_() {
  batch_len_ = 0U;
  batch_records_pending_ = 0U;
  batch_first_ms_ = 0U;
}

bool CsiStreamTransport::send_datagram_(const void *payload, size_t payload_len) {
  sockaddr_in collector_addr{};
  collector_addr.sin_family = AF_INET;
  collector_addr.sin_addr.s_addr = collector_ip_addr_.load(std::memory_order_relaxed);
  collector_addr.sin_port = htons(collector_port_);
  const ssize_t sent =
      sendto(stream_sock_, payload, payload_len, 0, reinterpret_cast<const sockaddr *>(&collector_addr), sizeof(collector_addr));
  if (sent == static_cast<ssize_t>(payload_len)) {
    stream_tx_total_.fetch_add(1U, std::memory_order_relaxed);
    return true;
  }

  const int send_errno = errno;
  stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
  const bool backpressure =
      send_errno == ENOMEM || send_errno == ENOBUFS || send_errno == EAGAIN || send_errno == EWOULDBLOCK;
  if (backpressure) {
    last_tx_backpressure_ = true;
    stream_tx_backpressure_total_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  close_stream_socket_();
  return false;
}

bool CsiStreamTransport::direct_credit_streaming_enabled_() const {
#if CONFIG_IDF_TARGET_ESP32C3
  return true;
#else
  return false;
#endif
}

bool CsiStreamTransport::consume_pending_pacing_credit_() {
  uint32_t available = pending_pacing_credits_.load(std::memory_order_relaxed);
  while (available > 0U) {
    if (pending_pacing_credits_.compare_exchange_weak(available, available - 1U, std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool CsiStreamTransport::ensure_direct_tx_worker_() {
#if defined(ESP_PLATFORM)
  if (!direct_credit_streaming_enabled_()) {
    return false;
  }
  if (direct_tx_task_running_.load(std::memory_order_relaxed)) {
    return true;
  }

  destroy_direct_tx_resources_();
  direct_tx_free_slots_ = xQueueCreate(kDirectTxQueueSlots, sizeof(uint8_t));
  direct_tx_ready_slots_ = xQueueCreate(kDirectTxQueueSlots, sizeof(uint8_t));
  direct_tx_stopped_ = xSemaphoreCreateBinary();
  if (direct_tx_free_slots_ == nullptr || direct_tx_ready_slots_ == nullptr || direct_tx_stopped_ == nullptr) {
    destroy_direct_tx_resources_();
    return false;
  }
  for (uint8_t idx = 0U; idx < kDirectTxQueueSlots; idx++) {
    if (xQueueSend(direct_tx_free_slots_, &idx, 0) != pdTRUE) {
      destroy_direct_tx_resources_();
      return false;
    }
  }

  direct_tx_task_running_.store(true, std::memory_order_relaxed);
  if (xTaskCreate(&CsiStreamTransport::direct_tx_task_entry_,
                  "espectre_stream_tx",
                  4096,
                  this,
                  7,
                  &direct_tx_task_handle_) != pdPASS) {
    direct_tx_task_running_.store(false, std::memory_order_relaxed);
    destroy_direct_tx_resources_();
    return false;
  }
  return true;
#else
  return false;
#endif
}

void CsiStreamTransport::stop_direct_tx_worker_() {
#if defined(ESP_PLATFORM)
  if (direct_tx_task_running_.exchange(false, std::memory_order_acq_rel)) {
    const uint8_t wake = kDirectTxQueueSlots;
    (void)xQueueSend(direct_tx_ready_slots_, &wake, 0);
    if (direct_tx_stopped_ != nullptr) {
      xSemaphoreTake(direct_tx_stopped_, portMAX_DELAY);
    }
  }
  direct_tx_task_handle_ = nullptr;
  destroy_direct_tx_resources_();
#endif
}

void CsiStreamTransport::destroy_direct_tx_resources_() {
#if defined(ESP_PLATFORM)
  if (direct_tx_free_slots_ != nullptr) {
    vQueueDelete(direct_tx_free_slots_);
    direct_tx_free_slots_ = nullptr;
  }
  if (direct_tx_ready_slots_ != nullptr) {
    vQueueDelete(direct_tx_ready_slots_);
    direct_tx_ready_slots_ = nullptr;
  }
  if (direct_tx_stopped_ != nullptr) {
    vSemaphoreDelete(direct_tx_stopped_);
    direct_tx_stopped_ = nullptr;
  }
#endif
}

void CsiStreamTransport::reset_direct_tx_queue_() {
#if defined(ESP_PLATFORM)
  if (direct_tx_free_slots_ == nullptr || direct_tx_ready_slots_ == nullptr) {
    return;
  }

  uint8_t slot_idx = 0U;
  while (xQueueReceive(direct_tx_ready_slots_, &slot_idx, 0) == pdTRUE) {
    (void)xQueueSend(direct_tx_free_slots_, &slot_idx, 0);
  }
#endif
}

void CsiStreamTransport::direct_tx_task_entry_(void *context) {
#if defined(ESP_PLATFORM)
  auto *transport = static_cast<CsiStreamTransport *>(context);
  if (transport != nullptr) {
    transport->run_direct_tx_task_();
  }
  vTaskDelete(nullptr);
#else
  (void)context;
#endif
}

void CsiStreamTransport::run_direct_tx_task_() {
#if defined(ESP_PLATFORM)
  int sock = -1;
  while (direct_tx_task_running_.load(std::memory_order_relaxed)) {
    uint8_t slot_idx = 0U;
    if (xQueueReceive(direct_tx_ready_slots_, &slot_idx, pdMS_TO_TICKS(250)) != pdTRUE) {
      continue;
    }
    if (!direct_tx_task_running_.load(std::memory_order_acquire) || slot_idx >= kDirectTxQueueSlots) {
      break;
    }

    DirectTxSlot &slot = direct_tx_slots_[slot_idx];
    const uint32_t collector_ip = collector_ip_addr_.load(std::memory_order_relaxed);
    if (collector_ip == 0U) {
      (void)xQueueSend(direct_tx_free_slots_, &slot_idx, 0);
      continue;
    }

    if (sock < 0) {
      sock = create_stream_socket();
    }

    if (sock >= 0) {
      sockaddr_in collector_addr{};
      collector_addr.sin_family = AF_INET;
      collector_addr.sin_addr.s_addr = collector_ip;
      collector_addr.sin_port = htons(collector_port_);
      const ssize_t sent =
          sendto(sock, slot.packet.data(), slot.packet_len, 0, reinterpret_cast<const sockaddr *>(&collector_addr),
                 sizeof(collector_addr));
      if (sent == static_cast<ssize_t>(slot.packet_len)) {
        stream_tx_total_.fetch_add(1U, std::memory_order_relaxed);
      } else {
        stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
        const int send_errno = errno;
        if (send_errno == ENOMEM || send_errno == ENOBUFS || send_errno == EAGAIN || send_errno == EWOULDBLOCK) {
          stream_tx_backpressure_total_.fetch_add(1U, std::memory_order_relaxed);
        } else {
          close(sock);
          sock = -1;
        }
      }
    } else {
      stream_tx_error_total_.fetch_add(1U, std::memory_order_relaxed);
    }

    (void)xQueueSend(direct_tx_free_slots_, &slot_idx, 0);
  }

  if (sock >= 0) {
    close(sock);
  }
  if (direct_tx_stopped_ != nullptr) {
    xSemaphoreGive(direct_tx_stopped_);
  }
#endif
}

void CsiStreamTransport::reset_runtime_telemetry_baseline_(const CsiCaptureService &capture_service,
                                                           const CsiTrafficService &traffic_service) {
  prev_capture_callback_total_ = capture_service.callback_invocations();
  prev_capture_valid_total_ = capture_service.valid_packets();
  prev_capture_invalid_total_ = capture_service.normalized_invalid_packets();
  prev_csi_callback_total_ = csi_callback_total_.load(std::memory_order_relaxed);
  prev_csi_accepted_total_ = csi_accepted_total_.load(std::memory_order_relaxed);
  prev_csi_filtered_total_ = csi_filtered_total_.load(std::memory_order_relaxed);
  prev_stream_fresh_total_ = stream_fresh_total_.load(std::memory_order_relaxed);
  prev_stream_repeat_total_ = stream_repeat_total_.load(std::memory_order_relaxed);
  prev_traffic_rx_total_ = traffic_service.get_packets_received();
  prev_tx_success_total_ = stream_tx_total_.load(std::memory_order_relaxed);
  prev_tx_error_total_ = stream_tx_error_total_.load(std::memory_order_relaxed);
  prev_tx_backpressure_total_ = stream_tx_backpressure_total_.load(std::memory_order_relaxed);
  prev_log_sample_ms_ = static_cast<uint64_t>(monotonic_now_ms());
}

}  // namespace espectre
