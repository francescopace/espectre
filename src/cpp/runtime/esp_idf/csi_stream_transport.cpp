#include "csi_stream_transport.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <net/if.h>

#include "espectre_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "runtime_time.h"
#include "sdkconfig.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace espectre {

namespace {

static const char *const TAG = "espectre.stream";
constexpr size_t kStreamRecordMaxBytes = sizeof(CsiStreamHeaderV5) + HT20_CSI_LEN;
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

bool is_zero_mac_(const uint8_t *mac) {
  if (mac == nullptr) {
    return true;
  }
  for (size_t idx = 0U; idx < 6U; idx++) {
    if (mac[idx] != 0U) {
      return false;
    }
  }
  return true;
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
#elif CONFIG_IDF_TARGET_ESP32S2
  return StreamChipType::S2;
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

esp_netif_t *get_sta_netif() { return esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"); }

bool get_sta_netif_index(uint32_t *out_index) {
  if (out_index == nullptr) {
    return false;
  }
  esp_netif_t *netif = get_sta_netif();
  if (netif == nullptr) {
    ESP_LOGW(TAG, "Failed to get STA netif for stream socket");
    return false;
  }

  const int if_index = esp_netif_get_netif_impl_index(netif);
  if (if_index <= 0) {
    ESP_LOGW(TAG, "Invalid STA netif index for stream socket: %d", if_index);
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
  if (if_indextoname(if_index, iface.ifr_name) == nullptr) {
    ESP_LOGW(TAG, "Failed to resolve STA interface name for stream socket index %" PRIu32, if_index);
    return false;
  }

  if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &iface, sizeof(iface)) != 0) {
    ESP_LOGW(TAG, "Failed to bind stream socket to %s (errno=%d)", iface.ifr_name, errno);
    return false;
  }

  return true;
}

int create_stream_socket() {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock < 0) {
    ESP_LOGW(TAG, "Failed to create stream socket (errno=%d)", errno);
    return -1;
  }

  if (!bind_socket_to_sta_interface(sock)) {
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

bool fill_rx_timestamp_metadata(const wifi_pkt_rx_ctrl_t &rx_ctrl, CsiStreamHeaderV5 *header) {
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

uint64_t counter_delta(uint64_t current, uint64_t previous) {
  return current >= previous ? current - previous : current;
}

}  // namespace

void CsiStreamTransport::configure(uint64_t device_id,
                                   uint16_t collector_port,
                                   uint32_t log_interval_ms,
                                   uint8_t tx_batch_records) {
  device_id_ = device_id;
  collector_port_ = collector_port;
  log_interval_ms_ = log_interval_ms;
  tx_batch_records_ = std::clamp<uint8_t>(tx_batch_records, 1U, STREAM_MAX_BATCH_RECORDS);
}

void CsiStreamTransport::reset_session() {
  close_stream_socket_();
  drop_stream_batch_();
  collector_ip_addr_ = 0U;
  stream_seq_.store(0U, std::memory_order_relaxed);
  last_pacing_streamed_total_ = 0U;
  last_csi_ms_.store(0U, std::memory_order_relaxed);
  csi_callback_total_.store(0U, std::memory_order_relaxed);
  csi_accepted_total_.store(0U, std::memory_order_relaxed);
  csi_filtered_total_.store(0U, std::memory_order_relaxed);
  stream_fresh_total_.store(0U, std::memory_order_relaxed);
  stream_repeat_total_.store(0U, std::memory_order_relaxed);
  stream_tx_total_ = 0U;
  stream_tx_error_total_ = 0U;
  stream_tx_backpressure_total_ = 0U;
  last_log_ms_ = 0U;
  prev_log_sample_ms_ = 0U;
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

  portENTER_CRITICAL(&latch_lock_);
  latest_csi_ = LatestCsiSample{};
  latest_csi_sent_total_ = 0U;
  portEXIT_CRITICAL(&latch_lock_);
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

  if (!is_zero_mac_(ap_bssid_.data()) && std::memcmp(info->mac, ap_bssid_.data(), ap_bssid_.size()) != 0) {
    csi_filtered_total_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  portENTER_CRITICAL(&latch_lock_);
  latest_csi_.rx_ctrl = info->rx_ctrl;
  std::memcpy(latest_csi_.csi.data(), normalized.data, normalized.len);
  latest_csi_.len = static_cast<uint16_t>(normalized.len);
  latest_csi_.first_word_invalid = info->first_word_invalid;
  latest_csi_.valid = true;
  latest_csi_.update_total++;
  portEXIT_CRITICAL(&latch_lock_);

  csi_accepted_total_.fetch_add(1U, std::memory_order_relaxed);
  last_csi_ms_.store(monotonic_now_ms(), std::memory_order_relaxed);
}

void CsiStreamTransport::update_from_traffic(const CsiTrafficService &traffic_service, bool streaming_ready) {
  sockaddr_in sender_addr{};
  if (traffic_service.get_last_sender(&sender_addr) && sender_addr.sin_addr.s_addr != 0U &&
      sender_addr.sin_addr.s_addr != collector_ip_addr_) {
    collector_ip_addr_ = sender_addr.sin_addr.s_addr;

    char addr_text[16];
    format_ipv4_addr(collector_ip_addr_, addr_text, sizeof(addr_text));
    ESP_LOGI(TAG,
             "Collector learned from UDP pacing: address=%s pacing_port=%u stream_port=%u",
             addr_text,
             static_cast<unsigned>(ntohs(sender_addr.sin_port)),
             static_cast<unsigned>(collector_port_));
  }

  const uint64_t pacing_rx_total = traffic_service.get_packets_received();
  const bool should_stream = collector_ip_addr_ != 0U && streaming_ready;
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
      stream_tx_error_total_ += pending_packets;
    } else {
      for (uint64_t idx = 0U; idx < pending_packets; idx++) {
        if (send_stream_datagram_()) {
          continue;
        }
        if (last_tx_backpressure_) {
          const uint64_t unsent_packets = pending_packets - idx - 1U;
          stream_tx_error_total_ += unsent_packets;
          stream_tx_backpressure_total_ += unsent_packets;
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
  const uint64_t now_ms = static_cast<uint64_t>(monotonic_now_ms());
  if (last_log_ms_ != 0U && now_ms - last_log_ms_ < log_interval_ms_) {
    return;
  }
  if (!streaming_ready) {
    last_log_ms_ = now_ms;
    return;
  }
  if (prev_log_sample_ms_ == 0U) {
    reset_runtime_telemetry_baseline_(traffic_service);
    prev_log_sample_ms_ = now_ms;
  }

  const uint64_t dt_ms = std::max<uint64_t>(1U, now_ms - prev_log_sample_ms_);
  const uint64_t csi_callback_total = csi_callback_total_.load(std::memory_order_relaxed);
  const uint64_t csi_accepted_total = csi_accepted_total_.load(std::memory_order_relaxed);
  const uint64_t csi_filtered_total = csi_filtered_total_.load(std::memory_order_relaxed);
  const uint64_t stream_fresh_total = stream_fresh_total_.load(std::memory_order_relaxed);
  const uint64_t stream_repeat_total = stream_repeat_total_.load(std::memory_order_relaxed);
  const uint64_t traffic_valid_total = traffic_service.get_packets_received();
  const uint64_t tx_success_total = stream_tx_total_;
  const uint64_t tx_error_total = stream_tx_error_total_;
  const uint64_t tx_backpressure_total = stream_tx_backpressure_total_;

  const auto to_pps = [dt_ms](uint64_t delta) { return static_cast<float>(delta) * 1000.0F / static_cast<float>(dt_ms); };
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
  const uint32_t csi_capture_callbacks = capture_service.callback_invocations();
  const uint32_t csi_capture_null = capture_service.null_or_empty_packets();
  const uint32_t csi_capture_raw_drop = capture_service.interceptor_drops();
  const uint32_t csi_capture_bad_sc = capture_service.normalized_invalid_packets();
  const uint32_t csi_capture_valid = capture_service.valid_packets();

  const bool csi_enabled = capture_service.is_enabled();
  const bool traffic_active = traffic_rx_pps > 1.0F;
  const char *csi_health = !csi_enabled                     ? "not_armed"
                           : !traffic_active                ? "idle"
                           : csi_capture_callbacks == 0U    ? "armed_no_callback"
                           : csi_capture_valid == 0U        ? "callback_no_valid_packets"
                           : csi_callback_pps < 1.0F        ? "callback_silent"
                                                            : "ok";

  ESP_LOGI(TAG,
           "state=%s health=%s csi_ap=%.0f csi_filt=%.0f cb=%" PRIu32 " valid=%" PRIu32
           " null=%" PRIu32 " raw_drop=%" PRIu32 " bad_sc=%" PRIu32
           " udp_rx=%.1f udp_tx=%.0f fresh=%.0f repeat=%.0f"
           " tx_err=%.0f/%" PRIu64 " tx_bp=%.0f/%" PRIu64
           " age_ms=%" PRIu32 " heap=%.1f min=%.1f",
           state_name != nullptr ? state_name : "unknown",
           csi_health,
           static_cast<double>(csi_accepted_pps),
           static_cast<double>(csi_filtered_pps),
           csi_capture_callbacks,
           csi_capture_valid,
           csi_capture_null,
           csi_capture_raw_drop,
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
    stream_repeat_total_.fetch_add(1U, std::memory_order_relaxed);
    return 0U;
  }

  auto *header = reinterpret_cast<CsiStreamHeaderV5 *>(buffer);
  *header = CsiStreamHeaderV5{};
  header->magic = STREAM_MAGIC;
  header->version = STREAM_VERSION;
  header->header_len = static_cast<uint8_t>(sizeof(*header));
  header->chip = static_cast<uint8_t>(detect_chip_code());
  header->flags = 0U;
  header->seq_num = stream_seq_.fetch_add(1U, std::memory_order_relaxed);
  header->device_id = device_id_;
  header->device_ticks_us = static_cast<uint64_t>(esp_timer_get_time());
  header->wifi_rx_ts_us = sample.rx_ctrl.timestamp;
  header->wifi_rx_start_ts_ns = 0U;
  header->channel = sample.rx_ctrl.channel;
  header->rssi_dbm = sample.rx_ctrl.rssi;
#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2 || CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || \
    CONFIG_IDF_TARGET_ESP32C2
  header->noise_floor_dbm = sample.rx_ctrl.noise_floor;
#else
  header->noise_floor_dbm = -128;
#endif
  header->tx_backpressure_total = stream_tx_backpressure_total_;

  if (sample.first_word_invalid) {
    header->flags |= STREAM_FLAG_FIRST_WORD_INVALID;
  }
  if (header->wifi_rx_ts_us != 0U) {
    header->flags |= STREAM_FLAG_WIFI_RX_TS_VALID;
  }
  if (fill_rx_timestamp_metadata(sample.rx_ctrl, header)) {
    header->flags |= STREAM_FLAG_WIFI_RX_START_TS_NS_VALID;
  }
  header->flags |= STREAM_FLAG_CSI_FRESH;
  stream_fresh_total_.fetch_add(1U, std::memory_order_relaxed);

  header->csi_len_bytes = static_cast<uint16_t>(
      build_transport_csi_payload_(sample.csi.data(), sample.len, buffer + sizeof(*header), &header->num_subcarriers));
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
  if (collector_ip_addr_ == 0U) {
    return false;
  }
  if (!ensure_stream_socket_()) {
    stream_tx_error_total_++;
    return false;
  }

  const size_t record_len = build_stream_packet_(batch_buffer_.data() + batch_len_, batch_buffer_.size() - batch_len_);
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
  const bool sent = send_datagram_(batch_buffer_.data(), batch_len_);
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
  collector_addr.sin_addr.s_addr = collector_ip_addr_;
  collector_addr.sin_port = htons(collector_port_);
  const ssize_t sent =
      sendto(stream_sock_, payload, payload_len, 0, reinterpret_cast<const sockaddr *>(&collector_addr), sizeof(collector_addr));
  if (sent == static_cast<ssize_t>(payload_len)) {
    stream_tx_total_++;
    return true;
  }

  const int send_errno = errno;
  stream_tx_error_total_++;
  const bool backpressure =
      send_errno == ENOMEM || send_errno == ENOBUFS || send_errno == EAGAIN || send_errno == EWOULDBLOCK;
  if (backpressure) {
    last_tx_backpressure_ = true;
    stream_tx_backpressure_total_++;
    return false;
  }

  close_stream_socket_();
  return false;
}

void CsiStreamTransport::reset_runtime_telemetry_baseline_(const CsiTrafficService &traffic_service) {
  prev_csi_callback_total_ = csi_callback_total_.load(std::memory_order_relaxed);
  prev_csi_accepted_total_ = csi_accepted_total_.load(std::memory_order_relaxed);
  prev_csi_filtered_total_ = csi_filtered_total_.load(std::memory_order_relaxed);
  prev_stream_fresh_total_ = stream_fresh_total_.load(std::memory_order_relaxed);
  prev_stream_repeat_total_ = stream_repeat_total_.load(std::memory_order_relaxed);
  prev_traffic_rx_total_ = traffic_service.get_packets_received();
  prev_tx_success_total_ = stream_tx_total_;
  prev_tx_error_total_ = stream_tx_error_total_;
  prev_tx_backpressure_total_ = stream_tx_backpressure_total_;
  prev_log_sample_ms_ = static_cast<uint64_t>(monotonic_now_ms());
}

}  // namespace espectre
