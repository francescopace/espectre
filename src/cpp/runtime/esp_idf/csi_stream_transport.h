#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "csi_capture_service.h"
#include "csi_format.h"
#include "csi_stream_protocol.h"
#include "csi_traffic_service.h"

#include "freertos/FreeRTOS.h"

#if !defined(ESP_PLATFORM)
using portMUX_TYPE = int;
#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(mux) (void)(mux)
#define portEXIT_CRITICAL(mux) (void)(mux)
#endif

namespace esphome {
namespace espectre {

class CsiStreamTransport {
 public:
  void configure(uint64_t device_id, uint16_t collector_port, uint32_t log_interval_ms);
  void reset_session();
  void clear_ap_bssid();
  void set_ap_bssid(const uint8_t *bssid, size_t len);

  void handle_csi_packet(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized, bool streaming_ready);
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
  };

  size_t build_stream_packet_(uint8_t *buffer, size_t buffer_len);
  bool ensure_stream_socket_();
  void close_stream_socket_();
  bool send_stream_datagram_();
  void reset_runtime_telemetry_baseline_(const CsiTrafficService &traffic_service);

  uint64_t device_id_{0U};
  uint16_t collector_port_{5001U};
  uint32_t log_interval_ms_{1000U};
  std::atomic<uint32_t> stream_seq_{0U};
  uint32_t collector_ip_addr_{0U};
  std::array<uint8_t, 6> ap_bssid_{};

  portMUX_TYPE latch_lock_ = portMUX_INITIALIZER_UNLOCKED;
  LatestCsiSample latest_csi_{};
  uint64_t latest_csi_sent_total_{0U};
  int stream_sock_{-1};

  std::atomic<uint32_t> last_csi_ms_{0U};
  std::atomic<uint64_t> csi_callback_total_{0U};
  std::atomic<uint64_t> csi_accepted_total_{0U};
  std::atomic<uint64_t> csi_filtered_total_{0U};
  std::atomic<uint64_t> stream_fresh_total_{0U};
  std::atomic<uint64_t> stream_repeat_total_{0U};
  uint64_t stream_tx_total_{0U};
  uint64_t stream_tx_error_total_{0U};
  uint64_t stream_tx_backpressure_total_{0U};
  uint64_t last_pacing_streamed_total_{0U};

  uint64_t last_log_ms_{0U};
  uint64_t prev_log_sample_ms_{0U};
  uint64_t prev_csi_callback_total_{0U};
  uint64_t prev_csi_accepted_total_{0U};
  uint64_t prev_csi_filtered_total_{0U};
  uint64_t prev_stream_fresh_total_{0U};
  uint64_t prev_stream_repeat_total_{0U};
  uint64_t prev_traffic_rx_total_{0U};
  uint64_t prev_tx_success_total_{0U};
  uint64_t prev_tx_error_total_{0U};
  uint64_t prev_tx_backpressure_total_{0U};
  bool stream_active_last_tick_{true};
  bool last_tx_backpressure_{false};
};

}  // namespace espectre
}  // namespace esphome
