/*
 * ESPectre - FTM Manager
 *
 * Standalone Wi-Fi FTM helper used by the streamer frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "esp_wifi.h"

namespace esphome {
namespace espectre {

class FtmManager {
 public:
  using packet_sink_t = std::function<bool(const uint8_t *data, size_t len)>;

  void init(uint64_t device_id, uint32_t boot_id, packet_sink_t packet_sink);
  void on_wifi_connected(const wifi_ap_record_t &ap_info);
  void on_wifi_disconnected();
  void maybe_start_session(bool allow_periodic_sessions, uint64_t drop_total, uint64_t send_fail_total);

#if defined(CONFIG_ESP_WIFI_FTM_ENABLE) && CONFIG_ESP_WIFI_FTM_ENABLE && defined(CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT) && \
    CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT
  void handle_ftm_report(const wifi_event_ftm_report_t &event);
#endif

  uint64_t event_total() const { return event_total_.load(std::memory_order_relaxed); }
  uint64_t success_total() const { return success_total_.load(std::memory_order_relaxed); }
  uint64_t drop_total() const { return drop_total_.load(std::memory_order_relaxed); }
  bool session_in_flight() const { return session_in_flight_; }
  bool periodic_active() const { return periodic_active_; }
  bool ap_supports_responder() const { return ap_ftm_responder_; }

 private:
  void queue_event_(wifi_ftm_status_t status,
                    uint32_t rtt_raw_ns,
                    uint32_t rtt_est_ns,
                    uint32_t dist_est_cm,
                    uint8_t report_entries,
                    const uint8_t peer_mac[6],
                    bool periodic);

  packet_sink_t packet_sink_{};
  uint64_t device_id_{0U};
  uint32_t boot_id_{0U};
  uint32_t seq_num_{0U};
  uint32_t session_id_{0U};
  uint8_t ap_bssid_[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  uint8_t ap_channel_{0U};
  int8_t ap_rssi_{0};
  bool ap_info_valid_{false};
  bool ap_ftm_responder_{false};
  bool ap_ftm_initiator_{false};
  bool session_in_flight_{false};
  bool current_session_periodic_{false};
  bool periodic_active_{false};
  uint32_t next_attempt_ms_{0U};
  uint64_t last_drop_total_seen_{0U};
  uint64_t last_fail_total_seen_{0U};
  std::atomic<uint64_t> event_total_{0U};
  std::atomic<uint64_t> success_total_{0U};
  std::atomic<uint64_t> drop_total_{0U};
};

}  // namespace espectre
}  // namespace esphome
