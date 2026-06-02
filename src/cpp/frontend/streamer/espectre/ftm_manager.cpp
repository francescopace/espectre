/*
 * ESPectre - FTM Manager
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "ftm_manager.h"

#include <cinttypes>
#include <cstring>
#include <utility>

#include "csi_stream_protocol.h"
#include "espectre_log.h"
#include "esp_timer.h"

namespace esphome {
namespace espectre {

namespace {
static const char *const TAG = "espectre.stream.ftm";

#ifdef CONFIG_ESPECTRE_WIFI_FTM_ENABLED
constexpr bool kWifiFtmEnabled = true;
#else
constexpr bool kWifiFtmEnabled = false;
#endif

#ifdef CONFIG_ESPECTRE_WIFI_FTM_VERBOSE
constexpr bool kWifiFtmVerbose = true;
#else
constexpr bool kWifiFtmVerbose = false;
#endif

#if defined(CONFIG_ESP_WIFI_FTM_ENABLE) && CONFIG_ESP_WIFI_FTM_ENABLE && defined(CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT) && \
    CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT
constexpr bool kWifiFtmInitiatorSupported = true;
#else
constexpr bool kWifiFtmInitiatorSupported = false;
#endif

constexpr uint32_t kWifiFtmPeriodMs = static_cast<uint32_t>(CONFIG_ESPECTRE_WIFI_FTM_PERIOD_MS);
constexpr uint32_t kWifiFtmRetryBackoffMs = static_cast<uint32_t>(CONFIG_ESPECTRE_WIFI_FTM_RETRY_BACKOFF_MS);

const char *ftm_status_name_(wifi_ftm_status_t status) {
  switch (status) {
    case FTM_STATUS_SUCCESS:
      return "SUCCESS";
    case FTM_STATUS_UNSUPPORTED:
      return "UNSUPPORTED";
    case FTM_STATUS_CONF_REJECTED:
      return "CONF_REJECTED";
    case FTM_STATUS_NO_RESPONSE:
      return "NO_RESPONSE";
    case FTM_STATUS_FAIL:
      return "FAIL";
#ifdef FTM_STATUS_NO_VALID_MSMT
    case FTM_STATUS_NO_VALID_MSMT:
      return "NO_VALID_MSMT";
#endif
    case FTM_STATUS_USER_TERM:
      return "USER_TERM";
    default:
      return "UNKNOWN";
  }
}
}  // namespace

void FtmManager::init(uint64_t device_id, uint32_t boot_id, packet_sink_t packet_sink) {
  device_id_ = device_id;
  boot_id_ = boot_id;
  packet_sink_ = std::move(packet_sink);
  on_wifi_disconnected();
}

void FtmManager::on_wifi_connected(const wifi_ap_record_t &ap_info) {
  ap_info_valid_ = true;
  std::memcpy(ap_bssid_, ap_info.bssid, sizeof(ap_bssid_));
  ap_channel_ = ap_info.primary;
  ap_rssi_ = ap_info.rssi;
  ap_ftm_responder_ = ap_info.ftm_responder;
  ap_ftm_initiator_ = ap_info.ftm_initiator;
  session_in_flight_ = false;
  current_session_periodic_ = false;
  periodic_active_ = false;
  next_attempt_ms_ = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
  last_drop_total_seen_ = 0U;
  last_fail_total_seen_ = 0U;

  if (kWifiFtmVerbose || ap_ftm_responder_) {
    ESP_LOGI(TAG,
             "FTM AP: channel=%u rssi=%d responder=%s initiator=%s",
             static_cast<unsigned>(ap_channel_),
             ap_rssi_,
             ap_ftm_responder_ ? "yes" : "no",
             ap_ftm_initiator_ ? "yes" : "no");
  }
}

void FtmManager::on_wifi_disconnected() {
  ap_info_valid_ = false;
  ap_ftm_responder_ = false;
  ap_ftm_initiator_ = false;
  ap_channel_ = 0U;
  ap_rssi_ = 0;
  session_in_flight_ = false;
  current_session_periodic_ = false;
  periodic_active_ = false;
  next_attempt_ms_ = 0U;
}

void FtmManager::maybe_start_session(bool allow_periodic_sessions, uint64_t drop_total, uint64_t send_fail_total) {
  if (!kWifiFtmEnabled || !kWifiFtmInitiatorSupported || !ap_info_valid_ || !ap_ftm_responder_ || session_in_flight_) {
    return;
  }

#if defined(CONFIG_ESP_WIFI_FTM_ENABLE) && CONFIG_ESP_WIFI_FTM_ENABLE && defined(CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT) && \
    CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT
  const uint32_t now_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
  if (drop_total > last_drop_total_seen_ && periodic_active_) {
    last_drop_total_seen_ = drop_total;
    next_attempt_ms_ = now_ms + kWifiFtmRetryBackoffMs;
    return;
  }
  if (send_fail_total > last_fail_total_seen_ && periodic_active_) {
    last_fail_total_seen_ = send_fail_total;
    next_attempt_ms_ = now_ms + kWifiFtmRetryBackoffMs;
    return;
  }
  if (now_ms < next_attempt_ms_) {
    return;
  }

  const bool periodic = periodic_active_;
  if (periodic && !allow_periodic_sessions) {
    return;
  }

  wifi_ftm_initiator_cfg_t cfg{};
  std::memcpy(cfg.resp_mac, ap_bssid_, sizeof(cfg.resp_mac));
  cfg.channel = ap_channel_;
  cfg.frm_count = 16;
  cfg.burst_period = 0;

  const esp_err_t err = esp_wifi_ftm_initiate_session(&cfg);
  if (err != ESP_OK) {
    next_attempt_ms_ = now_ms + kWifiFtmRetryBackoffMs;
    if (kWifiFtmVerbose) {
      ESP_LOGW(TAG, "Failed to start FTM session: %s", esp_err_to_name(err));
    }
    return;
  }

  session_in_flight_ = true;
  current_session_periodic_ = periodic;
  session_id_ += 1U;
  next_attempt_ms_ = periodic ? (now_ms + kWifiFtmPeriodMs) : (now_ms + kWifiFtmRetryBackoffMs);
  if (kWifiFtmVerbose) {
    ESP_LOGI(TAG, "FTM session started (%s)", periodic ? "periodic" : "probe");
  }
#else
  (void)allow_periodic_sessions;
  (void)drop_total;
  (void)send_fail_total;
#endif
}

#if defined(CONFIG_ESP_WIFI_FTM_ENABLE) && CONFIG_ESP_WIFI_FTM_ENABLE && defined(CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT) && \
    CONFIG_ESP_WIFI_FTM_INITIATOR_SUPPORT
void FtmManager::handle_ftm_report(const wifi_event_ftm_report_t &event) {
  if (!kWifiFtmEnabled || !kWifiFtmInitiatorSupported) {
    return;
  }

  const bool periodic = current_session_periodic_;
  session_in_flight_ = false;
  current_session_periodic_ = false;
  event_total_.fetch_add(1U, std::memory_order_relaxed);

  if (event.status == FTM_STATUS_SUCCESS) {
    success_total_.fetch_add(1U, std::memory_order_relaxed);
    periodic_active_ = true;
    next_attempt_ms_ = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL) + kWifiFtmPeriodMs;
  } else {
    periodic_active_ = false;
    next_attempt_ms_ = static_cast<uint32_t>(esp_timer_get_time() / 1000ULL) + kWifiFtmRetryBackoffMs;
  }

  queue_event_(event.status,
               event.rtt_raw,
               event.rtt_est,
               event.dist_est,
               event.ftm_report_num_entries,
               event.peer_mac,
               periodic);

  if (kWifiFtmVerbose || event.status == FTM_STATUS_SUCCESS) {
    ESP_LOGI(TAG,
             "FTM report: status=%s dist_cm=%" PRIu32 " rtt_ns=%" PRIu32 " entries=%u",
             ftm_status_name_(event.status),
             event.dist_est,
             event.rtt_est,
             static_cast<unsigned>(event.ftm_report_num_entries));
  }
}
#endif

void FtmManager::queue_event_(wifi_ftm_status_t status,
                              uint32_t rtt_raw_ns,
                              uint32_t rtt_est_ns,
                              uint32_t dist_est_cm,
                              uint8_t report_entries,
                              const uint8_t peer_mac[6],
                              bool periodic) {
  if (!packet_sink_) {
    return;
  }

  FtmStreamEventV1 event{};
  event.magic = FTM_MAGIC;
  event.version = FTM_VERSION;
  event.header_len = static_cast<uint8_t>(sizeof(event));
  event.event_type = static_cast<uint8_t>(StreamFtmEventType::REPORT);
#if CONFIG_IDF_TARGET_ESP32C6
  event.chip = static_cast<uint8_t>(StreamChipType::C6);
#elif CONFIG_IDF_TARGET_ESP32C5
  event.chip = static_cast<uint8_t>(StreamChipType::C5);
#elif CONFIG_IDF_TARGET_ESP32C3
  event.chip = static_cast<uint8_t>(StreamChipType::C3);
#elif CONFIG_IDF_TARGET_ESP32S3
  event.chip = static_cast<uint8_t>(StreamChipType::S3);
#elif CONFIG_IDF_TARGET_ESP32S2
  event.chip = static_cast<uint8_t>(StreamChipType::S2);
#elif CONFIG_IDF_TARGET_ESP32
  event.chip = static_cast<uint8_t>(StreamChipType::ESP32);
#else
  event.chip = static_cast<uint8_t>(StreamChipType::UNKNOWN);
#endif
  event.ftm_status = static_cast<uint8_t>(status);
  event.ftm_flags = 0U;
  if (ap_ftm_responder_) {
    event.ftm_flags |= STREAM_FTM_FLAG_AP_RESPONDER;
  }
  if (ap_ftm_initiator_) {
    event.ftm_flags |= STREAM_FTM_FLAG_AP_INITIATOR;
  }
  if (periodic) {
    event.ftm_flags |= STREAM_FTM_FLAG_PERIODIC;
  }
  if (status == FTM_STATUS_SUCCESS) {
    event.ftm_flags |= STREAM_FTM_FLAG_SUCCESS;
  }
  event.seq_num = seq_num_++;
  event.device_id = device_id_;
  event.boot_id = boot_id_;
  event.device_ticks_us = static_cast<uint64_t>(esp_timer_get_time());
  if (peer_mac != nullptr) {
    std::memcpy(event.peer_mac, peer_mac, sizeof(event.peer_mac));
  }
  event.channel = ap_channel_;
  event.rssi_dbm = ap_rssi_;
  event.ftm_report_num_entries = report_entries;
  event.rtt_raw_ns = rtt_raw_ns;
  event.rtt_est_ns = rtt_est_ns;
  event.dist_est_cm = dist_est_cm;
  event.session_id = session_id_;

  if (!packet_sink_(reinterpret_cast<const uint8_t *>(&event), sizeof(event))) {
    drop_total_.fetch_add(1U, std::memory_order_relaxed);
  }
}

}  // namespace espectre
}  // namespace esphome
