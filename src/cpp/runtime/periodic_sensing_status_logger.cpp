#include "periodic_sensing_status_logger.h"

#if __has_include("esp_wifi.h")
#include "esp_wifi.h"
#define ESPECTRE_HAVE_ESP_WIFI 1
#endif

#include "espectre_log.h"
#include "runtime_time.h"

namespace espectre {

void PeriodicSensingStatusLogger::log_status(const char *tag,
                                             const RuntimeSnapshot &snapshot,
                                             uint32_t packets_per_publish) {
  if (!tag) {
    return;
  }

  const float motion_metric = snapshot.movement_metric;
  const float threshold = snapshot.threshold;
  const bool is_motion = (snapshot.motion_state == MotionState::MOTION);

  const uint32_t now_ms = monotonic_now_ms();

  uint32_t rate_pps = 0;
  if (last_log_time_ms_ > 0 && now_ms > last_log_time_ms_) {
    const uint32_t elapsed_ms = now_ms - last_log_time_ms_;
    if (elapsed_ms > 0) {
      rate_pps = static_cast<uint32_t>((static_cast<uint64_t>(packets_per_publish) * 1000U) / elapsed_ms);
    }
  }
  last_log_time_ms_ = now_ms;

  int8_t rssi = -127;
  uint8_t channel = 0;
#ifdef ESPECTRE_HAVE_ESP_WIFI
  wifi_ap_record_t ap_info{};
  if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
    rssi = ap_info.rssi;
    channel = ap_info.primary;
  }
#endif

  const float progress = (threshold > 0.0f) ? (motion_metric / threshold) : 0.0f;
  const int percent = static_cast<int>(progress * 100.0f);

  log_progress_bar(tag, progress, 20, 15,
                   "%3d%% | mvmt:%.6f thr:%.6f | %s | %u pkt/s | ch:%u rssi:%d",
                   percent, motion_metric, threshold,
                   is_motion ? "MOTION" : "IDLE",
                   static_cast<unsigned>(rate_pps),
                   static_cast<unsigned>(channel),
                   static_cast<int>(rssi));
}

}  // namespace espectre
