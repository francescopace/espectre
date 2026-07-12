#pragma once

#include <atomic>
#include <functional>

#include "csi_payload_normalizer.h"
#include "esp_attr.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "pending_event.h"
#include "utils.h"
#include "wifi_csi_interface.h"

namespace espectre {

using csi_capture_raw_packet_interceptor_t = std::function<bool(const wifi_csi_info_t *)>;
using csi_capture_packet_callback_t = std::function<void(const wifi_csi_info_t *, const NormalizedCSIPayload &)>;

class CsiCaptureService {
 public:
  void init(IWiFiCSI *wifi_csi = nullptr);
  void reset_session();

  esp_err_t enable();
  esp_err_t disable();
  void loop();
  void process_packet(wifi_csi_info_t *data);

  bool is_enabled() const { return enabled_; }
  uint32_t filtered_packets() const { return filtered_packets_.load(std::memory_order_relaxed); }
  uint32_t callback_invocations() const { return callback_invocations_.load(std::memory_order_relaxed); }
  uint32_t null_or_empty_packets() const { return null_or_empty_packets_.load(std::memory_order_relaxed); }
  uint32_t interceptor_drops() const { return interceptor_drops_.load(std::memory_order_relaxed); }
  uint32_t normalized_invalid_packets() const {
    return normalized_invalid_packets_.load(std::memory_order_relaxed);
  }
  uint32_t valid_packets() const { return valid_packets_.load(std::memory_order_relaxed); }
  uint32_t enable_attempts() const { return enable_attempts_.load(std::memory_order_relaxed); }
  uint32_t disable_attempts() const { return disable_attempts_.load(std::memory_order_relaxed); }
  esp_err_t last_configure_err() const { return static_cast<esp_err_t>(last_configure_err_.load(std::memory_order_relaxed)); }
  esp_err_t last_set_callback_err() const {
    return static_cast<esp_err_t>(last_set_callback_err_.load(std::memory_order_relaxed));
  }
  esp_err_t last_set_enabled_err() const {
    return static_cast<esp_err_t>(last_set_enabled_err_.load(std::memory_order_relaxed));
  }
  esp_err_t last_disable_err() const { return static_cast<esp_err_t>(last_disable_err_.load(std::memory_order_relaxed)); }

  void set_raw_packet_interceptor(csi_capture_raw_packet_interceptor_t interceptor) {
    raw_packet_interceptor_ = std::move(interceptor);
  }
  void set_packet_callback(csi_capture_packet_callback_t callback) { packet_callback_ = std::move(callback); }

 private:
  static void IRAM_ATTR csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *data);
  esp_err_t configure_platform_specific_();

  bool enabled_{false};
  IWiFiCSI *wifi_csi_{nullptr};
  WiFiCSIReal default_wifi_csi_;
  csi_capture_raw_packet_interceptor_t raw_packet_interceptor_;
  csi_capture_packet_callback_t packet_callback_;
  std::atomic<uint32_t> filtered_packets_{0U};
  std::atomic<uint32_t> callback_invocations_{0U};
  std::atomic<uint32_t> null_or_empty_packets_{0U};
  std::atomic<uint32_t> interceptor_drops_{0U};
  std::atomic<uint32_t> normalized_invalid_packets_{0U};
  std::atomic<uint32_t> valid_packets_{0U};
  std::atomic<uint32_t> enable_attempts_{0U};
  std::atomic<uint32_t> disable_attempts_{0U};
  std::atomic<int32_t> last_configure_err_{ESP_OK};
  std::atomic<int32_t> last_set_callback_err_{ESP_OK};
  std::atomic<int32_t> last_set_enabled_err_{ESP_OK};
  std::atomic<int32_t> last_disable_err_{ESP_OK};
  std::atomic<bool> collapse_seen_{false};
  std::atomic<bool> remap_seen_{false};
  PendingEvent<> collapse_log_event_;
  PendingEvent<> remap_log_event_;
};

}  // namespace espectre
