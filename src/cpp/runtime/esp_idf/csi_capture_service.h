#pragma once

#include <functional>

#include "csi_payload_normalizer.h"
#include "esp_attr.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "gain_controller.h"
#include "utils.h"
#include "wifi_csi_interface.h"

namespace esphome {
namespace espectre {

using csi_capture_packet_callback_t = std::function<void(const wifi_csi_info_t *, const NormalizedCSIPayload &)>;
using csi_capture_gain_packet_callback_t = std::function<void(const wifi_csi_info_t *)>;

class CsiCaptureService {
 public:
  void init(GainLockMode gain_lock_mode = GainLockMode::AUTO, IWiFiCSI *wifi_csi = nullptr);
  void reset_session();

  esp_err_t enable();
  esp_err_t disable();
  void process_packet(wifi_csi_info_t *data);

  bool is_enabled() const { return enabled_; }
  bool is_gain_locked() const { return gain_controller_.is_locked(); }
  uint32_t filtered_packets() const { return filtered_packets_; }

  const GainController &get_gain_controller() const { return gain_controller_; }

  void set_packet_callback(csi_capture_packet_callback_t callback) { packet_callback_ = std::move(callback); }
  void set_gain_packet_callback(csi_capture_gain_packet_callback_t callback) {
    gain_packet_callback_ = std::move(callback);
  }
  void set_gain_lock_callback(GainController::lock_complete_callback_t callback) {
    gain_lock_callback_ = std::move(callback);
    gain_controller_.set_lock_complete_callback(gain_lock_callback_);
  }

 private:
  static void IRAM_ATTR csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *data);
  esp_err_t configure_platform_specific_();
  void log_wrong_sc_packet_(const wifi_csi_info_t *data, size_t csi_len) const;

  bool enabled_{false};
  GainLockMode gain_lock_mode_{GainLockMode::AUTO};
  IWiFiCSI *wifi_csi_{nullptr};
  WiFiCSIReal default_wifi_csi_;
  GainController gain_controller_;
  GainController::lock_complete_callback_t gain_lock_callback_;
  csi_capture_gain_packet_callback_t gain_packet_callback_;
  csi_capture_packet_callback_t packet_callback_;
  uint32_t filtered_packets_{0U};
  bool collapse_logged_{false};
  bool remap_logged_{false};
};

}  // namespace espectre
}  // namespace esphome
