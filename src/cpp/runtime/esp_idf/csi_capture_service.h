#pragma once

#include <functional>

#include "csi_payload_normalizer.h"
#include "esp_attr.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "utils.h"
#include "wifi_csi_interface.h"

namespace esphome {
namespace espectre {

using csi_capture_packet_callback_t = std::function<void(const wifi_csi_info_t *, const NormalizedCSIPayload &)>;

class CsiCaptureService {
 public:
  void init(IWiFiCSI *wifi_csi = nullptr);
  void reset_session();

  esp_err_t enable();
  esp_err_t disable();
  void process_packet(wifi_csi_info_t *data);

  bool is_enabled() const { return enabled_; }
  uint32_t filtered_packets() const { return filtered_packets_; }

  void set_packet_callback(csi_capture_packet_callback_t callback) { packet_callback_ = std::move(callback); }

 private:
  static void IRAM_ATTR csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *data);
  esp_err_t configure_platform_specific_();
  void log_wrong_sc_packet_(const wifi_csi_info_t *data, size_t csi_len) const;

  bool enabled_{false};
  IWiFiCSI *wifi_csi_{nullptr};
  WiFiCSIReal default_wifi_csi_;
  csi_capture_packet_callback_t packet_callback_;
  uint32_t filtered_packets_{0U};
  bool collapse_logged_{false};
  bool remap_logged_{false};
};

}  // namespace espectre
}  // namespace esphome
