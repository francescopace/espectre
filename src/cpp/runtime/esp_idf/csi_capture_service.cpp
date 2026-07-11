#include "csi_capture_service.h"

#include <cinttypes>

#include "csi_format.h"
#include "csi_platform_config.h"
#include "espectre_log.h"

namespace espectre {

namespace {

static const char *const TAG = "CsiCapture";

}  // namespace

void CsiCaptureService::init(IWiFiCSI *wifi_csi) {
  wifi_csi_ = wifi_csi ? wifi_csi : &default_wifi_csi_;
  reset_session();
}

void CsiCaptureService::reset_session() {
  filtered_packets_ = 0U;
  callback_invocations_.store(0U, std::memory_order_relaxed);
  null_or_empty_packets_.store(0U, std::memory_order_relaxed);
  interceptor_drops_.store(0U, std::memory_order_relaxed);
  normalized_invalid_packets_.store(0U, std::memory_order_relaxed);
  valid_packets_.store(0U, std::memory_order_relaxed);
  enable_attempts_.store(0U, std::memory_order_relaxed);
  disable_attempts_.store(0U, std::memory_order_relaxed);
  last_configure_err_.store(ESP_OK, std::memory_order_relaxed);
  last_set_callback_err_.store(ESP_OK, std::memory_order_relaxed);
  last_set_enabled_err_.store(ESP_OK, std::memory_order_relaxed);
  last_disable_err_.store(ESP_OK, std::memory_order_relaxed);
  collapse_logged_ = false;
  remap_logged_ = false;
}

esp_err_t CsiCaptureService::enable() {
  if (enabled_) {
    ESP_LOGW(TAG, "CSI already enabled");
    return ESP_OK;
  }

  const uint32_t attempt = enable_attempts_.fetch_add(1U, std::memory_order_relaxed) + 1U;
  ESP_LOGI(TAG, "Arming CSI attempt=%" PRIu32, attempt);

  esp_err_t err = configure_platform_specific_();
  last_configure_err_.store(err, std::memory_order_relaxed);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure CSI: %s", esp_err_to_name(err));
    return err;
  }

  err = wifi_csi_->set_csi_rx_cb(&CsiCaptureService::csi_rx_callback_wrapper_, this);
  last_set_callback_err_.store(err, std::memory_order_relaxed);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set CSI callback: %s", esp_err_to_name(err));
    return err;
  }

  err = wifi_csi_->set_csi(true);
  last_set_enabled_err_.store(err, std::memory_order_relaxed);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to enable CSI: %s", esp_err_to_name(err));
    return err;
  }

  enabled_ = true;
  ESP_LOGI(TAG,
           "CSI armed attempt=%" PRIu32 " configure=%s set_cb=%s set_enabled=%s",
           attempt,
           esp_err_to_name(last_configure_err()),
           esp_err_to_name(last_set_callback_err()),
           esp_err_to_name(last_set_enabled_err()));
  return ESP_OK;
}

esp_err_t CsiCaptureService::disable() {
  if (!enabled_) {
    return ESP_OK;
  }

  const uint32_t attempt = disable_attempts_.fetch_add(1U, std::memory_order_relaxed) + 1U;
  esp_err_t err = wifi_csi_->set_csi(false);
  last_disable_err_.store(err, std::memory_order_relaxed);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to disable CSI: %s", esp_err_to_name(err));
    return err;
  }

  err = wifi_csi_->set_csi_rx_cb(nullptr, nullptr);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to unregister CSI callback: %s", esp_err_to_name(err));
    return err;
  }

  enabled_ = false;
  ESP_LOGI(TAG, "CSI disabled attempt=%" PRIu32 " disable=%s", attempt, esp_err_to_name(last_disable_err()));
  return ESP_OK;
}

void CsiCaptureService::process_packet(wifi_csi_info_t *data) {
  if (data == nullptr || data->buf == nullptr || data->len == 0U) {
    null_or_empty_packets_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  if (raw_packet_interceptor_ && raw_packet_interceptor_(data)) {
    interceptor_drops_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  int8_t csi_remapped[HT20_CSI_LEN];
  const NormalizedCSIPayload normalized =
      normalize_ht20_csi_payload(data->buf, data->len, csi_remapped, sizeof(csi_remapped));

  if (!collapse_logged_ &&
      (normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT20 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI double-length collapse active: 256->128 and/or 228->114");
    collapse_logged_ = true;
  }

  if (!remap_logged_ &&
      (normalized.tag == NormalizedCSIPayloadTag::HT57_TO_64 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI remap active: 57->64 SC (left_pad=4, right_pad=3)");
    remap_logged_ = true;
  }

  if (packet_callback_) {
    packet_callback_(data, normalized);
  }
  if (!normalized.valid() || normalized.len != HT20_CSI_LEN) {
    filtered_packets_++;
    normalized_invalid_packets_.fetch_add(1U, std::memory_order_relaxed);
    if (filtered_packets_ % 100 == 1) {
      log_wrong_sc_packet_(data, data->len);
    }
    return;
  }

  valid_packets_.fetch_add(1U, std::memory_order_relaxed);
}

void IRAM_ATTR CsiCaptureService::csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *data) {
  auto *service = static_cast<CsiCaptureService *>(ctx);
  if (service != nullptr) {
    service->callback_invocations_.fetch_add(1U, std::memory_order_relaxed);
    service->process_packet(data);
  }
}

esp_err_t CsiCaptureService::configure_platform_specific_() {
#ifdef CONFIG_IDF_TARGET
  ESP_LOGI(TAG, "Using %s CSI configuration", CONFIG_IDF_TARGET);
#else
  ESP_LOGI(TAG, "Using host CSI configuration");
#endif
  return configure_ht20_csi(wifi_csi_);
}

void CsiCaptureService::log_wrong_sc_packet_(const wifi_csi_info_t *data, size_t csi_len) const {
  const auto &rx = data->rx_ctrl;
#if CONFIG_SOC_WIFI_HE_SUPPORT
  ESP_LOGW(TAG,
           "Filtered %lu packets with wrong SC count (got %zu bytes, expected %d) "
           "[ch=%u bb=%u est_len=%u est_vld=%u]",
           static_cast<unsigned long>(filtered_packets_),
           csi_len,
           HT20_CSI_LEN,
           static_cast<unsigned>(rx.channel),
           static_cast<unsigned>(rx.cur_bb_format),
           static_cast<unsigned>(rx.rx_channel_estimate_len),
           static_cast<unsigned>(rx.rx_channel_estimate_info_vld));
#else
  ESP_LOGW(TAG,
           "Filtered %lu packets with wrong SC count (got %zu bytes, expected %d) "
           "[ch=%u sig_mode=%u cwb=%u mcs=%u]",
           static_cast<unsigned long>(filtered_packets_),
           csi_len,
           HT20_CSI_LEN,
           static_cast<unsigned>(rx.channel),
           static_cast<unsigned>(rx.sig_mode),
           static_cast<unsigned>(rx.cwb),
           static_cast<unsigned>(rx.mcs));
#endif
}

}  // namespace espectre
