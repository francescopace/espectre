/*
 * ESPectre - CSI Capture Service
 *
 * Enables ESP-IDF CSI capture, normalizes payloads, and forwards valid
 * packets.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "csi_capture_service.h"

#include <cinttypes>

#include "csi_format.h"
#include "csi_platform_config.h"
#include "espectre_log.h"
#include "sdkconfig.h"

namespace espectre {

namespace {

static const char *const TAG = "CsiCapture";
constexpr uint32_t kHealthSampleIntervalMs = 2000U;
constexpr uint32_t kHealthRearmCooldownMs = 10000U;
constexpr uint64_t kHealthMinPacingPackets = 20U;
constexpr uint64_t kHealthMinCallbackPercent = 50U;
constexpr uint8_t kHealthLowSupplyWindowsBeforeRearm = 2U;

struct CsiPayloadView {
  const int8_t *data;
  size_t len;
  bool selected_ht_ltf;
};

CsiPayloadView select_csi_payload_(const wifi_csi_info_t *info) {
  CsiPayloadView view{info->buf, info->len, false};
#if CONFIG_IDF_TARGET_ESP32
  // With LLTF and HT-LTF enabled, ESP-IDF stores them in that order. Keep the
  // HT-LTF half for HT frames so enabling legacy fallback does not change the
  // established 802.11n sample representation.
  if (info->rx_ctrl.sig_mode == 1U) {
    if (view.len == HT20_CSI_LEN_DOUBLE) {
      view.data += HT20_CSI_LEN;
      view.len = HT20_CSI_LEN;
      view.selected_ht_ltf = true;
    } else if (view.len == HT20_CSI_LEN_SHORT_DOUBLE) {
      view.data += HT20_CSI_LEN_SHORT;
      view.len = HT20_CSI_LEN_SHORT;
      view.selected_ht_ltf = true;
    }
  }
#endif
  return view;
}

}  // namespace

void CsiCaptureService::init(IWiFiCSI *wifi_csi) {
  wifi_csi_ = wifi_csi ? wifi_csi : &default_wifi_csi_;
  reset_session();
}

void CsiCaptureService::reset_session() {
  filtered_packets_.store(0U, std::memory_order_relaxed);
  callback_invocations_.store(0U, std::memory_order_relaxed);
  null_or_empty_packets_.store(0U, std::memory_order_relaxed);
  normalized_invalid_packets_.store(0U, std::memory_order_relaxed);
  valid_packets_.store(0U, std::memory_order_relaxed);
  enable_attempts_.store(0U, std::memory_order_relaxed);
  disable_attempts_.store(0U, std::memory_order_relaxed);
  last_configure_err_.store(ESP_OK, std::memory_order_relaxed);
  last_set_callback_err_.store(ESP_OK, std::memory_order_relaxed);
  last_set_enabled_err_.store(ESP_OK, std::memory_order_relaxed);
  last_disable_err_.store(ESP_OK, std::memory_order_relaxed);
  collapse_seen_.store(false, std::memory_order_relaxed);
  remap_seen_.store(false, std::memory_order_relaxed);
  collapse_log_event_.clear();
  remap_log_event_.clear();
  health_prev_pacing_total_ = 0U;
  health_prev_callback_total_ = 0U;
  health_last_sample_ms_ = 0U;
  health_last_rearm_ms_ = 0U;
  health_low_supply_windows_ = 0U;
  health_baseline_valid_ = false;
}

void CsiCaptureService::loop() {
  if (collapse_log_event_.take()) {
    ESP_LOGI(TAG, "CSI double-length collapse active: 256->128 and/or 228->114");
  }
  if (remap_log_event_.take()) {
    ESP_LOGI(TAG, "CSI remap active: 57->64 SC (left_pad=4, right_pad=3)");
  }
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

  const CsiPayloadView payload = select_csi_payload_(data);
  int8_t csi_remapped[HT20_CSI_LEN];
  const NormalizedCSIPayload normalized =
      normalize_ht20_csi_payload(payload.data, payload.len, csi_remapped, sizeof(csi_remapped));

  if (normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT20 ||
      normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64) {
    if (!collapse_seen_.exchange(true, std::memory_order_relaxed)) {
      collapse_log_event_.post();
    }
  }

  if (normalized.tag == NormalizedCSIPayloadTag::HT57_TO_64 ||
      normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64) {
    if (!remap_seen_.exchange(true, std::memory_order_relaxed)) {
      remap_log_event_.post();
    }
  }

  if (packet_callback_) {
    if (payload.selected_ht_ltf) {
      wifi_csi_info_t selected_info = *data;
      selected_info.buf = const_cast<int8_t *>(payload.data);
      selected_info.len = static_cast<uint16_t>(payload.len);
      selected_info.first_word_invalid = false;
      packet_callback_(packet_callback_context_, &selected_info, normalized);
    } else {
      packet_callback_(packet_callback_context_, data, normalized);
    }
  }
  if (!normalized.valid() || normalized.len != HT20_CSI_LEN) {
    filtered_packets_.fetch_add(1U, std::memory_order_relaxed);
    normalized_invalid_packets_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }

  valid_packets_.fetch_add(1U, std::memory_order_relaxed);
}

CsiCaptureService::HealthAction CsiCaptureService::maintain_pacing_health(uint64_t pacing_total, uint32_t now_ms) {
#if !CONFIG_IDF_TARGET_ESP32
  (void)pacing_total;
  (void)now_ms;
  return HealthAction::NONE;
#else
  const uint32_t callback_total = callback_invocations();
  if (!enabled_) {
    health_baseline_valid_ = false;
    health_low_supply_windows_ = 0U;
    return HealthAction::NONE;
  }

  if (!health_baseline_valid_) {
    health_prev_pacing_total_ = pacing_total;
    health_prev_callback_total_ = callback_total;
    health_last_sample_ms_ = now_ms;
    health_baseline_valid_ = true;
    return HealthAction::NONE;
  }

  if (static_cast<uint32_t>(now_ms - health_last_sample_ms_) < kHealthSampleIntervalMs) {
    return HealthAction::NONE;
  }

  const uint64_t pacing_delta = pacing_total >= health_prev_pacing_total_
                                    ? pacing_total - health_prev_pacing_total_
                                    : pacing_total;
  const uint32_t callback_delta = callback_total >= health_prev_callback_total_
                                      ? callback_total - health_prev_callback_total_
                                      : callback_total;
  health_prev_pacing_total_ = pacing_total;
  health_prev_callback_total_ = callback_total;
  health_last_sample_ms_ = now_ms;

  if (pacing_delta < kHealthMinPacingPackets ||
      static_cast<uint64_t>(callback_delta) * 100U >= pacing_delta * kHealthMinCallbackPercent) {
    health_low_supply_windows_ = 0U;
    return HealthAction::NONE;
  }

  if (health_low_supply_windows_ < kHealthLowSupplyWindowsBeforeRearm) {
    health_low_supply_windows_++;
  }
  if (health_low_supply_windows_ < kHealthLowSupplyWindowsBeforeRearm) {
    return HealthAction::NONE;
  }
  if (health_last_rearm_ms_ != 0U &&
      static_cast<uint32_t>(now_ms - health_last_rearm_ms_) < kHealthRearmCooldownMs) {
    return HealthAction::NONE;
  }

  ESP_LOGW(TAG,
           "CSI callback supply stalled: callbacks=%" PRIu32 " pacing=%" PRIu64 "; rearming CSI",
           callback_delta,
           pacing_delta);
  health_last_rearm_ms_ = now_ms;
  health_low_supply_windows_ = 0U;
  if (disable() != ESP_OK || enable() != ESP_OK) {
    ESP_LOGE(TAG, "CSI rearm failed");
    return HealthAction::REARM_FAILED;
  }

  ESP_LOGI(TAG, "CSI rearmed after sustained callback deficit");
  return HealthAction::REARMED;
#endif
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

}  // namespace espectre
