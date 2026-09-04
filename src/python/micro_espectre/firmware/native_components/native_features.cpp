// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#ifndef NO_QSTR

#include "native_features.h"
#include "native_log_sink.h"

#include "espectre_core_sdk.h"

#include <new>

namespace {

struct DetectorHandle {
  espectre_native_detector_kind_t kind;
  espectre::BaseDetector *detector;
  uint8_t subcarriers[espectre::HT20_SELECTED_BAND_SIZE];
  uint8_t subcarrier_count;
};

bool set_subcarriers(
    DetectorHandle *handle,
    const uint8_t *subcarriers,
    uint8_t subcarrier_count) {
  if (handle == nullptr || subcarriers == nullptr || subcarrier_count == 0U ||
      subcarrier_count > espectre::HT20_SELECTED_BAND_SIZE) {
    return false;
  }
  for (uint8_t index = 0; index < subcarrier_count; ++index) {
    if (subcarriers[index] >= espectre::HT20_NUM_SUBCARRIERS) {
      return false;
    }
    handle->subcarriers[index] = subcarriers[index];
  }
  handle->subcarrier_count = subcarrier_count;
  return true;
}

DetectorHandle *as_detector(void *handle) {
  auto *resolved = static_cast<DetectorHandle *>(handle);
  return resolved != nullptr && resolved->detector != nullptr ? resolved : nullptr;
}

espectre::TemporalCsiSampler *as_sampler(void *handle) {
  return static_cast<espectre::TemporalCsiSampler *>(handle);
}

}  // namespace

extern "C" void *espectre_native_detector_create(
    espectre_native_detector_kind_t kind,
    uint16_t window_size,
    float threshold,
    uint16_t lag,
    bool enable_hampel,
    uint8_t hampel_window,
    float hampel_threshold,
    bool enable_lowpass,
    float lowpass_cutoff,
    const uint8_t *subcarriers,
    uint8_t subcarrier_count) {
  espectre_native_ensure_log_sink();
  auto *handle = new (std::nothrow) DetectorHandle{
      kind,
      nullptr,
      {},
      0U,
  };
  if (handle == nullptr || !set_subcarriers(handle, subcarriers, subcarrier_count)) {
    delete handle;
    return nullptr;
  }
  if (kind != ESPECTRE_NATIVE_DETECTOR_LIGHTWEIGHT) {
    delete handle;
    return nullptr;
  }
  handle->detector = new (std::nothrow) espectre::LightweightDetector(
      window_size,
      threshold,
      lag);
  if (handle->detector == nullptr || !handle->detector->is_valid()) {
    delete handle->detector;
    delete handle;
    return nullptr;
  }
  handle->detector->configure_hampel(
      enable_hampel,
      hampel_window,
      hampel_threshold);
  handle->detector->configure_lowpass(enable_lowpass, lowpass_cutoff);
  return handle;
}

extern "C" void espectre_native_detector_destroy(void *handle) {
  auto *resolved = static_cast<DetectorHandle *>(handle);
  if (resolved == nullptr) {
    return;
  }
  delete resolved->detector;
  resolved->detector = nullptr;
  delete resolved;
}

extern "C" bool espectre_native_detector_process(
    void *handle,
    const int8_t *csi_data,
    size_t csi_length,
    uint32_t timestamp_us) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr || csi_data == nullptr) {
    return false;
  }
  resolved->detector->set_packet_timestamp_us(timestamp_us);
  resolved->detector->process_packet(
      csi_data,
      csi_length,
      resolved->subcarriers,
      resolved->subcarrier_count);
  return true;
}

extern "C" bool espectre_native_detector_update(void *handle, float output[6]) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr || output == nullptr) {
    return false;
  }
  resolved->detector->update_state();
  output[0] = resolved->detector->get_state() == espectre::MotionState::MOTION ? 1.0f : 0.0f;
  output[1] = resolved->detector->get_motion_metric();
  output[2] = resolved->detector->get_threshold();
  output[3] = 0.0f;
  output[4] = 0.0f;
  output[5] = 0.0f;
  if (resolved->kind == ESPECTRE_NATIVE_DETECTOR_LIGHTWEIGHT) {
    auto *detector = static_cast<espectre::LightweightDetector *>(resolved->detector);
    output[3] = detector->get_turb_autocorr();
    output[4] = detector->get_turb_iqr_over_mean_aggr();
    output[5] = detector->get_logit();
  }
  return true;
}

extern "C" bool espectre_native_detector_set_subcarriers(
    void *handle,
    const uint8_t *subcarriers,
    uint8_t subcarrier_count) {
  return set_subcarriers(as_detector(handle), subcarriers, subcarrier_count);
}

extern "C" bool espectre_native_detector_advance_missing(void *handle, uint32_t count) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr) {
    return false;
  }
  resolved->detector->advance_missing_slots(count);
  return true;
}

extern "C" bool espectre_native_detector_set_minimum_valid(void *handle, uint16_t count) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr) {
    return false;
  }
  resolved->detector->set_minimum_valid_samples(count);
  return true;
}

extern "C" bool espectre_native_detector_is_ready(void *handle) {
  auto *resolved = as_detector(handle);
  return resolved != nullptr && resolved->detector->is_ready();
}

extern "C" bool espectre_native_detector_reset(void *handle) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr) {
    return false;
  }
  resolved->detector->clear_buffer();
  return true;
}

extern "C" bool espectre_native_detector_set_threshold(void *handle, float threshold) {
  auto *resolved = as_detector(handle);
  return resolved != nullptr && resolved->detector->set_threshold(threshold);
}

extern "C" float espectre_native_detector_get_threshold(void *handle) {
  auto *resolved = as_detector(handle);
  return resolved == nullptr ? 0.0f : resolved->detector->get_threshold();
}

extern "C" float espectre_native_detector_get_metric(void *handle) {
  auto *resolved = as_detector(handle);
  return resolved == nullptr ? 0.0f : resolved->detector->get_motion_metric();
}

extern "C" uint32_t espectre_native_detector_get_total_packets(void *handle) {
  auto *resolved = as_detector(handle);
  return resolved == nullptr ? 0U : resolved->detector->get_total_packets();
}

extern "C" bool espectre_native_detector_calibration_begin(void *handle) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr) {
    return false;
  }
  resolved->detector->on_startup_calibration_begin();
  return true;
}

extern "C" bool espectre_native_detector_calibration_complete(void *handle) {
  auto *resolved = as_detector(handle);
  if (resolved == nullptr) {
    return false;
  }
  resolved->detector->on_startup_calibration_complete();
  return true;
}

extern "C" bool espectre_native_detector_apply_adaptive_threshold(
    void *handle,
    float threshold) {
  auto *resolved = as_detector(handle);
  return resolved != nullptr && resolved->detector->set_adaptive_threshold(threshold);
}

extern "C" void *espectre_native_sampler_create(
    uint32_t target_pps,
    uint32_t window_size_ms) {
  espectre_native_ensure_log_sink();
  auto *sampler = new (std::nothrow) espectre::TemporalCsiSampler(
      target_pps,
      window_size_ms);
  if (sampler == nullptr || !sampler->is_valid()) {
    delete sampler;
    return nullptr;
  }
  return sampler;
}

extern "C" void espectre_native_sampler_destroy(void *handle) {
  delete as_sampler(handle);
}

extern "C" bool espectre_native_sampler_configure(
    void *handle,
    uint32_t target_pps,
    uint32_t window_size_ms) {
  auto *sampler = as_sampler(handle);
  return sampler != nullptr && sampler->configure(target_pps, window_size_ms);
}

extern "C" void espectre_native_sampler_reset(void *handle) {
  auto *sampler = as_sampler(handle);
  if (sampler != nullptr) {
    sampler->reset();
  }
}

extern "C" void espectre_native_sampler_clear_history(void *handle) {
  auto *sampler = as_sampler(handle);
  if (sampler != nullptr) {
    sampler->clear_history();
  }
}

extern "C" void espectre_native_sampler_clear_window_preserving_phase(void *handle) {
  auto *sampler = as_sampler(handle);
  if (sampler != nullptr) {
    sampler->clear_window_preserving_phase();
  }
}

extern "C" bool espectre_native_sampler_admit(
    void *handle,
    uint32_t timestamp_us,
    bool has_timestamp,
    uint32_t now_us,
    bool has_now) {
  auto *sampler = as_sampler(handle);
  return sampler != nullptr && sampler->admit(timestamp_us, has_timestamp, now_us, has_now);
}

extern "C" bool espectre_native_sampler_flush(void *handle) {
  auto *sampler = as_sampler(handle);
  return sampler != nullptr && sampler->flush();
}

extern "C" uint32_t espectre_native_sampler_get_u32(void *handle, uint8_t field) {
  auto *sampler = as_sampler(handle);
  if (sampler == nullptr) {
    return 0U;
  }
  switch (field) {
    case 0:
      return sampler->target_pps();
    case 1:
      return sampler->window_size_ms();
    case 2:
      return sampler->window_slots();
    case 3:
      return sampler->minimum_valid_slots();
    case 4:
      return sampler->minimum_sample_spacing_us();
    case 5:
      return sampler->occupancy_slots();
    default:
      return 0U;
  }
}

extern "C" uint64_t espectre_native_sampler_get_u64(void *handle, uint8_t field) {
  auto *sampler = as_sampler(handle);
  if (sampler == nullptr) {
    return 0U;
  }
  switch (field) {
    case 0:
      return sampler->current_slot();
    case 1:
      return sampler->slots_advanced();
    case 2:
      return sampler->missing_slots_before();
    case 3:
      return sampler->accepted_packets();
    case 4:
      return sampler->excess_packets();
    case 5:
      return sampler->duplicate_packets();
    case 6:
      return sampler->out_of_order_packets();
    case 7:
      return sampler->stale_packets();
    case 8:
      return sampler->missing_timestamp_packets();
    case 9:
      return sampler->missing_slots();
    case 10:
      return sampler->gap_resets();
    default:
      return 0U;
  }
}

extern "C" float espectre_native_sampler_get_occupancy_ratio(void *handle) {
  auto *sampler = as_sampler(handle);
  return sampler == nullptr ? 0.0f : sampler->occupancy_ratio();
}

extern "C" bool espectre_native_sampler_get_flag(void *handle, uint8_t field) {
  auto *sampler = as_sampler(handle);
  if (sampler == nullptr) {
    return false;
  }
  switch (field) {
    case 0:
      return sampler->is_ready();
    case 1:
      return sampler->accepted();
    case 2:
      return sampler->selected_current();
    case 3:
      return sampler->has_pending_candidate();
    case 4:
      return sampler->reset_required();
    case 5:
      return sampler->gap_reset_required();
    default:
      return false;
  }
}

#endif  // NO_QSTR
