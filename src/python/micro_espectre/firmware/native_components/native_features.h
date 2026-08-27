// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESPECTRE_NATIVE_DETECTOR_LIGHTWEIGHT = 0,
    ESPECTRE_NATIVE_DETECTOR_HIGH_ACCURACY = 1,
} espectre_native_detector_kind_t;

void *espectre_native_detector_create(
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
    uint8_t subcarrier_count
);
void espectre_native_detector_destroy(void *handle);
bool espectre_native_detector_process(
    void *handle,
    const int8_t *csi_data,
    size_t csi_length,
    uint32_t timestamp_us
);
bool espectre_native_detector_update(void *handle, float output[6]);
bool espectre_native_detector_set_subcarriers(
    void *handle,
    const uint8_t *subcarriers,
    uint8_t subcarrier_count
);
bool espectre_native_detector_advance_missing(void *handle, uint32_t count);
bool espectre_native_detector_set_minimum_valid(void *handle, uint16_t count);
bool espectre_native_detector_is_ready(void *handle);
bool espectre_native_detector_reset(void *handle);
bool espectre_native_detector_set_threshold(void *handle, float threshold);
float espectre_native_detector_get_threshold(void *handle);
float espectre_native_detector_get_metric(void *handle);
uint32_t espectre_native_detector_get_total_packets(void *handle);
bool espectre_native_detector_calibration_begin(void *handle);
bool espectre_native_detector_calibration_complete(void *handle);
bool espectre_native_detector_apply_adaptive_threshold(void *handle, float threshold);

void *espectre_native_sampler_create(uint32_t target_pps, uint32_t window_size_ms);
void espectre_native_sampler_destroy(void *handle);
bool espectre_native_sampler_configure(void *handle, uint32_t target_pps, uint32_t window_size_ms);
void espectre_native_sampler_reset(void *handle);
void espectre_native_sampler_clear_history(void *handle);
void espectre_native_sampler_clear_window_preserving_phase(void *handle);
bool espectre_native_sampler_admit(
    void *handle,
    uint32_t timestamp_us,
    bool has_timestamp,
    uint32_t now_us,
    bool has_now
);
bool espectre_native_sampler_flush(void *handle);
uint32_t espectre_native_sampler_get_u32(void *handle, uint8_t field);
uint64_t espectre_native_sampler_get_u64(void *handle, uint8_t field);
float espectre_native_sampler_get_occupancy_ratio(void *handle);
bool espectre_native_sampler_get_flag(void *handle, uint8_t field);

#ifdef __cplusplus
}
#endif
