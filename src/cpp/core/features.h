/*
 * ESPectre - Shared Feature Support
 *
 * Shared L1-delta constants plus C++ feature extraction helpers for the
 * production Core-6 ML detector. Port of src/python/micro_espectre/features.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "ml_weights.h"
#include "utils.h"

namespace esphome {
namespace espectre {

constexpr uint8_t L1_DELTA_LAG = 10;
constexpr float L1_DELTA_STARTUP_THRESHOLD_FACTOR = 1.1f;

// Number of features extracted, driven by the exported model.
constexpr uint8_t ML_NUM_FEATURES = ML_MODEL_INPUT_SIZE;
static_assert(ML_MODEL_INPUT_SIZE >= 1,
              "ML model must expose at least one input feature");

// Maximum buffer size for sorting (MAD)
constexpr uint16_t ML_MAX_SORT_SIZE = 200;

// Canonical ML feature identifiers, shared with the exporter in
// tools/10_train_ml_model.py (CPP_FEATURE_IDS).
enum MLFeatureId : uint8_t {
    ML_FEAT_TURB_SKEWNESS = 5,
    ML_FEAT_TURB_AUTOCORR = 6,
    ML_FEAT_TURB_MAD_OVER_MEAN = 13,
    ML_FEAT_L1_DELTA = 17,
    ML_FEAT_L1_DELTA_STD = 18,
    ML_FEAT_L1_DELTA_WAVEFORM_LENGTH = 23,
};

// First L1-delta feature id: ids >= this are computed from the delta series.
constexpr uint8_t ML_FEAT_L1_FIRST = ML_FEAT_L1_DELTA;

// True when the id is an L1-delta feature (needs the profile/delta rings).
inline bool ml_feature_is_l1(uint8_t id) { return id >= ML_FEAT_L1_FIRST; }

inline float calc_skewness(const float* values, uint16_t count, float mean, float std_dev) {
    if (count < 3 || std_dev < 1e-10f) return 0.0f;

    float m3 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        m3 += diff * diff * diff;
    }
    m3 /= count;

    return m3 / (std_dev * std_dev * std_dev);
}

inline float median_from_sorted(const float* sorted_values, uint16_t count) {
    if (count == 0 || sorted_values == nullptr) return 0.0f;
    if (count % 2 == 0) {
        return (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0f;
    }
    return sorted_values[count / 2];
}

inline float calc_autocorrelation(const float* values, uint16_t count, float mean,
                                  float variance, uint16_t lag = 1) {
    if (count < lag + 2 || variance < 1e-10f) return 0.0f;

    float autocovariance = 0.0f;
    for (uint16_t i = 0; i < count - lag; i++) {
        autocovariance += (values[i] - mean) * (values[i + lag] - mean);
    }
    autocovariance /= (count - lag);

    return autocovariance / variance;
}

inline float calc_mad(const float* values, uint16_t count, const float* sorted_values = nullptr) {
    if (count < 2 || count > ML_MAX_SORT_SIZE) return 0.0f;

    float sorted_scratch[ML_MAX_SORT_SIZE];
    const float* sorted = sorted_values;
    if (sorted == nullptr) {
        for (uint16_t i = 0; i < count; i++) {
            sorted_scratch[i] = values[i];
        }
        std::sort(sorted_scratch, sorted_scratch + count);
        sorted = sorted_scratch;
    }

    float median = median_from_sorted(sorted, count);

    float abs_devs[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        abs_devs[i] = std::fabs(values[i] - median);
    }

    return calculate_median_float(abs_devs, count);
}

inline float calc_waveform_length(const float* values, uint16_t count) {
    if (count < 2 || values == nullptr) return 0.0f;

    float total = 0.0f;
    float prev = values[0];
    for (uint16_t i = 1; i < count; i++) {
        float curr = values[i];
        total += std::fabs(curr - prev);
        prev = curr;
    }
    return total;
}

struct MLSeriesStats {
    uint16_t count = 0;
    float mean = 0.0f;
    float variance = 0.0f;
    float std = 0.0f;
    float mad = 0.0f;
    float skewness = 0.0f;
    float autocorr = 0.0f;
    float waveform_length = 0.0f;
    float mean_denom = 1e-6f;  // max(|mean|, 1e-6), matches Python
};

inline void compute_ml_series_stats(const float* values, uint16_t count,
                                    MLSeriesStats* out) {
    *out = MLSeriesStats{};
    if (values == nullptr || count < 2) {
        return;
    }
    out->count = count;

    float sum = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        sum += values[i];
    }
    out->mean = sum / count;

    float var_sum = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float d = values[i] - out->mean;
        var_sum += d * d;
    }
    out->variance = var_sum / count;
    out->std = out->variance > 0.0f ? std::sqrt(out->variance) : 0.0f;

    out->mad = calc_mad(values, count);
    out->skewness = calc_skewness(values, count, out->mean, out->std);
    out->autocorr = calc_autocorrelation(values, count, out->mean, out->variance, 1);
    out->waveform_length = calc_waveform_length(values, count);
    out->mean_denom = std::max(std::fabs(out->mean), 1e-6f);
}

inline float ml_feature_value_from_stats(uint8_t id, const MLSeriesStats& turb,
                                         const MLSeriesStats& delta) {
    switch (id) {
        case ML_FEAT_TURB_SKEWNESS: return turb.skewness;
        case ML_FEAT_TURB_AUTOCORR: return turb.autocorr;
        case ML_FEAT_TURB_MAD_OVER_MEAN: return turb.mad / turb.mean_denom;
        case ML_FEAT_L1_DELTA: return delta.mean;
        case ML_FEAT_L1_DELTA_STD: return delta.std;
        case ML_FEAT_L1_DELTA_WAVEFORM_LENGTH: return delta.waveform_length;
        default: return 0.0f;
    }
}

inline void extract_ml_features_by_id(const float* turb_buffer, uint16_t turb_count,
                                      const float* delta_buffer, uint16_t delta_count,
                                      const uint8_t* feature_ids, uint8_t num_features,
                                      float* features_out) {
    MLSeriesStats turb;
    compute_ml_series_stats(turb_buffer, turb_count, &turb);

    MLSeriesStats delta;
    compute_ml_series_stats(delta_buffer, delta_count, &delta);

    for (uint8_t i = 0; i < num_features; i++) {
        features_out[i] = ml_feature_value_from_stats(feature_ids[i], turb, delta);
    }
}

}  // namespace espectre
}  // namespace esphome
