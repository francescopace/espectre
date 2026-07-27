/*
 * ESPectre - Shared Feature Support
 *
 * Shared L1-delta constants plus C++ feature extraction helpers for the
 * production Coherence-7 ML detector, and for the two members Classic reads
 * directly. Port of src/python/micro_espectre/csi_features.py.
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

namespace espectre {

constexpr uint8_t L1_DELTA_LAG = 10;
constexpr float L1_DELTA_STARTUP_THRESHOLD_FACTOR = 1.1f;

// Number of features extracted, driven by the exported model.
constexpr uint8_t ML_NUM_FEATURES = ML_MODEL_INPUT_SIZE;
static_assert(ML_MODEL_INPUT_SIZE >= 1,
              "ML model must expose at least one input feature");

// Canonical ML feature identifiers, shared with the exporter in
// tools/train_ml_model.py (CPP_FEATURE_IDS).
enum MLFeatureId : uint8_t {
    ML_FEAT_TURB_AUTOCORR = 6,
    ML_FEAT_TURB_MAD_OVER_MEAN = 13,
    ML_FEAT_TURB_ZCR = 14,
    ML_FEAT_L1_DELTA = 17,
    ML_FEAT_L1_DELTA_STD = 18,
    ML_FEAT_L1_DELTA_AUTOCORR = 24,
    ML_FEAT_L1_DELTA_LAG_RATIO = 25,
};

// Where a feature's value comes from. Ids carry no ordering: a new turbulence
// feature may take any free number, so the mapping is spelled out rather than
// inferred from magnitude.
enum class MLFeatureSource : uint8_t {
    TURBULENCE_SERIES,
    L1_DELTA_SERIES,
    L1_TRACKER,
};

inline MLFeatureSource ml_feature_source(MLFeatureId id) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR:
        case ML_FEAT_TURB_MAD_OVER_MEAN:
        case ML_FEAT_TURB_ZCR:
            return MLFeatureSource::TURBULENCE_SERIES;
        case ML_FEAT_L1_DELTA:
        case ML_FEAT_L1_DELTA_STD:
        case ML_FEAT_L1_DELTA_AUTOCORR:
            return MLFeatureSource::L1_DELTA_SERIES;
        case ML_FEAT_L1_DELTA_LAG_RATIO:
            return MLFeatureSource::L1_TRACKER;
    }
    // No default label above, so -Wswitch reports a new enumerator here
    // instead of letting it inherit a neighbour's buffers. An id the enum does
    // not know needs nothing beyond the turbulence series.
    return MLFeatureSource::TURBULENCE_SERIES;
}

// True when the id needs the L1 profile rings running: the delta-series
// members do, and so does the lag ratio, which the tracker derives itself.
inline bool ml_feature_needs_l1_tracker(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) !=
           MLFeatureSource::TURBULENCE_SERIES;
}

// True when the id needs the rebuilt L1-delta series. The lag ratio does not:
// it arrives ready-made from the tracker.
inline bool ml_feature_needs_l1_series(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::L1_DELTA_SERIES;
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

/**
 * Median absolute deviation.
 *
 * Both the sorted view and the scratch are caller-owned: the only caller
 * already sorts for the zcr center, so MAD reuses that view instead of
 * sorting a second copy.
 *
 * @param sorted_values Ascending copy of `values`, `count` entries
 * @param abs_dev_scratch Scratch of at least `count` floats, overwritten
 * @param scratch_capacity Length of `abs_dev_scratch`
 */
inline float calc_mad(const float* values, uint16_t count,
                      const float* sorted_values,
                      float* abs_dev_scratch, uint16_t scratch_capacity) {
    if (values == nullptr || sorted_values == nullptr ||
        abs_dev_scratch == nullptr || count < 2 || scratch_capacity < count) {
        return 0.0f;
    }

    const float median = median_from_sorted(sorted_values, count);

    for (uint16_t i = 0; i < count; i++) {
        abs_dev_scratch[i] = std::fabs(values[i] - median);
    }

    return calculate_median_float(abs_dev_scratch, count);
}

// Crossing rate of the series around `center`. Shift and scale invariant when
// `center` tracks the window; matches the Python `calc_zero_crossing_rate`,
// whose zcr center is the upper median `sorted[count / 2]`.
inline float calc_zero_crossing_rate(const float* values, uint16_t count, float center) {
    if (count < 2 || values == nullptr) return 0.0f;

    uint16_t crossings = 0;
    bool prev_above = values[0] >= center;
    for (uint16_t i = 1; i < count; i++) {
        bool curr_above = values[i] >= center;
        if (curr_above != prev_above) {
            crossings++;
            prev_above = curr_above;
        }
    }
    return static_cast<float>(crossings) / (count - 1);
}

struct MLSeriesStats {
    uint16_t count = 0;
    float mean = 0.0f;
    float variance = 0.0f;
    float std = 0.0f;
    float mad = 0.0f;
    float autocorr = 0.0f;
    float zcr = 0.0f;
    float mean_denom = 1e-6f;  // max(|mean|, 1e-6), matches Python
};

// Which per-series statistics one feature set actually references. Lets the
// hot path skip the unused ones (notably the sort, which the delta series
// never needs under the production feature set). Mean, variance, std, and
// mean_denom are always computed: they are cheap and interdependent.
struct MLStatNeeds {
    bool sorted = false;   // mad and/or zcr (both share one sort)
    bool autocorr = false;
};

// Caller-owned working memory for the sorted statistics. The detectors size
// it to their window and keep it alive for their lifetime, so no feature
// helper allocates on the CSI callback stack.
struct MLSeriesScratch {
    float* sorted_values = nullptr;
    float* abs_devs = nullptr;
    uint16_t capacity = 0U;

    bool holds(uint16_t count) const {
        return sorted_values != nullptr && abs_devs != nullptr && capacity >= count;
    }
};

// Derive the needs of one series (turbulence when l1 is false, L1-delta when
// true) from the exported feature id list.
inline MLStatNeeds ml_series_needs(const uint8_t *feature_ids, uint8_t num_features,
                                   bool l1) {
    MLStatNeeds needs;
    const MLFeatureSource wanted = l1 ? MLFeatureSource::L1_DELTA_SERIES
                                      : MLFeatureSource::TURBULENCE_SERIES;
    for (uint8_t i = 0; i < num_features; i++) {
        const uint8_t id = feature_ids[i];
        if (ml_feature_source(static_cast<MLFeatureId>(id)) != wanted) {
            continue;
        }
        switch (id) {
            case ML_FEAT_TURB_MAD_OVER_MEAN:
            case ML_FEAT_TURB_ZCR:
                needs.sorted = true;
                break;
            case ML_FEAT_TURB_AUTOCORR:
            case ML_FEAT_L1_DELTA_AUTOCORR:
                needs.autocorr = true;
                break;
            default:  // mean / std features need no extra statistic
                break;
        }
    }
    return needs;
}

inline void compute_ml_series_stats(const float* values, uint16_t count,
                                    MLSeriesStats* out, const MLStatNeeds& needs,
                                    const MLSeriesScratch& scratch) {
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
    out->mean_denom = std::max(std::fabs(out->mean), 1e-6f);

    // Sort once; MAD and the zcr center share the sorted view.
    if (needs.sorted && scratch.holds(count)) {
        for (uint16_t i = 0; i < count; i++) {
            scratch.sorted_values[i] = values[i];
        }
        std::sort(scratch.sorted_values, scratch.sorted_values + count);
        out->mad = calc_mad(values, count, scratch.sorted_values,
                            scratch.abs_devs, scratch.capacity);
        // Python zcr centers on the upper median (sorted[count // 2]).
        out->zcr = calc_zero_crossing_rate(values, count, scratch.sorted_values[count / 2]);
    }
    if (needs.autocorr) {
        out->autocorr = calc_autocorrelation(values, count, out->mean, out->variance, 1);
    }
}

/**
 * @param l1_delta_lag_ratio Preprocessed tracker metric for
 *        ML_FEAT_L1_DELTA_LAG_RATIO. Deliberately without a default: the
 *        no-motion value of the ratio is 1.0, so a forgotten argument would
 *        read as a plausible measurement rather than as an error. The Python
 *        extractor raises for the same reason; here the compiler does it.
 */
inline float ml_feature_value_from_stats(uint8_t id, const MLSeriesStats& turb,
                                         const MLSeriesStats& delta,
                                         float l1_delta_lag_ratio) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR: return turb.autocorr;
        case ML_FEAT_TURB_MAD_OVER_MEAN: return turb.mad / turb.mean_denom;
        case ML_FEAT_TURB_ZCR: return turb.zcr;
        case ML_FEAT_L1_DELTA: return delta.mean;
        case ML_FEAT_L1_DELTA_STD: return delta.std;
        case ML_FEAT_L1_DELTA_AUTOCORR: return delta.autocorr;
        case ML_FEAT_L1_DELTA_LAG_RATIO: return l1_delta_lag_ratio;
        default: return 0.0f;
    }
}

inline void extract_ml_features_by_id(const float* turb_buffer, uint16_t turb_count,
                                      const float* delta_buffer, uint16_t delta_count,
                                      const uint8_t* feature_ids, uint8_t num_features,
                                      float* features_out,
                                      const MLSeriesScratch& series_scratch,
                                      float l1_delta_lag_ratio) {
    MLSeriesStats turb;
    compute_ml_series_stats(turb_buffer, turb_count, &turb,
                            ml_series_needs(feature_ids, num_features, /*l1=*/false),
                            series_scratch);

    MLSeriesStats delta;
    compute_ml_series_stats(delta_buffer, delta_count, &delta,
                            ml_series_needs(feature_ids, num_features, /*l1=*/true),
                            series_scratch);

    for (uint8_t i = 0; i < num_features; i++) {
        features_out[i] = ml_feature_value_from_stats(
            feature_ids[i], turb, delta, l1_delta_lag_ratio);
    }
}

}  // namespace espectre
