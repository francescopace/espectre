/*
 * ESPectre - Shared Feature Support
 *
 * Shared L1-delta constants plus C++ feature extraction helpers for the
 * production scale-invariant ML feature set, and for the two members Lightweight
 * reads directly. Every feature is a ratio, a correlation, or a crossing rate:
 * the per-packet CSI scaling factor is never recorded, so anything carrying
 * absolute magnitude carries the link's noise floor with it.
 * Port of src/python/micro_espectre/csi_features.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
constexpr uint8_t TURB_IQR_AGGREGATION_WIDTH = 5U;

// Number of features extracted, driven by the exported model.
constexpr uint8_t ML_NUM_FEATURES = ML_MODEL_INPUT_SIZE;
static_assert(ML_MODEL_INPUT_SIZE >= 1,
              "ML model must expose at least one input feature");

// Canonical ML feature identifiers, shared with the exporter in
// tools/train_ml_model.py (CPP_FEATURE_IDS).
enum MLFeatureId : uint8_t {
    ML_FEAT_TURB_AUTOCORR = 6,
    ML_FEAT_TURB_ZCR = 14,
    ML_FEAT_L1_DELTA_LAG_RATIO = 25,
    ML_FEAT_TURB_IQR_OVER_MEAN_AGGR = 45,
    ML_FEAT_CHAN_SHAPE_COHERENT_INNOVATION_ENERGY = 46,
    ML_FEAT_CHAN_SHAPE_EXCESS_PATH = 47,
    ML_FEAT_CHAN_SHAPE_SPREAD_SUBBAND = 48,
};

// Where a feature's value comes from. Ids carry no ordering: a new turbulence
// feature may take any free number, so the mapping is spelled out rather than
// inferred from magnitude.
enum class MLFeatureSource : uint8_t {
    TURBULENCE_SERIES,
    AGGREGATED_TURBULENCE_SERIES,
    L1_TRACKER,
    CHANNEL_SHAPE_TRAJECTORY_TRACKER,
};

inline MLFeatureSource ml_feature_source(MLFeatureId id) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR:
        case ML_FEAT_TURB_ZCR:
            return MLFeatureSource::TURBULENCE_SERIES;
        case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
            return MLFeatureSource::AGGREGATED_TURBULENCE_SERIES;
        case ML_FEAT_L1_DELTA_LAG_RATIO:
            return MLFeatureSource::L1_TRACKER;
        case ML_FEAT_CHAN_SHAPE_COHERENT_INNOVATION_ENERGY:
        case ML_FEAT_CHAN_SHAPE_EXCESS_PATH:
        case ML_FEAT_CHAN_SHAPE_SPREAD_SUBBAND:
            return MLFeatureSource::CHANNEL_SHAPE_TRAJECTORY_TRACKER;
    }
    // No default label above, so -Wswitch reports a new enumerator here
    // instead of letting it inherit a neighbour's buffers. An id the enum does
    // not know needs nothing beyond the turbulence series.
    return MLFeatureSource::TURBULENCE_SERIES;
}

// True when the id needs the L1 profile rings running: the delta-series
// members do, and so does the lag ratio, which the tracker derives itself.
inline bool ml_feature_needs_l1_tracker(uint8_t id) {
    const MLFeatureSource source =
        ml_feature_source(static_cast<MLFeatureId>(id));
    return source == MLFeatureSource::L1_TRACKER;
}

inline bool ml_feature_needs_channel_shape_trajectory_tracker(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::CHANNEL_SHAPE_TRAJECTORY_TRACKER;
}

inline bool ml_feature_needs_aggregated_turbulence(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::AGGREGATED_TURBULENCE_SERIES;
}

inline float median_from_sorted(const float* sorted_values, uint16_t count) {
    if (count == 0 || sorted_values == nullptr) return 0.0f;
    if (count % 2 == 0) {
        return (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0f;
    }
    return sorted_values[count / 2];
}

inline float percentile_from_sorted(const float* sorted_values, uint16_t count,
                                    float quantile) {
    if (sorted_values == nullptr || count == 0U) return 0.0f;
    const float position = static_cast<float>(count - 1U) * quantile;
    const uint16_t lower = static_cast<uint16_t>(position);
    if (lower >= count - 1U) return sorted_values[count - 1U];
    const float fraction = position - static_cast<float>(lower);
    return sorted_values[lower] * (1.0f - fraction) +
           sorted_values[lower + 1U] * fraction;
}

inline float order_statistic_in_place(float* values, uint16_t count,
                                      uint16_t index) {
    std::nth_element(values, values + index, values + count);
    return values[index];
}

inline float percentile_in_place(float* values, uint16_t count,
                                 float quantile) {
    if (values == nullptr || count == 0U) return 0.0f;
    const float position = static_cast<float>(count - 1U) * quantile;
    const uint16_t lower = static_cast<uint16_t>(position);
    const float lower_value = order_statistic_in_place(values, count, lower);
    if (lower >= count - 1U) return lower_value;
    const float upper_value = order_statistic_in_place(
        values, count, static_cast<uint16_t>(lower + 1U));
    const float fraction = position - static_cast<float>(lower);
    return lower_value * (1.0f - fraction) + upper_value * fraction;
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
    float iqr = 0.0f;
    float autocorr = 0.0f;
    float zcr = 0.0f;
    float mean_denom = 1e-6f;  // max(|mean|, 1e-6), matches Python
};

// Which per-series statistics one feature set actually references. Lets the
// hot path skip unused passes over the window.
struct MLStatNeeds {
    bool mean = false;
    bool variance = false;
    bool sorted = false;   // iqr and/or zcr share one sort
    bool iqr = false;
    bool zcr = false;
    bool autocorr = false;
};

// Caller-owned working memory for the sorted statistics. The detectors size
// it to their window and keep it alive for their lifetime, so no feature
// helper allocates on the CSI callback stack.
//
// One sorted view is reused by the turbulence and aggregated-turbulence
// series. MLSeriesStats contains only scalars, so each result is materialised
// before the next call overwrites the view.
struct MLSeriesScratch {
    float* sorted_values = nullptr;
    uint16_t capacity = 0U;

    bool holds(uint16_t count) const {
        return sorted_values != nullptr && capacity >= count;
    }
};

// Derive the statistics needed by one production series from the exported ids.
inline MLStatNeeds ml_series_needs(const uint8_t *feature_ids,
                                   uint8_t num_features,
                                   MLFeatureSource wanted) {
    MLStatNeeds needs;
    for (uint8_t i = 0; i < num_features; i++) {
        const uint8_t id = feature_ids[i];
        if (ml_feature_source(static_cast<MLFeatureId>(id)) != wanted) {
            continue;
        }
        switch (id) {
            case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
                needs.mean = true;
                needs.sorted = true;
                needs.iqr = true;
                break;
            case ML_FEAT_TURB_ZCR:
                needs.sorted = true;
                needs.zcr = true;
                break;
            case ML_FEAT_TURB_AUTOCORR:
                needs.mean = true;
                needs.variance = true;
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

    if (needs.variance) {
        const MeanVariance moments =
            calculate_mean_variance_two_pass(values, count);
        out->mean = moments.mean;
        out->variance = moments.variance;
    } else if (needs.mean) {
        out->mean = calculate_mean(values, count);
    }
    if (needs.mean) {
        out->mean_denom = std::max(std::fabs(out->mean), 1e-6f);
    }

    // Select only the order statistics each feature consumes. A full sort
    // produces the same values but orders the other window elements needlessly.
    if (needs.sorted && scratch.holds(count)) {
        for (uint16_t i = 0; i < count; i++) {
            scratch.sorted_values[i] = values[i];
        }
        if (needs.iqr) {
            out->iqr = percentile_in_place(
                scratch.sorted_values, count, 0.75f) -
                percentile_in_place(scratch.sorted_values, count, 0.25f);
        }
        if (needs.zcr) {
            // Python zcr centers on the upper median (sorted[count // 2]).
            const float center = order_statistic_in_place(
                scratch.sorted_values, count,
                static_cast<uint16_t>(count / 2U));
            out->zcr = calc_zero_crossing_rate(
                values, count, center);
        }
    }
    if (needs.autocorr) {
        out->autocorr = calc_autocorrelation(values, count, out->mean, out->variance, 1);
    }
}

/**
 * Resolve one exported ML feature from precomputed series and tracker stats.
 *
 * @param id Exported `MLFeatureId` value to resolve.
 * @param turb Statistics for the packet-level turbulence series.
 * @param aggregated_turb Statistics for the aggregated turbulence series.
 * @param l1_delta_lag_ratio Preprocessed tracker metric for
 *        ML_FEAT_L1_DELTA_LAG_RATIO. Deliberately without a default: the
 *        no-motion value of the ratio is 1.0, so a forgotten argument would
 *        read as a plausible measurement rather than as an error. The Python
 *        extractor raises for the same reason; here the compiler does it.
 * @param chan_shape_spread_subband Current physical-time subband spread.
 * @param chan_shape_coherent_innovation_energy Current coherent innovation
 *        energy from the channel-shape trajectory tracker.
 * @param chan_shape_excess_path Current channel-shape excess-path metric.
 * @return The requested feature value, or `0.0f` for an unknown identifier.
 */
inline float ml_feature_value_from_stats(uint8_t id, const MLSeriesStats& turb,
                                         const MLSeriesStats& aggregated_turb,
                                         float l1_delta_lag_ratio,
                                         float chan_shape_spread_subband,
                                         float chan_shape_coherent_innovation_energy,
                                         float chan_shape_excess_path) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR: return turb.autocorr;
        case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
            return aggregated_turb.iqr / aggregated_turb.mean_denom;
        case ML_FEAT_TURB_ZCR: return turb.zcr;
        case ML_FEAT_L1_DELTA_LAG_RATIO: return l1_delta_lag_ratio;
        case ML_FEAT_CHAN_SHAPE_SPREAD_SUBBAND:
            return chan_shape_spread_subband;
        case ML_FEAT_CHAN_SHAPE_COHERENT_INNOVATION_ENERGY:
            return chan_shape_coherent_innovation_energy;
        case ML_FEAT_CHAN_SHAPE_EXCESS_PATH: return chan_shape_excess_path;
        default: return 0.0f;
    }
}

inline void extract_ml_features_by_id(const float* turb_buffer, uint16_t turb_count,
                                      const float* aggregated_turb_buffer,
                                      uint16_t aggregated_turb_count,
                                      const uint8_t* feature_ids, uint8_t num_features,
                                      float* features_out,
                                      const MLSeriesScratch& series_scratch,
                                      float l1_delta_lag_ratio,
                                      float chan_shape_spread_subband,
                                      float chan_shape_coherent_innovation_energy,
                                      float chan_shape_excess_path) {
    // The aggregated chronological view may alias `series_scratch`; consume it
    // first so the normal turbulence sort can reuse the same block afterwards.
    MLSeriesStats aggregated_turb;
    compute_ml_series_stats(
        aggregated_turb_buffer, aggregated_turb_count, &aggregated_turb,
        ml_series_needs(
            feature_ids, num_features,
            MLFeatureSource::AGGREGATED_TURBULENCE_SERIES),
        series_scratch);

    MLSeriesStats turb;
    compute_ml_series_stats(turb_buffer, turb_count, &turb,
                            ml_series_needs(
                                feature_ids, num_features,
                                MLFeatureSource::TURBULENCE_SERIES),
                            series_scratch);

    for (uint8_t i = 0; i < num_features; i++) {
        features_out[i] = ml_feature_value_from_stats(
            feature_ids[i], turb, aggregated_turb,
            l1_delta_lag_ratio, chan_shape_spread_subband,
            chan_shape_coherent_innovation_energy, chan_shape_excess_path);
    }
}

}  // namespace espectre
