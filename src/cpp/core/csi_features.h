/*
 * ESPectre - Shared Feature Support
 *
 * Shared L1-delta constants plus C++ feature extraction helpers for the
 * production scale-invariant ML feature set, and for the two members Classic
 * reads directly. Every feature is a ratio, a correlation, or a crossing rate:
 * the per-packet CSI scaling factor is never recorded, so anything carrying
 * absolute magnitude carries the link's noise floor with it.
 * Port of src/python/micro_espectre/csi_features.py.
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
constexpr uint8_t TURB_IQR_AGGREGATION_WIDTH = 5U;

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
    ML_FEAT_L1_DELTA_AUTOCORR = 24,
    ML_FEAT_L1_DELTA_LAG_RATIO = 25,
    ML_FEAT_CHAN_SHAPE_SPREAD = 40,
    ML_FEAT_CHAN_FREQ_COH_CV = 41,
    ML_FEAT_CHAN_FREQ_COH_CURVE_STD = 42,
    ML_FEAT_CHAN_COH_GAP = 43,
    ML_FEAT_CHAN_COH_SUBBAND_GAP_MEDIAN = 44,
    ML_FEAT_TURB_IQR_OVER_MEAN_AGGR = 45,
};

// Where a feature's value comes from. Ids carry no ordering: a new turbulence
// feature may take any free number, so the mapping is spelled out rather than
// inferred from magnitude.
enum class MLFeatureSource : uint8_t {
    TURBULENCE_SERIES,
    AGGREGATED_TURBULENCE_SERIES,
    L1_DELTA_SERIES,
    L1_TRACKER,
    CHANNEL_SHAPE_TRACKER,
    CHANNEL_COHERENCE_TRACKER,
};

inline MLFeatureSource ml_feature_source(MLFeatureId id) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR:
        case ML_FEAT_TURB_MAD_OVER_MEAN:
        case ML_FEAT_TURB_ZCR:
            return MLFeatureSource::TURBULENCE_SERIES;
        case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
            return MLFeatureSource::AGGREGATED_TURBULENCE_SERIES;
        case ML_FEAT_L1_DELTA_AUTOCORR:
            return MLFeatureSource::L1_DELTA_SERIES;
        case ML_FEAT_L1_DELTA_LAG_RATIO:
            return MLFeatureSource::L1_TRACKER;
        case ML_FEAT_CHAN_SHAPE_SPREAD:
        case ML_FEAT_CHAN_FREQ_COH_CV:
        case ML_FEAT_CHAN_FREQ_COH_CURVE_STD:
            return MLFeatureSource::CHANNEL_SHAPE_TRACKER;
        case ML_FEAT_CHAN_COH_GAP:
        case ML_FEAT_CHAN_COH_SUBBAND_GAP_MEDIAN:
            return MLFeatureSource::CHANNEL_COHERENCE_TRACKER;
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
    return source == MLFeatureSource::L1_DELTA_SERIES ||
           source == MLFeatureSource::L1_TRACKER;
}

// True when the id needs the rebuilt L1-delta series. The lag ratio does not:
// it arrives ready-made from the tracker.
inline bool ml_feature_needs_l1_series(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::L1_DELTA_SERIES;
}

inline bool ml_feature_needs_channel_shape_tracker(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::CHANNEL_SHAPE_TRACKER;
}

inline bool ml_feature_needs_channel_coherence_tracker(uint8_t id) {
    return ml_feature_source(static_cast<MLFeatureId>(id)) ==
           MLFeatureSource::CHANNEL_COHERENCE_TRACKER;
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
    float iqr = 0.0f;
    float autocorr = 0.0f;
    float zcr = 0.0f;
    float mean_denom = 1e-6f;  // max(|mean|, 1e-6), matches Python
};

// Which per-series statistics one feature set actually references. Lets the
// hot path skip the unused ones (notably the sort, which the delta series
// never needs under the production feature set). Mean, variance, std, and
// mean_denom are always computed: they are cheap and interdependent.
struct MLStatNeeds {
    bool sorted = false;   // mad, iqr, and/or zcr share one sort
    bool mad = false;
    bool iqr = false;
    bool zcr = false;
    bool autocorr = false;
};

// Caller-owned working memory for the sorted statistics. The detectors size
// it to their window and keep it alive for their lifetime, so no feature
// helper allocates on the CSI callback stack.
//
// One scratch is reused across all series within a single feature extraction
// (normal turbulence, aggregated turbulence, then L1 delta). That is safe only
// because MLSeriesStats holds scalars: each call's results are fully
// materialised before the next call overwrites the buffers. A statistic that
// returned a pointer or a view into the sorted values would silently read the
// wrong series, with no compiler diagnostic and no test failure, because every
// series produces plausible numbers. Keep MLSeriesStats pointer-free, or give
// each series its own slice.
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
            case ML_FEAT_TURB_MAD_OVER_MEAN:
                needs.sorted = true;
                needs.mad = true;
                break;
            case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
                needs.sorted = true;
                needs.iqr = true;
                break;
            case ML_FEAT_TURB_ZCR:
                needs.sorted = true;
                needs.zcr = true;
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

    const MeanVariance moments = calculate_mean_variance_two_pass(values, count);
    out->mean = moments.mean;
    out->variance = moments.variance;
    out->std = out->variance > 0.0f ? std::sqrt(out->variance) : 0.0f;
    out->mean_denom = std::max(std::fabs(out->mean), 1e-6f);

    // Sort once; MAD and the zcr center share the sorted view.
    if (needs.sorted && scratch.holds(count)) {
        for (uint16_t i = 0; i < count; i++) {
            scratch.sorted_values[i] = values[i];
        }
        std::sort(scratch.sorted_values, scratch.sorted_values + count);
        if (needs.mad) {
            out->mad = calc_mad(values, count, scratch.sorted_values,
                                scratch.abs_devs, scratch.capacity);
        }
        if (needs.iqr) {
            out->iqr = percentile_from_sorted(
                scratch.sorted_values, count, 0.75f) -
                percentile_from_sorted(scratch.sorted_values, count, 0.25f);
        }
        if (needs.zcr) {
            // Python zcr centers on the upper median (sorted[count // 2]).
            out->zcr = calc_zero_crossing_rate(
                values, count, scratch.sorted_values[count / 2]);
        }
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
                                         const MLSeriesStats& aggregated_turb,
                                         const MLSeriesStats& delta,
                                         float l1_delta_lag_ratio,
                                         float chan_shape_spread,
                                         float chan_freq_coh_cv,
                                         float chan_freq_coh_curve_std,
                                         float chan_coh_gap,
                                         float chan_coh_subband_gap_median) {
    switch (id) {
        case ML_FEAT_TURB_AUTOCORR: return turb.autocorr;
        case ML_FEAT_TURB_MAD_OVER_MEAN: return turb.mad / turb.mean_denom;
        case ML_FEAT_TURB_IQR_OVER_MEAN_AGGR:
            return aggregated_turb.iqr / aggregated_turb.mean_denom;
        case ML_FEAT_TURB_ZCR: return turb.zcr;
        case ML_FEAT_L1_DELTA_AUTOCORR: return delta.autocorr;
        case ML_FEAT_L1_DELTA_LAG_RATIO: return l1_delta_lag_ratio;
        case ML_FEAT_CHAN_SHAPE_SPREAD: return chan_shape_spread;
        case ML_FEAT_CHAN_FREQ_COH_CV: return chan_freq_coh_cv;
        case ML_FEAT_CHAN_FREQ_COH_CURVE_STD: return chan_freq_coh_curve_std;
        case ML_FEAT_CHAN_COH_GAP: return chan_coh_gap;
        case ML_FEAT_CHAN_COH_SUBBAND_GAP_MEDIAN: return chan_coh_subband_gap_median;
        default: return 0.0f;
    }
}

inline void extract_ml_features_by_id(const float* turb_buffer, uint16_t turb_count,
                                      const float* aggregated_turb_buffer,
                                      uint16_t aggregated_turb_count,
                                      const float* delta_buffer, uint16_t delta_count,
                                      const uint8_t* feature_ids, uint8_t num_features,
                                      float* features_out,
                                      const MLSeriesScratch& series_scratch,
                                      float l1_delta_lag_ratio,
                                      float chan_shape_spread,
                                      float chan_freq_coh_cv,
                                      float chan_freq_coh_curve_std,
                                      float chan_coh_gap,
                                      float chan_coh_subband_gap_median) {
    MLSeriesStats turb;
    compute_ml_series_stats(turb_buffer, turb_count, &turb,
                            ml_series_needs(
                                feature_ids, num_features,
                                MLFeatureSource::TURBULENCE_SERIES),
                            series_scratch);

    MLSeriesStats aggregated_turb;
    compute_ml_series_stats(
        aggregated_turb_buffer, aggregated_turb_count, &aggregated_turb,
        ml_series_needs(
            feature_ids, num_features,
            MLFeatureSource::AGGREGATED_TURBULENCE_SERIES),
        series_scratch);

    MLSeriesStats delta;
    compute_ml_series_stats(delta_buffer, delta_count, &delta,
                            ml_series_needs(
                                feature_ids, num_features,
                                MLFeatureSource::L1_DELTA_SERIES),
                            series_scratch);

    for (uint8_t i = 0; i < num_features; i++) {
        features_out[i] = ml_feature_value_from_stats(
            feature_ids[i], turb, aggregated_turb, delta,
            l1_delta_lag_ratio, chan_shape_spread,
            chan_freq_coh_cv, chan_freq_coh_curve_std, chan_coh_gap,
            chan_coh_subband_gap_median);
    }
}

}  // namespace espectre
