/*
 * ESPectre - High-Accuracy Detector Implementation
 *
 * Neural network-based motion detection algorithm.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "high_accuracy_detector.h"
#include "ml_weights.h"
#include "threshold.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include "espectre_log.h"

namespace espectre {

static const char *TAG = "HighAccuracyDetector";
static_assert(ML_MODEL_INPUT_SIZE == ML_NUM_FEATURES,
              "Exported model input size must match extracted ML feature count");

// Construction and ownership

HighAccuracyDetector::HighAccuracyDetector(uint16_t window_size, float threshold, uint16_t lag)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , uses_l1_tracker_(false)
    , uses_shape_trajectory_tracker_(false)
    , uses_aggregated_turbulence_(false)
    , lag_(std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX))
    , feature_scratch_(nullptr)
    , aggregated_turbulence_buffer_(nullptr) {
    threshold_ = clamp_threshold(threshold_, HIGH_ACCURACY_MIN_THRESHOLD, HIGH_ACCURACY_MAX_THRESHOLD);
    // Maintain the L1-delta rings only when the exported model needs them, and
    // reserve the rebuilt series only for the features that read it.
    for (uint8_t i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        const uint8_t id = ML_FEATURE_IDS[i];
        uses_l1_tracker_ = uses_l1_tracker_ || ml_feature_needs_l1_tracker(id);
        uses_shape_trajectory_tracker_ =
            uses_shape_trajectory_tracker_ ||
            ml_feature_needs_channel_shape_trajectory_tracker(id);
        uses_aggregated_turbulence_ =
            uses_aggregated_turbulence_ ||
            ml_feature_needs_aggregated_turbulence(id);
    }

    // One block holds every working array the feature path needs; the
    // accessors below expose as the reusable sorted-series view.
    feature_scratch_ = alloc_zeroed_floats(feature_scratch_size_());
    if (feature_scratch_ == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate feature scratch (%u floats)",
                 static_cast<unsigned>(feature_scratch_size_()));
    }
    if (uses_aggregated_turbulence_) {
        aggregated_turbulence_buffer_ = alloc_zeroed_floats(window_size_);
        if (aggregated_turbulence_buffer_ == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate aggregated turbulence buffer");
        }
    }
    aggregated_turbulence_.bind(aggregated_turbulence_buffer_, window_size_);
    l1_tracker_.configure(uses_l1_tracker_ ? l1_delta_capacity_() : 0U, lag_);
    shape_trajectory_tracker_.configure(uses_shape_trajectory_tracker_);
    ESP_LOGI(TAG,
             "Initialized (window=%d, threshold=%.2f, l1=%d, trajectory=%d, aggr=%d)",
             window_size_, threshold_, uses_l1_tracker_ ? 1 : 0,
             uses_shape_trajectory_tracker_ ? 1 : 0,
             uses_aggregated_turbulence_ ? 1 : 0);
}

HighAccuracyDetector::~HighAccuracyDetector() {
    delete[] feature_scratch_;
    delete[] aggregated_turbulence_buffer_;
}

HighAccuracyDetector::HighAccuracyDetector(HighAccuracyDetector&& other) noexcept
    : BaseDetector(std::move(other))
    , threshold_(other.threshold_)
    , uses_l1_tracker_(other.uses_l1_tracker_)
    , uses_shape_trajectory_tracker_(other.uses_shape_trajectory_tracker_)
    , uses_aggregated_turbulence_(other.uses_aggregated_turbulence_)
    , lag_(other.lag_)
    , l1_tracker_(std::move(other.l1_tracker_))
    , shape_trajectory_tracker_(std::move(other.shape_trajectory_tracker_))
    , feature_scratch_(other.feature_scratch_)
    , aggregated_turbulence_buffer_(other.aggregated_turbulence_buffer_)
    , aggregated_turbulence_(std::move(other.aggregated_turbulence_)) {
    other.feature_scratch_ = nullptr;
    other.aggregated_turbulence_buffer_ = nullptr;
}

HighAccuracyDetector& HighAccuracyDetector::operator=(HighAccuracyDetector&& other) noexcept {
    if (this != &other) {
        BaseDetector::operator=(std::move(other));
        threshold_ = other.threshold_;
        uses_l1_tracker_ = other.uses_l1_tracker_;
        uses_shape_trajectory_tracker_ = other.uses_shape_trajectory_tracker_;
        uses_aggregated_turbulence_ = other.uses_aggregated_turbulence_;
        lag_ = other.lag_;
        l1_tracker_ = std::move(other.l1_tracker_);
        shape_trajectory_tracker_ = std::move(other.shape_trajectory_tracker_);
        delete[] feature_scratch_;
        feature_scratch_ = other.feature_scratch_;
        other.feature_scratch_ = nullptr;
        delete[] aggregated_turbulence_buffer_;
        aggregated_turbulence_buffer_ = other.aggregated_turbulence_buffer_;
        aggregated_turbulence_ = std::move(other.aggregated_turbulence_);
        other.aggregated_turbulence_buffer_ = nullptr;
    }
    return *this;
}

uint16_t HighAccuracyDetector::feature_scratch_size_() const {
    return std::max<uint16_t>(window_size_, HT20_NUM_SUBCARRIERS);
}

MLSeriesScratch HighAccuracyDetector::series_scratch_() const {
    if (feature_scratch_ == nullptr) {
        return MLSeriesScratch{};
    }
    return MLSeriesScratch{feature_scratch_, window_size_};
}

void HighAccuracyDetector::configure_hampel(bool enabled, uint8_t window_size,
                                  float threshold) {
    BaseDetector::configure_hampel(enabled, window_size, threshold);
    aggregated_turbulence_.configure_hampel(enabled, window_size, threshold);
    l1_tracker_.configure_hampel(enabled, window_size, threshold);
}

void HighAccuracyDetector::configure_lowpass(bool enabled, float cutoff_hz) {
    BaseDetector::configure_lowpass(enabled, cutoff_hz);
    aggregated_turbulence_.configure_lowpass(enabled, cutoff_hz);
}

// Detection

void HighAccuracyDetector::update_state() {
    if (!is_ready()) {
        clear_evaluation_state_();
        return;
    }

    // Extract ML features expected by the exported model
    float features[ML_NUM_FEATURES];
    extract_features(features);
    
    // Run MLP inference
    current_metric_ = predict(features);
    
    // Keep Python/C++ parity: ML state is decided directly from the
    // probability threshold at each evaluation tick, without hysteresis.
    state_ = current_metric_ > threshold_ ? MotionState::MOTION
                                               : MotionState::IDLE;
}

bool HighAccuracyDetector::set_threshold(float threshold) {
    if (!is_valid_threshold(threshold, HIGH_ACCURACY_MIN_THRESHOLD, HIGH_ACCURACY_MAX_THRESHOLD)) {
        ESP_LOGE(TAG, "Invalid threshold: %.6f (must be %.1f-%.1f)",
                 threshold, HIGH_ACCURACY_MIN_THRESHOLD, HIGH_ACCURACY_MAX_THRESHOLD);
        return false;
    }
    
    threshold_ = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.6f", threshold);
    return true;
}

// Feature extraction

void HighAccuracyDetector::extract_features(float* features_out) {
    uint16_t turb_count = 0U;
    const float* turb_series = ordered_turbulence(turb_count);
    if (turb_series == nullptr) {
        std::fill(features_out, features_out + ML_NUM_FEATURES, 0.0f);
        return;
    }

    uint16_t aggregated_turb_count = 0U;
    const float* aggregated_turb_series =
        ordered_aggregated_turbulence_(aggregated_turb_count);

    float trajectory_innovation = 0.0f;
    float trajectory_excess = 0.0f;
    float trajectory_spread = 0.0f;
    float trajectory_kendall = 0.0f;
    if (uses_shape_trajectory_tracker_) {
        shape_trajectory_tracker_.trajectory_features(
            trajectory_innovation, trajectory_excess, trajectory_spread,
            trajectory_kendall);
    }

    extract_ml_features_by_id(turb_series, turb_count,
                              aggregated_turb_series, aggregated_turb_count,
                              ML_FEATURE_IDS, ML_MODEL_INPUT_SIZE, features_out,
                              series_scratch_(),
                              l1_tracker_.delta_lag_ratio(),
                              trajectory_spread,
                              trajectory_innovation,
                              trajectory_excess,
                              trajectory_kendall);
}

// L1-delta profile pipeline

uint16_t HighAccuracyDetector::l1_delta_capacity_() const {
    // window_size profiles yield window_size - lag deltas. Use the configured
    // lag rather than the nominal constant so replay experiments cannot make
    // the ring and its readiness gate disagree.
    return window_size_ > lag_ ? static_cast<uint16_t>(window_size_ - lag_) : 0;
}

bool HighAccuracyDetector::is_ready() const {
    if (!BaseDetector::is_ready()) {
        return false;
    }
    // The L1 rings feed real features, so readiness has to wait for them too.
    // This used to hold only by arithmetic accident: at the nominal lag both
    // rings fill on the same packet. An alternate lag can break that, and an
    // ungated ML would infer on a partly filled ring whose lag ratio returns its
    // no-motion sentinel rather than signalling "not ready".
    return (!uses_l1_tracker_ || l1_delta_capacity_() == 0U ||
            l1_tracker_.count() > 0U) &&
           (!uses_aggregated_turbulence_ ||
            (aggregated_turbulence_.count() >= window_size_ &&
             aggregated_turbulence_.valid_count() >= minimum_valid_samples_));
}

void HighAccuracyDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers,
                                uint8_t num_subcarriers,
                                int8_t rssi_dbm) {
    if (csi_data == nullptr) {
        ESP_LOGE(TAG, "process_packet: null CSI data");
        return;
    }
    (void) rssi_dbm;

    const uint8_t* resolved_subcarriers = selected_subcarriers;
    uint8_t resolved_count = num_subcarriers;
    if (resolved_subcarriers == nullptr || resolved_count == 0U) {
        resolved_subcarriers = DEFAULT_SUBCARRIERS;
        resolved_count = HT20_SELECTED_BAND_SIZE;
    }

    float local_packet_values[HT20_NUM_SUBCARRIERS]{};
    float* packet_values = feature_scratch_ != nullptr
        ? feature_scratch_ : local_packet_values;
    const uint8_t packet_value_count = fill_packet_subcarrier_energies(
        csi_data, csi_len, packet_values, HT20_NUM_SUBCARRIERS);

    if (uses_shape_trajectory_tracker_) {
        const uint64_t fallback_timestamp =
            static_cast<uint64_t>(get_total_packets()) * 10000U;
        shape_trajectory_tracker_.process_packet(
            csi_data, csi_len, packet_timestamp_us_or(fallback_timestamp),
            packet_values, packet_value_count);
    }

    energies_to_amplitudes_in_place(packet_values, packet_value_count);

    float amplitudes[HT20_SELECTED_BAND_SIZE]{};
    const uint8_t amplitude_count = select_subcarrier_amplitudes(
        packet_values, packet_value_count, resolved_subcarriers,
        resolved_count, amplitudes, HT20_SELECTED_BAND_SIZE);
    const MeanVariance amplitude_stats =
        calculate_mean_variance_two_pass(amplitudes, amplitude_count);
    add_turbulence_to_buffer(apply_cv_normalization(
        std::sqrt(amplitude_stats.variance), amplitude_stats.mean));

    if (uses_aggregated_turbulence_) {
        float aggregated_amplitudes[HT20_SELECTED_BAND_SIZE]{};
        const uint8_t aggregated_count =
            select_adjacent_aggregated_subcarrier_amplitudes(
                packet_values, packet_value_count,
                resolved_subcarriers, resolved_count,
                TURB_IQR_AGGREGATION_WIDTH, aggregated_amplitudes,
                HT20_SELECTED_BAND_SIZE);
        add_aggregated_turbulence_(
            calculate_spatial_turbulence_from_amplitudes(
                aggregated_amplitudes, aggregated_count));
    }

    if (!uses_l1_tracker_) {
        return;
    }
    l1_tracker_.process(amplitudes, amplitude_count, amplitude_stats.mean);
}

void HighAccuracyDetector::clear_buffer() {
    BaseDetector::clear_buffer();
    l1_tracker_.clear();
    shape_trajectory_tracker_.clear();
    aggregated_turbulence_.clear();
}

void HighAccuracyDetector::add_aggregated_turbulence_(float turbulence) {
    aggregated_turbulence_.add(turbulence);
}

void HighAccuracyDetector::advance_missing_slots(uint32_t count) {
    BaseDetector::advance_missing_slots(count);
    if (uses_l1_tracker_) l1_tracker_.advance_missing_slots(count);
    aggregated_turbulence_.advance_missing_slots(count);
}

const float* HighAccuracyDetector::ordered_aggregated_turbulence_(uint16_t& count) const {
    return aggregated_turbulence_.ordered_view(feature_scratch_, window_size_, count);
}

// MLP inference

// Inference runs without floating-point contraction.
//
// The accumulation below is a long chain of `val += input * weight` in float,
// and a compiler that is free to fuse each pair into an FMA rounds it
// differently. The difference is tiny per step, but the MLP output feeds a
// threshold, so on recordings whose probabilities sit near `0.5` it flips whole
// decisions: with contraction on, ten of the twenty-eight paired replays moved,
// the worst by `3.2` points of recall, and the report parity gate failed on ML
// while Lightweight stayed clean. Contraction also made the result depend on the
// compiler and the surrounding code rather than on the model.
//
// Both runtimes must decide the same way, so contraction is disabled here
// rather than in one build system, which keeps ESP-IDF firmware and the host
// tests on the same arithmetic.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

float HighAccuracyDetector::predict(const float* features) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    constexpr size_t kBufferSize =
        (ML_MAX_LAYER_WIDTH > ML_MODEL_INPUT_SIZE) ? ML_MAX_LAYER_WIDTH : ML_MODEL_INPUT_SIZE;
    float buffer_a[kBufferSize] = {0.0f};
    float buffer_b[kBufferSize] = {0.0f};

    // Normalize features using pre-computed mean and scale
    for (int i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        buffer_a[i] = (features[i] - ML_FEATURE_MEAN[i]) / ML_FEATURE_SCALE[i];
    }

    float *current = buffer_a;
    float *next = buffer_b;
    float out = 0.0f;

    for (int layer = 0; layer < ML_MODEL_NUM_LAYERS; layer++) {
        const int in_size = ML_MODEL_LAYER_INPUT_SIZES[layer];
        const int out_size = ML_MODEL_LAYER_OUTPUT_SIZES[layer];
        const float *weights = ML_MODEL_WEIGHTS[layer];
        const float *biases = ML_MODEL_BIASES[layer];
        const bool is_output_layer = (layer == ML_MODEL_NUM_LAYERS - 1);

        for (int j = 0; j < out_size; j++) {
            float val = biases[j];
            for (int i = 0; i < in_size; i++) {
                val += current[i] * weights[i * out_size + j];
            }

            if (is_output_layer) {
                out = val;
            } else {
                next[j] = std::max(0.0f, val);
            }
        }

        if (!is_output_layer) {
            std::swap(current, next);
        }
    }

    // Sigmoid with overflow protection on the direct 0-1 probability scale
    if (out < -20.0f) return 0.0f;
    if (out > 20.0f) return HIGH_ACCURACY_METRIC_SCALE;
    return (1.0f / (1.0f + std::exp(-out))) * HIGH_ACCURACY_METRIC_SCALE;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

}  // namespace espectre
