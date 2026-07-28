/*
 * ESPectre - ML Detector Implementation
 *
 * Neural network-based motion detection algorithm.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "ml_detector.h"
#include "ml_weights.h"
#include "threshold.h"
#include <cmath>
#include <algorithm>
#include "espectre_log.h"

namespace espectre {

static const char *TAG = "MLDetector";
static_assert(ML_MODEL_INPUT_SIZE == ML_NUM_FEATURES,
              "Exported model input size must match extracted ML feature count");

// ============================================================================
// CONSTRUCTOR
// ============================================================================

MLDetector::MLDetector(uint16_t window_size, float threshold, uint16_t lag)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , uses_l1_tracker_(false)
    , uses_l1_series_(false)
    , lag_(std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX))
    , feature_scratch_(nullptr) {
    threshold_ = clamp_threshold(threshold_, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);

    // Maintain the L1-delta rings only when the exported model needs them, and
    // reserve the rebuilt series only for the features that read it.
    for (uint8_t i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        const uint8_t id = ML_FEATURE_IDS[i];
        uses_l1_tracker_ = uses_l1_tracker_ || ml_feature_needs_l1_tracker(id);
        uses_l1_series_ = uses_l1_series_ || ml_feature_needs_l1_series(id);
    }

    // One block holds every working array the feature path needs; the
    // accessors below carve it into the sort view, the absolute-deviation
    // view, and the rebuilt L1-delta series.
    feature_scratch_ = alloc_zeroed_floats(feature_scratch_size_());
    if (feature_scratch_ == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate feature scratch (%u floats)",
                 static_cast<unsigned>(feature_scratch_size_()));
    }
    l1_tracker_.configure(uses_l1_tracker_ ? l1_delta_capacity_() : 0U, lag_);

    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f, l1=%d, l1_series=%d)",
             window_size_, threshold_, uses_l1_tracker_ ? 1 : 0,
             uses_l1_series_ ? 1 : 0);
}

MLDetector::~MLDetector() {
    delete[] feature_scratch_;
}

MLDetector::MLDetector(MLDetector&& other) noexcept
    : BaseDetector(std::move(other))
    , threshold_(other.threshold_)
    , uses_l1_tracker_(other.uses_l1_tracker_)
    , uses_l1_series_(other.uses_l1_series_)
    , lag_(other.lag_)
    , l1_tracker_(std::move(other.l1_tracker_))
    , feature_scratch_(other.feature_scratch_) {
    other.feature_scratch_ = nullptr;
}

MLDetector& MLDetector::operator=(MLDetector&& other) noexcept {
    if (this != &other) {
        BaseDetector::operator=(std::move(other));
        threshold_ = other.threshold_;
        uses_l1_tracker_ = other.uses_l1_tracker_;
        uses_l1_series_ = other.uses_l1_series_;
        lag_ = other.lag_;
        l1_tracker_ = std::move(other.l1_tracker_);
        delete[] feature_scratch_;
        feature_scratch_ = other.feature_scratch_;
        other.feature_scratch_ = nullptr;
    }
    return *this;
}

uint16_t MLDetector::feature_scratch_size_() const {
    // Sort view + absolute-deviation view, plus the L1-delta series when the
    // exported model uses it.
    return static_cast<uint16_t>(2U * window_size_ +
                                 (uses_l1_series_ ? l1_delta_capacity_() : 0U));
}

float* MLDetector::delta_series_() const {
    if (feature_scratch_ == nullptr || !uses_l1_series_) {
        return nullptr;
    }
    return feature_scratch_ + 2U * window_size_;
}

MLSeriesScratch MLDetector::series_scratch_() const {
    if (feature_scratch_ == nullptr) {
        return MLSeriesScratch{};
    }
    return MLSeriesScratch{feature_scratch_, feature_scratch_ + window_size_, window_size_};
}

void MLDetector::configure_hampel(bool enabled, uint8_t window_size,
                                  float threshold) {
    BaseDetector::configure_hampel(enabled, window_size, threshold);
    l1_tracker_.configure_hampel(enabled, window_size, threshold);
}

// ============================================================================
// DETECTION LOGIC
// ============================================================================

void MLDetector::update_state() {
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

bool MLDetector::set_threshold(float threshold) {
    if (!is_valid_threshold(threshold, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD)) {
        ESP_LOGE(TAG, "Invalid threshold: %.6f (must be %.1f-%.1f)",
                 threshold, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);
        return false;
    }
    
    threshold_ = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.6f", threshold);
    return true;
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

void MLDetector::extract_features(float* features_out) {
    // Reconstruct the L1-delta series (chronological) when the model uses it.
    float* delta_series = delta_series_();
    const uint16_t delta_len =
        delta_series != nullptr ? l1_tracker_.build_series(delta_series) : 0U;

    uint16_t turb_count = 0U;
    const float* turb_series = ordered_turbulence(turb_count);
    if (turb_series == nullptr) {
        std::fill(features_out, features_out + ML_NUM_FEATURES, 0.0f);
        return;
    }

    extract_ml_features_by_id(turb_series, turb_count,
                              delta_series, delta_len,
                              ML_FEATURE_IDS, ML_MODEL_INPUT_SIZE, features_out,
                              series_scratch_(),
                              l1_tracker_.delta_lag_ratio());
}

// ============================================================================
// L1-DELTA PROFILE PIPELINE
// ============================================================================

uint16_t MLDetector::l1_delta_capacity_() const {
    // window_size profiles yield window_size - lag deltas, matching the Python
    // features.l1_delta_series window semantics. Use the configured lag rather
    // than the nominal constant so replay experiments cannot make the ring and
    // its readiness gate disagree.
    return window_size_ > lag_ ? static_cast<uint16_t>(window_size_ - lag_) : 0;
}

bool MLDetector::is_ready() const {
    if (!BaseDetector::is_ready()) {
        return false;
    }
    // The L1 rings feed real features, so readiness has to wait for them too.
    // This used to hold only by arithmetic accident: at the nominal lag both
    // rings fill on the same packet. An alternate lag can break that, and an
    // ungated ML would infer on a partly filled ring whose lag ratio returns its
    // no-motion sentinel rather than signalling "not ready".
    return !uses_l1_tracker_ || l1_tracker_.count() >= l1_delta_capacity_();
}

void MLDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers,
                                uint8_t num_subcarriers,
                                int8_t rssi_dbm) {
    if (csi_data == nullptr) {
        ESP_LOGE(TAG, "process_packet: null CSI data");
        return;
    }
    (void) rssi_dbm;

    float amplitudes[HT20_SELECTED_BAND_SIZE];
    const uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);
    process_amplitudes(amplitudes, amplitude_count);

    if (!uses_l1_tracker_) {
        return;
    }
    l1_tracker_.process(amplitudes, amplitude_count);
}

void MLDetector::clear_buffer() {
    BaseDetector::clear_buffer();
    l1_tracker_.clear();
}

// ============================================================================
// MLP INFERENCE
// ============================================================================

// Inference runs without floating-point contraction.
//
// The accumulation below is a long chain of `val += input * weight` in float,
// and a compiler that is free to fuse each pair into an FMA rounds it
// differently. The difference is tiny per step, but the MLP output feeds a
// threshold, so on recordings whose probabilities sit near `0.5` it flips whole
// decisions: with contraction on, ten of the twenty-eight paired replays moved,
// the worst by `3.2` points of recall, and the report parity gate failed on ML
// while Classic stayed clean. Contraction also made the result depend on the
// compiler and the surrounding code rather than on the model.
//
// Both runtimes must decide the same way, so contraction is disabled here
// rather than in one build system, which keeps ESP-IDF, PlatformIO, and the
// host tests on the same arithmetic.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

float MLDetector::predict(const float* features) {
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
    if (out > 20.0f) return ML_METRIC_SCALE;
    return (1.0f / (1.0f + std::exp(-out))) * ML_METRIC_SCALE;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

}  // namespace espectre
