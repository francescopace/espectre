/*
 * ESPectre - Gesture Detector Implementation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "gesture_detector.h"
#include "utils.h"
#include "ml_detector.h"
#include <cmath>
#include <algorithm>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "GestureDetector";

// ============================================================================
// CONSTRUCTOR
// ============================================================================

GestureDetector::GestureDetector()
    : ring_head_(0)
    , ring_count_(0)
    , event_len_(0)
    , active_(false)
    , last_gesture_(nullptr)
    , pending_live_gesture_(nullptr)
    , last_live_gesture_(nullptr)
    , candidate_live_gesture_(nullptr)
    , candidate_live_count_(0)
    , live_packets_since_start_(0)
    , use_cv_normalization_(false) {

    memset(ring_turb_, 0, sizeof(ring_turb_));
    memset(ring_phases_, 0, sizeof(ring_phases_));
    memset(event_turb_, 0, sizeof(event_turb_));
    memset(event_phases_, 0, sizeof(event_phases_));

    ESP_LOGI(TAG, "Initialized (ring=%d, window=%d, pre_roll=%d, weights=yes)",
             GESTURE_RING_SIZE, GESTURE_WINDOW_LEN, GESTURE_PREROLL_LEN);
}

// ============================================================================
// PACKET PROCESSING
// ============================================================================

void GestureDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* subcarriers, uint8_t num_sc) {
    const uint8_t* sc = (subcarriers && num_sc > 0) ? subcarriers : ML_SUBCARRIERS;
    const uint8_t n_sc = (subcarriers && num_sc > 0) ? num_sc : GESTURE_PHASE_SC;

    // Compute turbulence
    float amplitudes[GESTURE_PHASE_SC];
    float turbulence = calculate_spatial_turbulence_from_csi(
        csi_data, csi_len, sc, n_sc, 64, amplitudes, use_cv_normalization_
    );

    // Compute phase values (I/Q angle per selected subcarrier)
    float phases[GESTURE_PHASE_SC];
    const uint8_t limit = std::min<uint8_t>(n_sc, GESTURE_PHASE_SC);
    for (uint8_t k = 0; k < limit; k++) {
        phases[k] = extract_phase_from_csi(csi_data, csi_len, sc[k]);
    }

    // ---------------------------------------------------------
    // Update ring buffer (circular)
    // ---------------------------------------------------------
    uint16_t write_idx;
    if (ring_count_ < GESTURE_RING_SIZE) {
        write_idx = ring_count_;
        ring_count_++;
    } else {
        // Overwrite oldest slot; advance head
        write_idx = ring_head_;
        ring_head_ = (ring_head_ + 1) % GESTURE_RING_SIZE;
    }
    ring_turb_[write_idx] = turbulence;
    memcpy(ring_phases_ + write_idx * GESTURE_PHASE_SC, phases, limit * sizeof(float));

    // ---------------------------------------------------------
    // Append to event buffer when active
    // ---------------------------------------------------------
    if (active_) {
        if (event_len_ < GESTURE_MAX_EVENT_LEN) {
            event_turb_[event_len_] = turbulence;
            memcpy(event_phases_ + event_len_ * GESTURE_PHASE_SC, phases, limit * sizeof(float));
            event_len_++;
        } else {
            // Sliding window: drop oldest packet and append newest.
            memmove(event_turb_, event_turb_ + 1, (GESTURE_MAX_EVENT_LEN - 1) * sizeof(float));
            memmove(event_phases_, event_phases_ + GESTURE_PHASE_SC,
                    (GESTURE_MAX_EVENT_LEN - 1) * GESTURE_PHASE_SC * sizeof(float));
            event_turb_[GESTURE_MAX_EVENT_LEN - 1] = turbulence;
            memcpy(event_phases_ + (GESTURE_MAX_EVENT_LEN - 1) * GESTURE_PHASE_SC,
                   phases, limit * sizeof(float));
        }

        live_packets_since_start_++;

        // Continuous live inference at fixed cadence on the sliding window.
        if (event_len_ >= GESTURE_WINDOW_LEN &&
            (live_packets_since_start_ % GESTURE_LIVE_STRIDE) == 0) {
            float features[GESTURE_NUM_FEATURES];
            extract_gesture_model_features(
                event_turb_, event_phases_, event_len_, GESTURE_PHASE_SC,
                GESTURE_FEATURE_IDS, GESTURE_NUM_FEATURES, features
            );
            float conf = 0.0f;
            float margin = 0.0f;
            const char* name = predict(features, &conf, &margin);
            last_gesture_ = name;
            if (name != nullptr &&
                conf >= GESTURE_MIN_CONFIDENCE &&
                margin >= GESTURE_MIN_MARGIN) {
                if (candidate_live_gesture_ != nullptr && strcmp(candidate_live_gesture_, name) == 0) {
                    candidate_live_count_++;
                } else {
                    candidate_live_gesture_ = name;
                    candidate_live_count_ = 1;
                }

                if (candidate_live_count_ >= GESTURE_MIN_CONSECUTIVE) {
                    pending_live_gesture_ = name;
                    const bool changed = (last_live_gesture_ == nullptr || strcmp(last_live_gesture_, name) != 0);
                    if (changed) {
                        // Log at INFO when a concrete gesture label is first recognized.
                        if (strcmp(name, "movement") != 0 && strcmp(name, "no_gesture") != 0) {
                            ESP_LOGI(TAG, "Gesture detected: %s (conf=%.2f, margin=%.2f)", name, conf, margin);
                        } else {
                            ESP_LOGD(TAG, "Live prediction update: %s (conf=%.2f, margin=%.2f)", name, conf, margin);
                        }
                        last_live_gesture_ = name;
                    }
                }
            } else {
                candidate_live_gesture_ = nullptr;
                candidate_live_count_ = 0;
                pending_live_gesture_ = "no_gesture";
            }
        }
    }
}

// ============================================================================
// EVENT LIFECYCLE
// ============================================================================

void GestureDetector::start_detection() {
    // Reconstruct the latest fixed pre-roll slice from ring buffer.
    event_len_ = 0;

    const uint16_t pre_len = std::min<uint16_t>(GESTURE_PREROLL_LEN, ring_count_);
    if (pre_len > 0) {
        uint16_t src_start = 0;
        if (ring_count_ < GESTURE_RING_SIZE) {
            // Ring not full: data in [0..ring_count_) in chronological order.
            src_start = ring_count_ - pre_len;
        } else {
            // Ring full: oldest is ring_head_. Last pre_len starts at:
            // ring_head_ + (ring_size - pre_len).
            src_start = (ring_head_ + (GESTURE_RING_SIZE - pre_len)) % GESTURE_RING_SIZE;
        }

        for (uint16_t i = 0; i < pre_len; i++) {
            const uint16_t src_idx = (src_start + i) % GESTURE_RING_SIZE;
            event_turb_[i] = ring_turb_[src_idx];
            memcpy(event_phases_ + i * GESTURE_PHASE_SC,
                   ring_phases_ + src_idx * GESTURE_PHASE_SC,
                   GESTURE_PHASE_SC * sizeof(float));
        }
        event_len_ = pre_len;
    }

    active_ = true;
    pending_live_gesture_ = nullptr;
    last_live_gesture_ = nullptr;
    candidate_live_gesture_ = nullptr;
    candidate_live_count_ = 0;
    live_packets_since_start_ = 0;
    ESP_LOGD(TAG, "Motion started - pre-roll: %d packets (target window: %d)",
             event_len_, GESTURE_WINDOW_LEN);
}

const char* GestureDetector::finalize_detection() {
    active_ = false;

    if (event_len_ < GESTURE_WINDOW_LEN) {
        ESP_LOGD(TAG, "Event window incomplete (%d/%d), skipping classification",
                 event_len_, GESTURE_WINDOW_LEN);
        event_len_ = 0;
        pending_live_gesture_ = nullptr;
        last_live_gesture_ = nullptr;
        candidate_live_gesture_ = nullptr;
        candidate_live_count_ = 0;
        live_packets_since_start_ = 0;
        return nullptr;
    }
    ESP_LOGD(TAG, "Classifying fixed window (%d packets)", event_len_);

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_model_features(
        event_turb_, event_phases_, event_len_, GESTURE_PHASE_SC,
        GESTURE_FEATURE_IDS, GESTURE_NUM_FEATURES, features
    );

    float conf = 0.0f;
    float margin = 0.0f;
    const char* name = predict(features, &conf, &margin);
    if (name != nullptr &&
        (conf < GESTURE_MIN_CONFIDENCE || margin < GESTURE_MIN_MARGIN)) {
        name = "no_gesture";
    }
    last_gesture_ = name;
    event_len_ = 0;
    last_live_gesture_ = nullptr;
    candidate_live_gesture_ = nullptr;
    candidate_live_count_ = 0;
    live_packets_since_start_ = 0;

    if (name && strcmp(name, "movement") != 0) {
        ESP_LOGI(TAG, "Gesture detected: %s (conf=%.2f, margin=%.2f)", name, conf, margin);
    }
    return name;
}

const char* GestureDetector::consume_live_prediction() {
    const char* out = pending_live_gesture_;
    pending_live_gesture_ = nullptr;
    return out;
}

// ============================================================================
// RING BUFFER MANAGEMENT
// ============================================================================

void GestureDetector::clear_ring() {
    ring_head_ = 0;
    ring_count_ = 0;
    event_len_ = 0;
    active_ = false;
    pending_live_gesture_ = nullptr;
    last_live_gesture_ = nullptr;
    candidate_live_gesture_ = nullptr;
    candidate_live_count_ = 0;
    live_packets_since_start_ = 0;
    ESP_LOGD(TAG, "Ring buffer cleared");
}

void GestureDetector::reset() {
    event_len_ = 0;
    active_ = false;
    last_gesture_ = nullptr;
    pending_live_gesture_ = nullptr;
    last_live_gesture_ = nullptr;
    candidate_live_gesture_ = nullptr;
    candidate_live_count_ = 0;
    live_packets_since_start_ = 0;
}

// ============================================================================
// INFERENCE
// ============================================================================

const char* GestureDetector::predict(const float* features, float* out_confidence, float* out_margin) {
    constexpr int C = GESTURE_NUM_CLASSES;
    float probs[C];
    for (int j = 0; j < C; j++) probs[j] = 0.0f;

    // Tiny-MLP path: normalize input, apply ReLU hidden layers, softmax output.
    float layer_a[GESTURE_MLP_MAX_WIDTH];
    float layer_b[GESTURE_MLP_MAX_WIDTH];
    for (int i = 0; i < GESTURE_NUM_FEATURES; i++) {
        layer_a[i] = (features[i] - GESTURE_FEATURE_MEAN[i]) / GESTURE_FEATURE_SCALE[i];
    }

    float *in_buf = layer_a;
    float *out_buf = layer_b;
    for (int li = 0; li < (GESTURE_MLP_NUM_LAYERS - 1); li++) {
        const int in_size = GESTURE_MLP_LAYER_SIZES[li];
        const int out_size = GESTURE_MLP_LAYER_SIZES[li + 1];
        const int w_start = GESTURE_MLP_WEIGHT_OFFSETS[li];
        const int b_start = GESTURE_MLP_BIAS_OFFSETS[li];
        const bool is_hidden = (li < (GESTURE_MLP_NUM_LAYERS - 2));

        for (int o = 0; o < out_size; o++) {
            float acc = GESTURE_MLP_BIASES[b_start + o];
            const int row_base = w_start + o * in_size;
            for (int i = 0; i < in_size; i++) {
                acc += GESTURE_MLP_WEIGHTS[row_base + i] * in_buf[i];
            }
            out_buf[o] = is_hidden ? std::max(0.0f, acc) : acc;
        }
        float *tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
    }

    float max_logit = in_buf[0];
    for (int j = 1; j < C; j++) {
        if (in_buf[j] > max_logit) max_logit = in_buf[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < C; j++) {
        probs[j] = std::exp(in_buf[j] - max_logit);
        sum_exp += probs[j];
    }
    const float inv = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
    for (int j = 0; j < C; j++) probs[j] *= inv;

    int best = 0;
    float best_prob = probs[0];
    float second_prob = 0.0f;
    for (int j = 1; j < C; j++) {
        float p = probs[j];
        if (p > best_prob) {
            second_prob = best_prob;
            best_prob = p;
            best = j;
        } else if (p > second_prob) {
            second_prob = p;
        }
    }
    if (out_confidence != nullptr) *out_confidence = best_prob;
    if (out_margin != nullptr) *out_margin = best_prob - second_prob;

    ESP_LOGV(TAG, "Inference: class=%d (%s), confidence=%.2f",
             best, GESTURE_CLASS_LABELS[best], best_prob);
    return GESTURE_CLASS_LABELS[best];
}

}  // namespace espectre
}  // namespace esphome
