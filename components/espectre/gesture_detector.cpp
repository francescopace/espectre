/*
 * ESPectre - Gesture Detector Implementation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "gesture_detector.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "GestureDetector";

// Fixed subcarriers (same as MLDetector)
static constexpr uint8_t DEFAULT_SUBCARRIERS[12] = {11, 14, 17, 21, 24, 28, 31, 35, 39, 42, 46, 49};

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
    , live_packets_since_start_(0) {

    memset(ring_turb_, 0, sizeof(ring_turb_));
    memset(ring_phases_, 0, sizeof(ring_phases_));
    memset(event_turb_, 0, sizeof(event_turb_));
    memset(event_phases_, 0, sizeof(event_phases_));

    ESP_LOGI(TAG, "Initialized (ring=%d, window=%d, pre_roll=%d, weights=%s)",
             GESTURE_RING_SIZE, GESTURE_WINDOW_LEN, GESTURE_PREROLL_LEN,
             ESPECTRE_GESTURE_WEIGHTS_AVAILABLE ? "yes" : "no (train model first)");
}

// ============================================================================
// PACKET PROCESSING
// ============================================================================

void GestureDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* subcarriers, uint8_t num_sc) {
    const uint8_t* sc = (subcarriers && num_sc > 0) ? subcarriers : DEFAULT_SUBCARRIERS;
    const uint8_t n_sc = (subcarriers && num_sc > 0) ? num_sc : GESTURE_PHASE_SC;

    // Compute turbulence
    float amplitudes[GESTURE_PHASE_SC];
    float turbulence = calculate_spatial_turbulence_from_csi(
        csi_data, csi_len, sc, n_sc, 64, amplitudes, /*use_cv_norm=*/false
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
            extract_gesture_features(event_turb_, event_phases_,
                                     event_len_, GESTURE_PHASE_SC, features);
            const char* name = predict(features);
            last_gesture_ = name;
            if (name != nullptr) {
                pending_live_gesture_ = name;
                const bool changed = (last_live_gesture_ == nullptr || strcmp(last_live_gesture_, name) != 0);
                if (changed) {
                    // Log at INFO when a concrete gesture label is first recognized.
                    if (strcmp(name, "movement") != 0 && strcmp(name, "no_gesture") != 0) {
                        ESP_LOGI(TAG, "Gesture detected: %s", name);
                    } else {
                        ESP_LOGD(TAG, "Live prediction update: %s", name);
                    }
                    last_live_gesture_ = name;
                }
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
        live_packets_since_start_ = 0;
        return nullptr;
    }
    ESP_LOGD(TAG, "Classifying fixed window (%d packets)", event_len_);

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(event_turb_, event_phases_,
                             event_len_, GESTURE_PHASE_SC, features);

    const char* name = predict(features);
    last_gesture_ = name;
    event_len_ = 0;
    last_live_gesture_ = nullptr;
    live_packets_since_start_ = 0;

    if (name && strcmp(name, "movement") != 0) {
        ESP_LOGI(TAG, "Gesture detected: %s", name);
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
    live_packets_since_start_ = 0;
    ESP_LOGD(TAG, "Ring buffer cleared");
}

void GestureDetector::reset() {
    event_len_ = 0;
    active_ = false;
    last_gesture_ = nullptr;
    pending_live_gesture_ = nullptr;
    last_live_gesture_ = nullptr;
    live_packets_since_start_ = 0;
}

// ============================================================================
// INFERENCE
// ============================================================================

const char* GestureDetector::predict(const float* features) {
#if !ESPECTRE_GESTURE_WEIGHTS_AVAILABLE
    return nullptr;
#else
    // Normalize features
    float x_norm[GESTURE_NUM_FEATURES];
    for (int i = 0; i < GESTURE_NUM_FEATURES; i++) {
        x_norm[i] = (features[i] - GESTURE_FEATURE_MEAN[i]) / GESTURE_FEATURE_SCALE[i];
    }

    // LogReg export: single affine logits layer.
    constexpr int C = GESTURE_NUM_CLASSES;
    float logits[C];
    for (int j = 0; j < C; j++) {
        logits[j] = GESTURE_B1[j];
        for (int i = 0; i < GESTURE_NUM_FEATURES; i++) {
            logits[j] += x_norm[i] * GESTURE_W1[i][j];
        }
    }

    // Softmax with numerical stability
    float max_logit = logits[0];
    for (int j = 1; j < C; j++) {
        if (logits[j] > max_logit) max_logit = logits[j];
    }
    float probs[C];
    float sum_exp = 0.0f;
    for (int j = 0; j < C; j++) {
        probs[j] = std::exp(logits[j] - max_logit);
        sum_exp += probs[j];
    }

    int best = 0;
    float best_prob = probs[0] / sum_exp;
    for (int j = 1; j < C; j++) {
        float p = probs[j] / sum_exp;
        if (p > best_prob) { best_prob = p; best = j; }
    }

    ESP_LOGV(TAG, "Inference: class=%d (%s), confidence=%.2f",
             best, GESTURE_CLASS_LABELS[best], best_prob);
    return GESTURE_CLASS_LABELS[best];
#endif
}

}  // namespace espectre
}  // namespace esphome
