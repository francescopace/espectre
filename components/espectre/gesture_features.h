/*
 * ESPectre - Gesture Feature Extraction
 *
 * C++ port of micro-espectre/src/gesture_features.py.
 *
 * Extracts event-level features from a complete motion event buffer for
 * gesture classification. Unlike motion features (aggregate
 * statistics over a sliding window), these describe the morphology of
 * the full event.
 *
 * Canonical feature list (19 total):
 *   0. event_duration       - Log-compressed duration, quantized to 0.1, bounded
 *   1. peak_position        - Normalized peak position within event (0..1)
 *   2. peak_to_mean_ratio   - Peak / mean turbulence (clamped 0..10)
 *   3. rise_fall_asymmetry  - (rise-fall)/(rise+fall), bounded [-1,1]
 *   4. pre_post_energy_ratio - Energy ratio first half / second half (clamped 0.1..10)
 *   5. n_local_peaks        - Local maxima count with prominence threshold, scaled (count/10)
 *   6. peak_fwhm            - Full width at half maximum, normalized
 *   7. turb_mad             - Median absolute deviation of turbulence
 *   8. turb_iqr             - Interquartile range of turbulence
 *   9. turb_mid_mean        - Mean turbulence in middle third of event
 *  10. turb_late_mean       - Mean turbulence in late third of event
 *  11. turb_late_minus_mid  - Late-third mean minus middle-third mean
 *  12. turb_diff_abs_mean   - Mean absolute first difference of turbulence
 *  13. phase_diff_var       - Mean phase differential variance across event
 *  14. phase_entropy        - Mean phase Shannon entropy across event
 *  15. phase_circular_variance - Mean circular variance across event
 *  16. phase_inter_sc_coherence - 1/(1+var) of wrapped adjacent phase differences
 *  17. turb_range          - Global turbulence range (max-min)
 *  18. turb_delta_energy   - Mean squared first difference of turbulence
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace esphome {
namespace espectre {

// Gesture feature extractor output size (must match model feature count).
constexpr uint8_t GESTURE_FEATURE_VECTOR_SIZE = 17;
constexpr uint8_t GESTURE_TOTAL_AVAILABLE_FEATURES = 19;
constexpr uint8_t GESTURE_PHASE_ENTROPY_BINS = 5;
constexpr uint16_t GESTURE_MAX_EVENT_PACKETS = 512;

// ============================================================================
// Morphology features
// ============================================================================

inline float gesture_event_duration(const float* turb, uint16_t n) {
    if (n == 0) return 0.0f;
    const float ref = 200.0f;
    float value = std::log1p(static_cast<float>(n)) / std::log1p(ref);
    if (value < 0.0f) return 0.0f;
    if (value > 1.2f) value = 1.2f;
    // Coarse quantization (0.1 bins) to reduce fine-grained duration shortcut.
    return std::round(value * 10.0f) / 10.0f;
}

inline float gesture_peak_position(const float* turb, uint16_t n) {
    if (n < 2) return 0.5f;
    uint16_t peak_idx = 0;
    for (uint16_t i = 1; i < n; i++) {
        if (turb[i] > turb[peak_idx]) peak_idx = i;
    }
    return static_cast<float>(peak_idx) / (n - 1);
}

inline float gesture_peak_to_mean_ratio(const float* turb, uint16_t n) {
    if (n < 1) return 1.0f;
    float sum = 0.0f;
    float peak = turb[0];
    for (uint16_t i = 0; i < n; i++) {
        sum += turb[i];
        if (turb[i] > peak) peak = turb[i];
    }
    float mean = sum / n;
    if (mean < 1e-10f) return 0.0f;
    float ratio = peak / mean;
    return ratio > 10.0f ? 10.0f : ratio;
}

inline float gesture_rise_fall_asymmetry(const float* turb, uint16_t n) {
    if (n < 2) return 0.0f;
    uint16_t peak_idx = 0;
    for (uint16_t i = 1; i < n; i++) {
        if (turb[i] > turb[peak_idx]) peak_idx = i;
    }
    float rise = static_cast<float>(peak_idx) / n;
    float fall = static_cast<float>(n - 1 - peak_idx) / n;
    float denom = rise + fall;
    if (denom < 1e-10f) return 0.0f;
    float value = (rise - fall) / denom;
    if (value < -1.0f) return -1.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

inline float gesture_pre_post_energy_ratio(const float* turb, uint16_t n) {
    if (n < 2) return 1.0f;
    uint16_t mid = n / 2;
    float e_first = 0.0f, e_second = 0.0f;
    for (uint16_t i = 0; i < mid; i++) e_first += turb[i] * turb[i];
    for (uint16_t i = mid; i < n; i++) e_second += turb[i] * turb[i];
    if (e_second < 1e-10f) return 10.0f;
    float ratio = e_first / e_second;
    if (ratio < 0.1f) return 0.1f;
    if (ratio > 10.0f) return 10.0f;
    return ratio;
}

inline float gesture_n_local_peaks(const float* turb, uint16_t n,
                                   float min_prominence_frac = 0.1f) {
    if (n < 3) return 0.0f;

    float min_val = turb[0], max_val = turb[0];
    for (uint16_t i = 1; i < n; i++) {
        if (turb[i] < min_val) min_val = turb[i];
        if (turb[i] > max_val) max_val = turb[i];
    }
    float threshold = min_prominence_frac * (max_val - min_val);

    uint16_t count = 0;
    for (uint16_t i = 1; i < n - 1; i++) {
        if (turb[i] > turb[i - 1] && turb[i] > turb[i + 1]) {
            float prom = turb[i] - std::max(turb[i - 1], turb[i + 1]);
            if (prom >= threshold) count++;
        }
    }
    return count / 10.0f;
}

inline float gesture_peak_fwhm(const float* turb, uint16_t n) {
    if (n < 3) return 0.0f;

    float peak_val = turb[0], min_val = turb[0];
    uint16_t peak_idx = 0;
    for (uint16_t i = 1; i < n; i++) {
        if (turb[i] > peak_val) { peak_val = turb[i]; peak_idx = i; }
        if (turb[i] < min_val) min_val = turb[i];
    }
    float half_max = min_val + (peak_val - min_val) * 0.5f;

    uint16_t left = peak_idx;
    for (int16_t i = peak_idx; i >= 0; i--) {
        if (turb[i] <= half_max) { left = i; break; }
    }
    uint16_t right = peak_idx;
    for (uint16_t i = peak_idx; i < n; i++) {
        if (turb[i] <= half_max) { right = i; break; }
    }
    return static_cast<float>(right - left) / n;
}


inline float gesture_turb_mad(const float* turb, uint16_t n) {
    if (n < 2) return 0.0f;

    float sorted_vals[GESTURE_MAX_EVENT_PACKETS];
    if (n > GESTURE_MAX_EVENT_PACKETS) return 0.0f;
    for (uint16_t i = 0; i < n; i++) sorted_vals[i] = turb[i];
    std::sort(sorted_vals, sorted_vals + n);

    uint16_t mid = n / 2;
    float median = (n % 2 == 0) ? 0.5f * (sorted_vals[mid - 1] + sorted_vals[mid]) : sorted_vals[mid];

    float abs_dev[GESTURE_MAX_EVENT_PACKETS];
    for (uint16_t i = 0; i < n; i++) abs_dev[i] = std::fabs(sorted_vals[i] - median);
    std::sort(abs_dev, abs_dev + n);
    return (n % 2 == 0) ? 0.5f * (abs_dev[mid - 1] + abs_dev[mid]) : abs_dev[mid];
}


inline float gesture_turb_iqr(const float* turb, uint16_t n) {
    if (n < 4) return 0.0f;

    float sorted_vals[GESTURE_MAX_EVENT_PACKETS];
    if (n > GESTURE_MAX_EVENT_PACKETS) return 0.0f;
    for (uint16_t i = 0; i < n; i++) sorted_vals[i] = turb[i];
    std::sort(sorted_vals, sorted_vals + n);

    uint16_t q1_idx = static_cast<uint16_t>(0.25f * (n - 1));
    uint16_t q3_idx = static_cast<uint16_t>(0.75f * (n - 1));
    return sorted_vals[q3_idx] - sorted_vals[q1_idx];
}

inline float gesture_turb_mid_mean(const float* turb, uint16_t n) {
    if (n < 3) return 0.0f;
    uint16_t s = n / 3;
    uint16_t e = (2 * n) / 3;
    if (e <= s) return 0.0f;
    float sum = 0.0f;
    for (uint16_t i = s; i < e; i++) sum += turb[i];
    return sum / static_cast<float>(e - s);
}

inline float gesture_turb_late_mean(const float* turb, uint16_t n) {
    if (n < 3) return 0.0f;
    uint16_t s = (2 * n) / 3;
    if (n <= s) return 0.0f;
    float sum = 0.0f;
    for (uint16_t i = s; i < n; i++) sum += turb[i];
    return sum / static_cast<float>(n - s);
}

inline float gesture_turb_late_minus_mid(const float* turb, uint16_t n) {
    return gesture_turb_late_mean(turb, n) - gesture_turb_mid_mean(turb, n);
}

inline float gesture_turb_diff_abs_mean(const float* turb, uint16_t n) {
    if (n < 2) return 0.0f;
    float sum = 0.0f;
    for (uint16_t i = 1; i < n; i++) sum += std::fabs(turb[i] - turb[i - 1]);
    return sum / static_cast<float>(n - 1);
}

inline float gesture_turb_range(const float* turb, uint16_t n) {
    if (n < 1) return 0.0f;
    float min_val = turb[0], max_val = turb[0];
    for (uint16_t i = 1; i < n; i++) {
        if (turb[i] < min_val) min_val = turb[i];
        if (turb[i] > max_val) max_val = turb[i];
    }
    return max_val - min_val;
}

inline float gesture_turb_delta_energy(const float* turb, uint16_t n) {
    if (n < 2) return 0.0f;
    float sum = 0.0f;
    for (uint16_t i = 1; i < n; i++) {
        const float d = turb[i] - turb[i - 1];
        sum += d * d;
    }
    return sum / static_cast<float>(n - 1);
}

// ============================================================================
// Phase features (per-packet phase arrays, averaged over the event)
// ============================================================================

inline float gesture_phase_diff_var(const float* phases, uint8_t n_phases,
                                    uint16_t n_packets) {
    // Called with a flat array [packet0_phases... packet1_phases...] of shape
    // (n_packets * n_phases). Returns mean phase diff variance.
    if (n_packets == 0 || n_phases < 2) return 0.0f;
    float total = 0.0f;
    uint16_t valid = 0;
    for (uint16_t p = 0; p < n_packets; p++) {
        const float* ph = phases + p * n_phases;
        // Compute diffs and their variance
        float diff_sum = 0.0f;
        for (uint8_t i = 0; i < n_phases - 1; i++) diff_sum += ph[i + 1] - ph[i];
        float diff_mean = diff_sum / (n_phases - 1);
        float var = 0.0f;
        for (uint8_t i = 0; i < n_phases - 1; i++) {
            float d = (ph[i + 1] - ph[i]) - diff_mean;
            var += d * d;
        }
        var /= (n_phases - 1);
        total += var;
        valid++;
    }
    return valid > 0 ? total / valid : 0.0f;
}

inline float gesture_phase_entropy(const float* phases, uint8_t n_phases,
                                   uint16_t n_packets) {
    if (n_packets == 0 || n_phases < 2) return 0.0f;
    float total = 0.0f;
    uint16_t valid = 0;
    float log2val = std::log(2.0f);

    for (uint16_t p = 0; p < n_packets; p++) {
        const float* ph = phases + p * n_phases;
        float min_p = ph[0], max_p = ph[0];
        for (uint8_t i = 1; i < n_phases; i++) {
            if (ph[i] < min_p) min_p = ph[i];
            if (ph[i] > max_p) max_p = ph[i];
        }
        float p_range = max_p - min_p;
        if (p_range < 1e-10f) continue;

        uint16_t bins[GESTURE_PHASE_ENTROPY_BINS] = {};
        float bin_width = p_range / GESTURE_PHASE_ENTROPY_BINS;
        for (uint8_t i = 0; i < n_phases; i++) {
            int idx = static_cast<int>((ph[i] - min_p) / bin_width);
            if (idx >= GESTURE_PHASE_ENTROPY_BINS) idx = GESTURE_PHASE_ENTROPY_BINS - 1;
            bins[idx]++;
        }
        float entropy = 0.0f;
        for (uint8_t b = 0; b < GESTURE_PHASE_ENTROPY_BINS; b++) {
            if (bins[b] > 0) {
                float prob = static_cast<float>(bins[b]) / n_phases;
                entropy -= prob * std::log(prob) / log2val;
            }
        }
        total += entropy;
        valid++;
    }
    return valid > 0 ? total / valid : 0.0f;
}

inline float gesture_phase_circular_variance(const float* phases, uint8_t n_phases,
                                             uint16_t n_packets) {
    if (n_packets == 0 || n_phases < 2) return 0.0f;
    float total = 0.0f;
    uint16_t valid = 0;
    for (uint16_t p = 0; p < n_packets; p++) {
        const float* ph = phases + p * n_phases;
        float cos_sum = 0.0f;
        float sin_sum = 0.0f;
        for (uint8_t i = 0; i < n_phases; i++) {
            cos_sum += std::cos(ph[i]);
            sin_sum += std::sin(ph[i]);
        }
        float r = std::sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / n_phases;
        if (r < 0.0f) r = 0.0f;
        if (r > 1.0f) r = 1.0f;
        total += (1.0f - r);
        valid++;
    }
    return valid > 0 ? total / valid : 0.0f;
}

inline float wrap_phase_delta(float delta) {
    constexpr float kPi = 3.14159265358979323846f;
    const float two_pi = 2.0f * kPi;
    while (delta > kPi) delta -= two_pi;
    while (delta < -kPi) delta += two_pi;
    return delta;
}

inline float gesture_phase_inter_sc_coherence(const float* phases, uint8_t n_phases,
                                              uint16_t n_packets) {
    if (n_packets == 0 || n_phases < 3) return 0.0f;
    float total_var = 0.0f;
    uint16_t valid = 0;
    for (uint16_t p = 0; p < n_packets; p++) {
        const float* ph = phases + p * n_phases;
        float mean_d = 0.0f;
        const uint8_t n_diffs = n_phases - 1;
        for (uint8_t i = 0; i < n_diffs; i++) {
            mean_d += wrap_phase_delta(ph[i + 1] - ph[i]);
        }
        mean_d /= static_cast<float>(n_diffs);

        float var_d = 0.0f;
        for (uint8_t i = 0; i < n_diffs; i++) {
            const float d = wrap_phase_delta(ph[i + 1] - ph[i]) - mean_d;
            var_d += d * d;
        }
        var_d /= static_cast<float>(n_diffs);
        total_var += var_d;
        valid++;
    }
    if (valid == 0) return 0.0f;
    const float mean_var = total_var / static_cast<float>(valid);
    return 1.0f / (1.0f + mean_var);
}

// ============================================================================
// Combined feature extraction
// ============================================================================

/**
 * Extract all gesture features from an event buffer.
 *
 * @param turb_buf       Turbulence values for the full event (n_packets floats)
 * @param phases_buf     Phase arrays, flattened (n_packets * n_phases floats)
 * @param n_packets      Number of packets in the event
 * @param n_phases       Number of phases per packet (typically 12)
 * @param features_out   Output array for GESTURE_FEATURE_VECTOR_SIZE values
 */
inline void extract_gesture_features(const float* turb_buf,
                                     const float* phases_buf,
                                     uint16_t n_packets,
                                     uint8_t n_phases,
                                     float* features_out) {
    for (uint8_t i = 0; i < GESTURE_FEATURE_VECTOR_SIZE; i++) features_out[i] = 0.0f;
    if (n_packets < 2) return;

    features_out[0]  = gesture_event_duration(turb_buf, n_packets);
    features_out[1]  = gesture_peak_position(turb_buf, n_packets);
    features_out[2]  = gesture_peak_to_mean_ratio(turb_buf, n_packets);
    features_out[3]  = gesture_rise_fall_asymmetry(turb_buf, n_packets);
    features_out[4]  = gesture_pre_post_energy_ratio(turb_buf, n_packets);
    features_out[5]  = gesture_n_local_peaks(turb_buf, n_packets);
    features_out[6]  = gesture_peak_fwhm(turb_buf, n_packets);
    features_out[7]  = gesture_turb_mad(turb_buf, n_packets);
    features_out[8]  = gesture_turb_iqr(turb_buf, n_packets);
    features_out[9]  = gesture_turb_mid_mean(turb_buf, n_packets);
    features_out[10] = gesture_turb_late_mean(turb_buf, n_packets);
    features_out[11] = gesture_turb_late_minus_mid(turb_buf, n_packets);
    features_out[12] = gesture_turb_diff_abs_mean(turb_buf, n_packets);
    features_out[13] = gesture_phase_diff_var(phases_buf, n_phases, n_packets);
    features_out[14] = gesture_phase_entropy(phases_buf, n_phases, n_packets);
    features_out[15] = gesture_phase_circular_variance(phases_buf, n_phases, n_packets);
    features_out[16] = gesture_phase_inter_sc_coherence(phases_buf, n_phases, n_packets);
}

inline float extract_gesture_feature_by_id(const float* turb_buf,
                                           const float* phases_buf,
                                           uint16_t n_packets,
                                           uint8_t n_phases,
                                           uint8_t feature_id) {
    switch (feature_id) {
        case 0: return gesture_event_duration(turb_buf, n_packets);
        case 1: return gesture_peak_position(turb_buf, n_packets);
        case 2: return gesture_peak_to_mean_ratio(turb_buf, n_packets);
        case 3: return gesture_rise_fall_asymmetry(turb_buf, n_packets);
        case 4: return gesture_pre_post_energy_ratio(turb_buf, n_packets);
        case 5: return gesture_n_local_peaks(turb_buf, n_packets);
        case 6: return gesture_peak_fwhm(turb_buf, n_packets);
        case 7: return gesture_turb_mad(turb_buf, n_packets);
        case 8: return gesture_turb_iqr(turb_buf, n_packets);
        case 9: return gesture_turb_mid_mean(turb_buf, n_packets);
        case 10: return gesture_turb_late_mean(turb_buf, n_packets);
        case 11: return gesture_turb_late_minus_mid(turb_buf, n_packets);
        case 12: return gesture_turb_diff_abs_mean(turb_buf, n_packets);
        case 13: return gesture_phase_diff_var(phases_buf, n_phases, n_packets);
        case 14: return gesture_phase_entropy(phases_buf, n_phases, n_packets);
        case 15: return gesture_phase_circular_variance(phases_buf, n_phases, n_packets);
        case 16: return gesture_phase_inter_sc_coherence(phases_buf, n_phases, n_packets);
        case 17: return gesture_turb_range(turb_buf, n_packets);
        case 18: return gesture_turb_delta_energy(turb_buf, n_packets);
        default: return 0.0f;
    }
}

inline void extract_gesture_model_features(const float* turb_buf,
                                           const float* phases_buf,
                                           uint16_t n_packets,
                                           uint8_t n_phases,
                                           const uint8_t* feature_ids,
                                           uint8_t n_features,
                                           float* features_out) {
    for (uint8_t i = 0; i < n_features; i++) {
        features_out[i] = extract_gesture_feature_by_id(
            turb_buf, phases_buf, n_packets, n_phases, feature_ids[i]
        );
    }
}

}  // namespace espectre
}  // namespace esphome
