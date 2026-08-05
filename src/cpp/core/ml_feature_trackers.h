/*
 * ESPectre - Production ML Feature Trackers
 *
 * Minimal shared trackers for the promoted production ML features that rely on
 * HT20 live-band coherence and normalized amplitude-shape dynamics.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "csi_format.h"
#include "detector_limits.h"

namespace espectre {

constexpr uint8_t HT20_LIVE_BAND_SIZE = 56U;
constexpr std::array<uint8_t, HT20_LIVE_BAND_SIZE> HT20_LIVE_BINS = {
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
};
constexpr uint8_t HT20_COHERENCE_SUBBAND_SIZE = 14U;
constexpr uint8_t HT20_COHERENCE_SUBBAND_COUNT = 4U;
// The DC null splits the live band into two runs that are contiguous in both
// bin number and profile index, so a pair separated by `offset` bins is always
// `left + offset` inside one run.
constexpr uint8_t HT20_LIVE_HALF_SIZE = HT20_LIVE_BAND_SIZE / 2U;
constexpr uint8_t FREQUENCY_COHERENCE_COUNT = 3U;
constexpr std::array<uint8_t, FREQUENCY_COHERENCE_COUNT> FREQUENCY_COHERENCE_OFFSETS = {
    2U, 4U, 12U,
};

inline void extract_ht20_live_complex_profile(const int8_t* csi_data, size_t csi_len,
                                              std::complex<float>* out) {
    if (out == nullptr) {
        return;
    }
    if (csi_data == nullptr || csi_len < HT20_CSI_LEN) {
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            out[i] = std::complex<float>(0.0f, 0.0f);
        }
        return;
    }
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        const uint8_t sc_idx = HT20_LIVE_BINS[i];
        const float imag = static_cast<float>(csi_data[sc_idx * 2U]);
        const float real = static_cast<float>(csi_data[sc_idx * 2U + 1U]);
        out[i] = std::complex<float>(real, imag);
    }
}

// The four subbands tile the live band exactly (0-13, 14-27, 28-41, 42-55), so
// the full-band and subband coherences of one reference read the very same
// cross products. Building them once here lets both consumers share the array
// instead of each recomputing all 56 products and their magnitudes.
inline void fill_coherence_cross(const std::complex<float>* current,
                                 const std::complex<float>* reference,
                                 std::complex<float>* cross, float* magnitude) {
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        cross[i] = current[i] * std::conj(reference[i]);
        magnitude[i] = std::abs(cross[i]);
    }
}

// The aligned sum is `sum_k cross[k] * w^bin[k]` for `w = exp(-i * ramp)`, and
// the live bins of one contiguous band are consecutive, so the sum is a
// polynomial in `w`. Horner evaluates it with one sine and cosine pair for the
// whole band instead of one per bin. Only the magnitude is used, so the shared
// `w^start_bin` factor cancels and the band's absolute position drops out.
inline float delay_compensated_coherence_band_from_cross(const std::complex<float>* cross,
                                                         const float* magnitude,
                                                         uint8_t start, uint8_t count) {
    if (count < 2U) {
        return 0.0f;
    }
    float total = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        total += magnitude[start + i];
    }
    if (total <= 0.0f) {
        return 0.0f;
    }
    std::complex<float> ramp_sum(0.0f, 0.0f);
    for (uint8_t i = 1; i < count; i++) {
        ramp_sum += cross[start + i] * std::conj(cross[start + i - 1U]);
    }
    const float ramp = std::atan2(ramp_sum.imag(), ramp_sum.real());
    const std::complex<float> w(std::cos(-ramp), std::sin(-ramp));
    std::complex<float> aligned = cross[start + count - 1U];
    for (int16_t i = static_cast<int16_t>(count) - 2; i >= 0; i--) {
        aligned = aligned * w + cross[start + i];
    }
    return std::abs(aligned) / total;
}

inline float delay_compensated_coherence_from_cross(const std::complex<float>* cross,
                                                    const float* magnitude) {
    float total = 0.0f;
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        total += magnitude[i];
    }
    if (total <= 0.0f) {
        return 0.0f;
    }
    std::complex<float> ramp_sum(0.0f, 0.0f);
    for (uint8_t i = 1; i < HT20_LIVE_BAND_SIZE; i++) {
        if (HT20_LIVE_BINS[i] - HT20_LIVE_BINS[i - 1U] != 1U) {
            continue;
        }
        ramp_sum += cross[i] * std::conj(cross[i - 1U]);
    }
    const float ramp = std::atan2(ramp_sum.imag(), ramp_sum.real());
    const std::complex<float> w(std::cos(-ramp), std::sin(-ramp));
    // One Horner pass per half. The DC null makes the bins jump between them,
    // so the upper half re-enters through `w^(bin[half] - bin[0])`; the shared
    // `w^bin[0]` factor cancels under the magnitude.
    std::complex<float> lower = cross[HT20_LIVE_HALF_SIZE - 1U];
    std::complex<float> upper = cross[HT20_LIVE_BAND_SIZE - 1U];
    for (int16_t j = static_cast<int16_t>(HT20_LIVE_HALF_SIZE) - 2; j >= 0; j--) {
        lower = lower * w + cross[j];
        upper = upper * w + cross[HT20_LIVE_HALF_SIZE + j];
    }
    const float gap = static_cast<float>(HT20_LIVE_BINS[HT20_LIVE_HALF_SIZE]) -
                      static_cast<float>(HT20_LIVE_BINS[0]);
    const std::complex<float> shift(std::cos(-ramp * gap), std::sin(-ramp * gap));
    return std::abs(lower + shift * upper) / total;
}

inline void subband_coherences_from_cross(const std::complex<float>* cross,
                                          const float* magnitude, float* out) {
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        out[b] = delay_compensated_coherence_band_from_cross(
            cross, magnitude, static_cast<uint8_t>(b * HT20_COHERENCE_SUBBAND_SIZE),
            HT20_COHERENCE_SUBBAND_SIZE);
    }
}

inline float delay_compensated_coherence_band(const std::complex<float>* current,
                                              const std::complex<float>* reference,
                                              uint8_t start, uint8_t count) {
    if (current == nullptr || reference == nullptr || count < 2U) {
        return 0.0f;
    }
    std::complex<float> cross[HT20_LIVE_BAND_SIZE]{};
    float magnitude[HT20_LIVE_BAND_SIZE]{};
    fill_coherence_cross(current, reference, cross, magnitude);
    return delay_compensated_coherence_band_from_cross(cross, magnitude, start, count);
}

inline float delay_compensated_coherence(const std::complex<float>* current,
                                         const std::complex<float>* reference) {
    if (current == nullptr || reference == nullptr) {
        return 0.0f;
    }
    std::complex<float> cross[HT20_LIVE_BAND_SIZE]{};
    float magnitude[HT20_LIVE_BAND_SIZE]{};
    fill_coherence_cross(current, reference, cross, magnitude);
    return delay_compensated_coherence_from_cross(cross, magnitude);
}

inline void subband_coherences(const std::complex<float>* current,
                               const std::complex<float>* reference,
                               float* out) {
    if (out == nullptr) {
        return;
    }
    if (current == nullptr || reference == nullptr) {
        for (uint8_t i = 0; i < HT20_COHERENCE_SUBBAND_COUNT; i++) {
            out[i] = 0.0f;
        }
        return;
    }
    std::complex<float> cross[HT20_LIVE_BAND_SIZE]{};
    float magnitude[HT20_LIVE_BAND_SIZE]{};
    fill_coherence_cross(current, reference, cross, magnitude);
    subband_coherences_from_cross(cross, magnitude, out);
}

inline void normalized_amplitude_profile(const std::complex<float>* profile,
                                         float* out) {
    if (profile == nullptr || out == nullptr) {
        return;
    }
    float squared_sum = 0.0f;
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        out[i] = std::abs(profile[i]);
        squared_sum += out[i] * out[i];
    }
    const float norm = std::sqrt(squared_sum);
    if (norm <= 0.0f) {
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            out[i] = 0.0f;
        }
        return;
    }
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        out[i] /= norm;
    }
}

inline float motion_participation(const float* energy, uint8_t count) {
    if (energy == nullptr || count == 0U) {
        return 0.0f;
    }
    float total = 0.0f;
    float squared = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        total += energy[i];
        squared += energy[i] * energy[i];
    }
    if (total <= 0.0f || squared <= 0.0f) {
        return 0.0f;
    }
    return (total * total) / (static_cast<float>(count) * squared);
}

// Cache the squared magnitude of every live subcarrier once per packet. Every
// offset reads the same magnitudes, so this replaces the repeated per-pair work.
inline void fill_frequency_coherence_squares(const std::complex<float>* profile,
                                             float* squares) {
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        squares[i] = std::norm(profile[i]);
    }
}

// Coherence at one offset, reusing the cached squared magnitudes. Walking the
// two halves visits pairs in ascending left index, exactly the order the full
// 56x56 scan produced, so the numerator accumulates identically.
inline float frequency_coherence_from_squares(const std::complex<float>* profile,
                                              const float* squares, uint8_t offset) {
    const int span = static_cast<int>(HT20_LIVE_HALF_SIZE) - static_cast<int>(offset);
    if (span <= 0) {
        return 0.0f;
    }
    std::complex<float> numerator(0.0f, 0.0f);
    float left_norm = 0.0f;
    float right_norm = 0.0f;
    for (uint8_t start = 0U; start < HT20_LIVE_BAND_SIZE;
         start = static_cast<uint8_t>(start + HT20_LIVE_HALF_SIZE)) {
        const uint8_t stop = static_cast<uint8_t>(start + span);
        for (uint8_t left = start; left < stop; left++) {
            const uint8_t right = static_cast<uint8_t>(left + offset);
            numerator += std::conj(profile[left]) * profile[right];
            left_norm += squares[left];
            right_norm += squares[right];
        }
    }
    const float denominator = std::sqrt(left_norm) * std::sqrt(right_norm);
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    return std::abs(numerator) / denominator;
}

inline float frequency_coherence(const std::complex<float>* profile, uint8_t offset = 4U) {
    if (profile == nullptr) {
        return 0.0f;
    }
    float squares[HT20_LIVE_BAND_SIZE]{};
    fill_frequency_coherence_squares(profile, squares);
    return frequency_coherence_from_squares(profile, squares, offset);
}

// Fill `out` with the FREQUENCY_COHERENCE_OFFSETS coherences of one packet,
// sharing the squared magnitudes across the three offsets. Stack only, so the
// per-packet path stays allocation free.
inline void frequency_coherences(const std::complex<float>* profile, float* out) {
    if (out == nullptr) {
        return;
    }
    if (profile == nullptr) {
        for (uint8_t i = 0; i < FREQUENCY_COHERENCE_COUNT; i++) {
            out[i] = 0.0f;
        }
        return;
    }
    float squares[HT20_LIVE_BAND_SIZE]{};
    fill_frequency_coherence_squares(profile, squares);
    for (uint8_t i = 0; i < FREQUENCY_COHERENCE_COUNT; i++) {
        out[i] = frequency_coherence_from_squares(profile, squares,
                                                  FREQUENCY_COHERENCE_OFFSETS[i]);
    }
}

class ChannelShapeTracker {
 public:
  ChannelShapeTracker() = default;

  void configure(uint16_t capacity, uint16_t lag) {
    capacity_ = std::min<uint16_t>(capacity, DETECTOR_MAX_WINDOW_SIZE);
    lag_ = std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX);
    lag_distance_ring_.assign(capacity_, 0.0f);
    adjacent_distance_ring_.assign(capacity_, 0.0f);
    frequency_coherence_ring_.assign(capacity_, 0.0f);
    frequency_curve_ring_.assign(capacity_, 0.0f);
    motion_energy_ring_.assign(static_cast<size_t>(capacity_) * HT20_LIVE_BAND_SIZE, 0.0f);
    ring_.assign(lag_profile_size_(), 0.0f);
    clear();
  }

  void clear() {
    ring_index_ = 0U;
    has_previous_ = false;
    ring_filled_.fill(false);
    previous_.fill(0.0f);
    motion_energy_.fill(0.0f);
    lag_distance_slot_ = 0U;
    lag_distance_count_ = 0U;
    lag_distance_sum_ = 0.0f;
    adjacent_distance_slot_ = 0U;
    adjacent_distance_count_ = 0U;
    adjacent_distance_sum_ = 0.0f;
    motion_energy_slot_ = 0U;
    motion_energy_count_ = 0U;
    frequency_coherence_slot_ = 0U;
    frequency_coherence_count_ = 0U;
    frequency_coherence_sum_ = 0.0f;
    frequency_coherence_square_sum_ = 0.0f;
    frequency_curve_slot_ = 0U;
    frequency_curve_count_ = 0U;
    frequency_curve_sum_ = 0.0f;
    frequency_curve_square_sum_ = 0.0f;
    std::fill(lag_distance_ring_.begin(), lag_distance_ring_.end(), 0.0f);
    std::fill(adjacent_distance_ring_.begin(), adjacent_distance_ring_.end(), 0.0f);
    std::fill(frequency_coherence_ring_.begin(), frequency_coherence_ring_.end(), 0.0f);
    std::fill(frequency_curve_ring_.begin(), frequency_curve_ring_.end(), 0.0f);
    std::fill(motion_energy_ring_.begin(), motion_energy_ring_.end(), 0.0f);
  }

  void process_packet(const int8_t* csi_data, size_t csi_len) {
    if (capacity_ == 0U) {
        return;
    }
    std::complex<float> complex_values[HT20_LIVE_BAND_SIZE]{};
    float profile[HT20_LIVE_BAND_SIZE]{};
    extract_ht20_live_complex_profile(csi_data, csi_len, complex_values);
    normalized_amplitude_profile(complex_values, profile);

    const uint16_t slot = ring_index_;
    const size_t ring_base = static_cast<size_t>(slot) * HT20_LIVE_BAND_SIZE;
    if (ring_filled_[slot]) {
        float squared_sum = 0.0f;
        float delta[HT20_LIVE_BAND_SIZE]{};
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            const float diff = profile[i] - ring_[ring_base + i];
            delta[i] = diff * diff;
            squared_sum += delta[i];
        }
        push_scalar_(std::sqrt(squared_sum), lag_distance_ring_,
                     lag_distance_slot_, lag_distance_count_, lag_distance_sum_);
        push_motion_energy_(delta);
    }
    if (has_previous_) {
        float squared_sum = 0.0f;
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            const float diff = profile[i] - previous_[i];
            squared_sum += diff * diff;
        }
        push_scalar_(std::sqrt(squared_sum), adjacent_distance_ring_,
                     adjacent_distance_slot_, adjacent_distance_count_,
                     adjacent_distance_sum_);
    }

    static_assert(FREQUENCY_COHERENCE_OFFSETS[0] == 2U &&
                      FREQUENCY_COHERENCE_OFFSETS[1] == 4U &&
                      FREQUENCY_COHERENCE_OFFSETS[2] == 12U,
                  "short, base, and long coherence are read by index below");
    float coherences[FREQUENCY_COHERENCE_COUNT]{};
    frequency_coherences(complex_values, coherences);
    push_frequency_coherence_(coherences[1]);
    const float short_coherence = coherences[0];
    const float long_coherence = coherences[2];
    const float coherence_sum = short_coherence + long_coherence;
    const float curve_contrast = coherence_sum > 0.0f
        ? (short_coherence - long_coherence) / coherence_sum
        : 0.0f;
    push_frequency_curve_(curve_contrast);

    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        previous_[i] = profile[i];
        ring_[ring_base + i] = profile[i];
    }
    has_previous_ = true;
    ring_filled_[slot] = true;
    ring_index_ = static_cast<uint16_t>((ring_index_ + 1U) % lag_);
  }

  uint16_t count() const { return lag_distance_count_; }

  float shape_spread() const {
    return motion_participation(motion_energy_.data(), HT20_LIVE_BAND_SIZE);
  }

  float frequency_coherence_cv() const {
    if (frequency_coherence_count_ == 0U) {
        return 0.0f;
    }
    const double mean =
        frequency_coherence_sum_ / static_cast<double>(frequency_coherence_count_);
    const double variance = std::max(
        0.0,
        frequency_coherence_square_sum_ /
            static_cast<double>(frequency_coherence_count_) -
            mean * mean);
    if (mean <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::sqrt(variance) / mean);
  }

  float frequency_coherence_curve_std() const {
    if (frequency_curve_count_ == 0U) {
        return 0.0f;
    }
    const double mean = frequency_curve_sum_ / static_cast<double>(frequency_curve_count_);
    const double variance = std::max(
        0.0,
        frequency_curve_square_sum_ / static_cast<double>(frequency_curve_count_) -
            mean * mean);
    return static_cast<float>(std::sqrt(variance));
  }

 private:
  // A tracker configured with no capacity is never fed, so it holds no ring.
  size_t lag_profile_size_() const {
    return capacity_ == 0U ? 0U
                           : static_cast<size_t>(lag_) * HT20_LIVE_BAND_SIZE;
  }

  void push_scalar_(float value, std::vector<float>& ring, uint16_t& slot,
                    uint16_t& count, float& total) {
    if (ring.empty()) {
        return;
    }
    if (count < capacity_) {
        count++;
    } else {
        total -= ring[slot];
    }
    ring[slot] = value;
    total += value;
    slot = static_cast<uint16_t>((slot + 1U) % capacity_);
  }

  void push_motion_energy_(const float* values) {
    if (motion_energy_ring_.empty() || values == nullptr) {
        return;
    }
    const size_t base = static_cast<size_t>(motion_energy_slot_) * HT20_LIVE_BAND_SIZE;
    if (motion_energy_count_ < capacity_) {
        motion_energy_count_++;
    } else {
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            motion_energy_[i] -= motion_energy_ring_[base + i];
        }
    }
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        motion_energy_ring_[base + i] = values[i];
        motion_energy_[i] += values[i];
    }
    motion_energy_slot_ = static_cast<uint16_t>((motion_energy_slot_ + 1U) % capacity_);
  }

  void push_frequency_coherence_(float value) {
    if (frequency_coherence_ring_.empty()) {
        return;
    }
    if (frequency_coherence_count_ < capacity_) {
        frequency_coherence_count_++;
    } else {
        const float old = frequency_coherence_ring_[frequency_coherence_slot_];
        frequency_coherence_sum_ -= old;
        frequency_coherence_square_sum_ -= old * old;
    }
    frequency_coherence_ring_[frequency_coherence_slot_] = value;
    frequency_coherence_sum_ += value;
    frequency_coherence_square_sum_ += value * value;
    frequency_coherence_slot_ =
        static_cast<uint16_t>((frequency_coherence_slot_ + 1U) % capacity_);
  }

  void push_frequency_curve_(float value) {
    if (frequency_curve_ring_.empty()) {
        return;
    }
    if (frequency_curve_count_ < capacity_) {
        frequency_curve_count_++;
    } else {
        const float old = frequency_curve_ring_[frequency_curve_slot_];
        frequency_curve_sum_ -= old;
        frequency_curve_square_sum_ -= old * old;
    }
    frequency_curve_ring_[frequency_curve_slot_] = value;
    frequency_curve_sum_ += value;
    frequency_curve_square_sum_ += value * value;
    frequency_curve_slot_ =
        static_cast<uint16_t>((frequency_curve_slot_ + 1U) % capacity_);
  }

  uint16_t capacity_{0U};
  uint16_t lag_{1U};
  uint16_t ring_index_{0U};
  bool has_previous_{false};
  // Sized to the configured lag rather than L1_DELTA_LAG_MAX: the ceiling is
  // 32 while the 100 ms contract resolves to 10 packets at the nominal rate,
  // so a static array would leave two thirds of 32 x 56 floats unused.
  std::vector<float> ring_{};
  std::array<bool, L1_DELTA_LAG_MAX> ring_filled_{};
  std::array<float, HT20_LIVE_BAND_SIZE> previous_{};
  std::array<float, HT20_LIVE_BAND_SIZE> motion_energy_{};
  std::vector<float> lag_distance_ring_{};
  std::vector<float> adjacent_distance_ring_{};
  std::vector<float> frequency_coherence_ring_{};
  std::vector<float> frequency_curve_ring_{};
  std::vector<float> motion_energy_ring_{};
  uint16_t lag_distance_slot_{0U};
  uint16_t lag_distance_count_{0U};
  float lag_distance_sum_{0.0f};
  uint16_t adjacent_distance_slot_{0U};
  uint16_t adjacent_distance_count_{0U};
  float adjacent_distance_sum_{0.0f};
  uint16_t motion_energy_slot_{0U};
  uint16_t motion_energy_count_{0U};
  uint16_t frequency_coherence_slot_{0U};
  uint16_t frequency_coherence_count_{0U};
  double frequency_coherence_sum_{0.0};
  double frequency_coherence_square_sum_{0.0};
  uint16_t frequency_curve_slot_{0U};
  uint16_t frequency_curve_count_{0U};
  double frequency_curve_sum_{0.0};
  double frequency_curve_square_sum_{0.0};
};

class ChannelCoherenceTracker {
 public:
  ChannelCoherenceTracker() = default;

  void configure(uint16_t capacity, uint16_t lag) {
    capacity_ = std::min<uint16_t>(capacity, DETECTOR_MAX_WINDOW_SIZE);
    lag_ = std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX);
    lag_ring_.assign(capacity_, 0.0f);
    adjacent_ring_.assign(capacity_, 0.0f);
    subband_lag_ring_.assign(static_cast<size_t>(capacity_) * HT20_COHERENCE_SUBBAND_COUNT, 0.0f);
    subband_adjacent_ring_.assign(static_cast<size_t>(capacity_) * HT20_COHERENCE_SUBBAND_COUNT, 0.0f);
    ring_.assign(lag_profile_size_(), std::complex<float>(0.0f, 0.0f));
    clear();
  }

  void clear() {
    ring_index_ = 0U;
    has_previous_ = false;
    ring_filled_.fill(false);
    previous_.fill(std::complex<float>(0.0f, 0.0f));
    lag_sum_ = 0.0f;
    lag_count_ = 0U;
    adjacent_sum_ = 0.0f;
    adjacent_count_ = 0U;
    lag_slot_ = 0U;
    adjacent_slot_ = 0U;
    subband_lag_sum_.fill(0.0f);
    subband_adjacent_sum_.fill(0.0f);
    subband_lag_slot_ = 0U;
    subband_lag_count_ = 0U;
    subband_adjacent_slot_ = 0U;
    subband_adjacent_count_ = 0U;
    std::fill(lag_ring_.begin(), lag_ring_.end(), 0.0f);
    std::fill(adjacent_ring_.begin(), adjacent_ring_.end(), 0.0f);
    std::fill(subband_lag_ring_.begin(), subband_lag_ring_.end(), 0.0f);
    std::fill(subband_adjacent_ring_.begin(), subband_adjacent_ring_.end(), 0.0f);
  }

  void process_packet(const int8_t* csi_data, size_t csi_len) {
    if (capacity_ == 0U) {
        return;
    }
    std::complex<float> profile[HT20_LIVE_BAND_SIZE]{};
    extract_ht20_live_complex_profile(csi_data, csi_len, profile);
    // One cross-product array per reference, shared by the full band and the
    // four subbands that tile it.
    std::complex<float> cross[HT20_LIVE_BAND_SIZE]{};
    float magnitude[HT20_LIVE_BAND_SIZE]{};
    const uint16_t slot = ring_index_;
    const size_t ring_base = static_cast<size_t>(slot) * HT20_LIVE_BAND_SIZE;
    if (ring_filled_[slot]) {
        fill_coherence_cross(profile, &ring_[ring_base], cross, magnitude);
        const float lag_value = delay_compensated_coherence_from_cross(cross, magnitude);
        push_scalar_(lag_value, lag_ring_, lag_slot_, lag_count_, lag_sum_);
        float lag_subbands[HT20_COHERENCE_SUBBAND_COUNT]{};
        subband_coherences_from_cross(cross, magnitude, lag_subbands);
        push_subbands_(lag_subbands, subband_lag_ring_, subband_lag_slot_,
                       subband_lag_count_, subband_lag_sum_);
    }
    if (has_previous_) {
        fill_coherence_cross(profile, previous_.data(), cross, magnitude);
        const float adjacent_value = delay_compensated_coherence_from_cross(cross, magnitude);
        push_scalar_(adjacent_value, adjacent_ring_, adjacent_slot_,
                     adjacent_count_, adjacent_sum_);
        float adjacent_subbands[HT20_COHERENCE_SUBBAND_COUNT]{};
        subband_coherences_from_cross(cross, magnitude, adjacent_subbands);
        push_subbands_(adjacent_subbands, subband_adjacent_ring_,
                       subband_adjacent_slot_, subband_adjacent_count_,
                       subband_adjacent_sum_);
    }
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        previous_[i] = profile[i];
        ring_[ring_base + i] = profile[i];
    }
    has_previous_ = true;
    ring_filled_[slot] = true;
    ring_index_ = static_cast<uint16_t>((ring_index_ + 1U) % lag_);
  }

  uint16_t count() const { return lag_count_; }

  float coherence_gap() const {
    if (lag_count_ == 0U || adjacent_count_ == 0U) {
        return 0.0f;
    }
    const double adjacent_mean =
        adjacent_sum_ / static_cast<double>(adjacent_count_);
    const double lag_mean = lag_sum_ / static_cast<double>(lag_count_);
    return static_cast<float>(adjacent_mean - lag_mean);
  }

  float coherence_subband_gap_median() const {
    if (subband_lag_count_ == 0U || subband_adjacent_count_ == 0U) {
        return 0.0f;
    }
    float gaps[HT20_COHERENCE_SUBBAND_COUNT]{};
    for (uint8_t i = 0; i < HT20_COHERENCE_SUBBAND_COUNT; i++) {
        const double adjacent_mean =
            subband_adjacent_sum_[i] /
            static_cast<double>(subband_adjacent_count_);
        const double lag_mean =
            subband_lag_sum_[i] / static_cast<double>(subband_lag_count_);
        gaps[i] = static_cast<float>(adjacent_mean - lag_mean);
    }
    std::sort(gaps, gaps + HT20_COHERENCE_SUBBAND_COUNT);
    return 0.5f * (gaps[1] + gaps[2]);
  }

 private:
  // A tracker configured with no capacity is never fed, so it holds no ring.
  size_t lag_profile_size_() const {
    return capacity_ == 0U ? 0U
                           : static_cast<size_t>(lag_) * HT20_LIVE_BAND_SIZE;
  }

  void push_scalar_(float value, std::vector<float>& ring, uint16_t& slot,
                    uint16_t& count, double& total) {
    if (ring.empty()) {
        return;
    }
    if (count < capacity_) {
        count++;
    } else {
        total -= static_cast<double>(ring[slot]);
    }
    ring[slot] = value;
    total += static_cast<double>(value);
    slot = static_cast<uint16_t>((slot + 1U) % capacity_);
  }

  void push_subbands_(const float* values, std::vector<float>& ring, uint16_t& slot,
                      uint16_t& count,
                      std::array<double, HT20_COHERENCE_SUBBAND_COUNT>& total) {
    if (ring.empty() || values == nullptr) {
        return;
    }
    const size_t base =
        static_cast<size_t>(slot) * HT20_COHERENCE_SUBBAND_COUNT;
    if (count < capacity_) {
        count++;
    } else {
        for (uint8_t i = 0; i < HT20_COHERENCE_SUBBAND_COUNT; i++) {
            total[i] -= static_cast<double>(ring[base + i]);
        }
    }
    for (uint8_t i = 0; i < HT20_COHERENCE_SUBBAND_COUNT; i++) {
        ring[base + i] = values[i];
        total[i] += static_cast<double>(values[i]);
    }
    slot = static_cast<uint16_t>((slot + 1U) % capacity_);
  }

  uint16_t capacity_{0U};
  uint16_t lag_{1U};
  uint16_t ring_index_{0U};
  bool has_previous_{false};
  // Sized to the configured lag, as in ChannelShapeTracker. This ring holds
  // complex samples, so a static L1_DELTA_LAG_MAX array is the single largest
  // allocation in the whole ML detector.
  std::vector<std::complex<float>> ring_{};
  std::array<bool, L1_DELTA_LAG_MAX> ring_filled_{};
  std::array<std::complex<float>, HT20_LIVE_BAND_SIZE> previous_{};
  std::vector<float> lag_ring_{};
  std::vector<float> adjacent_ring_{};
  std::vector<float> subband_lag_ring_{};
  std::vector<float> subband_adjacent_ring_{};
  double lag_sum_{0.0};
  uint16_t lag_count_{0U};
  double adjacent_sum_{0.0};
  uint16_t adjacent_count_{0U};
  uint16_t lag_slot_{0U};
  uint16_t adjacent_slot_{0U};
  std::array<double, HT20_COHERENCE_SUBBAND_COUNT> subband_lag_sum_{};
  std::array<double, HT20_COHERENCE_SUBBAND_COUNT> subband_adjacent_sum_{};
  uint16_t subband_lag_slot_{0U};
  uint16_t subband_lag_count_{0U};
  uint16_t subband_adjacent_slot_{0U};
  uint16_t subband_adjacent_count_{0U};
};

}  // namespace espectre
