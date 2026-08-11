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
// The DC null splits the live band into two runs that are contiguous in both
// bin number and profile index, so a pair separated by `offset` bins is always
// `left + offset` inside one run.
constexpr uint8_t HT20_LIVE_HALF_SIZE = HT20_LIVE_BAND_SIZE / 2U;
constexpr uint8_t FREQUENCY_COHERENCE_COUNT = 2U;
constexpr std::array<uint8_t, FREQUENCY_COHERENCE_COUNT> FREQUENCY_COHERENCE_OFFSETS = {
    2U, 12U,
};
constexpr uint8_t CHANNEL_SHAPE_SUBBAND_COUNT = 8U;
constexpr uint8_t CHANNEL_SHAPE_SUBBAND_SIZE =
    HT20_LIVE_BAND_SIZE / CHANNEL_SHAPE_SUBBAND_COUNT;
constexpr uint32_t CHANNEL_SHAPE_BIN_US = 80000U;
constexpr uint32_t CHANNEL_SHAPE_WINDOW_US = 1000000U;
constexpr uint8_t CHANNEL_SHAPE_WINDOW_BINS =
    static_cast<uint8_t>((CHANNEL_SHAPE_WINDOW_US + CHANNEL_SHAPE_BIN_US - 1U) /
                         CHANNEL_SHAPE_BIN_US);
constexpr uint8_t CHANNEL_SHAPE_MAX_PROFILES_PER_BIN = L1_DELTA_LAG_MAX;
constexpr std::array<std::array<float, CHANNEL_SHAPE_SUBBAND_COUNT>,
                     CHANNEL_SHAPE_SUBBAND_COUNT>
    CHANNEL_SHAPE_DCT = {{
        {{0.3535533906f, 0.4903926402f, 0.4619397663f, 0.4157348062f, 0.3535533906f, 0.2777851165f, 0.1913417162f, 0.0975451610f}},
        {{0.3535533906f, 0.4157348062f, 0.1913417162f, -0.0975451610f, -0.3535533906f, -0.4903926402f, -0.4619397663f, -0.2777851165f}},
        {{0.3535533906f, 0.2777851165f, -0.1913417162f, -0.4903926402f, -0.3535533906f, 0.0975451610f, 0.4619397663f, 0.4157348062f}},
        {{0.3535533906f, 0.0975451610f, -0.4619397663f, -0.2777851165f, 0.3535533906f, 0.4157348062f, -0.1913417162f, -0.4903926402f}},
        {{0.3535533906f, -0.0975451610f, -0.4619397663f, 0.2777851165f, 0.3535533906f, -0.4157348062f, -0.1913417162f, 0.4903926402f}},
        {{0.3535533906f, -0.2777851165f, -0.1913417162f, 0.4903926402f, -0.3535533906f, -0.0975451610f, 0.4619397663f, -0.4157348062f}},
        {{0.3535533906f, -0.4157348062f, 0.1913417162f, 0.0975451610f, -0.3535533906f, 0.4903926402f, -0.4619397663f, 0.2777851165f}},
        {{0.3535533906f, -0.4903926402f, 0.4619397663f, -0.4157348062f, 0.3535533906f, -0.2777851165f, 0.1913417162f, -0.0975451610f}},
    }};

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

inline float frequency_coherence(const std::complex<float>* profile, uint8_t offset = 2U) {
    if (profile == nullptr ||
        (offset != FREQUENCY_COHERENCE_OFFSETS[0] &&
         offset != FREQUENCY_COHERENCE_OFFSETS[1])) {
        return 0.0f;
    }
    float squares[HT20_LIVE_BAND_SIZE]{};
    fill_frequency_coherence_squares(profile, squares);
    return frequency_coherence_from_squares(profile, squares, offset);
}

// Fill `out` with the FREQUENCY_COHERENCE_OFFSETS coherences of one packet,
// sharing the squared magnitudes across the two offsets Classic consumes.
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

class ChannelShapeTrajectoryTracker {
 public:
  void configure(bool enabled) {
    enabled_ = enabled;
    clear();
  }

  void clear() {
    bin_count_ = 0U;
    current_profile_count_ = 0U;
    current_bin_ = 0U;
    has_current_bin_ = false;
    has_previous_raw_ = false;
    previous_raw_.fill(0);
    for (auto& bin : bins_) {
      bin.index = 0U;
      bin.modes.fill(0.0f);
    }
    for (auto& profile : current_profiles_) {
      profile.fill(0.0f);
    }
  }

  void process_packet(const int8_t* csi_data, size_t csi_len,
                      uint64_t timestamp_us,
                      const float* subcarrier_energies = nullptr,
                      uint8_t subcarrier_count = 0U) {
    if (!enabled_ || csi_data == nullptr || csi_len < HT20_CSI_LEN) {
      return;
    }
    bool duplicate = has_previous_raw_;
    for (size_t i = 0; i < HT20_CSI_LEN; i++) {
      if (!has_previous_raw_ || previous_raw_[i] != csi_data[i]) {
        duplicate = false;
      }
      previous_raw_[i] = csi_data[i];
    }
    has_previous_raw_ = true;
    if (duplicate) {
      return;
    }

    const uint64_t bin_index = timestamp_us / CHANNEL_SHAPE_BIN_US;
    if (!has_current_bin_) {
      current_bin_ = bin_index;
      has_current_bin_ = true;
    } else if (bin_index != current_bin_) {
      finalize_current_bin_();
      current_bin_ = bin_index;
      current_profile_count_ = 0U;
      trim_(bin_index);
    }
    if (current_profile_count_ >= CHANNEL_SHAPE_MAX_PROFILES_PER_BIN) {
      return;
    }
    fill_profile_(csi_data, subcarrier_energies, subcarrier_count,
                  current_profiles_[current_profile_count_]);
    current_profile_count_++;
  }

  void trajectory_features(float& coherent_innovation_energy,
                           float& excess_path) const {
    coherent_innovation_energy = 0.0f;
    excess_path = 0.0f;
    std::array<PathPoint, CHANNEL_SHAPE_WINDOW_BINS + 1U> path{};
    const uint8_t count = build_path_(path);
    if (count < 3U) {
      return;
    }
    std::array<float, CHANNEL_SHAPE_WINDOW_BINS - 1U> innovation_samples{};
    std::array<float, CHANNEL_SHAPE_WINDOW_BINS - 1U> excess_samples{};
    uint8_t innovation_count = 0U;
    uint8_t excess_count = 0U;
    Profile first_modes = path[0].modes;
    Profile middle_modes = path[1].modes;
    for (uint8_t i = 2U; i < count; i++) {
      const Profile last_modes = path[i].modes;
      const uint64_t previous_dt = path[i - 1U].index - path[i - 2U].index;
      const uint64_t current_dt = path[i].index - path[i - 1U].index;
      float first_norm_squared = 0.0f;
      float second_norm_squared = 0.0f;
      float chord_norm_squared = 0.0f;
      float first_high_squared = 0.0f;
      float second_high_squared = 0.0f;
      float chord_high_squared = 0.0f;
      float innovation_low_squared = 0.0f;
      float innovation_high_squared = 0.0f;
      const float ratio = previous_dt > 0U
                              ? static_cast<float>(current_dt) /
                                    static_cast<float>(previous_dt)
                              : 0.0f;
      for (uint8_t j = 0U; j < CHANNEL_SHAPE_SUBBAND_COUNT; j++) {
        const float first_delta = middle_modes[j] - first_modes[j];
        const float second_delta = last_modes[j] - middle_modes[j];
        const float chord_delta = last_modes[j] - first_modes[j];
        first_norm_squared += first_delta * first_delta;
        second_norm_squared += second_delta * second_delta;
        chord_norm_squared += chord_delta * chord_delta;
        if (j >= 4U) {
          first_high_squared += first_delta * first_delta;
          second_high_squared += second_delta * second_delta;
          chord_high_squared += chord_delta * chord_delta;
        }
        if (previous_dt > 0U && current_dt > 0U && j > 0U) {
          const float residual = second_delta - ratio * first_delta;
          if (j < 4U) {
            innovation_low_squared += residual * residual;
          } else {
            innovation_high_squared += residual * residual;
          }
        }
      }
      if (previous_dt > 0U && current_dt > 0U) {
        innovation_samples[innovation_count++] = std::max(
            0.0f, innovation_low_squared - innovation_high_squared);
      }
      // Parseval: the orthonormal DCT preserves full-profile L2 distances.
      const float raw_excess = std::sqrt(first_norm_squared) +
                               std::sqrt(second_norm_squared) -
                               std::sqrt(chord_norm_squared);
      const float high_excess = std::sqrt(first_high_squared) +
                                std::sqrt(second_high_squared) -
                                std::sqrt(chord_high_squared);
      excess_samples[excess_count++] =
          std::max(0.0f, raw_excess - std::max(0.0f, high_excess));
      first_modes = middle_modes;
      middle_modes = last_modes;
    }
    coherent_innovation_energy =
        median_(innovation_samples.data(), innovation_count);
    excess_path = median_(excess_samples.data(), excess_count);
  }

  float coherent_innovation_energy() const {
    float innovation = 0.0f;
    float excess = 0.0f;
    trajectory_features(innovation, excess);
    return innovation;
  }

  float excess_path() const {
    float innovation = 0.0f;
    float excess = 0.0f;
    trajectory_features(innovation, excess);
    return excess;
  }

 private:
  using Profile = std::array<float, CHANNEL_SHAPE_SUBBAND_COUNT>;
  struct PathPoint {
    uint64_t index{0U};
    Profile modes{};
  };

  static float median_(float* values, uint8_t count) {
    if (count == 0U) return 0.0f;
    std::sort(values, values + count);
    const uint8_t middle = count / 2U;
    return count % 2U == 0U ? 0.5f * (values[middle - 1U] + values[middle])
                            : values[middle];
  }

  static float norm_(const Profile& values, uint8_t start) {
    float total = 0.0f;
    for (uint8_t i = start; i < CHANNEL_SHAPE_SUBBAND_COUNT; i++) {
      total += values[i] * values[i];
    }
    return std::sqrt(total);
  }

  static Profile dct_modes_(const Profile& values) {
    Profile modes{};
    for (uint8_t mode = 0U; mode < CHANNEL_SHAPE_SUBBAND_COUNT; mode++) {
      for (uint8_t i = 0U; i < CHANNEL_SHAPE_SUBBAND_COUNT; i++) {
        modes[mode] += values[i] * CHANNEL_SHAPE_DCT[i][mode];
      }
    }
    return modes;
  }

  static void fill_profile_(const int8_t* csi_data,
                            const float* subcarrier_energies,
                            uint8_t subcarrier_count,
                            Profile& out) {
    out.fill(0.0f);
    float total = 0.0f;
    for (uint8_t i = 0U; i < HT20_LIVE_BAND_SIZE; i++) {
      const uint8_t subcarrier = HT20_LIVE_BINS[i];
      float energy = 0.0f;
      if (subcarrier_energies != nullptr && subcarrier < subcarrier_count) {
        energy = subcarrier_energies[subcarrier];
      } else {
        const float imag = static_cast<float>(csi_data[subcarrier * 2U]);
        const float real = static_cast<float>(csi_data[subcarrier * 2U + 1U]);
        energy = real * real + imag * imag;
      }
      out[i / CHANNEL_SHAPE_SUBBAND_SIZE] += energy;
      total += energy;
    }
    if (total <= 0.0f) return;
    for (float& value : out) value = std::sqrt(value / total);
  }

  Profile median_current_profile_() const {
    Profile result{};
    if (current_profile_count_ == 0U) return result;
    for (uint8_t dimension = 0U; dimension < CHANNEL_SHAPE_SUBBAND_COUNT;
         dimension++) {
      std::array<float, CHANNEL_SHAPE_MAX_PROFILES_PER_BIN> values{};
      for (uint8_t i = 0U; i < current_profile_count_; i++) {
        values[i] = current_profiles_[i][dimension];
      }
      result[dimension] = median_(values.data(), current_profile_count_);
    }
    const float length = norm_(result, 0U);
    if (length > 0.0f) {
      for (float& value : result) value /= length;
    }
    return result;
  }

  void finalize_current_bin_() {
    if (!has_current_bin_ || current_profile_count_ == 0U) return;
    if (bin_count_ >= CHANNEL_SHAPE_WINDOW_BINS) {
      for (uint8_t i = 1U; i < bin_count_; i++) bins_[i - 1U] = bins_[i];
      bin_count_--;
    }
    bins_[bin_count_].index = current_bin_;
    bins_[bin_count_].modes = dct_modes_(median_current_profile_());
    bin_count_++;
  }

  void trim_(uint64_t current_bin) {
    const uint64_t first_bin = current_bin >= CHANNEL_SHAPE_WINDOW_BINS - 1U
                                   ? current_bin - CHANNEL_SHAPE_WINDOW_BINS + 1U
                                   : 0U;
    uint8_t first = 0U;
    while (first < bin_count_ && bins_[first].index < first_bin) first++;
    if (first == 0U) return;
    for (uint8_t i = first; i < bin_count_; i++) bins_[i - first] = bins_[i];
    bin_count_ = static_cast<uint8_t>(bin_count_ - first);
  }

  uint8_t build_path_(
      std::array<PathPoint, CHANNEL_SHAPE_WINDOW_BINS + 1U>& path) const {
    for (uint8_t i = 0U; i < bin_count_; i++) path[i] = bins_[i];
    uint8_t count = bin_count_;
    if (current_profile_count_ > 0U && count < path.size()) {
      path[count].index = current_bin_;
      path[count].modes = dct_modes_(median_current_profile_());
      count++;
    }
    return count;
  }

  bool enabled_{false};
  std::array<PathPoint, CHANNEL_SHAPE_WINDOW_BINS> bins_{};
  uint8_t bin_count_{0U};
  uint64_t current_bin_{0U};
  bool has_current_bin_{false};
  std::array<Profile, CHANNEL_SHAPE_MAX_PROFILES_PER_BIN> current_profiles_{};
  uint8_t current_profile_count_{0U};
  std::array<int8_t, HT20_CSI_LEN> previous_raw_{};
  bool has_previous_raw_{false};
};

class ChannelShapeTracker {
 public:
  ChannelShapeTracker() = default;

  void configure(uint16_t capacity, uint16_t lag, bool track_frequency = true) {
    capacity_ = std::min<uint16_t>(capacity, DETECTOR_MAX_WINDOW_SIZE);
    lag_ = std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX);
    track_frequency_ = track_frequency;
    frequency_curve_ring_.assign(track_frequency_ ? capacity_ : 0U, 0.0f);
    motion_energy_ring_.assign(static_cast<size_t>(capacity_) * HT20_LIVE_BAND_SIZE, 0.0f);
    ring_.assign(lag_profile_size_(), 0.0f);
    clear();
  }

  void clear() {
    ring_index_ = 0U;
    ring_filled_.fill(false);
    motion_energy_.fill(0.0f);
    lag_distance_count_ = 0U;
    motion_energy_slot_ = 0U;
    motion_energy_count_ = 0U;
    frequency_curve_slot_ = 0U;
    frequency_curve_count_ = 0U;
    frequency_curve_sum_ = 0.0f;
    frequency_curve_square_sum_ = 0.0f;
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
    process_profile_(profile, complex_values);
  }

  void process_subcarrier_amplitudes(const float* amplitudes,
                                     uint8_t amplitude_count) {
    if (capacity_ == 0U || amplitudes == nullptr) {
      return;
    }
    float profile[HT20_LIVE_BAND_SIZE]{};
    float squared_sum = 0.0f;
    for (uint8_t i = 0U; i < HT20_LIVE_BAND_SIZE; i++) {
      const uint8_t subcarrier = HT20_LIVE_BINS[i];
      const float value =
          subcarrier < amplitude_count ? amplitudes[subcarrier] : 0.0f;
      profile[i] = value;
      squared_sum += value * value;
    }
    const float norm = std::sqrt(squared_sum);
    if (norm > 0.0f) {
      for (float& value : profile) value /= norm;
    }
    process_profile_(profile, nullptr);
  }

  uint16_t count() const { return lag_distance_count_; }
  bool tracks_frequency() const { return track_frequency_; }

  float shape_spread() const {
    return motion_participation(motion_energy_.data(), HT20_LIVE_BAND_SIZE);
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
  void process_profile_(const float* profile,
                        const std::complex<float>* complex_values) {
    const uint16_t slot = ring_index_;
    const size_t ring_base = static_cast<size_t>(slot) * HT20_LIVE_BAND_SIZE;
    if (ring_filled_[slot]) {
        float delta[HT20_LIVE_BAND_SIZE]{};
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            const float diff = profile[i] - ring_[ring_base + i];
            delta[i] = diff * diff;
        }
        lag_distance_count_ = std::min<uint16_t>(
            capacity_, static_cast<uint16_t>(lag_distance_count_ + 1U));
        push_motion_energy_(delta);
    }

    if (track_frequency_ && complex_values != nullptr) {
      static_assert(FREQUENCY_COHERENCE_OFFSETS[0] == 2U &&
                        FREQUENCY_COHERENCE_OFFSETS[1] == 12U,
                    "short and long coherence are read by index below");
      float coherences[FREQUENCY_COHERENCE_COUNT]{};
      frequency_coherences(complex_values, coherences);
      const float short_coherence = coherences[0];
      const float long_coherence = coherences[1];
      const float coherence_sum = short_coherence + long_coherence;
      const float curve_contrast = coherence_sum > 0.0f
          ? (short_coherence - long_coherence) / coherence_sum
          : 0.0f;
      push_frequency_curve_(curve_contrast);
    }

    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        ring_[ring_base + i] = profile[i];
    }
    ring_filled_[slot] = true;
    ring_index_ = static_cast<uint16_t>((ring_index_ + 1U) % lag_);
  }
  // A tracker configured with no capacity is never fed, so it holds no ring.
  size_t lag_profile_size_() const {
    return capacity_ == 0U ? 0U
                           : static_cast<size_t>(lag_) * HT20_LIVE_BAND_SIZE;
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
  bool track_frequency_{true};
  uint16_t ring_index_{0U};
  // Sized to the configured lag rather than L1_DELTA_LAG_MAX: the ceiling is
  // 32 while the 100 ms contract resolves to 10 packets at the nominal rate,
  // so a static array would leave two thirds of 32 x 56 floats unused.
  std::vector<float> ring_{};
  std::array<bool, L1_DELTA_LAG_MAX> ring_filled_{};
  std::array<float, HT20_LIVE_BAND_SIZE> motion_energy_{};
  std::vector<float> frequency_curve_ring_{};
  std::vector<float> motion_energy_ring_{};
  uint16_t lag_distance_count_{0U};
  uint16_t motion_energy_slot_{0U};
  uint16_t motion_energy_count_{0U};
  uint16_t frequency_curve_slot_{0U};
  uint16_t frequency_curve_count_{0U};
  double frequency_curve_sum_{0.0};
  double frequency_curve_square_sum_{0.0};
};


}  // namespace espectre
