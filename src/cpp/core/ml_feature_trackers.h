/*
 * ESPectre - Production ML Feature Trackers
 *
 * Minimal shared tracker for promoted normalized amplitude-shape dynamics.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "csi_format.h"
#include "detector_limits.h"

namespace espectre {

constexpr uint8_t HT20_LIVE_BAND_SIZE = 56U;
constexpr std::array<uint8_t, HT20_LIVE_BAND_SIZE> HT20_LIVE_BINS = {
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
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
constexpr float CHANNEL_SHAPE_KENDALL_RELATIVE_DEADBAND = 0.02f;
constexpr uint8_t CHANNEL_SHAPE_KENDALL_MIN_COMPARABLE_PAIRS = 8U;
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
      bin.kendall_order = 0U;
      bin.kendall_valid = 0U;
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
    // Duplicate payloads still advance the physical-time window.
    if (has_previous_raw_ &&
        std::memcmp(previous_raw_.data(), csi_data, HT20_CSI_LEN) == 0) {
      return;
    }
    std::memcpy(previous_raw_.data(), csi_data, HT20_CSI_LEN);
    has_previous_raw_ = true;
    if (current_profile_count_ >= CHANNEL_SHAPE_MAX_PROFILES_PER_BIN) {
      return;
    }
    fill_profile_(csi_data, subcarrier_energies, subcarrier_count,
                  current_profiles_[current_profile_count_]);
    current_profile_count_++;
  }

  void trajectory_features(float& coherent_innovation_energy,
                           float& excess_path,
                           float& shape_spread_subband,
                           float& kendall_lag_excess) const {
    coherent_innovation_energy = 0.0f;
    excess_path = 0.0f;
    shape_spread_subband = 0.0f;
    kendall_lag_excess = 0.0f;
    std::array<PathPoint, CHANNEL_SHAPE_WINDOW_BINS + 1U> path{};
    const uint8_t count = build_path_(path);
    if (count < 2U) {
      return;
    }

    Profile spread_energy{};
    for (uint8_t i = 1U; i < count; i++) {
      if (path[i].index - path[i - 1U].index != 1U) continue;
      for (uint8_t subband = 0U;
           subband < CHANNEL_SHAPE_SUBBAND_COUNT; subband++) {
        float delta = 0.0f;
        for (uint8_t mode = 0U;
             mode < CHANNEL_SHAPE_SUBBAND_COUNT; mode++) {
          delta += (path[i].modes[mode] - path[i - 1U].modes[mode]) *
                   CHANNEL_SHAPE_DCT[subband][mode];
        }
        spread_energy[subband] += delta * delta;
      }
    }
    shape_spread_subband = motion_participation(
        spread_energy.data(), CHANNEL_SHAPE_SUBBAND_COUNT);
    if (count < 3U) return;

    std::array<float, CHANNEL_SHAPE_WINDOW_BINS - 1U> innovation_samples{};
    std::array<float, CHANNEL_SHAPE_WINDOW_BINS - 1U> excess_samples{};
    uint8_t innovation_count = 0U;
    uint8_t excess_count = 0U;
    Profile first_modes = path[0].modes;
    Profile middle_modes = path[1].modes;
    float previous_norm = 0.0f;
    float previous_high_norm = 0.0f;
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
        if (i == 2U) first_norm_squared += first_delta * first_delta;
        second_norm_squared += second_delta * second_delta;
        chord_norm_squared += chord_delta * chord_delta;
        if (j >= 4U) {
          if (i == 2U) first_high_squared += first_delta * first_delta;
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
      if (i == 2U) {
        previous_norm = std::sqrt(first_norm_squared);
        previous_high_norm = std::sqrt(first_high_squared);
      }
      const float second_norm = std::sqrt(second_norm_squared);
      const float second_high_norm = std::sqrt(second_high_squared);
      const float raw_excess = previous_norm + second_norm -
                               std::sqrt(chord_norm_squared);
      const float high_excess = previous_high_norm + second_high_norm -
                                std::sqrt(chord_high_squared);
      excess_samples[excess_count++] =
          std::max(0.0f, raw_excess - std::max(0.0f, high_excess));
      previous_norm = second_norm;
      previous_high_norm = second_high_norm;
      first_modes = middle_modes;
      middle_modes = last_modes;
    }
    coherent_innovation_energy =
        median_(innovation_samples.data(), innovation_count);
    excess_path = median_(excess_samples.data(), excess_count);

    std::array<float, CHANNEL_SHAPE_WINDOW_BINS - 2U> kendall_samples{};
    uint8_t kendall_count = 0U;
    // Cache adjacent comparisons lazily, including invalid comparisons. Keep
    // the original summation order in each overlapping four-point window.
    std::array<float, CHANNEL_SHAPE_WINDOW_BINS> adjacent_distances{};
    std::array<uint8_t, CHANNEL_SHAPE_WINDOW_BINS> adjacent_status{};
    for (uint8_t i = 3U; i < count; i++) {
      if (path[i].index - path[i - 3U].index != 3U) continue;
      float long_distance = 0.0f;
      if (!kendall_distance_(path[i], path[i - 3U], long_distance)) continue;
      float local_sum = 0.0f;
      bool local_ok = true;
      for (uint8_t lag = 0U; lag < 3U; lag++) {
        const uint8_t edge = i - lag - 1U;
        if (adjacent_status[edge] == 0U) {
          adjacent_status[edge] = kendall_distance_(
              path[edge + 1U], path[edge], adjacent_distances[edge]) ? 1U : 2U;
        }
        if (adjacent_status[edge] != 1U) {
          local_ok = false;
          break;
        }
        local_sum += adjacent_distances[edge];
      }
      if (!local_ok) continue;
      kendall_samples[kendall_count++] =
          std::max(0.0f, long_distance - local_sum / 3.0f);
    }
    kendall_lag_excess = median_(kendall_samples.data(), kendall_count);
  }

  void trajectory_features(float& coherent_innovation_energy,
                           float& excess_path,
                           float& shape_spread_subband) const {
    float kendall = 0.0f;
    trajectory_features(coherent_innovation_energy, excess_path,
                        shape_spread_subband, kendall);
  }

  void trajectory_features(float& coherent_innovation_energy,
                           float& excess_path) const {
    float spread = 0.0f;
    float kendall = 0.0f;
    trajectory_features(coherent_innovation_energy, excess_path, spread,
                        kendall);
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

  float shape_spread_subband() const {
    float innovation = 0.0f;
    float excess = 0.0f;
    float spread = 0.0f;
    trajectory_features(innovation, excess, spread);
    return spread;
  }

  float subband_kendall_lag_excess() const {
    float innovation = 0.0f;
    float excess = 0.0f;
    float spread = 0.0f;
    float kendall = 0.0f;
    trajectory_features(innovation, excess, spread, kendall);
    return kendall;
  }

 private:
  using Profile = std::array<float, CHANNEL_SHAPE_SUBBAND_COUNT>;
  struct PathPoint {
    uint64_t index{0U};
    Profile modes{};
    uint32_t kendall_order{0U};
    uint32_t kendall_valid{0U};
  };

  static uint8_t popcount32_(uint32_t value) {
    uint8_t count = 0U;
    while (value != 0U) {
      count = static_cast<uint8_t>(count + (value & 1U));
      value >>= 1U;
    }
    return count;
  }

  static void kendall_signature_(const Profile& profile, uint32_t& order,
                                 uint32_t& valid) {
    order = 0U;
    valid = 0U;
    float maximum = 0.0f;
    for (float value : profile) {
      if (value > maximum) maximum = value;
    }
    if (maximum <= 0.0f) return;
    const float threshold =
        CHANNEL_SHAPE_KENDALL_RELATIVE_DEADBAND * maximum;
    uint8_t bit = 0U;
    for (uint8_t left = 0U; left + 1U < CHANNEL_SHAPE_SUBBAND_COUNT; left++) {
      for (uint8_t right = left + 1U; right < CHANNEL_SHAPE_SUBBAND_COUNT;
           right++) {
        const float difference = profile[left] - profile[right];
        const uint32_t flag = 1U << bit;
        if (std::fabs(difference) > threshold) {
          valid |= flag;
          if (difference > 0.0f) order |= flag;
        }
        bit++;
      }
    }
  }

  static bool kendall_distance_(const PathPoint& current,
                                const PathPoint& reference, float& distance) {
    const uint32_t common = current.kendall_valid & reference.kendall_valid;
    const uint8_t comparable = popcount32_(common);
    if (comparable < CHANNEL_SHAPE_KENDALL_MIN_COMPARABLE_PAIRS) {
      distance = 0.0f;
      return false;
    }
    const uint8_t discordant = popcount32_(
        (current.kendall_order ^ reference.kendall_order) & common);
    distance = static_cast<float>(discordant) / static_cast<float>(comparable);
    return true;
  }

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
    const Profile profile = median_current_profile_();
    kendall_signature_(profile, bins_[bin_count_].kendall_order,
                       bins_[bin_count_].kendall_valid);
    bins_[bin_count_].modes = dct_modes_(profile);
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
      const Profile profile = median_current_profile_();
      path[count].index = current_bin_;
      kendall_signature_(profile, path[count].kendall_order,
                         path[count].kendall_valid);
      path[count].modes = dct_modes_(profile);
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

}  // namespace espectre
