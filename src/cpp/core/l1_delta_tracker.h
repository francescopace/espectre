/*
 * ESPectre - L1 Delta Tracker
 *
 * Tracks L1 amplitude deltas across CSI subcarriers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "csi_format.h"
#include "detector_limits.h"
#include "csi_features.h"
#include "filters.h"
#include "utils.h"

namespace espectre {

/**
 * One sliding window of displacements over a caller-owned ring.
 *
 * The tracker keeps two of these, one per lag, and they share a single
 * allocation, so the pair costs no extra bookkeeping and no second heap block.
 */
struct L1DeltaWindow {
  float* ring{nullptr};
  uint16_t index{0U};
  uint16_t slots{0U};
  uint16_t count{0U};
  float sum{0.0f};

  void push(float value, uint16_t capacity) {
    if (capacity == 0U || ring == nullptr) {
      return;
    }
    if (slots >= capacity && std::isfinite(ring[index])) {
      sum -= ring[index];
      --count;
    }
    ring[index] = value;
    if (std::isfinite(value)) {
      sum += value;
      ++count;
    }
    index++;
    if (index >= capacity) {
      index = 0U;
    }
    if (slots < capacity) ++slots;
  }

  float mean() const { return count > 0U ? sum / static_cast<float>(count) : 0.0f; }

  void clear(uint16_t capacity) {
    index = 0U;
    slots = 0U;
    count = 0U;
    sum = 0.0f;
    if (ring != nullptr && capacity > 0U) {
      std::memset(ring, 0, capacity * sizeof(float));
    }
  }
};

class L1DeltaTracker {
 public:
  L1DeltaTracker() = default;
  ~L1DeltaTracker() { delete[] storage_; }
  L1DeltaTracker(L1DeltaTracker&& other) noexcept
      : capacity_(other.capacity_),
        lag_(other.lag_),
        profile_index_(other.profile_index_),
        storage_(other.storage_),
        lagged_(other.lagged_),
        adjacent_(other.adjacent_),
        hampel_state_(other.hampel_state_),
        hampel_adjacent_(other.hampel_adjacent_) {
    std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
    std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
    other.capacity_ = 0U;
    other.profile_index_ = 0U;
    other.storage_ = nullptr;
    other.lagged_ = L1DeltaWindow{};
    other.adjacent_ = L1DeltaWindow{};
  }
  L1DeltaTracker& operator=(L1DeltaTracker&& other) noexcept {
    if (this != &other) {
      delete[] storage_;
      capacity_ = other.capacity_;
      std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
      std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
      lag_ = other.lag_;
      profile_index_ = other.profile_index_;
      storage_ = other.storage_;
      lagged_ = other.lagged_;
      adjacent_ = other.adjacent_;
      hampel_state_ = other.hampel_state_;
      hampel_adjacent_ = other.hampel_adjacent_;
      other.capacity_ = 0U;
      other.profile_index_ = 0U;
      other.storage_ = nullptr;
      other.lagged_ = L1DeltaWindow{};
      other.adjacent_ = L1DeltaWindow{};
    }
    return *this;
  }
  L1DeltaTracker(const L1DeltaTracker&) = delete;
  L1DeltaTracker& operator=(const L1DeltaTracker&) = delete;

  /**
   * @param capacity Delta ring capacity in packets
   * @param lag Profile-displacement distance in packets, bounded by
   *        L1_DELTA_LAG_MAX because the profile ring is statically sized
   */
  void configure(uint16_t capacity, uint16_t lag = L1_DELTA_LAG) {
    allocate_delta_ring_(std::min<uint16_t>(capacity, DETECTOR_MAX_WINDOW_SIZE));
    lag_ = std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX);
    clear();
  }

  void configure_hampel(bool enabled,
                        uint8_t window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                        float threshold = HAMPEL_TURBULENCE_THRESHOLD_DEFAULT) {
    hampel_turbulence_init(&hampel_state_, window_size, threshold, enabled);
    // The ratio divides one displacement by the other, so both must be filtered
    // alike: an outlier surviving only in the denominator would depress the
    // ratio and read as less motion.
    hampel_turbulence_init(&hampel_adjacent_, window_size, threshold, enabled);
  }

  void clear() {
    std::memset(profile_ring_, 0, sizeof(profile_ring_));
    std::memset(profile_len_, 0, sizeof(profile_len_));
    lagged_.clear(capacity_);
    adjacent_.clear(capacity_);
    profile_index_ = 0U;
    if (hampel_state_.window_size >= HAMPEL_TURBULENCE_WINDOW_MIN) {
      hampel_turbulence_init(&hampel_adjacent_, hampel_state_.window_size,
                             hampel_state_.threshold, hampel_state_.enabled);
      hampel_turbulence_init(&hampel_state_, hampel_state_.window_size,
                             hampel_state_.threshold, hampel_state_.enabled);
    }
  }

  void process(const float *amplitudes, uint8_t amplitude_count) {
    const float mean =
        amplitudes != nullptr && amplitude_count >= 2U
            ? calculate_mean(amplitudes, amplitude_count)
            : 0.0f;
    process(amplitudes, amplitude_count, mean);
  }

  void process(const float *amplitudes, uint8_t amplitude_count,
               float amplitude_mean) {
    if (capacity_ == 0U) {
      return;
    }

    float profile[HT20_SELECTED_BAND_SIZE]{};
    const float *reference = profile_ring_[profile_index_];
    const uint8_t reference_len = profile_len_[profile_index_];
    // The packet before this one sits in the slot behind the lagged reference,
    // so the adjacent displacement needs no storage of its own.
    const uint16_t previous_index =
        profile_index_ > 0U ? static_cast<uint16_t>(profile_index_ - 1U)
                            : static_cast<uint16_t>(lag_ - 1U);
    const float *previous = profile_ring_[previous_index];
    const uint8_t previous_len = profile_len_[previous_index];
    uint8_t profile_len = 0U;
    float lagged_value = std::numeric_limits<float>::quiet_NaN();
    float adjacent_value = std::numeric_limits<float>::quiet_NaN();

    if (amplitudes != nullptr && amplitude_count >= 2U &&
        amplitude_count <= HT20_SELECTED_BAND_SIZE) {
      profile_len = normalize_amplitude_profile(
          amplitudes, amplitude_count, amplitude_mean, profile);
      const bool lagged = profile_len > 0U && reference_len == profile_len;
      const bool adjacent = profile_len > 0U && previous_len == profile_len;
      if (lagged || adjacent) {
        // One pass over the normalized profile feeds both displacements.
        float lagged_sum = 0.0f;
        float adjacent_sum = 0.0f;
        for (uint8_t i = 0U; i < profile_len; i++) {
          if (lagged) {
            lagged_sum += std::fabs(profile[i] - reference[i]);
          }
          if (adjacent) {
            adjacent_sum += std::fabs(profile[i] - previous[i]);
          }
        }
        if (lagged) {
          lagged_value = hampel_filter_turbulence(
              &hampel_state_, lagged_sum / profile_len);
        }
        if (adjacent) {
          adjacent_value = hampel_filter_turbulence(
              &hampel_adjacent_, adjacent_sum / profile_len);
        }
      }
    }

    std::memcpy(profile_ring_[profile_index_], profile, profile_len * sizeof(float));
    profile_len_[profile_index_] = profile_len;
    profile_index_++;
    if (profile_index_ >= lag_) {
      profile_index_ = 0U;
    }
    lagged_.push(lagged_value, capacity_);
    adjacent_.push(adjacent_value, capacity_);
  }

  void advance_missing_slots(uint32_t count) {
    const float missing = std::numeric_limits<float>::quiet_NaN();
    for (uint32_t slot = 0U; slot < count; ++slot) {
      profile_len_[profile_index_] = 0U;
      profile_index_++;
      if (profile_index_ >= lag_) profile_index_ = 0U;
      lagged_.push(missing, capacity_);
      adjacent_.push(missing, capacity_);
    }
  }

  uint16_t count() const { return lagged_.count; }
  float mean() const { return lagged_.mean(); }

  /**
   * Mean lagged displacement over mean adjacent displacement.
   *
   * Noise saturates the displacement immediately, so its ratio sits near 1.0;
   * real channel evolution keeps growing with the lag and lifts it. Both terms
   * share the same units, so the ratio drops the noise floor that makes the raw
   * mean unusable when the link is weak.
   */
  float delta_lag_ratio() const {
    const float adjacent_mean = adjacent_.mean();
    if (lagged_.count == 0U || adjacent_.count == 0U || adjacent_mean <= 0.0f) {
      return 1.0f;
    }
    return lagged_.mean() / adjacent_mean;
  }

  uint16_t build_series(float *out) const {
    if (out == nullptr || capacity_ == 0U || lagged_.count == 0U) {
      return 0U;
    }
    const uint16_t slots = lagged_.slots;
    uint16_t source = slots < capacity_ ? 0U : lagged_.index;
    uint16_t written = 0U;
    for (uint16_t offset = 0U; offset < slots; ++offset) {
      const float value = lagged_.ring[source];
      if (std::isfinite(value)) out[written++] = value;
      source++;
      if (source >= capacity_) source = 0U;
    }
    return written;
  }

 private:
  void allocate_delta_ring_(uint16_t capacity) {
    if (capacity == capacity_ && (capacity == 0U || storage_ != nullptr)) {
      return;
    }
    delete[] storage_;
    storage_ = nullptr;
    capacity_ = 0U;
    lagged_ = L1DeltaWindow{};
    adjacent_ = L1DeltaWindow{};
    if (capacity == 0U) {
      return;
    }
    // One block, two views: the lagged window first, the adjacent one behind it.
    float* block = new (std::nothrow) float[2U * static_cast<size_t>(capacity)];
    if (block == nullptr) {
      return;
    }
    storage_ = block;
    capacity_ = capacity;
    lagged_.ring = block;
    adjacent_.ring = block + capacity;
  }

  uint16_t capacity_{0U};
  float profile_ring_[L1_DELTA_LAG_MAX][HT20_SELECTED_BAND_SIZE]{};
  uint8_t profile_len_[L1_DELTA_LAG_MAX]{};
  uint16_t lag_{L1_DELTA_LAG};
  uint16_t profile_index_{0U};
  float* storage_{nullptr};
  L1DeltaWindow lagged_{};
  L1DeltaWindow adjacent_{};
  hampel_filter_state_t hampel_state_{};
  hampel_filter_state_t hampel_adjacent_{};
};

}  // namespace espectre
