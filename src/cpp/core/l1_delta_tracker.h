/*
 * ESPectre - L1 Delta Tracker
 *
 * Tracks L1 amplitude deltas across CSI subcarriers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

#include "csi_format.h"
#include "detector_limits.h"
#include "csi_features.h"
#include "filters.h"
#include "utils.h"

namespace espectre {

class L1DeltaTracker {
 public:
  L1DeltaTracker() = default;
  ~L1DeltaTracker() { delete[] delta_ring_; }
  L1DeltaTracker(L1DeltaTracker&& other) noexcept
      : capacity_(other.capacity_),
        lag_(other.lag_),
        profile_index_(other.profile_index_),
        delta_ring_(other.delta_ring_),
        delta_index_(other.delta_index_),
        delta_count_(other.delta_count_),
        delta_sum_(other.delta_sum_),
        hampel_state_(other.hampel_state_) {
    std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
    std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
    other.capacity_ = 0U;
    other.profile_index_ = 0U;
    other.delta_ring_ = nullptr;
    other.delta_index_ = 0U;
    other.delta_count_ = 0U;
    other.delta_sum_ = 0.0f;
  }
  L1DeltaTracker& operator=(L1DeltaTracker&& other) noexcept {
    if (this != &other) {
      delete[] delta_ring_;
      capacity_ = other.capacity_;
      std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
      std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
      lag_ = other.lag_;
      profile_index_ = other.profile_index_;
      delta_ring_ = other.delta_ring_;
      delta_index_ = other.delta_index_;
      delta_count_ = other.delta_count_;
      delta_sum_ = other.delta_sum_;
      hampel_state_ = other.hampel_state_;
      other.capacity_ = 0U;
      other.profile_index_ = 0U;
      other.delta_ring_ = nullptr;
      other.delta_index_ = 0U;
      other.delta_count_ = 0U;
      other.delta_sum_ = 0.0f;
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
  }

  void clear() {
    std::memset(profile_ring_, 0, sizeof(profile_ring_));
    std::memset(profile_len_, 0, sizeof(profile_len_));
    if (delta_ring_ != nullptr && capacity_ > 0U) {
      std::memset(delta_ring_, 0, capacity_ * sizeof(float));
    }
    profile_index_ = 0U;
    delta_index_ = 0U;
    delta_count_ = 0U;
    delta_sum_ = 0.0f;
    if (hampel_state_.window_size >= HAMPEL_TURBULENCE_WINDOW_MIN) {
      hampel_turbulence_init(&hampel_state_, hampel_state_.window_size,
                             hampel_state_.threshold, hampel_state_.enabled);
    }
  }

  void process(const float *amplitudes, uint8_t amplitude_count) {
    if (capacity_ == 0U) {
      return;
    }

    float profile[HT20_SELECTED_BAND_SIZE]{};
    const float *reference = profile_ring_[profile_index_];
    const uint8_t reference_len = profile_len_[profile_index_];
    uint8_t profile_len = 0U;

    if (amplitudes != nullptr && amplitude_count >= 2U &&
        amplitude_count <= HT20_SELECTED_BAND_SIZE) {
      float amplitude_sum = 0.0f;
      for (uint8_t i = 0U; i < amplitude_count; i++) {
        amplitude_sum += amplitudes[i];
      }
      if (amplitude_sum > 0.0f) {
        const float mean = amplitude_sum / amplitude_count;
        profile_len = amplitude_count;
        if (reference_len == profile_len) {
          float delta_sum = 0.0f;
          for (uint8_t i = 0U; i < profile_len; i++) {
            const float value = amplitudes[i] / mean;
            profile[i] = value;
            delta_sum += std::fabs(value - reference[i]);
          }
          const float delta = hampel_filter_turbulence(
              &hampel_state_, delta_sum / profile_len);
          if (delta_count_ >= capacity_) {
            delta_sum_ -= delta_ring_[delta_index_];
          }
          delta_ring_[delta_index_] = delta;
          delta_sum_ += delta;
          delta_index_ = (delta_index_ + 1U) % capacity_;
          if (delta_count_ < capacity_) {
            delta_count_++;
          }
        } else {
          for (uint8_t i = 0U; i < profile_len; i++) {
            profile[i] = amplitudes[i] / mean;
          }
        }
      }
    }

    std::memcpy(profile_ring_[profile_index_], profile, profile_len * sizeof(float));
    profile_len_[profile_index_] = profile_len;
    profile_index_ = static_cast<uint16_t>((profile_index_ + 1U) % lag_);
  }

  uint16_t count() const { return delta_count_; }
  float mean() const { return delta_count_ > 0U ? delta_sum_ / delta_count_ : 0.0f; }

  uint16_t build_series(float *out) const {
    if (out == nullptr || capacity_ == 0U || delta_count_ == 0U) {
      return 0U;
    }
    const uint16_t start = delta_count_ < capacity_ ? 0U : delta_index_;
    for (uint16_t i = 0U; i < delta_count_; i++) {
      out[i] = delta_ring_[(start + i) % capacity_];
    }
    return delta_count_;
  }

 private:
  void allocate_delta_ring_(uint16_t capacity) {
    if (capacity == capacity_ && (capacity == 0U || delta_ring_ != nullptr)) {
      return;
    }
    delete[] delta_ring_;
    delta_ring_ = nullptr;
    capacity_ = 0U;
    if (capacity == 0U) {
      return;
    }
    float* ring = new (std::nothrow) float[capacity];
    if (ring == nullptr) {
      return;
    }
    delta_ring_ = ring;
    capacity_ = capacity;
  }

  uint16_t capacity_{0U};
  float profile_ring_[L1_DELTA_LAG_MAX][HT20_SELECTED_BAND_SIZE]{};
  uint8_t profile_len_[L1_DELTA_LAG_MAX]{};
  uint16_t lag_{L1_DELTA_LAG};
  uint16_t profile_index_{0U};
  float* delta_ring_{nullptr};
  uint16_t delta_index_{0U};
  uint16_t delta_count_{0U};
  float delta_sum_{0.0f};
  hampel_filter_state_t hampel_state_{};
};

}  // namespace espectre
