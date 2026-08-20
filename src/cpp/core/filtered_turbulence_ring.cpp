/*
 * ESPectre - Filtered Turbulence Ring
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "filtered_turbulence_ring.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace espectre {

FilteredTurbulenceRing::FilteredTurbulenceRing() {
  hampel_turbulence_init(&hampel_state_, HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                         HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, false);
  lowpass_filter_init(&lowpass_state_, LOWPASS_CUTOFF_DEFAULT, LOWPASS_SAMPLE_RATE, false);
}

void FilteredTurbulenceRing::bind(float *storage, uint16_t capacity) {
  storage_ = storage;
  capacity_ = storage != nullptr ? capacity : 0U;
  clear();
}

void FilteredTurbulenceRing::clear() {
  if (storage_ != nullptr) {
    std::fill(storage_, storage_ + capacity_, 0.0f);
  }
  index_ = 0U;
  count_ = 0U;
  valid_count_ = 0U;
  hampel_turbulence_init(&hampel_state_, hampel_state_.window_size,
                         hampel_state_.threshold, hampel_state_.enabled);
  lowpass_filter_reset(&lowpass_state_);
}

void FilteredTurbulenceRing::configure_hampel(bool enabled, uint8_t window_size, float threshold) {
  hampel_turbulence_init(&hampel_state_, window_size, threshold, enabled);
}

void FilteredTurbulenceRing::configure_lowpass(bool enabled, float cutoff_hz) {
  lowpass_filter_init(&lowpass_state_, cutoff_hz, LOWPASS_SAMPLE_RATE, enabled);
}

void FilteredTurbulenceRing::add(float turbulence) {
  if (storage_ == nullptr || capacity_ == 0U) {
    return;
  }
  const float hampel = hampel_filter_turbulence(&hampel_state_, turbulence);
  const float filtered = lowpass_filter_apply(&lowpass_state_, hampel);
  if (!std::isfinite(storage_[index_]) || count_ < capacity_) {
    ++valid_count_;
  }
  storage_[index_] = filtered;
  index_ = static_cast<uint16_t>(index_ + 1U == capacity_ ? 0U : index_ + 1U);
  if (count_ < capacity_) {
    ++count_;
  }
}

void FilteredTurbulenceRing::advance_missing_slots(uint32_t count) {
  if (storage_ == nullptr || capacity_ == 0U) {
    return;
  }
  const float missing = std::numeric_limits<float>::quiet_NaN();
  if (count >= capacity_) {
    std::fill(storage_, storage_ + capacity_, missing);
    const uint16_t remainder = static_cast<uint16_t>(count % capacity_);
    index_ = static_cast<uint16_t>((index_ + remainder) % capacity_);
    count_ = capacity_;
    valid_count_ = 0U;
    return;
  }
  for (uint32_t slot = 0U; slot < count; ++slot) {
    if (count_ >= capacity_ && std::isfinite(storage_[index_])) {
      --valid_count_;
    }
    storage_[index_] = missing;
    index_ = static_cast<uint16_t>(index_ + 1U == capacity_ ? 0U : index_ + 1U);
    if (count_ < capacity_) {
      ++count_;
    }
  }
}

const float *FilteredTurbulenceRing::ordered_view(float *scratch,
                                                  uint16_t scratch_capacity,
                                                  uint16_t &count) const {
  count = count_;
  if (storage_ == nullptr || count_ == 0U) {
    return nullptr;
  }
  if (count_ < capacity_) {
    return storage_;
  }
  if (scratch == nullptr || scratch_capacity < capacity_) {
    count = 0U;
    return nullptr;
  }
  const uint16_t tail = static_cast<uint16_t>(capacity_ - index_);
  std::copy(storage_ + index_, storage_ + capacity_, scratch);
  std::copy(storage_, storage_ + index_, scratch + tail);
  return scratch;
}

}  // namespace espectre
