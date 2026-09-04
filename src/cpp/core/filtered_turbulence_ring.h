/*
 * ESPectre - Filtered Turbulence Ring
 *
 * Shared filtered ring used by detector feature streams.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "filters.h"

namespace espectre {

/** @brief Fixed-capacity turbulence history with optional Hampel and low-pass filtering. */
class FilteredTurbulenceRing {
 public:
  /** @brief Construct an unbound history with default filter parameters. */
  FilteredTurbulenceRing();

  /** @brief Attach caller-owned storage and reset the history. */
  void bind(float *storage, uint16_t capacity);
  /** @brief Clear samples while preserving the configured filters. */
  void clear();
  /** @brief Configure and reset the Hampel filter. */
  void configure_hampel(bool enabled, uint8_t window_size, float threshold);
  /** @brief Configure and reset the low-pass filter. */
  void configure_lowpass(bool enabled, float cutoff_hz);
  /** @brief Filter and append one turbulence sample. */
  void add(float turbulence);
  /** @brief Append missing-sample markers without running the filters. */
  void advance_missing_slots(uint32_t count);

  /**
   * @brief Return the samples in chronological order.
   * @param scratch Caller-owned storage used only after the ring wraps.
   * @param scratch_capacity Number of elements available in @p scratch.
   * @param count Receives the returned element count, or zero when scratch is too small.
   * @return Internal storage before wrap, @p scratch after wrap, or nullptr when empty or invalid.
   */
  const float *ordered_view(float *scratch, uint16_t scratch_capacity, uint16_t &count) const;
  /** @brief Return the number of occupied slots, including missing markers. */
  uint16_t count() const { return count_; }
  /** @brief Return the number of finite samples in the history. */
  uint16_t valid_count() const { return valid_count_; }
  /** @brief Return the bound storage capacity. */
  uint16_t capacity() const { return capacity_; }
  /** @brief Return whether Hampel filtering is enabled. */
  bool hampel_enabled() const { return hampel_state_.enabled; }

 private:
  float *storage_{nullptr};
  uint16_t capacity_{0U};
  uint16_t index_{0U};
  uint16_t count_{0U};
  uint16_t valid_count_{0U};
  hampel_filter_state_t hampel_state_{};
  lowpass_filter_state_t lowpass_state_{};
};

}  // namespace espectre
