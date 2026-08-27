/*
 * ESPectre - Temporal CSI Sampler
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "temporal_csi_sampler.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace espectre {

uint32_t temporal_window_slots(uint32_t target_pps,
                               uint32_t window_size_ms) {
  if (target_pps == 0U || window_size_ms == 0U) return 0U;
  const uint64_t product = static_cast<uint64_t>(target_pps) * window_size_ms;
  const uint64_t slots = (product + 999U) / 1000U;
  return static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(1U, slots), std::numeric_limits<uint32_t>::max()));
}

uint32_t temporal_minimum_valid_slots(uint32_t window_slots) {
  const uint64_t slots = std::max<uint32_t>(1U, window_slots);
  return static_cast<uint32_t>(
      (slots * TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR +
       TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR - 1U) /
      TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR);
}

uint32_t temporal_minimum_sample_spacing_us(uint32_t target_pps) {
  if (target_pps == 0U) return 0U;
  const uint64_t numerator = TEMPORAL_CSI_MICROSECONDS_PER_SECOND;
  const uint64_t denominator = static_cast<uint64_t>(target_pps) *
                               TEMPORAL_CSI_SLOT_HALF_DENOMINATOR;
  return static_cast<uint32_t>(
      std::max<uint64_t>(1U, (numerator + denominator - 1U) / denominator));
}

TemporalCsiSampler::TemporalCsiSampler(uint32_t target_pps,
                                       uint32_t window_size_ms) {
  configure(target_pps, window_size_ms);
}

bool TemporalCsiSampler::configure(uint32_t target_pps,
                                   uint32_t window_size_ms) {
  const uint32_t slots = temporal_window_slots(target_pps, window_size_ms);
  const uint64_t window_us = static_cast<uint64_t>(window_size_ms) * 1000U;
  if (slots == 0U || window_us == 0U ||
      window_us > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  std::unique_ptr<uint64_t[]> slot_ids(
      new (std::nothrow) uint64_t[static_cast<size_t>(slots)]);
  if (slot_ids == nullptr) {
    return false;
  }
  std::fill(slot_ids.get(), slot_ids.get() + slots, kEmptySlot);
  target_pps_ = target_pps;
  window_size_ms_ = window_size_ms;
  window_size_us_ = static_cast<uint32_t>(window_us);
  window_slots_ = slots;
  minimum_valid_slots_ = temporal_minimum_valid_slots(slots);
  minimum_sample_spacing_us_ = temporal_minimum_sample_spacing_us(target_pps);
  slot_ids_ = std::move(slot_ids);
  reset();
  return true;
}

void TemporalCsiSampler::clear_window_() {
  if (slot_ids_ != nullptr) {
    std::fill(slot_ids_.get(), slot_ids_.get() + window_slots_, kEmptySlot);
  }
  occupancy_slots_ = 0U;
}

void TemporalCsiSampler::reset() {
  clear_history();
  accepted_packets_ = 0U;
  excess_packets_ = 0U;
  duplicate_packets_ = 0U;
  out_of_order_packets_ = 0U;
  stale_packets_ = 0U;
  missing_timestamp_packets_ = 0U;
  missing_slots_ = 0U;
  gap_resets_ = 0U;
}

void TemporalCsiSampler::clear_history() {
  clear_window_();
  has_last_timestamp_ = false;
  last_timestamp_ = 0U;
  elapsed_us_ = 0U;
  has_last_admitted_slot_ = false;
  last_admitted_slot_ = 0U;
  last_admitted_elapsed_us_ = 0U;
  has_window_origin_ = false;
  window_origin_slot_ = 0U;
  has_active_slot_ = false;
  active_slot_ = 0U;
  has_pending_candidate_ = false;
  pending_slot_ = 0U;
  pending_elapsed_us_ = 0U;
  pending_center_error_ = 0U;
  pending_reset_required_ = false;
  accepted_ = false;
  selected_current_ = false;
  reset_required_ = false;
  gap_reset_required_ = false;
  slots_advanced_ = 0U;
  missing_slots_before_ = 0U;
}

void TemporalCsiSampler::clear_window_preserving_phase() {
  clear_window_();
  has_window_origin_ = has_active_slot_;
  window_origin_slot_ = active_slot_;
  has_pending_candidate_ = false;
  pending_slot_ = 0U;
  pending_elapsed_us_ = 0U;
  pending_center_error_ = 0U;
  pending_reset_required_ = false;
  accepted_ = false;
  selected_current_ = false;
  reset_required_ = false;
  gap_reset_required_ = false;
  slots_advanced_ = 0U;
  missing_slots_before_ = 0U;
}

bool TemporalCsiSampler::drop_() {
  accepted_ = false;
  selected_current_ = false;
  reset_required_ = false;
  gap_reset_required_ = false;
  slots_advanced_ = 0U;
  missing_slots_before_ = 0U;
  return false;
}

bool TemporalCsiSampler::select_candidate_(uint64_t slot, uint64_t elapsed_us,
                                           bool reset_required) {
  const uint64_t scaled_elapsed = elapsed_us * target_pps_;
  const uint64_t scaled_center = slot * TEMPORAL_CSI_MICROSECONDS_PER_SECOND;
  const uint64_t center_error = scaled_elapsed >= scaled_center
      ? scaled_elapsed - scaled_center
      : scaled_center - scaled_elapsed;

  if (has_last_admitted_slot_ && slot <= last_admitted_slot_) {
    ++excess_packets_;
    return false;
  }
  if (has_last_admitted_slot_ &&
      elapsed_us - last_admitted_elapsed_us_ < minimum_sample_spacing_us_) {
    ++excess_packets_;
    return false;
  }
  if (has_pending_candidate_ && center_error >= pending_center_error_) {
    ++excess_packets_;
    return false;
  }
  if (has_pending_candidate_) {
    ++excess_packets_;
  }
  has_pending_candidate_ = true;
  pending_slot_ = slot;
  pending_elapsed_us_ = elapsed_us;
  pending_center_error_ = center_error;
  pending_reset_required_ = reset_required;
  selected_current_ = true;
  return true;
}

bool TemporalCsiSampler::commit_candidate_() {
  if (!has_pending_candidate_) return false;
  const uint64_t slot = pending_slot_;
  const uint64_t advanced = has_last_admitted_slot_
      ? slot - last_admitted_slot_
      : 0U;
  const uint64_t missing_before = has_last_admitted_slot_ && advanced > 0U
      ? advanced - 1U
      : 0U;

  if (has_last_admitted_slot_) {
    if (advanced >= window_slots_) {
      clear_window_();
    } else {
      for (uint64_t expired = last_admitted_slot_ + 1U;
           expired <= slot; ++expired) {
        const size_t index = static_cast<size_t>(expired % window_slots_);
        if (slot_ids_[index] != kEmptySlot) {
          slot_ids_[index] = kEmptySlot;
          --occupancy_slots_;
        }
      }
    }
  }

  const size_t index = static_cast<size_t>(slot % window_slots_);
  if (slot_ids_[index] == kEmptySlot) ++occupancy_slots_;
  slot_ids_[index] = slot;
  has_last_admitted_slot_ = true;
  last_admitted_slot_ = slot;
  last_admitted_elapsed_us_ = pending_elapsed_us_;
  accepted_ = true;
  reset_required_ = pending_reset_required_;
  slots_advanced_ = advanced;
  missing_slots_before_ = missing_before;
  ++accepted_packets_;
  missing_slots_ += missing_before;
  has_pending_candidate_ = false;
  pending_reset_required_ = false;
  return true;
}

bool TemporalCsiSampler::admit(uint32_t timestamp_us, bool has_timestamp,
                               uint32_t now_us, bool has_now) {
  drop_();
  if (!has_timestamp) {
    ++missing_timestamp_packets_;
    return false;
  }

  if (has_now) {
    const uint32_t age = now_us - timestamp_us;
    if (age >= window_size_us_ && age < kHalfTimestampRange) {
      ++stale_packets_;
      return false;
    }
  }

  if (!has_last_timestamp_) {
    has_last_timestamp_ = true;
    last_timestamp_ = timestamp_us;
    elapsed_us_ = 0U;
    has_active_slot_ = true;
    active_slot_ = 0U;
    has_window_origin_ = true;
    window_origin_slot_ = 0U;
    select_candidate_(0U, 0U, false);
    return false;
  }

  const uint32_t delta = timestamp_us - last_timestamp_;
  if (delta == 0U) {
    ++duplicate_packets_;
    return false;
  }
  if (delta >= kHalfTimestampRange) {
    ++out_of_order_packets_;
    return false;
  }

  last_timestamp_ = timestamp_us;
  if (delta >= window_size_us_) {
    ++gap_resets_;
    const bool emitted = commit_candidate_();
    clear_window_();
    elapsed_us_ = 0U;
    has_last_admitted_slot_ = false;
    last_admitted_elapsed_us_ = 0U;
    has_window_origin_ = true;
    window_origin_slot_ = 0U;
    has_active_slot_ = true;
    active_slot_ = 0U;
    select_candidate_(0U, 0U, true);
    gap_reset_required_ = true;
    return emitted;
  }

  elapsed_us_ += delta;
  // Center bins on their ideal sampling instant. Flooring makes ordinary
  // +/- scheduling jitter pathological: 0, 9, 20, 29 ms at 100 pps maps to
  // 0, 0, 2, 2 instead of four independent samples.
  const uint64_t slot =
      (elapsed_us_ * target_pps_ +
       TEMPORAL_CSI_MICROSECONDS_PER_SECOND / 2U) /
      TEMPORAL_CSI_MICROSECONDS_PER_SECOND;
  if (has_active_slot_ && slot < active_slot_) {
    ++excess_packets_;
    return false;
  }
  bool emitted = false;
  if (!has_active_slot_ || slot > active_slot_) {
    emitted = commit_candidate_();
    has_active_slot_ = true;
    active_slot_ = slot;
  }
  select_candidate_(slot, elapsed_us_, false);
  return emitted;
}

bool TemporalCsiSampler::flush() {
  drop_();
  return commit_candidate_();
}

float TemporalCsiSampler::occupancy_ratio() const {
  return window_slots_ == 0U
             ? 0.0f
             : static_cast<float>(occupancy_slots_) /
                   static_cast<float>(window_slots_);
}

bool TemporalCsiSampler::is_ready() const {
  return has_window_origin_ && has_last_admitted_slot_ &&
         last_admitted_slot_ >= window_origin_slot_ &&
         last_admitted_slot_ - window_origin_slot_ + 1U >= window_slots_ &&
         occupancy_slots_ >= minimum_valid_slots_;
}

}  // namespace espectre
