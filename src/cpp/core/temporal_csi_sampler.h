/*
 * ESPectre - Temporal CSI Sampler
 *
 * Fixed-grid CSI admission shared by production runtime and host-side C++
 * validation. Mirrors tools/lib/temporal_csi_sampler.py.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <memory>

namespace espectre {

constexpr uint32_t TEMPORAL_CSI_MICROSECONDS_PER_SECOND = 1000000U;
/**
 * A window is ready when at least this fraction of its slots holds a sample:
 * `TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR / TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR`,
 * rounded up.
 */
constexpr uint8_t TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR = 7U;
/** See `TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR`. */
constexpr uint8_t TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR = 10U;
/** Selected candidates are at least one slot period divided by this apart. */
constexpr uint8_t TEMPORAL_CSI_SLOT_HALF_DENOMINATOR = 2U;

static_assert(TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR > 0U);
static_assert(TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR <=
              TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR);

/** Return the fixed-grid slot count for a target rate and window duration. */
uint32_t temporal_window_slots(uint32_t target_pps, uint32_t window_size_ms);
/** Return the minimum occupied slots required for a ready window. */
uint32_t temporal_minimum_valid_slots(uint32_t window_slots);
/** Return the minimum spacing between selected candidates at a target rate. */
uint32_t temporal_minimum_sample_spacing_us(uint32_t target_pps);

/**
 * Admit timestamped CSI packets onto the production fixed-time grid.
 *
 * The sampler retains at most one candidate per target-rate slot, preserves
 * missing slots, rejects invalid timestamp progress, and reports when a gap
 * requires detector history to be cleared. A core-only integration should
 * apply its admission result before forwarding CSI to a detector.
 *
 * The sampler stores timing and slot state, not CSI payloads. The caller keeps
 * the currently selected payload. When admit() commits a prior slot, consume
 * that retained payload before replacing it when selected_current() is true.
 *
 * Not thread-safe. Construct, configure, admit, and read it from the task that
 * owns the custom capture pipeline. Construction and configure() use
 * non-throwing storage allocation; check is_valid() after construction and
 * the configure() result before consuming packets. The sampler is movable and
 * intentionally non-copyable because the slot window owns live temporal state.
 */
class TemporalCsiSampler {
 public:
  /** Construct a sampler for the requested target rate and window duration. */
  explicit TemporalCsiSampler(uint32_t target_pps = 100U,
                              uint32_t window_size_ms = 1000U);
  TemporalCsiSampler(TemporalCsiSampler&&) noexcept = default;
  TemporalCsiSampler& operator=(TemporalCsiSampler&&) noexcept = default;
  TemporalCsiSampler(const TemporalCsiSampler&) = delete;
  TemporalCsiSampler& operator=(const TemporalCsiSampler&) = delete;

  /** Reconfigure the grid and clear its timestamp epoch and window state. */
  bool configure(uint32_t target_pps, uint32_t window_size_ms);
  /** Return whether the active slot window owns complete storage. */
  bool is_valid() const { return slot_occupied_ != nullptr && window_slots_ > 0U; }
  /** Clear the timestamp epoch, window state, and lifetime counters. */
  void reset();
  /** Clear the window and timestamp grid while retaining lifetime counters. */
  void clear_history();
  /** Clear admitted window data while retaining the active timestamp grid. */
  void clear_window_preserving_phase();

  /**
   * Observe one candidate and report whether the retained payload was committed.
   *
   * `now_us` is optional processing time on the same unsigned 32-bit clock as
   * `timestamp_us`. Omit it when the clocks differ, including classic ESP32
   * Wi-Fi RX timestamps versus `esp_timer`.
   */
  bool admit(uint32_t timestamp_us, bool has_timestamp = true,
             uint32_t now_us = 0U, bool has_now = false);
  /** Commit the retained payload when the input stream ends. */
  bool flush();

  /** @name Grid configuration */
  /** @{ */
  uint32_t target_pps() const { return target_pps_; }
  uint32_t window_size_ms() const { return window_size_ms_; }
  /** Slots in one window: `target_pps` times the window duration, rounded up. */
  uint32_t window_slots() const { return window_slots_; }
  /** Occupied slots a window needs to be ready. */
  uint32_t minimum_valid_slots() const { return minimum_valid_slots_; }
  /** Smallest time between two committed candidates, in microseconds. */
  uint32_t minimum_sample_spacing_us() const {
    return minimum_sample_spacing_us_;
  }
  /** @} */

  /** @name Window state */
  /** @{ */
  /** Slots in the current window that hold a committed sample. */
  uint32_t occupancy_slots() const { return occupancy_slots_; }
  /** occupancy_slots() divided by window_slots(). */
  float occupancy_ratio() const;
  /** Whether the window spans all its slots and meets the occupancy floor. */
  bool is_ready() const;
  /** @} */

  /**
   * @name Result of the latest admit() or flush()
   * Each call overwrites these values.
   * @{
   */
  /** Whether the call committed the retained payload; same as the return value. */
  bool accepted() const { return accepted_; }
  /** Whether the current packet became the retained candidate; store its payload. */
  bool selected_current() const { return selected_current_; }
  /** Whether a candidate is retained and not yet committed. */
  bool has_pending_candidate() const { return has_pending_candidate_; }
  /** Whether the committed payload is the first after a gap; clear detector history before it. */
  bool reset_required() const { return reset_required_; }
  /** Whether this packet followed a gap of at least one window; clear detector history. */
  bool gap_reset_required() const { return gap_reset_required_; }
  /** Grid slot of the latest committed payload, counted from the timestamp epoch. */
  uint64_t current_slot() const { return last_admitted_slot_; }
  /** Slots between the previous committed payload and this one. */
  uint64_t slots_advanced() const { return slots_advanced_; }
  /** Empty slots before the committed payload; pass to `BaseDetector::advance_missing_slots()`. */
  uint64_t missing_slots_before() const { return missing_slots_before_; }
  /** @} */

  /**
   * @name Lifetime counters
   * Cleared by reset(), kept by clear_history().
   * @{
   */
  /** Payloads committed to the grid. */
  uint64_t accepted_packets() const { return accepted_packets_; }
  /** Packets not kept because their slot already had a better candidate or they came too soon. */
  uint64_t excess_packets() const { return excess_packets_; }
  /** Packets whose timestamp moved backwards. */
  uint64_t out_of_order_packets() const { return out_of_order_packets_; }
  /** Packets older than one window when processed; needs `now_us`. */
  uint64_t stale_packets() const { return stale_packets_; }
  /** Empty slots skipped between committed payloads. */
  uint64_t missing_slots() const { return missing_slots_; }
  /** @} */

 private:
  static constexpr uint32_t kHalfTimestampRange = 0x80000000U;

  void clear_window_();
  bool drop_();
  bool select_candidate_(uint64_t slot, uint64_t elapsed_us,
                         bool reset_required);
  bool commit_candidate_();

  uint32_t target_pps_{100U};
  uint32_t window_size_ms_{1000U};
  uint32_t window_size_us_{1000000U};
  uint32_t window_slots_{100U};
  uint32_t minimum_valid_slots_{70U};
  uint32_t minimum_sample_spacing_us_{5000U};
  std::unique_ptr<uint8_t[]> slot_occupied_;
  uint32_t occupancy_slots_{0U};

  bool has_last_timestamp_{false};
  uint32_t last_timestamp_{0U};
  uint64_t elapsed_us_{0U};
  bool has_last_admitted_slot_{false};
  uint64_t last_admitted_slot_{0U};
  uint64_t last_admitted_elapsed_us_{0U};
  bool has_window_origin_{false};
  uint64_t window_origin_slot_{0U};

  bool has_active_slot_{false};
  uint64_t active_slot_{0U};
  bool has_pending_candidate_{false};
  uint64_t pending_slot_{0U};
  uint64_t pending_elapsed_us_{0U};
  uint64_t pending_center_error_{0U};
  bool pending_reset_required_{false};

  bool accepted_{false};
  bool selected_current_{false};
  bool reset_required_{false};
  bool gap_reset_required_{false};
  uint64_t slots_advanced_{0U};
  uint64_t missing_slots_before_{0U};

  uint64_t accepted_packets_{0U};
  uint64_t excess_packets_{0U};
  uint64_t out_of_order_packets_{0U};
  uint64_t stale_packets_{0U};
  uint64_t missing_slots_{0U};
};

}  // namespace espectre
