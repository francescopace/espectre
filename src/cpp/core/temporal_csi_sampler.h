/*
 * ESPectre - Temporal CSI Sampler
 *
 * Fixed-grid CSI admission shared by production runtime and host-side C++
 * validation. Mirrors src/python/micro_espectre/temporal_csi_sampler.py.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace espectre {

constexpr uint32_t TEMPORAL_CSI_MICROSECONDS_PER_SECOND = 1000000U;
constexpr uint8_t TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR = 4U;
constexpr uint8_t TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR = 5U;

uint32_t temporal_window_slots(uint32_t target_pps, uint32_t window_size_ms);
uint32_t temporal_minimum_valid_slots(uint32_t window_slots);
uint32_t temporal_minimum_sample_spacing_us(uint32_t target_pps);

class TemporalCsiSampler {
 public:
  explicit TemporalCsiSampler(uint32_t target_pps = 100U,
                              uint32_t window_size_ms = 1000U);

  bool configure(uint32_t target_pps, uint32_t window_size_ms);
  void reset();
  void clear_history();

  bool admit(uint32_t timestamp_us, bool has_timestamp = true,
             uint32_t now_us = 0U, bool has_now = false);

  uint32_t target_pps() const { return target_pps_; }
  uint32_t window_size_ms() const { return window_size_ms_; }
  uint32_t window_slots() const { return window_slots_; }
  uint32_t minimum_valid_slots() const { return minimum_valid_slots_; }
  uint32_t minimum_sample_spacing_us() const {
    return minimum_sample_spacing_us_;
  }
  uint32_t occupancy_slots() const { return occupancy_slots_; }
  float occupancy_ratio() const;
  bool is_ready() const;

  bool accepted() const { return accepted_; }
  bool reset_required() const { return reset_required_; }
  uint64_t current_slot() const { return last_admitted_slot_; }
  uint64_t slots_advanced() const { return slots_advanced_; }
  uint64_t missing_slots_before() const { return missing_slots_before_; }

  uint64_t accepted_packets() const { return accepted_packets_; }
  uint64_t excess_packets() const { return excess_packets_; }
  uint64_t duplicate_packets() const { return duplicate_packets_; }
  uint64_t out_of_order_packets() const { return out_of_order_packets_; }
  uint64_t stale_packets() const { return stale_packets_; }
  uint64_t missing_timestamp_packets() const { return missing_timestamp_packets_; }
  uint64_t missing_slots() const { return missing_slots_; }
  uint64_t gap_resets() const { return gap_resets_; }

 private:
  static constexpr uint64_t kEmptySlot = UINT64_MAX;
  static constexpr uint32_t kHalfTimestampRange = 0x80000000U;

  void clear_window_();
  bool drop_();
  bool accept_slot_(uint64_t slot, uint64_t advanced,
                    uint64_t missing_before);

  uint32_t target_pps_{100U};
  uint32_t window_size_ms_{1000U};
  uint32_t window_size_us_{1000000U};
  uint32_t window_slots_{100U};
  uint32_t minimum_valid_slots_{80U};
  uint32_t minimum_sample_spacing_us_{8000U};
  std::vector<uint64_t> slot_ids_;
  uint32_t occupancy_slots_{0U};

  bool has_last_timestamp_{false};
  uint32_t last_timestamp_{0U};
  uint64_t elapsed_us_{0U};
  bool has_last_admitted_slot_{false};
  uint64_t last_admitted_slot_{0U};
  uint64_t last_admitted_elapsed_us_{0U};

  bool accepted_{false};
  bool reset_required_{false};
  uint64_t slots_advanced_{0U};
  uint64_t missing_slots_before_{0U};

  uint64_t accepted_packets_{0U};
  uint64_t excess_packets_{0U};
  uint64_t duplicate_packets_{0U};
  uint64_t out_of_order_packets_{0U};
  uint64_t stale_packets_{0U};
  uint64_t missing_timestamp_packets_{0U};
  uint64_t missing_slots_{0U};
  uint64_t gap_resets_{0U};
};

}  // namespace espectre
