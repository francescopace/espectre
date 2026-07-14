#include "traffic_rate_controller.h"

#include <algorithm>

namespace espectre {

void TrafficRateController::init(uint32_t target_pps, bool adaptive_enabled) {
  target_pps_ = target_pps;
  current_pps_ = target_pps;
  observed_pps_ = 0U;
  previous_accepted_csi_total_ = 0U;
  previous_observation_us_ = 0;
  adaptive_enabled_ = adaptive_enabled;
}

bool TrafficRateController::observe(uint64_t accepted_csi_total, int64_t now_us) {
  if (previous_observation_us_ == 0 || accepted_csi_total < previous_accepted_csi_total_) {
    previous_accepted_csi_total_ = accepted_csi_total;
    previous_observation_us_ = now_us;
    return false;
  }

  const int64_t elapsed_us = now_us - previous_observation_us_;
  if (elapsed_us < CONTROL_WINDOW_US) {
    return false;
  }

  const uint64_t accepted_delta = accepted_csi_total - previous_accepted_csi_total_;
  observed_pps_ = static_cast<uint32_t>((accepted_delta * 1000000ULL) / static_cast<uint64_t>(elapsed_us));
  previous_accepted_csi_total_ = accepted_csi_total;
  previous_observation_us_ = now_us;

  if (!adaptive_enabled_ || target_pps_ == 0U) {
    return false;
  }

  const uint32_t lower_bound = (target_pps_ * (100U - TOLERANCE_PERCENT)) / 100U;
  const uint32_t upper_bound = (target_pps_ * (100U + TOLERANCE_PERCENT) + 99U) / 100U;
  const uint32_t minimum_rate = std::min(MIN_RATE_PPS, target_pps_);
  const uint32_t maximum_rate = std::min(
      MAX_RATE_PPS,
      std::max(minimum_rate,
               (target_pps_ * MAX_RATE_NUMERATOR + MAX_RATE_DENOMINATOR - 1U) / MAX_RATE_DENOMINATOR));
  uint32_t next_rate = current_pps_;

  if (observed_pps_ > upper_bound) {
    const uint32_t proportional_rate = static_cast<uint32_t>(
        (static_cast<uint64_t>(current_pps_) * target_pps_) / std::max<uint32_t>(observed_pps_, 1U));
    const uint32_t reduction_floor = (current_pps_ * (100U - MAX_REDUCTION_PERCENT)) / 100U;
    next_rate = std::max(minimum_rate, std::max(proportional_rate, reduction_floor));
  } else if (observed_pps_ < lower_bound) {
    const uint32_t additive_step = std::max<uint32_t>(1U, (target_pps_ * 2U + 99U) / 100U);
    next_rate = std::min(maximum_rate, current_pps_ + additive_step);
  }

  if (next_rate == current_pps_) {
    return false;
  }
  current_pps_ = next_rate;
  return true;
}

}  // namespace espectre
