/*
 * ESPectre - Traffic Rate Controller
 *
 * Adapts CSI traffic-generator rate from observed sensing load.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "traffic_rate_controller.h"

#include <algorithm>

namespace espectre {

void TrafficRateController::init(uint32_t target_pps, bool adaptive_enabled) {
  target_pps_ = target_pps;
  current_pps_ = target_pps;
  observed_pps_ = 0U;
  previous_accepted_csi_total_ = 0U;
  previous_send_success_total_ = 0U;
  previous_send_error_total_ = 0U;
  previous_observation_us_ = 0;
  last_adjustment_us_ = 0;
  oversupply_windows_ = 0U;
  adaptive_enabled_ = adaptive_enabled;
}

bool TrafficRateController::observe(uint64_t accepted_csi_total,
                                    uint64_t send_success_total,
                                    uint64_t send_error_total,
                                    int64_t now_us) {
  if (previous_observation_us_ == 0 || accepted_csi_total < previous_accepted_csi_total_ ||
      send_success_total < previous_send_success_total_ || send_error_total < previous_send_error_total_) {
    previous_accepted_csi_total_ = accepted_csi_total;
    previous_send_success_total_ = send_success_total;
    previous_send_error_total_ = send_error_total;
    previous_observation_us_ = now_us;
    return false;
  }

  const int64_t elapsed_us = now_us - previous_observation_us_;
  if (elapsed_us < CONTROL_WINDOW_US) {
    return false;
  }

  const uint64_t accepted_delta = accepted_csi_total - previous_accepted_csi_total_;
  const uint64_t send_success_delta = send_success_total - previous_send_success_total_;
  const uint64_t send_error_delta = send_error_total - previous_send_error_total_;
  observed_pps_ = static_cast<uint32_t>((accepted_delta * 1000000ULL) / static_cast<uint64_t>(elapsed_us));
  previous_accepted_csi_total_ = accepted_csi_total;
  previous_send_success_total_ = send_success_total;
  previous_send_error_total_ = send_error_total;
  previous_observation_us_ = now_us;

  if (!adaptive_enabled_ || target_pps_ == 0U) {
    return false;
  }

  const uint32_t lower_bound = (target_pps_ * (100U - TOLERANCE_PERCENT)) / 100U;
  const uint32_t upper_bound = (target_pps_ * (100U + TOLERANCE_PERCENT) + 99U) / 100U;
  const uint32_t minimum_rate = std::max(
      std::min(MIN_RATE_PPS, target_pps_),
      (target_pps_ * PACING_FLOOR_PERCENT + 99U) / 100U);
  const uint32_t maximum_rate = std::min(
      MAX_RATE_PPS,
      std::max(minimum_rate,
               (target_pps_ * MAX_RATE_NUMERATOR + MAX_RATE_DENOMINATOR - 1U) / MAX_RATE_DENOMINATOR));
  uint32_t next_rate = current_pps_;
  const uint64_t send_attempt_delta = send_success_delta + send_error_delta;
  const uint64_t backpressure_threshold = std::max<uint64_t>(
      MIN_BACKPRESSURE_EVENTS,
      (send_attempt_delta * BACKPRESSURE_PERCENT + 99U) / 100U);
  const bool significant_backpressure = send_error_delta >= backpressure_threshold;
  const bool settling = last_adjustment_us_ != 0 && now_us - last_adjustment_us_ < ADJUSTMENT_SETTLE_US;
  const uint32_t additive_step = std::max<uint32_t>(1U, (target_pps_ * 2U + 99U) / 100U);

  if (significant_backpressure) {
    oversupply_windows_ = 0U;
    if (!settling) {
      next_rate = std::max(minimum_rate, (current_pps_ * (100U - REDUCTION_PERCENT)) / 100U);
    }
  } else if (observed_pps_ * 100U < target_pps_ * SEVERE_DEFICIT_PERCENT) {
    oversupply_windows_ = 0U;
  } else if (observed_pps_ > upper_bound) {
    if (oversupply_windows_ < OVERSUPPLY_WINDOWS_BEFORE_REDUCTION) {
      oversupply_windows_++;
    }
    if (!settling && oversupply_windows_ >= OVERSUPPLY_WINDOWS_BEFORE_REDUCTION) {
      next_rate = std::max(minimum_rate, (current_pps_ * (100U - REDUCTION_PERCENT)) / 100U);
      oversupply_windows_ = 0U;
    }
  } else if (observed_pps_ < lower_bound) {
    oversupply_windows_ = 0U;
    if (!settling) {
      next_rate = std::min(maximum_rate, current_pps_ + additive_step);
    }
  } else {
    oversupply_windows_ = 0U;
    if (!settling && current_pps_ < target_pps_) {
      next_rate = std::min(target_pps_, current_pps_ + additive_step);
    } else if (!settling && current_pps_ > target_pps_) {
      next_rate = std::max(target_pps_, current_pps_ - additive_step);
    }
  }

  if (next_rate == current_pps_) {
    return false;
  }
  current_pps_ = next_rate;
  last_adjustment_us_ = now_us;
  return true;
}

}  // namespace espectre
