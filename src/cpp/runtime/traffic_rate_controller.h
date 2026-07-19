/*
 * ESPectre - Traffic Rate Controller
 *
 * Adapts CSI traffic-generator rate from observed sensing load.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

class TrafficRateController {
 public:
  void init(uint32_t target_pps, bool adaptive_enabled);

  bool observe(uint64_t accepted_csi_total,
               uint64_t send_success_total,
               uint64_t send_error_total,
               int64_t now_us);

  uint32_t target_pps() const { return target_pps_; }
  uint32_t current_pps() const { return current_pps_; }
  uint32_t observed_pps() const { return observed_pps_; }
  bool adaptive_enabled() const { return adaptive_enabled_; }

 private:
  static constexpr int64_t CONTROL_WINDOW_US = 2000000;
  static constexpr uint32_t MIN_RATE_PPS = 5U;
  static constexpr uint32_t MAX_RATE_PPS = 1000U;
  static constexpr uint32_t MAX_RATE_NUMERATOR = 5U;
  static constexpr uint32_t MAX_RATE_DENOMINATOR = 4U;
  static constexpr uint32_t TOLERANCE_PERCENT = 5U;
  static constexpr uint32_t SEVERE_DEFICIT_PERCENT = 50U;
  static constexpr uint32_t PACING_FLOOR_PERCENT = 70U;
  static constexpr uint32_t REDUCTION_PERCENT = 15U;
  static constexpr uint32_t BACKPRESSURE_PERCENT = 5U;
  static constexpr uint32_t MIN_BACKPRESSURE_EVENTS = 3U;
  static constexpr uint8_t OVERSUPPLY_WINDOWS_BEFORE_REDUCTION = 2U;
  static constexpr int64_t ADJUSTMENT_SETTLE_US = CONTROL_WINDOW_US * 3;

  uint32_t target_pps_{0U};
  uint32_t current_pps_{0U};
  uint32_t observed_pps_{0U};
  uint64_t previous_accepted_csi_total_{0U};
  uint64_t previous_send_success_total_{0U};
  uint64_t previous_send_error_total_{0U};
  int64_t previous_observation_us_{0};
  int64_t last_adjustment_us_{0};
  uint8_t oversupply_windows_{0U};
  bool adaptive_enabled_{true};
};

}  // namespace espectre
