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

  bool observe(uint64_t accepted_csi_total, int64_t now_us);

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
  static constexpr uint32_t MAX_REDUCTION_PERCENT = 30U;

  uint32_t target_pps_{0U};
  uint32_t current_pps_{0U};
  uint32_t observed_pps_{0U};
  uint64_t previous_accepted_csi_total_{0U};
  int64_t previous_observation_us_{0};
  bool adaptive_enabled_{true};
};

}  // namespace espectre
