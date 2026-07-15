/*
 * ESPectre - Runtime Debug Telemetry
 *
 * Aggregates loop and detection timing metrics for periodic debug logs.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

class RuntimeDebugTelemetry {
 public:
  void reset();
  void record_loop_duration(uint32_t duration_us);
  void record_detection_timing(uint64_t duration_sum_us,
                               uint32_t samples,
                               uint32_t minimum_us,
                               uint32_t maximum_us);
  void log_if_due(const char *tag);

 private:
  static constexpr uint64_t LOG_INTERVAL_US = 10000000ULL;

  uint64_t window_start_us_{0U};
  uint64_t loop_busy_us_{0U};
  uint64_t loop_duration_sum_us_{0U};
  uint32_t loop_duration_max_us_{0U};
  uint32_t loop_samples_{0U};
  uint64_t detection_duration_sum_us_{0U};
  uint32_t detection_duration_min_us_{0U};
  uint32_t detection_duration_max_us_{0U};
  uint32_t detection_samples_{0U};
};

class RuntimeDebugLoopScope {
 public:
  RuntimeDebugLoopScope(RuntimeDebugTelemetry &telemetry, const char *tag);
  ~RuntimeDebugLoopScope();

  RuntimeDebugLoopScope(const RuntimeDebugLoopScope &) = delete;
  RuntimeDebugLoopScope &operator=(const RuntimeDebugLoopScope &) = delete;

 private:
  RuntimeDebugTelemetry &telemetry_;
  const char *tag_;
  int64_t start_us_;
};

}  // namespace espectre
