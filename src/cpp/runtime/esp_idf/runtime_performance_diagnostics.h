/*
 * ESPectre - Runtime Performance Diagnostics
 *
 * Aggregates runtime loop and detector timing into bounded diagnostic windows.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

struct RuntimePerformanceDiagnosticsSnapshot {
  bool window_ready{false};
  uint32_t window_duration_us{0U};
  float runtime_load_percent{0.0f};
  uint32_t loop_samples{0U};
  uint32_t loop_average_us{0U};
  uint32_t loop_maximum_us{0U};
  uint32_t detection_samples{0U};
  uint64_t detection_sum_us{0U};
  uint32_t detection_average_us{0U};
  uint32_t detection_minimum_us{0U};
  uint32_t detection_maximum_us{0U};
};

class RuntimePerformanceDiagnostics {
 public:
  void reset();
  void record_loop_duration(uint32_t duration_us);
  void record_detection_timing(uint64_t duration_sum_us,
                               uint32_t samples,
                               uint32_t minimum_us,
                               uint32_t maximum_us);
  void update_if_due();
  RuntimePerformanceDiagnosticsSnapshot snapshot() const { return latest_; }

 private:
  static constexpr uint64_t WINDOW_INTERVAL_US = 10000000ULL;

  uint64_t window_start_us_{0U};
  uint64_t loop_busy_us_{0U};
  uint64_t loop_duration_sum_us_{0U};
  uint32_t loop_duration_max_us_{0U};
  uint32_t loop_samples_{0U};
  uint64_t detection_duration_sum_us_{0U};
  uint32_t detection_duration_min_us_{0U};
  uint32_t detection_duration_max_us_{0U};
  uint32_t detection_samples_{0U};
  RuntimePerformanceDiagnosticsSnapshot latest_{};
};

class RuntimePerformanceLoopScope {
 public:
  explicit RuntimePerformanceLoopScope(RuntimePerformanceDiagnostics &diagnostics);
  ~RuntimePerformanceLoopScope();

  RuntimePerformanceLoopScope(const RuntimePerformanceLoopScope &) = delete;
  RuntimePerformanceLoopScope &operator=(const RuntimePerformanceLoopScope &) = delete;

 private:
  RuntimePerformanceDiagnostics &diagnostics_;
  int64_t start_us_;
};

}  // namespace espectre
