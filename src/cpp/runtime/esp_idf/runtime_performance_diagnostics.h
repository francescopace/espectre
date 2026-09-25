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

#include <array>
#include <cstddef>
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

/**
 * Attributes one runtime loop iteration to named steps. When the iteration,
 * or the interval since the previous one, reaches SLOW_THRESHOLD_US, finish()
 * logs one warning that names the slow steps. Log-sink time and listener time
 * are separate, listener time does not include the sink, and the summary
 * includes the wait before the iteration.
 */
class RuntimeLoopStepTimer {
 public:
  static constexpr uint32_t SLOW_THRESHOLD_US = 100000U;

  /** Start an iteration and close the interval since the previous one. */
  void begin();
  /** Attribute the time since the previous mark to `step`, a string literal. */
  void mark(const char *step);
  /** Close the iteration and log it with `tag` when it was slow. */
  void finish(const char *tag);
  /** Forget the previous iteration, so a restart is not reported as a gap. */
  void reset();

  /** Account time a frontend listener callback held the calling thread. */
  static void record_listener_time(uint32_t duration_us);

 private:
  struct Segment {
    const char *name{nullptr};
    uint32_t duration_us{0U};
    uint32_t sink_us{0U};
    uint32_t listener_us{0U};
  };

  static constexpr size_t MAX_STEPS = 12U;

  Segment close_segment_(const char *name, uint64_t now_us);

  bool active_{false};
  bool has_previous_{false};
  uint64_t iteration_start_us_{0U};
  uint64_t segment_start_us_{0U};
  uint64_t previous_end_us_{0U};
  Segment gap_{};
  std::array<Segment, MAX_STEPS> steps_{};
  size_t step_count_{0U};
  uint32_t sink_writes_{0U};
  uint32_t sink_maximum_us_{0U};
};

}  // namespace espectre
