/*
 * ESPectre - Runtime Performance Diagnostics
 *
 * Aggregates runtime loop and detector timing into bounded diagnostic windows.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_performance_diagnostics.h"

#include <algorithm>
#include <limits>

#include "esp_timer.h"

namespace espectre {

namespace {

uint32_t elapsed_us_since(uint64_t start_us, uint64_t end_us) {
  if (end_us <= start_us) {
    return 0U;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(end_us - start_us,
                                                  std::numeric_limits<uint32_t>::max()));
}

}  // namespace

void RuntimePerformanceDiagnostics::reset() {
  window_start_us_ = 0U;
  loop_busy_us_ = 0U;
  loop_duration_sum_us_ = 0U;
  loop_duration_max_us_ = 0U;
  loop_samples_ = 0U;
  detection_duration_sum_us_ = 0U;
  detection_duration_min_us_ = 0U;
  detection_duration_max_us_ = 0U;
  detection_samples_ = 0U;
  latest_ = {};
}

void RuntimePerformanceDiagnostics::record_loop_duration(uint32_t duration_us) {
  loop_busy_us_ += duration_us;
  loop_duration_sum_us_ += duration_us;
  loop_duration_max_us_ = std::max(loop_duration_max_us_, duration_us);
  loop_samples_ += 1U;
}

void RuntimePerformanceDiagnostics::record_detection_timing(uint64_t duration_sum_us,
                                                             uint32_t samples,
                                                             uint32_t minimum_us,
                                                             uint32_t maximum_us) {
  if (samples == 0U) {
    return;
  }
  detection_duration_sum_us_ += duration_sum_us;
  detection_duration_min_us_ =
      detection_samples_ == 0U ? minimum_us : std::min(detection_duration_min_us_, minimum_us);
  detection_duration_max_us_ = std::max(detection_duration_max_us_, maximum_us);
  detection_samples_ += samples;
}

void RuntimePerformanceDiagnostics::update_if_due() {
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (window_start_us_ == 0U) {
    window_start_us_ = now_us;
    return;
  }
  if (now_us <= window_start_us_ || now_us - window_start_us_ < WINDOW_INTERVAL_US) {
    return;
  }

  const uint64_t elapsed_us = now_us - window_start_us_;
  latest_.window_ready = true;
  latest_.window_duration_us = static_cast<uint32_t>(
      std::min<uint64_t>(elapsed_us, std::numeric_limits<uint32_t>::max()));
  latest_.runtime_load_percent = static_cast<float>(
      std::min(100.0, static_cast<double>(loop_busy_us_) * 100.0 / static_cast<double>(elapsed_us)));
  latest_.loop_samples = loop_samples_;
  latest_.loop_average_us =
      loop_samples_ > 0U ? static_cast<uint32_t>(loop_duration_sum_us_ / loop_samples_) : 0U;
  latest_.loop_maximum_us = loop_duration_max_us_;
  latest_.detection_samples = detection_samples_;
  latest_.detection_sum_us = detection_duration_sum_us_;
  latest_.detection_average_us = detection_samples_ > 0U
                                    ? static_cast<uint32_t>(detection_duration_sum_us_ / detection_samples_)
                                    : 0U;
  latest_.detection_minimum_us = detection_duration_min_us_;
  latest_.detection_maximum_us = detection_duration_max_us_;

  window_start_us_ = now_us;
  loop_busy_us_ = 0U;
  loop_duration_sum_us_ = 0U;
  loop_duration_max_us_ = 0U;
  loop_samples_ = 0U;
  detection_duration_sum_us_ = 0U;
  detection_duration_min_us_ = 0U;
  detection_duration_max_us_ = 0U;
  detection_samples_ = 0U;
}

RuntimePerformanceLoopScope::RuntimePerformanceLoopScope(RuntimePerformanceDiagnostics &diagnostics)
    : diagnostics_(diagnostics), start_us_(esp_timer_get_time()) {}

RuntimePerformanceLoopScope::~RuntimePerformanceLoopScope() {
  const int64_t end_us = esp_timer_get_time();
  diagnostics_.record_loop_duration(
      elapsed_us_since(static_cast<uint64_t>(start_us_), static_cast<uint64_t>(end_us)));
  diagnostics_.update_if_due();
}

}  // namespace espectre
