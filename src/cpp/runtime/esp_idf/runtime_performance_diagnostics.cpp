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
#include <cstdarg>
#include <cstdio>
#include <limits>

#include "core/espectre_log.h"
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

uint32_t clamp_us(uint64_t duration_us) {
  return static_cast<uint32_t>(std::min<uint64_t>(duration_us, std::numeric_limits<uint32_t>::max()));
}

unsigned rounded_ms(uint32_t duration_us) {
  return static_cast<unsigned>((static_cast<uint64_t>(duration_us) + 500U) / 1000U);
}

void append_text(char *buffer, size_t capacity, size_t *length, const char *format, ...) {
  if (*length >= capacity) {
    return;
  }
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(buffer + *length, capacity - *length, format, args);
  va_end(args);
  if (written > 0) {
    *length = std::min(capacity, *length + static_cast<size_t>(written));
  }
}

// Frontend listener time recorded on this thread and not yet attributed.
thread_local uint64_t thread_listener_us = 0U;

// Steps shorter than this carry no evidence about a stall.
constexpr uint32_t REPORTED_STEP_MINIMUM_US = 1000U;

}  // namespace

void RuntimePerformanceDiagnostics::reset() {
  window_start_us_ = 0U;
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
      std::min(100.0, static_cast<double>(loop_duration_sum_us_) * 100.0 / static_cast<double>(elapsed_us)));
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

void RuntimeLoopStepTimer::record_listener_time(uint32_t duration_us) {
  thread_listener_us += duration_us;
}

RuntimeLoopStepTimer::Segment RuntimeLoopStepTimer::close_segment_(const char *name, uint64_t now_us) {
  const detail::LogSinkTiming sink = detail::take_log_sink_timing();
  Segment segment;
  segment.name = name;
  segment.duration_us = elapsed_us_since(segment_start_us_, now_us);
  segment.sink_us = clamp_us(sink.total_us);
  segment.listener_us = clamp_us(thread_listener_us);
  thread_listener_us = 0U;
  sink_writes_ += sink.writes;
  sink_maximum_us_ = std::max(sink_maximum_us_, sink.maximum_us);
  segment_start_us_ = now_us;
  return segment;
}

void RuntimeLoopStepTimer::begin() {
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  segment_start_us_ = has_previous_ ? previous_end_us_ : now_us;
  // The gap's sink and listener counts belong to this report. Zero before
  // closing it so a slow log between iterations stays in the summary.
  sink_writes_ = 0U;
  sink_maximum_us_ = 0U;
  gap_ = close_segment_("gap", now_us);
  step_count_ = 0U;
  iteration_start_us_ = now_us;
  active_ = true;
}

void RuntimeLoopStepTimer::mark(const char *step) {
  if (!active_) {
    return;
  }
  const Segment segment = close_segment_(step, static_cast<uint64_t>(esp_timer_get_time()));
  if (step_count_ < steps_.size()) {
    steps_[step_count_++] = segment;
  }
}

void RuntimeLoopStepTimer::finish(const char *tag) {
  if (!active_) {
    return;
  }
  active_ = false;
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  const Segment trailing = close_segment_("other", now_us);
  const bool trailing_listed =
      trailing.duration_us >= REPORTED_STEP_MINIMUM_US && step_count_ < steps_.size();
  if (trailing_listed) {
    steps_[step_count_++] = trailing;
  }
  const uint32_t loop_us = elapsed_us_since(iteration_start_us_, now_us);
  has_previous_ = true;
  previous_end_us_ = now_us;
  if (loop_us < SLOW_THRESHOLD_US && gap_.duration_us < SLOW_THRESHOLD_US) {
    return;
  }

  // Every named step with a sink and listener breakdown, the gap, and the summary.
  char text[768];
  size_t length = 0U;
  uint32_t sink_us = gap_.sink_us;
  uint32_t listener_us = gap_.listener_us;
  if (!trailing_listed) {
    sink_us += trailing.sink_us;
    listener_us += trailing.listener_us;
  }
  append_text(text, sizeof(text), &length, "Runtime loop took %u ms, %u ms after the previous one",
              rounded_ms(loop_us), rounded_ms(gap_.duration_us));
  if (gap_.sink_us >= REPORTED_STEP_MINIMUM_US || gap_.listener_us >= REPORTED_STEP_MINIMUM_US) {
    append_text(text, sizeof(text), &length, " (log sink %u ms, listener %u ms)",
                rounded_ms(gap_.sink_us), rounded_ms(gap_.listener_us));
  }
  append_text(text, sizeof(text), &length, ":");
  bool listed = false;
  for (size_t index = 0U; index < step_count_; ++index) {
    const Segment &step = steps_[index];
    sink_us += step.sink_us;
    listener_us += step.listener_us;
    if (step.duration_us < REPORTED_STEP_MINIMUM_US) {
      continue;
    }
    append_text(text, sizeof(text), &length, "%s %s %u ms", listed ? "," : "", step.name,
                rounded_ms(step.duration_us));
    if (step.sink_us >= REPORTED_STEP_MINIMUM_US || step.listener_us >= REPORTED_STEP_MINIMUM_US) {
      append_text(text, sizeof(text), &length, " (log sink %u ms, listener %u ms)",
                  rounded_ms(step.sink_us), rounded_ms(step.listener_us));
    }
    listed = true;
  }
  if (!listed) {
    append_text(text, sizeof(text), &length, " no step over 1 ms");
  }
  append_text(text, sizeof(text), &length,
              "; log sink %u ms in %u write%s (max %u ms), listener %u ms",
              rounded_ms(sink_us), static_cast<unsigned>(sink_writes_),
              sink_writes_ == 1U ? "" : "s", rounded_ms(sink_maximum_us_),
              rounded_ms(listener_us));
  ESPECTRE_LOGW(tag, "%s", text);
  // Exclude this report from the next interval, so a slow sink cannot turn one
  // stall into a chain of reports about the previous report.
  (void) detail::take_log_sink_timing();
  previous_end_us_ = static_cast<uint64_t>(esp_timer_get_time());
}

void RuntimeLoopStepTimer::reset() {
  active_ = false;
  has_previous_ = false;
  step_count_ = 0U;
  thread_listener_us = 0U;
  (void) detail::take_log_sink_timing();
}

}  // namespace espectre
