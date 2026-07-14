#include "runtime_debug_telemetry.h"

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "espectre_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace espectre {

namespace {

uint32_t cpu_frequency_mhz() {
#if defined(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)
  return CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#elif defined(CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ)
  return CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ;
#else
  return 0U;
#endif
}

uint32_t elapsed_us_since(uint64_t start_us, uint64_t end_us) {
  if (end_us <= start_us) {
    return 0U;
  }
  return static_cast<uint32_t>(std::min<uint64_t>(end_us - start_us,
                                                  std::numeric_limits<uint32_t>::max()));
}

}  // namespace

void RuntimeDebugTelemetry::reset() {
  window_start_us_ = 0U;
  loop_busy_us_ = 0U;
  loop_duration_sum_us_ = 0U;
  loop_duration_max_us_ = 0U;
  loop_samples_ = 0U;
  detection_duration_sum_us_ = 0U;
  detection_duration_min_us_ = 0U;
  detection_duration_max_us_ = 0U;
  detection_samples_ = 0U;
}

void RuntimeDebugTelemetry::record_loop_duration(uint32_t duration_us) {
  loop_busy_us_ += duration_us;
  loop_duration_sum_us_ += duration_us;
  loop_duration_max_us_ = std::max(loop_duration_max_us_, duration_us);
  loop_samples_++;
}

void RuntimeDebugTelemetry::record_detection_timing(uint64_t duration_sum_us,
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

void RuntimeDebugTelemetry::log_if_due(const char *tag) {
  if (tag == nullptr) {
    return;
  }

  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (window_start_us_ == 0U) {
    window_start_us_ = now_us;
    return;
  }
  if (now_us <= window_start_us_ || now_us - window_start_us_ < LOG_INTERVAL_US) {
    return;
  }

  const uint64_t elapsed_us = now_us - window_start_us_;
  const double runtime_load_percent =
      std::min(100.0, static_cast<double>(loop_busy_us_) * 100.0 / static_cast<double>(elapsed_us));
  const uint32_t loop_average_us =
      loop_samples_ > 0U ? static_cast<uint32_t>(loop_duration_sum_us_ / loop_samples_) : 0U;
  const uint32_t detection_average_us = detection_samples_ > 0U
                                            ? static_cast<uint32_t>(detection_duration_sum_us_ / detection_samples_)
                                            : 0U;

  unsigned long heap_free = 0UL;
  unsigned long heap_minimum = 0UL;
  unsigned long heap_largest = 0UL;
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  heap_free = static_cast<unsigned long>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT));
  heap_minimum = static_cast<unsigned long>(heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT));
  heap_largest = static_cast<unsigned long>(heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));
#endif

  ESP_LOGD(tag,
           "[telemetry] heap_free=%lu heap_min=%lu heap_largest=%lu cpu_mhz=%" PRIu32
           " runtime_load=%.2f%% loop_avg_us=%" PRIu32 " loop_max_us=%" PRIu32
           " detection_samples=%" PRIu32 " detection_sum_us=%" PRIu64
           " detection_avg_us=%" PRIu32
           " detection_min_us=%" PRIu32 " detection_max_us=%" PRIu32,
           heap_free,
           heap_minimum,
           heap_largest,
           cpu_frequency_mhz(),
           runtime_load_percent,
           loop_average_us,
           loop_duration_max_us_,
           detection_samples_,
           detection_duration_sum_us_,
           detection_average_us,
           detection_duration_min_us_,
           detection_duration_max_us_);

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

RuntimeDebugLoopScope::RuntimeDebugLoopScope(RuntimeDebugTelemetry &telemetry, const char *tag)
    : telemetry_(telemetry), tag_(tag), start_us_(esp_timer_get_time()) {}

RuntimeDebugLoopScope::~RuntimeDebugLoopScope() {
  const int64_t end_us = esp_timer_get_time();
  telemetry_.record_loop_duration(
      elapsed_us_since(static_cast<uint64_t>(start_us_), static_cast<uint64_t>(end_us)));
  telemetry_.log_if_due(tag_);
}

}  // namespace espectre
