// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include <esp_timer.h>

namespace espectre {

/** Record the complete frontend loop body on scope exit, including early returns. */
class FrontendLoopTimer {
 public:
  explicit FrontendLoopTimer(float &latest_ms)
      : latest_ms_(latest_ms), started_us_(esp_timer_get_time()) {}

  ~FrontendLoopTimer() {
    latest_ms_ = static_cast<float>(esp_timer_get_time() - started_us_) / 1000.0f;
  }

  FrontendLoopTimer(const FrontendLoopTimer &) = delete;
  FrontendLoopTimer &operator=(const FrontendLoopTimer &) = delete;

 private:
  float &latest_ms_;
  int64_t started_us_;
};

}  // namespace espectre
