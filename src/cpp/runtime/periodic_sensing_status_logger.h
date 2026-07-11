#pragma once

#include <cstdint>

#include "runtime_snapshot.h"

namespace espectre {

class PeriodicSensingStatusLogger {
 public:
  void log_status(const char *tag, const RuntimeSnapshot &snapshot, uint32_t packets_per_publish);
  void reset() { last_log_time_ms_ = 0; }

 private:
  uint32_t last_log_time_ms_{0};
};

}  // namespace espectre
