#pragma once

#include <utility>

#include "espectre_log.h"
#include "runtime_frontend_controller.h"

namespace espectre {

template <typename ResetStatusLogger>
inline void finalize_frontend_calibration(RuntimeFrontendController &runtime,
                                          const RuntimeSnapshot &snapshot,
                                          ResetStatusLogger &&reset_status_logger,
                                          bool success,
                                          const char *tag) {
  runtime.record_snapshot(snapshot);
  std::forward<ResetStatusLogger>(reset_status_logger)();
  if (!success) {
    ESP_LOGW(tag, "Calibration finished without a valid update");
  }
}

}  // namespace espectre
