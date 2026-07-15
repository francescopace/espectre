/*
 * ESPectre - Runtime Detector Store
 *
 * Persists the selected runtime detector algorithm across reboots.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "esp_err.h"
#include "runtime_sensing_schema.h"

namespace espectre {

esp_err_t load_runtime_detection_algorithm(DetectionAlgorithm *algorithm, bool *has_saved_value);
esp_err_t save_runtime_detection_algorithm(DetectionAlgorithm algorithm);

}  // namespace espectre
