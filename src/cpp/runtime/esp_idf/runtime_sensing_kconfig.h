/*
 * ESPectre - Runtime Sensing Kconfig
 *
 * Builds the default sensing runtime configuration from Kconfig values.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "runtime_interface.h"

namespace espectre {

RuntimeConfig make_runtime_sensing_config_from_kconfig();

}  // namespace espectre
