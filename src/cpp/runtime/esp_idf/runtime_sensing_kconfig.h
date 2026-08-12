/*
 * ESPectre - Runtime Sensing Kconfig
 *
 * Builds the default sensing runtime configuration from Kconfig values.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "runtime_interface.h"

namespace espectre {

/**
 * Build a `RuntimeConfig` from the `ESPECTRE_*` menuconfig options.
 *
 * This is the ergonomic path on ESP-IDF: expose the sensing settings in
 * menuconfig, call this at boot, and override only the fields your product
 * computes at runtime.
 *
 * @code
 * espectre::RuntimeConfig config = espectre::make_runtime_sensing_config_from_kconfig();
 * config.device_id = my_device_id();
 * controller.set_config(config);
 * @endcode
 *
 * Every value is range-checked against the schema in
 * `runtime_sensing_schema.h`. An out-of-range or unparseable option falls back
 * to the documented default and logs a warning rather than failing the boot,
 * so a bad `sdkconfig` degrades instead of bricking the device.
 *
 * Options absent from the build, for instance when the SDK Kconfig is not
 * sourced, compile to their defaults. Fields with no Kconfig option, such as
 * `device_id` and the stream settings, keep their `RuntimeConfig` defaults.
 *
 * @return A validated configuration, ready for
 *         `RuntimeFrontendController::set_config()`.
 */
RuntimeConfig make_runtime_sensing_config_from_kconfig();

}  // namespace espectre
