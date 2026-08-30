/*
 * ESPectre - ESPHome Log Sink
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "espectre_log.h"

namespace esphome {
namespace espectre_component {

::espectre::LogSink make_esphome_log_sink();

}  // namespace espectre_component
}  // namespace esphome
