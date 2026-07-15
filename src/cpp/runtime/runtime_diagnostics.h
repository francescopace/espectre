/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics snapshot helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <functional>

#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace espectre {

using runtime_diagnostic_visitor_t = std::function<void(const char *key, const char *value)>;

void visit_runtime_diagnostics(const RuntimeConfig &config,
                               const RuntimeSnapshot &snapshot,
                               runtime_diagnostic_visitor_t visitor);

}  // namespace espectre
