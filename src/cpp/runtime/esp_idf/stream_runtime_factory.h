/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#pragma once

#include <memory>

#include "runtime_interface.h"

namespace espectre {

std::unique_ptr<IEspectreRuntime> make_stream_runtime(const RuntimeConfig &config);

}  // namespace espectre
