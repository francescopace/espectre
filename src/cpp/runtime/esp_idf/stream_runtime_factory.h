#pragma once

#include <memory>

#include "runtime_interface.h"

namespace espectre {

std::unique_ptr<IEspectreRuntime> make_stream_runtime(const RuntimeConfig &config);

}  // namespace espectre
