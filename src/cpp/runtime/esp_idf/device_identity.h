#pragma once

#include <cstdint>
#include <string>

namespace espectre {

uint64_t derive_runtime_device_id();
std::string derive_runtime_device_id_string();

}  // namespace espectre
