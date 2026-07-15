/*
 * ESPectre - Device Identity
 *
 * Derives stable runtime device identifiers from platform identity data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <string>

namespace espectre {

uint64_t derive_runtime_device_id();
std::string derive_runtime_device_id_string();

}  // namespace espectre
