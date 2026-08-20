/*
 * ESPectre - Device Identity
 *
 * Derives stable runtime device identifiers from platform identity data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

namespace espectre {

/** Return the cached 64-bit SHA-256 pseudonym derived from the station MAC. */
uint64_t derive_runtime_device_id();
/** Return the cached canonical text for `derive_runtime_device_id()`. */
std::string derive_runtime_device_id_string();

}  // namespace espectre
