/*
 * ESPectre - Mock ESP-IDF Random Source
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace esp_random_mock {
inline uint32_t value = 0U;
}

inline uint32_t esp_random() { return esp_random_mock::value; }
