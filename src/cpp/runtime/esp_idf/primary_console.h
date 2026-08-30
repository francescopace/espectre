/*
 * ESPectre - ESP-IDF Primary Console
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <esp_err.h>

namespace espectre {

// Initialize the configured application console transport. UART and USB
// Serial/JTAG remain owned by ESP-IDF and require no work here.
esp_err_t initialize_primary_console();

}  // namespace espectre
