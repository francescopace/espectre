/*
 * ESPectre - Firmware Primary Console
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <esp_err.h>

namespace espectre {

// Accommodate a complete pending Improv response without splitting its bytes
// around unrelated log writes. USB output is still bounded and non-blocking.
constexpr size_t PRIMARY_CONSOLE_TX_BUFFER_SIZE = 1024U;

// Initialize the configured application console transport. UART and USB
// Serial/JTAG remain owned by ESP-IDF and require no work here.
esp_err_t initialize_primary_console();

// Route ESP-IDF logs straight to the installed USB Serial/JTAG driver. Full
// buffers drop whole lines instead of waiting for a reader, and lines longer
// than about 500 bytes are truncated. Call it only where the application owns the
// console; ESPHome keeps its own logger hook.
void use_nonblocking_usb_log_output();

}  // namespace espectre
