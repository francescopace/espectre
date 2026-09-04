/*
 * ESPectre - Station Socket Helpers
 *
 * Shared ESP-IDF helpers for binding sockets to the Wi-Fi station interface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

bool bind_socket_to_sta_interface(int sock, const char *log_tag, const char *purpose);

}  // namespace espectre
