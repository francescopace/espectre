/*
 * ESPectre - Banner
 *
 * ASCII banner helper printed by sensing frontends at startup.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

inline constexpr const char *kEspectreAsciiLogoLines[] = {
    "  _____ ____  ____           __            ",
    " | ____/ ___||  _ \\ ___  ___| |_ _ __ ___ ",
    " |  _| \\___ \\| |_) / _ \\/ __| __| '__/ _ \\",
    " | |___ ___) |  __/  __/ (__| |_| | |  __/",
    " |_____|____/|_|   \\___|\\___|\\__|_|  \\___|",
};

template<typename Logger>
inline void log_espectre_banner(Logger &&log_line) {
  log_line("");
  for (const char *line : kEspectreAsciiLogoLines) {
    log_line(line);
  }
  log_line("");
}

}  // namespace espectre
