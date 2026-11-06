/*
 * ESPectre - Emulator CSI Replay
 *
 * Delivers recorded CSI from UART1 to the firmware's CSI callback when the
 * firmware runs in esp-emu, which models Wi-Fi but not CSI.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

/**
 * True when this firmware runs in esp-emu with the replay marker.
 *
 * Firmware links this unit with `--wrap` for esp_wifi_set_csi_config,
 * esp_wifi_set_csi_rx_cb, and esp_wifi_set_csi. On silicon the wrappers pass
 * every call to ESP-IDF and never touch UART1. When the factory MAC carries
 * `kEmulatorReplayMacPrefix`, they keep the CSI callback and feed it frames
 * read from UART1 instead. Each frame keeps its recorded RX timestamp and
 * arrives as an external host's ping to the station, which the `external`
 * traffic filter keeps.
 */
bool emulator_csi_replay_active();

}  // namespace espectre
