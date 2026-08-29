/*
 * ESPectre - CSI Traffic Types
 *
 * Platform-agnostic CSI traffic mode shared between the runtime interface
 * and the ESP-IDF traffic service implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

/**
 * Where the CSI-bearing traffic comes from.
 *
 * CSI is only produced when packets actually arrive, so something has to keep
 * the link busy. This picks who does it.
 */
enum class CsiTrafficMode {
  /**
   * The runtime generates its own traffic at `csi_target_pps`.
   * Default, and the only self-sufficient mode.
   */
  INTERNAL,
  /**
   * Another device supplies exact UDP markers or unicast ICMP Echo Requests;
   * the runtime does not start its internal generator.
   */
  EXTERNAL,
};

}  // namespace espectre
