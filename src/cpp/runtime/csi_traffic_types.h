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
   * The runtime generates its own traffic at `traffic_generator_rate`.
   *
   * Default, and the only self-sufficient mode. A rate of zero degrades it to
   * `DISABLED`.
   */
  INTERNAL,
  /** Another device supplies the traffic; the runtime only listens. */
  EXTERNAL,
  /** External traffic, with the runtime pacing the sender to hold the rate. */
  PACING,
  /**
   * No traffic management at all.
   *
   * Only sensible when ambient traffic already sustains the packet rate the
   * detector needs; otherwise the detector starves.
   */
  DISABLED,
};

}  // namespace espectre
