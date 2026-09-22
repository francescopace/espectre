// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include <cstdint>

namespace espectre {

/** Default Wi-Fi station packet counters, each wrapping modulo 2^32. */
struct NetworkTrafficSnapshot {
  /** Packets accepted by the station driver, excluding failed sends. */
  uint32_t tx_packets;
  /** Packets delivered by the station driver, including later stack drops. */
  uint32_t rx_packets;
};

/**
 * Read cumulative station counters and refresh the tracked WIFI_STA_DEF handle.
 *
 * Call once after creating the station to start tracking and establish a rate
 * baseline, then on each diagnostic interval. Counters persist across sensing
 * restarts and wrap modulo 2^32. Reads and packet updates are thread-safe;
 * the two counters are sampled independently. No payload is inspected.
 *
 * Compile ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES and link with
 * ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS when consuming the source groups
 * directly. The SDK component already supplies these link options.
 */
NetworkTrafficSnapshot read_network_traffic();

}  // namespace espectre
