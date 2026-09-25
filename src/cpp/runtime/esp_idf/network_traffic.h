// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include <cstdint>

#include "esp_err.h"

namespace espectre {

/** Default Wi-Fi station packet counters, each wrapping modulo 2^32. */
struct NetworkTrafficSnapshot {
  /** Packets accepted by the station driver, excluding failed sends. */
  uint32_t tx_packets;
  /** Packets delivered by the station driver, including later stack drops. */
  uint32_t rx_packets;
};

/**
 * Read cumulative counters for the WIFI_STA_DEF station interface.
 *
 * Call once to establish a rate baseline, then on each diagnostic interval.
 * The station is recognized when the WIFI_STA_DEF netif is created or
 * recreated, so a read never takes the lwIP lock. The packet hooks only
 * compare that pointer, which keeps the per-packet cost minimal. Counters persist across sensing restarts
 * and wrap modulo 2^32. Reads are lock-free and packet updates are
 * thread-safe; the two counters are sampled independently. No payload is
 * inspected.
 *
 * Compile ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES and link with
 * ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS when consuming the source groups
 * directly. The SDK component already supplies these link options.
 */
NetworkTrafficSnapshot read_network_traffic();

/**
 * Apply the station transmit-rate policy to the associated access point.
 *
 * `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` selects Auto, OFDM 6 Mbps, or HT20 MCS0
 * with long GI; Auto and TX A-MPDU builds leave driver rate selection
 * unchanged. The full runtime applies it itself. Integrations that own the
 * Wi-Fi station call it after every association, including reassociation,
 * and before starting CSI. Requires ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES.
 *
 * @return `ESP_OK`, or the driver error that prevented applying the policy.
 */
esp_err_t apply_station_tx_rate();

}  // namespace espectre
