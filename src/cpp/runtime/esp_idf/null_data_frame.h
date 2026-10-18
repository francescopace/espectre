/*
 * ESPectre - Direct Null Data frame contract
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

/// @cond INTERNAL
constexpr size_t TRAFFIC_NULL_DATA_FRAME_SIZE = 24U;
// Direct HT20/VHT20 peers use two zero bytes; C6 needs them for HT20 estimation.
constexpr size_t TRAFFIC_NULL_DATA_PEER_FRAME_SIZE = TRAFFIC_NULL_DATA_FRAME_SIZE + 2U;

size_t build_null_data_frame(const uint8_t *bssid, const uint8_t *station_mac,
                             uint8_t *buffer, size_t buffer_len, bool direct_peer = false);
/// @endcond

}  // namespace espectre
