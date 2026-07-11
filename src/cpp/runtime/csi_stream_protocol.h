/*
 * ESPectre - CSI Stream Protocol
 *
 * Protocol definitions for standalone CSI UDP streaming.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

enum class StreamChipType : uint8_t {
  UNKNOWN = 0,
  ESP32 = 1,
  S2 = 2,
  S3 = 3,
  C3 = 4,
  C5 = 5,
  C6 = 6,
};

enum StreamFlags : uint8_t {
  STREAM_FLAG_FIRST_WORD_INVALID = 1u << 0,
  STREAM_FLAG_WIFI_RX_TS_VALID = 1u << 1,
  STREAM_FLAG_WIFI_RX_START_TS_NS_VALID = 1u << 2,
  // Set when the packet carries a CSI sample not sent in a previous packet.
  // Cleared on repeats of the latest available sample emitted to keep the
  // traffic-paced stream at the target rate.
  STREAM_FLAG_CSI_FRESH = 1u << 3,
};

static constexpr uint16_t STREAM_MAGIC = 0x4353U;
static constexpr uint8_t STREAM_VERSION = 5U;

#pragma pack(push, 1)
struct CsiStreamHeaderV5 {
  uint16_t magic;
  uint8_t version;
  uint8_t header_len;

  uint8_t chip;
  uint8_t flags;
  uint32_t seq_num;
  uint16_t num_subcarriers;
  uint16_t csi_len_bytes;

  uint64_t device_id;
  uint64_t device_ticks_us;
  uint32_t wifi_rx_ts_us;
  uint64_t wifi_rx_start_ts_ns;

  uint8_t channel;
  int8_t rssi_dbm;
  int8_t noise_floor_dbm;
  uint64_t tx_backpressure_total;
};
#pragma pack(pop)

static_assert(sizeof(CsiStreamHeaderV5) == 53U, "CSI stream header size must remain stable");

static constexpr size_t STREAM_MAX_CSI_LEN_BYTES = 512U;
static constexpr size_t STREAM_MAX_PACKET_BYTES = sizeof(CsiStreamHeaderV5) + STREAM_MAX_CSI_LEN_BYTES;

}  // namespace espectre
