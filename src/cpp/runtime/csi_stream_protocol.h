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
  RESERVED_LEGACY_S2 = 2,
  S3 = 3,
  C3 = 4,
  C5 = 5,
  C6 = 6,
};

enum StreamFlags : uint8_t {
  STREAM_FLAG_FIRST_WORD_INVALID = 1u << 0,
  STREAM_FLAG_WIFI_RX_TS_VALID = 1u << 1,
  STREAM_FLAG_WIFI_RX_START_TS_NS_VALID = 1u << 2,
  // Set on every emitted CSI record. The streamer only transmits fresh CSI
  // samples, so stale repeats are dropped instead of being sent.
  STREAM_FLAG_CSI_FRESH = 1u << 3,
};

enum class StreamPhyMode : uint8_t {
  UNKNOWN = 0,
  LEGACY = 1,
  HT = 2,
  VHT = 3,
  HE_SU = 4,
  HE_MU = 5,
  HE_ERSU = 6,
  HE_TB = 7,
};

enum class StreamLtfType : uint8_t {
  UNKNOWN = 0,
  LLTF = 1,
  HT_LTF = 2,
  VHT_LTF = 3,
  HE_LTF = 4,
};

enum class StreamChannelWidth : uint8_t {
  UNKNOWN = 0,
  MHZ_20 = 1,
  MHZ_40 = 2,
  MHZ_80 = 3,
  MHZ_160 = 4,
  MHZ_80_80 = 5,
};

static constexpr uint16_t STREAM_MAGIC = 0x4353U;
static constexpr uint8_t STREAM_VERSION = 7U;

#pragma pack(push, 1)
struct CsiStreamHeaderV7 {
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
  uint32_t stream_fresh_total;
  uint32_t pacing_rx_total;

  uint8_t phy_mode;
  uint8_t ltf_type;
  uint8_t channel_width;
};
#pragma pack(pop)

static_assert(sizeof(CsiStreamHeaderV7) == 64U, "CSI stream header size must remain stable");

static constexpr size_t STREAM_MAX_CSI_LEN_BYTES = 512U;
static constexpr size_t STREAM_MAX_PACKET_BYTES = sizeof(CsiStreamHeaderV7) + STREAM_MAX_CSI_LEN_BYTES;

// Senders may concatenate up to this many complete records (header followed by
// payload) into one UDP datagram; receivers parse records back-to-back until
// the datagram is exhausted.
static constexpr size_t STREAM_MAX_BATCH_RECORDS = 8U;
// Seven current V7 HT20 records fit in one Ethernet MTU after UDP/IP headers.
static constexpr size_t STREAM_MAX_TX_BATCH_RECORDS = 7U;

}  // namespace espectre
