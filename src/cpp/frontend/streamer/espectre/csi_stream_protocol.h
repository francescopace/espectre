/*
 * ESPectre - CSI Stream Protocol
 *
 * Protocol definitions for standalone CSI UDP streaming.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

namespace esphome {
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
  STREAM_FLAG_STIMULUS_ID_VALID = 1u << 3,
  STREAM_FLAG_REFERENCE_FRAME = 1u << 4,
};

static constexpr uint16_t STREAM_MAGIC = 0x4353U;
static constexpr uint8_t STREAM_VERSION = 3U;

#pragma pack(push, 1)
struct CsiStreamHeaderV3 {
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
  uint32_t stimulus_id;

  uint8_t channel;
  int8_t rssi_dbm;
  int8_t noise_floor_dbm;
};
#pragma pack(pop)

static_assert(sizeof(CsiStreamHeaderV3) == 49U, "CSI stream header size must remain stable");

inline void stream_set_stimulus_id(CsiStreamHeaderV3 *header, uint32_t stimulus_id) {
  if (header == nullptr) {
    return;
  }

  header->stimulus_id = stimulus_id;
}

inline uint32_t stream_get_stimulus_id(const CsiStreamHeaderV3 &header) {
  return header.stimulus_id;
}

}  // namespace espectre
}  // namespace esphome
