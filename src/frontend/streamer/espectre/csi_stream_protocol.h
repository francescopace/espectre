/*
 * ESPectre - CSI Stream Protocol
 *
 * Protocol definitions for standalone CSI and FTM UDP streaming.
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
  STREAM_FLAG_GAIN_INFO_VALID = 1u << 1,
  STREAM_FLAG_WIFI_RX_TS_VALID = 1u << 2,
  STREAM_FLAG_FRAME_ID_FIELDS_VALID = 1u << 3,
  STREAM_FLAG_RAW_CSI_UNMODIFIED = 1u << 4,
  STREAM_FLAG_REFERENCE_FRAME = 1u << 5,
  STREAM_FLAG_STIMULUS_ID_VALID = 1u << 6,
  STREAM_FLAG_RX_FREQ_OFFSET_VALID = 1u << 7,
};

enum StreamFtmFlags : uint8_t {
  STREAM_FTM_FLAG_AP_RESPONDER = 1u << 0,
  STREAM_FTM_FLAG_AP_INITIATOR = 1u << 1,
  STREAM_FTM_FLAG_PERIODIC = 1u << 2,
  STREAM_FTM_FLAG_SUCCESS = 1u << 3,
};

enum class StreamFtmEventType : uint8_t {
  REPORT = 1,
};

static constexpr uint16_t STREAM_MAGIC = 0x4353U;
static constexpr uint8_t STREAM_VERSION = 1U;
static constexpr uint16_t FTM_MAGIC = 0x4654U;
static constexpr uint8_t FTM_VERSION = 1U;

#pragma pack(push, 1)
struct CsiStreamHeaderV1 {
  uint16_t magic;
  uint8_t version;
  uint8_t header_len;

  uint8_t chip;
  uint8_t flags;
  uint16_t stimulus_id_lo16;

  uint32_t seq_num;
  uint16_t num_subcarriers;
  uint16_t csi_len_bytes;

  uint16_t frame_ctrl;
  uint16_t rx_seq;

  uint64_t device_id;
  uint32_t boot_id;
  uint64_t device_ticks_us;
  uint32_t wifi_rx_ts_us;
  uint64_t wifi_rx_start_ts_ns;

  uint8_t tx_mac[6];
  uint8_t dmac[6];

  uint8_t channel;
  uint8_t secondary_channel;
  int8_t rssi_dbm;
  int8_t noise_floor_dbm;

  uint8_t agc_gain;
  int8_t fft_gain;
  uint16_t stimulus_id_hi16;
  int16_t rx_freq_offset_step;
  uint16_t reserved0;
};

struct FtmStreamEventV1 {
  uint16_t magic;
  uint8_t version;
  uint8_t header_len;

  uint8_t event_type;
  uint8_t chip;
  uint8_t ftm_status;
  uint8_t ftm_flags;

  uint32_t seq_num;
  uint64_t device_id;
  uint32_t boot_id;
  uint64_t device_ticks_us;

  uint8_t peer_mac[6];
  uint8_t channel;
  int8_t rssi_dbm;
  uint8_t ftm_report_num_entries;
  uint8_t reserved0;

  uint32_t rtt_raw_ns;
  uint32_t rtt_est_ns;
  uint32_t dist_est_cm;
  uint32_t session_id;
  uint16_t reserved1;
};
#pragma pack(pop)

static_assert(sizeof(CsiStreamHeaderV1) == 76U, "CSI stream header size must remain stable");
static_assert(sizeof(FtmStreamEventV1) == 60U, "FTM stream event size must remain stable");

inline void stream_set_stimulus_id(CsiStreamHeaderV1 *header, uint32_t stimulus_id) {
  if (header == nullptr) {
    return;
  }

  header->stimulus_id_lo16 = static_cast<uint16_t>(stimulus_id & 0xFFFFU);
  header->stimulus_id_hi16 = static_cast<uint16_t>((stimulus_id >> 16U) & 0xFFFFU);
}

inline uint32_t stream_get_stimulus_id(const CsiStreamHeaderV1 &header) {
  return static_cast<uint32_t>(header.stimulus_id_lo16) |
         (static_cast<uint32_t>(header.stimulus_id_hi16) << 16U);
}

}  // namespace espectre
}  // namespace esphome
