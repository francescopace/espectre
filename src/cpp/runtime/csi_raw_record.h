/*
 * ESPectre - Raw CSI Record Format
 *
 * Transport-neutral raw CSI record definitions. V7 remains readable for
 * historical captures, while current raw HTTP sessions emit V8.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

enum class RawCsiChipType : uint8_t {
  UNKNOWN = 0,
  ESP32 = 1,
  S2 = 2,
  RESERVED_LEGACY_S2 = S2,
  S3 = 3,
  C3 = 4,
  C5 = 5,
  C6 = 6,
};

enum RawCsiRecordFlags : uint8_t {
  RAW_CSI_FLAG_FIRST_WORD_INVALID = 1u << 0,
  RAW_CSI_FLAG_WIFI_RX_TS_VALID = 1u << 1,
  RAW_CSI_FLAG_WIFI_RX_START_TS_NS_VALID = 1u << 2,
  // Set on every emitted raw CSI record.
  RAW_CSI_FLAG_FRESH = 1u << 3,
};

enum class RawCsiPhyMode : uint8_t {
  UNKNOWN = 0,
  LEGACY = 1,
  HT = 2,
  VHT = 3,
  HE_SU = 4,
  HE_MU = 5,
  HE_ERSU = 6,
  HE_TB = 7,
};

enum class RawCsiLtfType : uint8_t {
  UNKNOWN = 0,
  LLTF = 1,
  HT_LTF = 2,
  VHT_LTF = 3,
  HE_LTF = 4,
};

enum class RawCsiChannelWidth : uint8_t {
  UNKNOWN = 0,
  MHZ_20 = 1,
  MHZ_40 = 2,
  MHZ_80 = 3,
  MHZ_160 = 4,
  MHZ_80_80 = 5,
};

static constexpr uint16_t RAW_CSI_RECORD_MAGIC = 0x4353U;
static constexpr uint8_t RAW_CSI_RECORD_VERSION_V7 = 7U;
static constexpr uint8_t RAW_CSI_RECORD_VERSION_V8 = 8U;
static constexpr uint8_t RAW_CSI_RECORD_VERSION = RAW_CSI_RECORD_VERSION_V8;

#pragma pack(push, 1)
struct RawCsiRecordHeaderV7 {
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
  uint64_t transport_backpressure_total;
  uint32_t fresh_record_total;
  uint32_t traffic_packets_total;

  uint8_t phy_mode;
  uint8_t ltf_type;
  uint8_t channel_width;
};

/** Transport-neutral raw CSI record emitted by Native Direct collection. */
struct RawCsiRecordHeaderV8 {
  uint16_t magic;
  uint8_t version;
  uint8_t header_len;

  uint8_t chip;
  uint8_t flags;
  uint32_t seq_num;
  uint16_t num_subcarriers;
  uint16_t csi_len_bytes;

  uint64_t device_id;
  /** Monotonic device time captured with the CSI sample. */
  uint64_t device_ticks_us;
  uint32_t wifi_rx_ts_us;
  uint64_t wifi_rx_start_ts_ns;

  uint8_t channel;
  int8_t rssi_dbm;
  int8_t noise_floor_dbm;
  uint64_t transport_backpressure_total;
  uint32_t fresh_record_total;
  uint32_t request_accepted_total;

  uint8_t phy_mode;
  uint8_t ltf_type;
  uint8_t channel_width;
};
#pragma pack(pop)

static_assert(sizeof(RawCsiRecordHeaderV7) == 64U, "CSI V7 raw record header size must remain stable");
static_assert(sizeof(RawCsiRecordHeaderV8) == 64U, "CSI V8 raw record header size must remain stable");

static constexpr size_t RAW_CSI_MAX_PAYLOAD_BYTES = 512U;
static constexpr size_t RAW_CSI_MAX_RECORD_BYTES =
    sizeof(RawCsiRecordHeaderV8) + RAW_CSI_MAX_PAYLOAD_BYTES;

}  // namespace espectre
