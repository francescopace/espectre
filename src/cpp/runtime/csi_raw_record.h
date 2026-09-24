/*
 * ESPectre - Raw CSI Record Format
 *
 * Transport-neutral raw CSI V8 record definitions.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

/** Chip that captured a raw record. */
enum class RawCsiChipType : uint8_t {
  UNKNOWN = 0,
  ESP32 = 1,
  S2 = 2,
  /** Older name for `S2`. */
  RESERVED_LEGACY_S2 = S2,
  S3 = 3,
  C3 = 4,
  C5 = 5,
  C6 = 6,
};

/** Bits of `RawCsiPacketView::record_flags` and `RawCsiRecordHeaderV8::flags`. */
enum RawCsiRecordFlags : uint8_t {
  /** The hardware marked the first four source bytes invalid. */
  RAW_CSI_FLAG_FIRST_WORD_INVALID = 1u << 0,
  /** `wifi_rx_ts_us` holds a driver timestamp. */
  RAW_CSI_FLAG_WIFI_RX_TS_VALID = 1u << 1,
  /** `wifi_rx_start_ts_ns` holds a receive start time. */
  RAW_CSI_FLAG_WIFI_RX_START_TS_NS_VALID = 1u << 2,
  /** Set on every emitted raw CSI record. */
  RAW_CSI_FLAG_FRESH = 1u << 3,
};

/** PHY of the frame that produced the CSI. */
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

/** Training field the CSI was estimated from. */
enum class RawCsiLtfType : uint8_t {
  UNKNOWN = 0,
  LLTF = 1,
  HT_LTF = 2,
  VHT_LTF = 3,
  HE_LTF = 4,
};

/** Channel width of the frame that produced the CSI. */
enum class RawCsiChannelWidth : uint8_t {
  UNKNOWN = 0,
  MHZ_20 = 1,
  MHZ_40 = 2,
  MHZ_80 = 3,
  MHZ_160 = 4,
  MHZ_80_80 = 5,
};

/** Value of `RawCsiRecordHeaderV8::magic`, the bytes `SC` on the wire. */
static constexpr uint16_t RAW_CSI_RECORD_MAGIC = 0x4353U;
static constexpr uint8_t RAW_CSI_RECORD_VERSION_V8 = 8U;
/** Record version this SDK writes. */
static constexpr uint8_t RAW_CSI_RECORD_VERSION = RAW_CSI_RECORD_VERSION_V8;

/**
 * Little-endian, 64-byte header of one raw CSI record, followed by
 * `csi_len_bytes` of interleaved I/Q payload.
 *
 * Transport-neutral: Direct raw collection sends it after a
 * `RawCsiHttpFramePrefix`. The capture fields mirror `RawCsiPacketView`.
 */
#pragma pack(push, 1)
struct RawCsiRecordHeaderV8 {
  /** `RAW_CSI_RECORD_MAGIC`. */
  uint16_t magic;
  /** `RAW_CSI_RECORD_VERSION_V8`. */
  uint8_t version;
  /** Size of this header in bytes. */
  uint8_t header_len;

  /** `RawCsiChipType` value. */
  uint8_t chip;
  /** Bit set of `RawCsiRecordFlags`. */
  uint8_t flags;
  /** Stream sequence, saturated at `UINT32_MAX`. */
  uint32_t seq_num;
  /** Complex subcarriers in the payload: `csi_len_bytes / 2`. */
  uint16_t num_subcarriers;
  /** Payload size in bytes, at most `RAW_CSI_MAX_PAYLOAD_BYTES`. */
  uint16_t csi_len_bytes;

  uint64_t device_id;
  /** Monotonic device time captured with the CSI sample, in microseconds. */
  uint64_t device_ticks_us;
  /** See `RawCsiPacketView::wifi_rx_ts_us`. */
  uint32_t wifi_rx_ts_us;
  /** See `RawCsiPacketView::wifi_rx_start_ts_ns`. */
  uint64_t wifi_rx_start_ts_ns;

  uint8_t channel;
  int8_t rssi_dbm;
  int8_t noise_floor_dbm;
  /** Copy of `RawCsiHttpFramePrefix::raw_send_backpressure_total`. */
  uint64_t transport_backpressure_total;
  /** Records sent including this one, saturated at `UINT32_MAX`. */
  uint32_t fresh_record_total;
  /** Packets offered to the stream up to this one, saturated at `UINT32_MAX`. */
  uint32_t request_accepted_total;

  /** `RawCsiPhyMode` value. */
  uint8_t phy_mode;
  /** `RawCsiLtfType` value. */
  uint8_t ltf_type;
  /** `RawCsiChannelWidth` value. */
  uint8_t channel_width;
};
#pragma pack(pop)

static_assert(sizeof(RawCsiRecordHeaderV8) == 64U, "CSI V8 raw record header size must remain stable");

/** Largest payload a record may carry; the built-in pipeline sends 128 bytes. */
static constexpr size_t RAW_CSI_MAX_PAYLOAD_BYTES = 512U;
/** Largest record: header plus the maximum payload. */
static constexpr size_t RAW_CSI_MAX_RECORD_BYTES =
    sizeof(RawCsiRecordHeaderV8) + RAW_CSI_MAX_PAYLOAD_BYTES;

}  // namespace espectre
