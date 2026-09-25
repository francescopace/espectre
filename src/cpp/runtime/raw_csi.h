/*
 * ESPectre - Raw CSI Session Contract
 *
 * Transport-neutral runtime and binary framing types for bounded raw CSI
 * collection.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "csi_raw_record.h"

namespace espectre {

/** Transient operation the runtime is performing. */
enum class RuntimeOperationState : uint8_t {
  /** Normal motion sensing. */
  SENSING = 0U,
  /** Raw CSI collection; motion detection and derived events are paused. */
  RAW_COLLECTION = 1U,
};

/** Why a raw CSI collection session ended. */
enum class RawCsiStopReason : uint8_t {
  /** The owner stopped it. */
  REQUESTED = 0U,
  /** The binary stream connection closed. */
  RAW_DISCONNECTED = 2U,
  /** The Wi-Fi link dropped. */
  WIFI_LOST = 3U,
  /** The association moved to another channel. */
  CHANNEL_CHANGED = 4U,
  /** A send to the client failed. */
  SLOW_CLIENT = 6U,
  /** The service or runtime shut down. */
  SHUTDOWN = 7U,
  /** An unexpected failure ended the session. */
  INTERNAL_ERROR = 8U,
};

/**
 * Callback-scoped view of one normalized raw CSI packet.
 *
 * The struct and the bytes addressed by `csi` are valid only for the duration
 * of the capture callback. Copy them before returning if another task needs the
 * sample. The built-in capture pipeline supplies HT20_CSI_LEN bytes (64 complex
 * subcarriers) after LLTF, HT, or VHT normalization. This normalized capture
 * bound is independent of RAW_CSI_MAX_PAYLOAD_BYTES, the record-format limit.
 */
struct RawCsiPacketView {
  /** Interleaved I/Q bytes in the centered HT20 convention. */
  const int8_t *csi{nullptr};
  /** Length of `csi` in bytes. */
  uint16_t csi_len{0U};
  /** `esp_timer` time when the callback received the packet, in microseconds. */
  uint64_t captured_at_us{0U};
  /**
   * Wi-Fi driver receive timestamp, in microseconds.
   *
   * Valid when `record_flags` has `RAW_CSI_FLAG_WIFI_RX_TS_VALID`. On the
   * classic ESP32 it runs on a different clock than `esp_timer`.
   */
  uint32_t wifi_rx_ts_us{0U};
  /** Bit set of `RawCsiRecordFlags`. */
  uint8_t record_flags{0U};
  /** Primary channel the packet arrived on. */
  uint8_t channel{0U};
  int8_t rssi_dbm{0};
  /** Noise floor as reported by the Wi-Fi driver. */
  int8_t noise_floor_dbm{0};
  RawCsiPhyMode phy_mode{RawCsiPhyMode::UNKNOWN};
  RawCsiLtfType ltf_type{RawCsiLtfType::UNKNOWN};
  /** Always 20 MHz from the built-in pipeline. */
  RawCsiChannelWidth channel_width{RawCsiChannelWidth::UNKNOWN};
};

/**
 * Consume one packet synchronously from the Wi-Fi CSI capture context.
 *
 * The callback must remain bounded, non-blocking, and allocation-free. It may
 * copy the packet into a preallocated bounded queue for another task. Returning
 * false reports that the consumer did not accept this packet, for example
 * because that queue was full; collection continues, and the consumer owns any
 * drop or backpressure accounting.
 *
 * `context` is the opaque caller-owned value supplied to
 * `start_raw_collection()`. `packet` is a normalized CSI view valid only
 * during the call. Return true when the consumer accepted the packet, or false
 * when it dropped it; the runtime does not stop collection on false.
 */
using raw_csi_packet_callback_t = bool (*)(void *context, const RawCsiPacketView &packet);

/**
 * @name Raw CSI stream
 * Direct HTTP endpoint and binary framing for raw collection. See
 * [API.md](https://github.com/francescopace/espectre/blob/main/docs/API.md#csi-collection)
 * for the session rules.
 * @{
 */

/** Path of the binary collection endpoint. */
constexpr char ESPECTRE_RAW_CSI_ENDPOINT[] = "/espectre/v1/csi";
/** Version of the stream framing in `RawCsiHttpFramePrefix::version`. */
constexpr uint8_t ESPECTRE_RAW_CSI_PROTOCOL_VERSION = 1U;
/** Version of the record that follows each prefix. */
constexpr uint8_t ESPECTRE_RAW_CSI_RECORD_VERSION = RAW_CSI_RECORD_VERSION_V8;
/** Size of a session identifier. */
constexpr size_t ESPECTRE_RAW_CSI_SESSION_ID_BYTES = 16U;
/** Value of `RawCsiHttpFramePrefix::magic`, the bytes `ESPR` on the wire. */
constexpr uint32_t ESPECTRE_RAW_CSI_RESPONSE_MAGIC = 0x52505345U; // "ESPR"

/**
 * Little-endian prefix before every record on the binary stream.
 *
 * A frame is this 60-byte prefix, a `RawCsiRecordHeaderV8`, and the CSI
 * payload. The counters let a client detect loss without a side channel.
 */
#pragma pack(push, 1)
struct RawCsiHttpFramePrefix {
  /** `ESPECTRE_RAW_CSI_RESPONSE_MAGIC`. */
  uint32_t magic;
  /** `ESPECTRE_RAW_CSI_PROTOCOL_VERSION`. */
  uint8_t version;
  /** `ESPECTRE_RAW_CSI_RECORD_VERSION`. */
  uint8_t record_version;
  /** Size of this prefix in bytes. */
  uint16_t header_len;
  /** Session identifier, constant for the whole connection. */
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES];
  /**
   * One-based sequence of packets offered to the stream, including dropped
   * ones, so a gap shows how many were lost.
   */
  uint64_t stream_sequence;
  /** Size of the record header and payload that follow, in bytes. */
  uint16_t record_len;
  /** Reserved; zero. */
  uint16_t flags;
  /** Records sent in this session, including this one. */
  uint64_t fresh_record_total;
  /** Records not sent: the queue was full, the packet was invalid, or its batch failed to send. */
  uint64_t raw_drop_total;
  /** Batch sends the client did not accept. */
  uint64_t raw_send_backpressure_total;
};
#pragma pack(pop)

/** @} */

static_assert(sizeof(RawCsiHttpFramePrefix) == 60U, "Raw CSI HTTP frame prefix size must remain stable");

/** Identity stamped on every record of one raw collection session. */
struct RawCsiSessionConfig {
  /** Identifier echoed in every frame prefix. */
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES]{};
  /** Device identity written to each record header. */
  uint64_t device_id{0U};
  /** Chip written to each record header. */
  RawCsiChipType chip{RawCsiChipType::UNKNOWN};
};

/** Counters of the current or last raw collection session. */
struct RawCsiSessionDiagnostics {
  /** Whether a session is open. */
  bool active{false};
  /** Whether a client is attached to the binary stream. */
  bool binary_bound{false};
  /** See `RawCsiHttpFramePrefix::raw_drop_total`. */
  uint64_t raw_drop_total{0U};
  /** See `RawCsiHttpFramePrefix::raw_send_backpressure_total`. */
  uint64_t raw_send_backpressure_total{0U};
  /** See `RawCsiHttpFramePrefix::fresh_record_total`. */
  uint64_t fresh_record_total{0U};
  /** Latest offered sequence; see `RawCsiHttpFramePrefix::stream_sequence`. */
  uint64_t stream_sequence{0U};
};

}  // namespace espectre
