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

enum class RuntimeOperationState : uint8_t {
  SENSING = 0U,
  RAW_COLLECTION = 1U,
};

inline const char *runtime_operation_state_name(RuntimeOperationState state) {
  return state == RuntimeOperationState::RAW_COLLECTION ? "raw_collection" : "sensing";
}

enum class RawCsiStopReason : uint8_t {
  REQUESTED = 0U,
  OWNER_DISCONNECTED,
  RAW_DISCONNECTED,
  WIFI_LOST,
  CHANNEL_CHANGED,
  BIND_TIMEOUT,
  SLOW_CLIENT,
  SHUTDOWN,
  INTERNAL_ERROR,
};

/**
 * Callback-scoped view of one normalized raw CSI packet.
 *
 * The struct and the bytes addressed by `csi` are valid only for the duration
 * of the capture callback. Copy them before returning if another task needs the
 * sample.
 */
struct RawCsiPacketView {
  const int8_t *csi{nullptr};
  uint16_t csi_len{0U};
  uint64_t captured_at_us{0U};
  uint32_t wifi_rx_ts_us{0U};
  uint64_t wifi_rx_start_ts_ns{0U};
  uint8_t record_flags{0U};
  uint8_t channel{0U};
  int8_t rssi_dbm{0};
  int8_t noise_floor_dbm{0};
  RawCsiPhyMode phy_mode{RawCsiPhyMode::UNKNOWN};
  RawCsiLtfType ltf_type{RawCsiLtfType::UNKNOWN};
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
 * @param context Opaque caller-owned value supplied to `start_raw_collection()`.
 * @param packet Callback-scoped normalized CSI view.
 * @return true when the consumer accepted the packet, or false when it dropped
 *         it. The runtime does not stop collection on false.
 */
using raw_csi_packet_callback_t = bool (*)(void *context, const RawCsiPacketView &packet);

constexpr char ESPECTRE_RAW_CSI_ENDPOINT[] = "/espectre/v1/csi";
constexpr uint8_t ESPECTRE_RAW_CSI_PROTOCOL_VERSION = 1U;
constexpr uint8_t ESPECTRE_RAW_CSI_RECORD_VERSION = RAW_CSI_RECORD_VERSION_V8;
constexpr size_t ESPECTRE_RAW_CSI_SESSION_ID_BYTES = 16U;
constexpr uint32_t ESPECTRE_RAW_CSI_RESPONSE_MAGIC = 0x52505345U; // "ESPR"

#pragma pack(push, 1)
struct RawCsiHttpFramePrefix {
  uint32_t magic;
  uint8_t version;
  uint8_t record_version;
  uint16_t header_len;
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES];
  uint64_t stream_sequence;
  uint16_t record_len;
  uint16_t flags;
  uint64_t fresh_record_total;
  uint64_t raw_drop_total;
  uint64_t raw_send_backpressure_total;
};
#pragma pack(pop)

static_assert(sizeof(RawCsiHttpFramePrefix) == 60U, "Raw CSI HTTP frame prefix size must remain stable");

struct RawCsiSessionConfig {
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES]{};
  uint64_t device_id{0U};
  RawCsiChipType chip{RawCsiChipType::UNKNOWN};
};

struct RawCsiSessionDiagnostics {
  bool active{false};
  bool binary_bound{false};
  uint64_t raw_drop_total{0U};
  uint64_t raw_send_backpressure_total{0U};
  uint64_t fresh_record_total{0U};
  uint64_t stream_sequence{0U};
};

}  // namespace espectre
