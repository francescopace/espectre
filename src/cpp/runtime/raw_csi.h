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

#include "csi_stream_protocol.h"

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
  IDLE_TIMEOUT,
  SLOW_CLIENT,
  SHUTDOWN,
  INTERNAL_ERROR,
};

/** View valid only for the duration of the capture callback. */
struct RawCsiPacketView {
  const int8_t *csi{nullptr};
  uint16_t csi_len{0U};
  uint64_t captured_at_us{0U};
  uint32_t wifi_rx_ts_us{0U};
  uint64_t wifi_rx_start_ts_ns{0U};
  uint8_t stream_flags{0U};
  uint8_t channel{0U};
  int8_t rssi_dbm{0};
  int8_t noise_floor_dbm{0};
  StreamPhyMode phy_mode{StreamPhyMode::UNKNOWN};
  StreamLtfType ltf_type{StreamLtfType::UNKNOWN};
  StreamChannelWidth channel_width{StreamChannelWidth::UNKNOWN};
};

using raw_csi_packet_callback_t = bool (*)(void *context, const RawCsiPacketView &packet);

constexpr char ESPECTRE_RAW_CSI_ENDPOINT[] = "/espectre/v1/csi";
constexpr uint8_t ESPECTRE_RAW_CSI_PROTOCOL_VERSION = 1U;
constexpr uint8_t ESPECTRE_RAW_CSI_RECORD_VERSION = STREAM_VERSION_V8;
constexpr size_t ESPECTRE_RAW_CSI_SESSION_ID_BYTES = 16U;
constexpr uint32_t ESPECTRE_RAW_CSI_RESPONSE_MAGIC = 0x52505345U; // "ESPR"

enum class RawCsiResponseStatus : uint8_t {
  FRESH = 0U,
  NO_SAMPLE = 1U,
  ERROR = 2U,
};

enum class RawCsiErrorCode : uint16_t {
  NONE = 0U,
  INVALID_SESSION = 1U,
  INVALID_SEQUENCE = 2U,
  PROTOCOL_MISMATCH = 3U,
  SESSION_INACTIVE = 4U,
  INTERNAL_ERROR = 5U,
};

#pragma pack(push, 1)
struct RawCsiHttpFramePrefixV1 {
  uint32_t magic;
  uint8_t version;
  uint8_t status;
  uint16_t header_len;
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES];
  uint64_t stream_sequence;
  uint16_t record_len;
  uint16_t error_code;
  uint64_t fresh_record_total;
  uint64_t no_sample_total;
  uint64_t replaced_sample_total;
  uint64_t dropped_sample_total;
  uint64_t raw_send_backpressure_total;
};
#pragma pack(pop)

static_assert(sizeof(RawCsiHttpFramePrefixV1) == 76U, "Raw CSI HTTP frame prefix size must remain stable");

struct RawCsiSessionConfig {
  uint8_t session_id[ESPECTRE_RAW_CSI_SESSION_ID_BYTES]{};
  uint64_t device_id{0U};
  StreamChipType chip{StreamChipType::UNKNOWN};
  uint64_t max_sample_age_us{20000U};
  uint32_t target_pps{100U};
};

struct RawCsiSessionDiagnostics {
  bool active{false};
  bool binary_bound{false};
  uint64_t no_sample_total{0U};
  uint64_t replaced_sample_total{0U};
  uint64_t dropped_sample_total{0U};
  uint64_t raw_send_backpressure_total{0U};
  uint64_t fresh_record_total{0U};
  uint64_t stream_sequence{0U};
};

}  // namespace espectre
