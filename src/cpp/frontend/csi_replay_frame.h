/*
 * ESPectre - CSI Replay Frame
 *
 * Wire format, packet synthesis, and the emulator marker for replaying
 * recorded CSI into the capture path inside esp-emu.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/csi_format.h"
#include "esp_wifi.h"

namespace espectre {

/**
 * First three bytes of the factory MAC that marks esp-emu replay runs.
 *
 * The first byte sets the locally administered bit. A real chip's factory MAC
 * comes from an IEEE-assigned OUI, which always clears that bit, and its
 * factory eFuse block cannot be rewritten, so silicon never carries this
 * prefix. CI passes esp-emu an eFuse image with it.
 */
constexpr uint8_t kEmulatorReplayMacPrefix[3] = {0x02U, 0x45U, 0x53U};

/** True when a factory MAC carries the esp-emu replay marker. */
bool is_emulator_replay_mac(const uint8_t factory_mac[6]);

/**
 * One recorded CSI packet as it arrives on the replay link.
 *
 * On the wire: `CSIR`, an int8 RSSI, a length byte, the recorded 32-bit RX
 * timestamp (little-endian, microseconds), the stored CSI bytes, and an 8-bit
 * sum of every byte after the magic.
 */
struct CsiReplayFrame {
  // Stored captures hold the normalized 64-bin HT20 layout.
  static constexpr size_t kMaxCsiBytes = HT20_CSI_LEN;

  int8_t rssi_dbm{0};
  uint8_t len{0U};
  uint32_t rx_timestamp_us{0U};
  std::array<int8_t, kMaxCsiBytes> csi{};
};

/** Byte-stream decoder that resynchronizes on the magic after any malformed frame. */
class CsiReplayFrameParser {
 public:
  /** Consume one byte; true when it completes a valid frame, written to `out`. */
  bool push(uint8_t byte, CsiReplayFrame *out);

 private:
  static constexpr size_t kMagicBytes = 4U;
  static constexpr size_t kHeaderBytes = 6U;

  void reset_();

  size_t magic_matched_{0U};
  size_t received_{0U};
  uint8_t header_[kHeaderBytes]{};
  uint8_t sum_{0U};
  CsiReplayFrame frame_{};
};

/**
 * Releases replayed frames at their recorded spacing behind a jitter buffer.
 *
 * The host sends frames on wall-clock time, so an emulator stall delivers a
 * burst over UART. Releasing each frame at its recorded RX timestamp,
 * offset by a fixed buffer, keeps that burst out of the firmware's callback
 * queue. A late frame waits at least `kCatchUpSpacingUs` after the previous
 * one, so the backlog drains at a bounded rate. Times are local microseconds.
 */
class CsiReplayPacer {
 public:
  static constexpr int64_t kJitterBufferUs = 100000;
  static constexpr int64_t kCatchUpSpacingUs = 4000;
  /** A frame further than this from its expected time restarts the timeline. */
  static constexpr int64_t kResyncUs = 1000000;

  /** When a frame recorded at `rx_timestamp_us`, arriving at `now_us`, is due. */
  int64_t due(uint32_t rx_timestamp_us, int64_t now_us);
  /** The earliest time a frame due at `due_us` may be released. */
  int64_t release_at(int64_t due_us) const;
  /** Record a release scheduled for `release_us` and performed at `now_us`. */
  void mark_released(int64_t release_us, int64_t now_us);

 private:
  bool anchored_{false};
  int64_t anchor_local_us_{0};
  uint32_t anchor_rx_us_{0U};
  bool has_release_{false};
  int64_t last_release_us_{0};
};

/** Station addresses a replayed frame carries; IPv4 values keep network byte order. */
struct CsiReplayIdentity {
  uint32_t local_ip_addr{0U};
  uint32_t peer_ip_addr{0U};
  uint8_t local_mac_addr[6]{};
};

/**
 * A replayed frame dressed as an external host's ping to the station.
 *
 * `info` points into `header`, `payload`, and the frame's CSI, so the packet
 * and its frame must outlive every use of `info`.
 */
struct CsiReplayPacket {
  static constexpr size_t kMacHeaderBytes = 24U;
  static constexpr size_t kPayloadBytes = 36U;

  wifi_csi_info_t info{};
  std::array<uint8_t, kMacHeaderBytes> header{};
  std::array<uint8_t, kPayloadBytes> payload{};
};

/**
 * Fill `out` with an HT20 CSI packet that the `external` traffic filter keeps:
 * an ICMP Echo Request to the station's own IP and MAC.
 *
 * Returns false until the station has an IP address.
 */
bool build_csi_replay_packet(const CsiReplayFrame &frame,
                             const CsiReplayIdentity &identity,
                             uint8_t channel,
                             uint16_t sequence,
                             CsiReplayPacket *out);

}  // namespace espectre
