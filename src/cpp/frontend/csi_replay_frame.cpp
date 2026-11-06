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
#include "csi_replay_frame.h"

#include <algorithm>
#include <cstring>

#include "sdkconfig.h"

namespace espectre {

namespace {

constexpr uint8_t kMagic[4] = {'C', 'S', 'I', 'R'};
constexpr size_t kLlcSnapBytes = 8U;
constexpr size_t kIpv4HeaderBytes = 20U;
constexpr size_t kIcmpHeaderBytes = 8U;
constexpr size_t kFcsBytes = 4U;
constexpr uint8_t kIpProtoIcmp = 1U;
constexpr uint8_t kIcmpEchoRequest = 8U;

static_assert(CsiReplayPacket::kPayloadBytes == kLlcSnapBytes + kIpv4HeaderBytes + kIcmpHeaderBytes,
              "replay payload is LLC/SNAP, IPv4, and an ICMP echo header");

void write_be16(uint8_t *out, uint16_t value) {
  out[0] = static_cast<uint8_t>(value >> 8U);
  out[1] = static_cast<uint8_t>(value & 0xFFU);
}

}  // namespace

bool is_emulator_replay_mac(const uint8_t factory_mac[6]) {
  return factory_mac != nullptr &&
         std::memcmp(factory_mac, kEmulatorReplayMacPrefix, sizeof(kEmulatorReplayMacPrefix)) == 0;
}

int64_t CsiReplayPacer::due(uint32_t rx_timestamp_us, int64_t now_us) {
  if (anchored_) {
    // The unsigned difference keeps the spacing across a 32-bit wrap.
    const int64_t due_us =
        anchor_local_us_ + static_cast<int32_t>(rx_timestamp_us - anchor_rx_us_);
    if (due_us <= now_us + kResyncUs && due_us >= now_us - kResyncUs) {
      return due_us;
    }
  }
  anchored_ = true;
  anchor_local_us_ = now_us + kJitterBufferUs;
  anchor_rx_us_ = rx_timestamp_us;
  return anchor_local_us_;
}

int64_t CsiReplayPacer::release_at(int64_t due_us) const {
  return has_release_ ? std::max(due_us, last_release_us_ + kCatchUpSpacingUs) : due_us;
}

void CsiReplayPacer::mark_released(int64_t release_us, int64_t now_us) {
  // A release long after its slot must not grant a burst of past slots.
  last_release_us_ = std::max(release_us, now_us - kCatchUpSpacingUs);
  has_release_ = true;
}

void CsiReplayFrameParser::reset_() {
  magic_matched_ = 0U;
  received_ = 0U;
  sum_ = 0U;
}

bool CsiReplayFrameParser::push(uint8_t byte, CsiReplayFrame *out) {
  if (magic_matched_ < kMagicBytes) {
    // A mismatch may itself start the next magic, as in "CSCSIR".
    magic_matched_ = byte == kMagic[magic_matched_] ? magic_matched_ + 1U
                                                    : (byte == kMagic[0] ? 1U : 0U);
    return false;
  }

  if (received_ < kHeaderBytes) {
    header_[received_++] = byte;
    sum_ = static_cast<uint8_t>(sum_ + byte);
    if (received_ == kHeaderBytes) {
      frame_.rssi_dbm = static_cast<int8_t>(header_[0]);
      frame_.len = header_[1];
      frame_.rx_timestamp_us = static_cast<uint32_t>(header_[2]) |
                               (static_cast<uint32_t>(header_[3]) << 8U) |
                               (static_cast<uint32_t>(header_[4]) << 16U) |
                               (static_cast<uint32_t>(header_[5]) << 24U);
      if (frame_.len == 0U || frame_.len > CsiReplayFrame::kMaxCsiBytes) {
        reset_();
      }
    }
    return false;
  }

  const size_t csi_index = received_ - kHeaderBytes;
  if (csi_index < frame_.len) {
    frame_.csi[csi_index] = static_cast<int8_t>(byte);
    sum_ = static_cast<uint8_t>(sum_ + byte);
    received_++;
    return false;
  }

  const bool valid = byte == sum_;
  if (valid && out != nullptr) {
    *out = frame_;
  }
  reset_();
  return valid && out != nullptr;
}

bool build_csi_replay_packet(const CsiReplayFrame &frame,
                             const CsiReplayIdentity &identity,
                             uint8_t channel,
                             uint16_t sequence,
                             CsiReplayPacket *out) {
  if (out == nullptr || frame.len == 0U || identity.local_ip_addr == 0U) {
    return false;
  }
  *out = {};

  // IPv4 addresses keep network byte order, as lwIP stores them.
  uint8_t *payload = out->payload.data();
  const uint8_t llc_snap[kLlcSnapBytes] = {0xAAU, 0xAAU, 0x03U, 0x00U, 0x00U, 0x00U, 0x08U, 0x00U};
  std::memcpy(payload, llc_snap, sizeof(llc_snap));
  uint8_t *ip = payload + kLlcSnapBytes;
  ip[0] = 0x45U;
  write_be16(ip + 2U, static_cast<uint16_t>(kIpv4HeaderBytes + kIcmpHeaderBytes));
  ip[8] = 64U;
  ip[9] = kIpProtoIcmp;
  std::memcpy(ip + 12U, &identity.peer_ip_addr, 4U);
  std::memcpy(ip + 16U, &identity.local_ip_addr, 4U);
  uint8_t *icmp = ip + kIpv4HeaderBytes;
  icmp[0] = kIcmpEchoRequest;
  write_be16(icmp + 6U, sequence);

  // A FromDS data frame addressed to the station.
  out->header[0] = 0x08U;
  out->header[1] = 0x02U;
  std::memcpy(out->header.data() + 4U, identity.local_mac_addr, 6U);

  wifi_csi_info_t &info = out->info;
  info.rx_ctrl.rssi = frame.rssi_dbm;
  info.rx_ctrl.channel = channel;
  // The recorded MAC-clock timestamp keeps the capture's packet spacing.
  info.rx_ctrl.timestamp = frame.rx_timestamp_us;
  info.rx_ctrl.sig_len = static_cast<uint16_t>(CsiReplayPacket::kMacHeaderBytes +
                                               CsiReplayPacket::kPayloadBytes + kFcsBytes);
#if CONFIG_SOC_WIFI_HE_SUPPORT
  info.rx_ctrl.cur_bb_format = RX_BB_FORMAT_HT;
  info.rx_ctrl.second = 0U;
  info.rx_ctrl.rx_channel_estimate_info_vld = 1U;
#else
  info.rx_ctrl.sig_mode = 1U;
  info.rx_ctrl.cwb = 0U;
#endif
  std::memcpy(info.dmac, identity.local_mac_addr, 6U);
  info.buf = const_cast<int8_t *>(frame.csi.data());
  info.len = frame.len;
  info.hdr = out->header.data();
  info.payload = payload;
  info.payload_len = static_cast<uint16_t>(out->payload.size());
  info.rx_seq = sequence;
  return true;
}

}  // namespace espectre
