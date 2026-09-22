/*
 * ESPectre - Shared mDNS Bootstrap Responder
 *
 * Bounded IPv4 responses for one-shot browser bootstrap hostnames.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace espectre {

class MdnsBootstrapResponder {
 public:
  static constexpr size_t NONCE_HEX_LENGTH = 24U;
  static constexpr uint32_t RESPONSE_TTL_SECONDS = 10U;

  ~MdnsBootstrapResponder();

  bool setup();
  bool update(uint32_t ipv4_address);
  void loop();
  void shutdown();
  bool active() const { return configured_.load() && ipv4_address_.load() != 0U; }

  // Called by the mDNS receive wrapper before the Espressif responder filters
  // questions for hostnames that it owns.
  void ingest_query(const uint8_t *packet,
                    size_t length,
                    size_t interface,
                    uint32_t source_ipv4,
                    uint16_t source_port);

 private:
  static constexpr size_t MAX_PENDING_RESPONSES = 4U;
  static constexpr size_t MAX_DEFERRED_KNOWN_ANSWERS = 2U * MAX_PENDING_RESPONSES;
  static constexpr size_t MAX_RESPONSE_BYTES = 256U;
  static constexpr uint8_t MAX_RESPONSES_PER_SECOND = 8U;

  struct PendingResponse {
    std::array<char, 64U> host{};
    std::array<char, 6U> domain{};
    size_t interface{0U};
    uint32_t source_ipv4{0U};
    uint16_t source_port{0U};
    uint16_t query_id{0U};
    uint16_t query_type{0U};
    uint16_t query_class{0U};
    int64_t due_us{0};
    int64_t expires_us{0};
    uint8_t requested{0U};
    uint8_t known{0U};
    bool unicast{false};
    bool truncated{false};
    bool used{false};
  };

  struct RecentResponse {
    std::array<char, 64U> host{};
    size_t interface{0U};
    std::array<int64_t, 2U> sent_us{{-1000000, -1000000}};
    bool used{false};
  };

  struct DeferredKnownAnswer {
    std::array<char, 64U> host{};
    int64_t received_us{0};
    size_t interface{0U};
    uint32_t source_ipv4{0U};
    uint32_t ipv4_address{0U};
    uint32_t generation{0U};
    uint16_t source_port{0U};
    uint8_t records{0U};
    bool truncated{false};
  };

  void defer_known_answers_(const uint8_t *packet, size_t length, size_t offset,
                           uint16_t answer_count, size_t interface,
                           uint32_t source_ipv4, uint16_t source_port, bool truncated);
  void apply_deferred_known_answers_();
  bool begin_send_(uint32_t generation);
  void end_send_();
  void wait_for_sends_();
  void clear_pending_();
  void finish_send_();

  std::array<PendingResponse, MAX_PENDING_RESPONSES> pending_{};
  std::array<RecentResponse, MAX_RESPONSES_PER_SECOND> recent_{};
  std::array<int64_t, MAX_RESPONSES_PER_SECOND> response_times_{};
  // The mDNS task publishes continuations; consumers are serialized by mutex_.
  // Retain decoded records instead of packets, without blocking packet ingestion.
  std::array<DeferredKnownAnswer, MAX_DEFERRED_KNOWN_ANSWERS> deferred_{};
  std::atomic<uint32_t> deferred_head_{0U};
  std::atomic<uint32_t> deferred_tail_{0U};
  void *mutex_{nullptr};
  std::atomic<uint32_t> ipv4_address_{0U};
  std::atomic<uint32_t> generation_{0U};
  std::atomic<uint32_t> sends_in_flight_{0U};
  size_t response_time_count_{0U};
  bool sending_{false};
  PendingResponse sent_{};
  uint32_t sent_generation_{0U};
  uint8_t sent_records_{0U};
  bool sent_success_{false};
  int64_t sent_time_us_{0};
  std::atomic<bool> send_complete_{false};
  std::atomic<bool> configured_{false};
};

}  // namespace espectre
