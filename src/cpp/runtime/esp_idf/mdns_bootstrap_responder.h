/*
 * ESPectre - Shared mDNS Bootstrap Responder
 *
 * Stateless IPv4 responses for one-shot browser bootstrap hostnames.
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
  static constexpr size_t MAX_RESPONSE_BYTES = 256U;
  static constexpr uint8_t MAX_RESPONSES_PER_SECOND = 8U;

  struct PendingResponse {
    std::array<uint8_t, MAX_RESPONSE_BYTES> bytes{};
    size_t length{0U};
    size_t interface{0U};
    uint32_t destination_ipv4{0U};
    uint16_t destination_port{0U};
    int64_t due_us{0};
    bool used{false};
  };

  void clear_pending_();

  std::array<PendingResponse, MAX_PENDING_RESPONSES> pending_{};
  std::array<int64_t, MAX_RESPONSES_PER_SECOND> response_times_{};
  void *mutex_{nullptr};
  std::atomic<uint32_t> ipv4_address_{0U};
  size_t response_time_count_{0U};
  std::atomic<bool> configured_{false};
};

}  // namespace espectre
