/*
 * ESPectre - Peer-Assisted Local Discovery
 *
 * Bounded, read-only peer metadata shared by the Direct transport and the
 * ESP-IDF mDNS query adapter.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace espectre {

inline constexpr const char *ESPECTRE_PEER_DISCOVERY_METHOD = "discover_peers";
inline constexpr uint32_t ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS = 3000U;
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_DEVICES = 8U;
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES = 2U;
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE = 3584U;

struct PeerDiscoveryCandidate {
  std::string instance;
  std::string hostname;
  std::string device_id;
  std::string name;
  std::string frontend;
  std::string txt_version;
  std::string protocol_version;
  std::string transport;
  std::string path;
  std::string events;
  std::string firmware;
  std::string chip;
  std::string capabilities;
  uint16_t port{0U};
  std::vector<uint32_t> ipv4_addresses;
};

struct PeerDiscoverySnapshot {
  uint32_t elapsed_ms{0U};
  bool timed_out{false};
  bool truncated{false};
  size_t rejected_results{0U};
  std::vector<PeerDiscoveryCandidate> devices;
};

/** Validate, deduplicate, sort, and serialize one bounded discovery result. */
PeerDiscoverySnapshot validate_peer_discovery_candidates(
    const std::vector<PeerDiscoveryCandidate> &candidates,
    uint32_t station_address,
    uint32_t station_netmask,
    uint32_t elapsed_ms,
    bool timed_out);
std::string peer_discovery_snapshot_json(const PeerDiscoverySnapshot &snapshot);

class IPeerDiscoveryService {
 public:
  using Completion = std::function<void(PeerDiscoverySnapshot snapshot)>;

  virtual ~IPeerDiscoveryService() = default;
  virtual void set_local_candidate(PeerDiscoveryCandidate candidate) {
    (void) candidate;
  }
  virtual void set_wifi_ready(bool ready) = 0;
  virtual bool ready() const = 0;
  virtual bool active() const = 0;
  virtual bool start(Completion completion) = 0;
  virtual void loop() = 0;
  virtual void shutdown() = 0;
};

}  // namespace espectre
