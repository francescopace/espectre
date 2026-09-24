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

/** Duration of one DNS-SD search. */
inline constexpr uint32_t ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS = 3000U;
/** Devices kept in one result. */
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_DEVICES = 8U;
/** IPv4 addresses kept per device. */
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES = 2U;
/** Largest serialized result, in bytes; devices that do not fit are dropped. */
inline constexpr size_t ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE = 3584U;

/**
 * One ESPectre device found through DNS-SD, as advertised in its TXT record.
 *
 * See [DISCOVERY.md](https://github.com/francescopace/espectre/blob/main/docs/DISCOVERY.md)
 * for the meaning and format of each advertised key.
 */
struct PeerDiscoveryCandidate {
  /** DNS-SD service instance name. */
  std::string instance;
  /** mDNS host name. */
  std::string hostname;
  /** Canonical device id; devices are deduplicated by it. */
  std::string device_id;
  /** User-facing device name. */
  std::string name;
  std::string frontend;
  /** TXT schema version, `txtvers`. */
  std::string txt_version;
  /** `ESPECTRE_PROTOCOL_VERSION` of the device. */
  std::string protocol_version;
  /** Transport name, such as `http`. */
  std::string transport;
  /** API base path, such as `/espectre/v1`. */
  std::string path;
  /** Application version. */
  std::string firmware;
  std::string chip;
  /** Advertised capability summary. */
  std::string capabilities;
  /** Service port. */
  uint16_t port{0U};
  /** IPv4 addresses in network byte order. */
  std::vector<uint32_t> ipv4_addresses;
};

/** Validated result of one discovery search. */
struct PeerDiscoverySnapshot {
  /** Search duration. */
  uint32_t elapsed_ms{0U};
  /** Whether the search ended before it completed. */
  bool timed_out{false};
  /** Whether devices or addresses were dropped to respect the limits. */
  bool truncated{false};
  /** Results rejected as invalid, off-link, or conflicting with another device. */
  size_t rejected_results{0U};
  /** Accepted devices, sorted by device id. */
  std::vector<PeerDiscoveryCandidate> devices;
};

/**
 * Validate, deduplicate, sort, and bound one discovery result.
 *
 * Keeps only addresses on the station's subnet, merges results that share a
 * device id and endpoint, and rejects device ids advertised by conflicting
 * endpoints. Addresses are in network byte order.
 */
PeerDiscoverySnapshot validate_peer_discovery_candidates(
    const std::vector<PeerDiscoveryCandidate> &candidates,
    uint32_t station_address,
    uint32_t station_netmask,
    uint32_t elapsed_ms,
    bool timed_out);
/**
 * Serialize a snapshot as the `devices` resource.
 *
 * Stays within `ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE`, dropping devices and
 * marking the result truncated when needed.
 */
std::string peer_discovery_snapshot_json(const PeerDiscoverySnapshot &snapshot);

/**
 * Finds other ESPectre devices on the local network.
 *
 * One search runs at a time. Call every method from the owner task.
 */
class IPeerDiscoveryService {
 public:
  /** Receives the result of a search; delivered from loop(). */
  using Completion = std::function<void(PeerDiscoverySnapshot snapshot)>;

  virtual ~IPeerDiscoveryService() = default;
  /** Describe this device, so it appears in its own results. */
  virtual void set_local_candidate(PeerDiscoveryCandidate candidate) {
    (void) candidate;
  }
  /** Report whether the station has an IPv4 address; searches need one. */
  virtual void set_wifi_ready(bool ready) = 0;
  /** Whether a search can start now. */
  virtual bool ready() const = 0;
  /** Whether a search is running. */
  virtual bool active() const = 0;
  /** Start a search; false when not ready or `completion` is empty. */
  virtual bool start(Completion completion) = 0;
  /** Collect results and deliver the completion when the search ends. */
  virtual void loop() = 0;
  /** Stop a running search without delivering its completion. */
  virtual void shutdown() = 0;
};

}  // namespace espectre
