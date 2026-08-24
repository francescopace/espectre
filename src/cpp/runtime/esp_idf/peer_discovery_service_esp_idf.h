/*
 * ESPectre - ESP-IDF Peer Discovery Service
 *
 * One asynchronous, bounded DNS-SD query using the existing mDNS responder.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <mdns.h>

#include "peer_discovery.h"

namespace espectre {

class EspIdfPeerDiscoveryService final : public IPeerDiscoveryService {
 public:
  ~EspIdfPeerDiscoveryService() override;

  void set_local_candidate(PeerDiscoveryCandidate candidate);
  void set_wifi_ready(bool ready) override;
  bool ready() const override;
  bool active() const override;
  bool start(Completion completion) override;
  void loop() override;
  void shutdown() override;

 private:
  void finish_(mdns_result_t *results, bool deliver);

  mdns_search_once_t *search_{nullptr};
  Completion completion_{};
  PeerDiscoveryCandidate local_candidate_{};
  int64_t started_us_{0};
  bool wifi_ready_{false};
};

}  // namespace espectre
