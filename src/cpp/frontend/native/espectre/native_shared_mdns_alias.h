/*
 * ESPectre - Native Shared mDNS Bootstrap Alias
 *
 * IPv4 publication of espectre-devices.local as a shared RRset.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

namespace espectre {

class NativeSharedMdnsAlias {
 public:
  bool setup(const std::string &hostname);
  bool update(uint32_t ipv4_address);
  void shutdown();
  bool published() const { return published_; }

 private:
  bool send_record_(bool goodbye);

  std::string hostname_;
  uint32_t address_{0U};
  bool published_{false};
};

}  // namespace espectre
