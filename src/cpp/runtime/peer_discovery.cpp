/*
 * ESPectre - Peer-Assisted Local Discovery
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "peer_discovery.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

#include "direct_http_protocol.h"
#include "protocol_json.h"

namespace espectre {

namespace {

constexpr size_t kMaxInstanceLength = 63U;
constexpr size_t kMaxHostnameLength = 63U;
constexpr size_t kMaxNameLength = 63U;
constexpr size_t kMaxFrontendLength = 16U;
constexpr size_t kMaxFirmwareLength = 48U;
constexpr size_t kMaxChipLength = 16U;
constexpr size_t kMaxCapabilitiesLength = 128U;
constexpr size_t kMaxCapabilities = 8U;

bool printable_text(const std::string &value, size_t max_length, bool allow_empty = false) {
  if ((!allow_empty && value.empty()) || value.size() > max_length) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= 0x20U && character <= 0x7eU;
  });
}

bool token(const std::string &value, size_t max_length) {
  return !value.empty() && value.size() <= max_length &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isalnum(character) || character == '-' || character == '_';
         });
}

bool device_id(const std::string &value) {
  return value.size() == 16U && std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return std::isdigit(character) || (character >= 'a' && character <= 'f');
  });
}

bool on_link_unicast(uint32_t address, uint32_t station_address, uint32_t netmask) {
  if (address == 0U || address == 0xffffffffU || station_address == 0U || netmask == 0U ||
      (address & netmask) != (station_address & netmask)) {
    return false;
  }
  const uint32_t host_bits = address & ~netmask;
  if (host_bits == 0U || host_bits == ~netmask) {
    return false;
  }
  const uint8_t first = static_cast<uint8_t>(address & 0xffU);
  return first != 0U && first != 127U && first < 224U;
}

std::vector<std::string> capability_tokens(const std::string &value) {
  std::vector<std::string> out;
  size_t start = 0U;
  while (start <= value.size()) {
    const size_t end = value.find(',', start);
    const std::string item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!token(item, 32U) || out.size() >= kMaxCapabilities ||
        std::find(out.begin(), out.end(), item) != out.end()) {
      return {};
    }
    out.push_back(item);
    if (end == std::string::npos) {
      break;
    }
    start = end + 1U;
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool valid_candidate(const PeerDiscoveryCandidate &candidate) {
  return printable_text(candidate.instance, kMaxInstanceLength) && token(candidate.hostname, kMaxHostnameLength) &&
         device_id(candidate.device_id) && printable_text(candidate.name, kMaxNameLength, true) &&
         token(candidate.frontend, kMaxFrontendLength) &&
         (candidate.frontend == "native" || candidate.frontend == "streamer" ||
          candidate.frontend == "esphome" || candidate.frontend == "matter") &&
         candidate.txt_version == ESPECTRE_DIRECT_DISCOVERY_TXT_VERSION &&
         candidate.protocol_version == "1" &&
         candidate.transport == ESPECTRE_DIRECT_HTTP_TRANSPORT &&
         candidate.path == ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT &&
         candidate.events == ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT &&
         printable_text(candidate.firmware, kMaxFirmwareLength) && token(candidate.chip, kMaxChipLength) &&
         printable_text(candidate.capabilities, kMaxCapabilitiesLength) &&
         !capability_tokens(candidate.capabilities).empty() && candidate.port != 0U;
}

std::string ipv4_text(uint32_t address) {
  return std::to_string(address & 0xffU) + "." + std::to_string((address >> 8U) & 0xffU) + "." +
         std::to_string((address >> 16U) & 0xffU) + "." + std::to_string((address >> 24U) & 0xffU);
}

std::string device_json(const PeerDiscoveryCandidate &device) {
  std::string out{"{"};
  append_json_pair(&out, "device_id", device.device_id.c_str(), true);
  append_json_pair(&out, "instance", device.instance.c_str());
  append_json_pair(&out, "hostname", device.hostname.c_str());
  append_json_pair(&out, "name", device.name.c_str());
  append_json_pair(&out, "frontend", device.frontend.c_str());
  out += ",\"schema_version\":2,\"txt_version\":2,\"protocol_version\":1";
  append_json_pair(&out, "transport", device.transport.c_str());
  append_json_pair(&out, "path", device.path.c_str());
  append_json_pair(&out, "events", device.events.c_str());
  append_json_pair(&out, "firmware", device.firmware.c_str());
  append_json_pair(&out, "chip", device.chip.c_str());
  out += ",\"port\":" + std::to_string(device.port) + ",\"capabilities\":[";
  const auto capabilities = capability_tokens(device.capabilities);
  for (size_t index = 0U; index < capabilities.size(); ++index) {
    if (index != 0U) out += ",";
    out += "\"" + capabilities[index] + "\"";
  }
  out += "],\"addresses\":[";
  for (size_t index = 0U; index < device.ipv4_addresses.size(); ++index) {
    if (index != 0U) out += ",";
    out += "\"" + ipv4_text(device.ipv4_addresses[index]) + "\"";
  }
  out += "]}";
  return out;
}

}  // namespace

PeerDiscoverySnapshot validate_peer_discovery_candidates(
    const std::vector<PeerDiscoveryCandidate> &candidates,
    uint32_t station_address,
    uint32_t station_netmask,
    uint32_t elapsed_ms,
    bool timed_out) {
  PeerDiscoverySnapshot snapshot;
  snapshot.elapsed_ms = elapsed_ms;
  snapshot.timed_out = timed_out;

  std::map<std::string, PeerDiscoveryCandidate> accepted;
  std::set<std::string> conflicted;
  for (PeerDiscoveryCandidate candidate : candidates) {
    if (!valid_candidate(candidate)) {
      snapshot.rejected_results += 1U;
      continue;
    }
    candidate.ipv4_addresses.erase(
        std::remove_if(candidate.ipv4_addresses.begin(), candidate.ipv4_addresses.end(),
                       [station_address, station_netmask](uint32_t address) {
                         return !on_link_unicast(address, station_address, station_netmask);
                       }),
        candidate.ipv4_addresses.end());
    std::sort(candidate.ipv4_addresses.begin(), candidate.ipv4_addresses.end());
    candidate.ipv4_addresses.erase(
        std::unique(candidate.ipv4_addresses.begin(), candidate.ipv4_addresses.end()),
        candidate.ipv4_addresses.end());
    if (candidate.ipv4_addresses.size() > ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES) {
      candidate.ipv4_addresses.resize(ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES);
      snapshot.truncated = true;
    }
    if (candidate.ipv4_addresses.empty()) {
      snapshot.rejected_results += 1U;
      continue;
    }

    const auto existing = accepted.find(candidate.device_id);
    if (existing == accepted.end()) {
      accepted.emplace(candidate.device_id, std::move(candidate));
      continue;
    }
    PeerDiscoveryCandidate &current = existing->second;
    const bool same_endpoint = current.hostname == candidate.hostname && current.frontend == candidate.frontend &&
                               current.port == candidate.port && current.path == candidate.path;
    if (!same_endpoint) {
      conflicted.insert(candidate.device_id);
      continue;
    }
    current.ipv4_addresses.insert(current.ipv4_addresses.end(),
                                  candidate.ipv4_addresses.begin(), candidate.ipv4_addresses.end());
    std::sort(current.ipv4_addresses.begin(), current.ipv4_addresses.end());
    current.ipv4_addresses.erase(
        std::unique(current.ipv4_addresses.begin(), current.ipv4_addresses.end()),
        current.ipv4_addresses.end());
    if (current.ipv4_addresses.size() > ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES) {
      current.ipv4_addresses.resize(ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES);
      snapshot.truncated = true;
    }
  }
  for (const std::string &identity : conflicted) {
    accepted.erase(identity);
    snapshot.rejected_results += 1U;
  }
  for (auto &entry : accepted) {
    if (snapshot.devices.size() >= ESPECTRE_PEER_DISCOVERY_MAX_DEVICES) {
      snapshot.truncated = true;
      break;
    }
    snapshot.devices.push_back(std::move(entry.second));
  }
  return snapshot;
}

std::string peer_discovery_snapshot_json(const PeerDiscoverySnapshot &snapshot) {
  std::string out{"{\"schema_version\":2,\"elapsed_ms\":"};
  out += std::to_string(snapshot.elapsed_ms);
  out += ",\"status\":\"";
  out += snapshot.timed_out ? "timeout" : "complete";
  out += "\",\"truncated\":";
  out += snapshot.truncated ? "true" : "false";
  out += ",\"rejected_results\":" + std::to_string(snapshot.rejected_results) + ",\"devices\":[";
  bool first = true;
  bool size_truncated = false;
  for (const PeerDiscoveryCandidate &device : snapshot.devices) {
    const std::string item = device_json(device);
    if (out.size() + item.size() + 2U > ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE) {
      size_truncated = true;
      break;
    }
    if (!first) out += ",";
    out += item;
    first = false;
  }
  out += "]}";
  if (size_truncated && !snapshot.truncated) {
    const std::string marker = "\"truncated\":false";
    const size_t position = out.find(marker);
    if (position != std::string::npos) {
      out.replace(position, marker.size(), "\"truncated\":true");
    }
  }
  return out;
}

}  // namespace espectre
