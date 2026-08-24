/*
 * ESPectre - ESP-IDF Peer Discovery Service
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "peer_discovery_service_esp_idf.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <esp_log.h>
#include <esp_netif.h>
#include <esp_timer.h>

namespace espectre {

namespace {

[[maybe_unused]] const char *const TAG = "espectre.peers";
constexpr const char *kStaNetifKey = "WIFI_STA_DEF";
constexpr size_t kMaxRawResults = ESPECTRE_PEER_DISCOVERY_MAX_DEVICES * 2U;

std::string text(const char *value, size_t maximum) {
  if (value == nullptr) {
    return {};
  }
  const size_t length = strnlen(value, maximum + 1U);
  return length <= maximum ? std::string(value, length) : std::string{};
}

std::string txt_value(const mdns_result_t *result, const char *key, size_t maximum) {
  if (result == nullptr || key == nullptr) {
    return {};
  }
  for (size_t index = 0U; index < result->txt_count; ++index) {
    if (result->txt[index].key == nullptr || std::strcmp(result->txt[index].key, key) != 0) {
      continue;
    }
    const size_t length = result->txt_value_len != nullptr
                              ? static_cast<size_t>(result->txt_value_len[index])
                              : strnlen(result->txt[index].value, maximum + 1U);
    if (result->txt[index].value == nullptr || length > maximum) {
      return {};
    }
    return std::string(result->txt[index].value, length);
  }
  return {};
}

std::vector<PeerDiscoveryCandidate> copy_candidates(mdns_result_t *results) {
  std::vector<PeerDiscoveryCandidate> candidates;
  candidates.reserve(ESPECTRE_PEER_DISCOVERY_MAX_DEVICES);
  for (mdns_result_t *result = results; result != nullptr; result = result->next) {
    if (result->service_type == nullptr || result->proto == nullptr ||
        std::strcmp(result->service_type, "_espectre") != 0 || std::strcmp(result->proto, "_tcp") != 0) {
      continue;
    }
    PeerDiscoveryCandidate candidate;
    candidate.instance = text(result->instance_name, 63U);
    candidate.hostname = text(result->hostname, 63U);
    candidate.device_id = txt_value(result, "device_id", 16U);
    candidate.name = txt_value(result, "name", 63U);
    candidate.frontend = txt_value(result, "frontend", 16U);
    candidate.txt_version = txt_value(result, "txtvers", 8U);
    candidate.protocol_version = txt_value(result, "protovers", 8U);
    candidate.path = txt_value(result, "path", 64U);
    candidate.firmware = txt_value(result, "firmware", 48U);
    candidate.chip = txt_value(result, "chip", 16U);
    candidate.tls = txt_value(result, "tls", 8U);
    candidate.capabilities = txt_value(result, "capabilities", 128U);
    candidate.port = result->port;
    for (mdns_ip_addr_t *address = result->addr; address != nullptr; address = address->next) {
      if (address->addr.type == ESP_IPADDR_TYPE_V4) {
        candidate.ipv4_addresses.push_back(address->addr.u_addr.ip4.addr);
      }
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

}  // namespace

EspIdfPeerDiscoveryService::~EspIdfPeerDiscoveryService() { shutdown(); }

void EspIdfPeerDiscoveryService::set_local_candidate(PeerDiscoveryCandidate candidate) {
  local_candidate_ = std::move(candidate);
}

void EspIdfPeerDiscoveryService::set_wifi_ready(bool ready) { wifi_ready_ = ready; }

bool EspIdfPeerDiscoveryService::ready() const { return wifi_ready_ && search_ == nullptr; }

bool EspIdfPeerDiscoveryService::active() const { return search_ != nullptr; }

bool EspIdfPeerDiscoveryService::start(Completion completion) {
  if (!ready() || !completion) {
    return false;
  }
  search_ = mdns_query_async_new(nullptr,
                                 "_espectre",
                                 "_tcp",
                                 MDNS_TYPE_PTR,
                                 ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS,
                                 kMaxRawResults,
                                 nullptr);
  if (search_ == nullptr) {
    ESP_LOGW(TAG, "Failed to start peer DNS-SD query");
    return false;
  }
  completion_ = std::move(completion);
  started_us_ = esp_timer_get_time();
  return true;
}

void EspIdfPeerDiscoveryService::loop() {
  if (search_ == nullptr) {
    return;
  }
  mdns_result_t *results = nullptr;
  if (!mdns_query_async_get_results(search_, 0U, &results, nullptr)) {
    return;
  }
  finish_(results, wifi_ready_);
}

void EspIdfPeerDiscoveryService::shutdown() {
  wifi_ready_ = false;
  if (search_ == nullptr) {
    completion_ = {};
    return;
  }
  const int64_t elapsed_us = std::max<int64_t>(0, esp_timer_get_time() - started_us_);
  const uint32_t elapsed_ms = static_cast<uint32_t>(elapsed_us / 1000LL);
  const uint32_t remaining_ms = elapsed_ms < ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS
                                    ? ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS - elapsed_ms + 10U
                                    : 0U;
  mdns_result_t *results = nullptr;
  (void) mdns_query_async_get_results(search_, remaining_ms, &results, nullptr);
  finish_(results, false);
}

void EspIdfPeerDiscoveryService::finish_(mdns_result_t *results, bool deliver) {
  const int64_t elapsed_us = std::max<int64_t>(0, esp_timer_get_time() - started_us_);
  const uint32_t elapsed_ms = static_cast<uint32_t>(elapsed_us / 1000LL);
  esp_netif_ip_info_t ip_info{};
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey(kStaNetifKey);
  const bool have_ip = netif != nullptr && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK;
  std::vector<PeerDiscoveryCandidate> candidates = copy_candidates(results);
  if (have_ip && !local_candidate_.device_id.empty()) {
    PeerDiscoveryCandidate local_candidate = local_candidate_;
    local_candidate.ipv4_addresses = {ip_info.ip.addr};
    candidates.push_back(std::move(local_candidate));
  }
  PeerDiscoverySnapshot snapshot = validate_peer_discovery_candidates(
      candidates,
      have_ip ? ip_info.ip.addr : 0U,
      have_ip ? ip_info.netmask.addr : 0U,
      elapsed_ms,
      false);

  mdns_query_results_free(results);
  const esp_err_t delete_result = mdns_query_async_delete(search_);
  if (delete_result != ESP_OK) {
    ESP_LOGW(TAG, "Failed to release peer query: %s", esp_err_to_name(delete_result));
  }
  search_ = nullptr;
  started_us_ = 0;
  Completion completion = std::move(completion_);
  completion_ = {};
  if (deliver && completion) {
    completion(std::move(snapshot));
  }
}

}  // namespace espectre
