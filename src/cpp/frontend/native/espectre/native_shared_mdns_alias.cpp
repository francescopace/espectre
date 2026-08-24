/*
 * ESPectre - Native Shared mDNS Bootstrap Alias
 *
 * This extension reuses the Espressif mDNS socket and responder. The public
 * delegated-host API owns the record, while a link wrapper clears the cache-
 * flush bit only for the shared Native bootstrap hostname.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_shared_mdns_alias.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>

#include <esp_log.h>
#include <mdns.h>

#include "mdns_netif.h"
#include "mdns_networking.h"
#include "mdns_responder.h"
#include "mdns_send.h"

namespace {

[[maybe_unused]] const char *const TAG = "espectre.alias";
constexpr uint32_t SHARED_ALIAS_TTL_SECONDS = 10U;
std::array<char, MDNS_NAME_BUF_LEN> g_shared_hostname{};
bool g_shared_alias_active = false;
bool g_shared_alias_goodbye = false;

uint16_t read_u16(const uint8_t *data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8U) | data[1]);
}

uint32_t read_u32(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24U) | (static_cast<uint32_t>(data[1]) << 16U) |
         (static_cast<uint32_t>(data[2]) << 8U) | data[3];
}

void write_u32(uint8_t *data, uint32_t value) {
  data[0] = static_cast<uint8_t>(value >> 24U);
  data[1] = static_cast<uint8_t>(value >> 16U);
  data[2] = static_cast<uint8_t>(value >> 8U);
  data[3] = static_cast<uint8_t>(value);
}

bool decode_name(const uint8_t *packet,
                 size_t length,
                 size_t offset,
                 std::string *name,
                 size_t *next_offset) {
  if (packet == nullptr || name == nullptr || next_offset == nullptr || offset >= length) {
    return false;
  }
  name->clear();
  size_t cursor = offset;
  size_t consumed = 0U;
  size_t jumps = 0U;
  bool jumped = false;
  while (cursor < length && jumps <= 16U) {
    const uint8_t label_length = packet[cursor];
    if (label_length == 0U) {
      if (!jumped) consumed += 1U;
      *next_offset = offset + consumed;
      return true;
    }
    if ((label_length & 0xc0U) == 0xc0U) {
      if (cursor + 1U >= length) return false;
      if (!jumped) consumed += 2U;
      cursor = static_cast<size_t>(((label_length & 0x3fU) << 8U) | packet[cursor + 1U]);
      jumped = true;
      jumps += 1U;
      continue;
    }
    if ((label_length & 0xc0U) != 0U || label_length > 63U || cursor + 1U + label_length > length) {
      return false;
    }
    if (!name->empty()) name->push_back('.');
    for (size_t index = 0U; index < label_length; ++index) {
      name->push_back(static_cast<char>(std::tolower(packet[cursor + 1U + index])));
    }
    if (!jumped) consumed += 1U + label_length;
    cursor += 1U + label_length;
  }
  return false;
}

void normalize_shared_alias_records(uint8_t *packet, size_t length) {
  if (!g_shared_alias_active || packet == nullptr || length < 12U) {
    return;
  }
  const std::string expected = std::string(g_shared_hostname.data()) + ".local";
  const uint16_t questions = read_u16(packet + 4U);
  const size_t records = static_cast<size_t>(read_u16(packet + 6U)) + read_u16(packet + 8U) +
                         read_u16(packet + 10U);
  size_t offset = 12U;
  std::string name;
  for (uint16_t index = 0U; index < questions; ++index) {
    size_t next = 0U;
    if (!decode_name(packet, length, offset, &name, &next) || next + 4U > length) return;
    offset = next + 4U;
  }
  for (size_t index = 0U; index < records; ++index) {
    size_t next = 0U;
    if (!decode_name(packet, length, offset, &name, &next) || next + 10U > length) return;
    const uint16_t type = read_u16(packet + next);
    const size_t class_offset = next + 2U;
    const size_t ttl_offset = next + 4U;
    const uint16_t data_length = read_u16(packet + next + 8U);
    if (name == expected && (type == MDNS_TYPE_A || type == MDNS_TYPE_AAAA)) {
      packet[class_offset] &= 0x7fU;
      if (g_shared_alias_goodbye) {
        write_u32(packet + ttl_offset, 0U);
      } else if (read_u32(packet + ttl_offset) != 0U) {
        write_u32(packet + ttl_offset, SHARED_ALIAS_TTL_SECONDS);
      }
    }
    offset = next + 10U + data_length;
    if (offset > length) return;
  }
}

}  // namespace

extern "C" size_t __real_mdns_priv_if_write(mdns_if_t tcpip_if,
                                             mdns_ip_protocol_t ip_protocol,
                                             const esp_ip_addr_t *ip,
                                             uint16_t port,
                                             uint8_t *data,
                                             size_t len);

extern "C" size_t __wrap_mdns_priv_if_write(mdns_if_t tcpip_if,
                                             mdns_ip_protocol_t ip_protocol,
                                             const esp_ip_addr_t *ip,
                                             uint16_t port,
                                             uint8_t *data,
                                             size_t len) {
  normalize_shared_alias_records(data, len);
  return __real_mdns_priv_if_write(tcpip_if, ip_protocol, ip, port, data, len);
}

namespace espectre {

bool NativeSharedMdnsAlias::setup(const std::string &hostname) {
  shutdown();
  if (hostname.empty() || hostname.size() >= g_shared_hostname.size() ||
      !std::all_of(hostname.begin(), hostname.end(), [](unsigned char character) {
        return std::islower(character) || std::isdigit(character) || character == '-';
      })) {
    return false;
  }
  hostname_ = hostname;
  std::copy(hostname.begin(), hostname.end(), g_shared_hostname.begin());
  g_shared_hostname[hostname.size()] = '\0';
  return true;
}

bool NativeSharedMdnsAlias::update(uint32_t ipv4_address) {
  if (hostname_.empty() || ipv4_address == 0U) {
    shutdown();
    return false;
  }
  mdns_ip_addr_t address{};
  address.addr.type = ESP_IPADDR_TYPE_V4;
  address.addr.u_addr.ip4.addr = ipv4_address;
  esp_err_t result = ESP_OK;
  const bool first_publication = !published_;
  if (first_publication) {
    result = mdns_delegate_hostname_add(hostname_.c_str(), &address);
  } else if (address_ != ipv4_address) {
    (void) send_record_(true);
    result = mdns_delegate_hostname_set_address(hostname_.c_str(), &address);
  }
  if (result != ESP_OK) {
    ESP_LOGW(TAG, "Failed to publish shared alias: %s", esp_err_to_name(result));
    return false;
  }
  address_ = ipv4_address;
  published_ = true;
  g_shared_alias_active = true;
  if (send_record_(false)) {
    return true;
  }
  if (first_publication) {
    (void) mdns_delegate_hostname_remove(hostname_.c_str());
    address_ = 0U;
    published_ = false;
    g_shared_alias_active = false;
  }
  return false;
}

void NativeSharedMdnsAlias::shutdown() {
  if (published_) {
    [[maybe_unused]] const bool goodbye_sent = send_record_(true);
    ESP_LOGI(TAG, "Shared alias goodbye %s", goodbye_sent ? "sent" : "unavailable");
    const esp_err_t result = mdns_delegate_hostname_remove(hostname_.c_str());
    if (result != ESP_OK && result != ESP_ERR_NOT_FOUND) {
      ESP_LOGW(TAG, "Failed to remove shared alias: %s", esp_err_to_name(result));
    }
  }
  published_ = false;
  address_ = 0U;
  g_shared_alias_active = false;
  g_shared_hostname.fill('\0');
  hostname_.clear();
}

bool NativeSharedMdnsAlias::send_record_(bool goodbye) {
  mdns_host_item_t *host = mdns_priv_get_hosts();
  while (host != nullptr && std::strcmp(host->hostname, hostname_.c_str()) != 0) {
    host = host->next;
  }
  if (host == nullptr) {
    return false;
  }
  bool sent = false;
  for (mdns_if_t interface = 0U; interface < MDNS_MAX_INTERFACES; ++interface) {
    if (!mdns_priv_if_ready(interface, MDNS_IP_PROTOCOL_V4)) continue;
    mdns_tx_packet_t *packet = mdns_priv_alloc_packet(interface, MDNS_IP_PROTOCOL_V4);
    if (packet == nullptr) continue;
    packet->flags = MDNS_FLAGS_QR_AUTHORITATIVE;
    if (!mdns_priv_create_answer(&packet->answers, MDNS_TYPE_A, nullptr, host, false, goodbye)) {
      mdns_priv_free_tx_packet(packet);
      continue;
    }
    g_shared_alias_goodbye = goodbye;
    mdns_priv_dispatch_tx_packet(packet);
    g_shared_alias_goodbye = false;
    mdns_priv_free_tx_packet(packet);
    sent = true;
  }
  return sent;
}

}  // namespace espectre
