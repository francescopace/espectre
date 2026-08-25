/*
 * ESPectre - Shared mDNS Bootstrap Responder
 *
 * This extension observes bootstrap mDNS questions before the Espressif responder
 * filters unregistered hostnames. Matching one-shot bootstrap questions are
 * answered through the responder's existing socket without registering or
 * retaining the queried nonce.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mdns_bootstrap_responder.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>

#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <mdns.h>

#include "mdns_networking.h"
#include "mdns_private.h"

namespace {

[[maybe_unused]] const char *const TAG = "espectre.bootstrap";
constexpr char BOOTSTRAP_PREFIX[] = "espectre-devices-";
constexpr char LOCAL_LABEL[] = "local";
constexpr uint16_t DNS_HEADER_SIZE = 12U;
constexpr uint16_t DNS_FLAG_RESPONSE_AUTHORITATIVE = 0x8400U;
constexpr uint16_t DNS_FLAG_RESPONSE = 0x8000U;
constexpr uint16_t DNS_FLAG_TRUNCATED = 0x0200U;
constexpr uint16_t DNS_OPCODE_MASK = 0x7800U;
constexpr uint16_t DNS_CLASS_IN = 0x0001U;
constexpr uint16_t DNS_CLASS_UNICAST_RESPONSE = 0x8000U;
constexpr uint16_t DNS_TYPE_A = 0x0001U;
constexpr uint16_t DNS_TYPE_AAAA = 0x001cU;
constexpr uint16_t DNS_TYPE_NSEC = 0x002fU;
constexpr uint16_t MDNS_PORT = 5353U;
constexpr int64_t RATE_WINDOW_US = 1000000;
constexpr int64_t RESPONSE_DELAY_STEP_US = 25000;
constexpr uint32_t MDNS_MULTICAST_IPV4 =
    static_cast<uint32_t>(224U) | (static_cast<uint32_t>(251U) << 24U);

std::atomic<espectre::MdnsBootstrapResponder *> g_bootstrap_responder{nullptr};

uint16_t read_u16(const uint8_t *data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8U) | data[1]);
}

void write_u16(uint8_t *data, uint16_t value) {
  data[0] = static_cast<uint8_t>(value >> 8U);
  data[1] = static_cast<uint8_t>(value);
}

void write_u32(uint8_t *data, uint32_t value) {
  data[0] = static_cast<uint8_t>(value >> 24U);
  data[1] = static_cast<uint8_t>(value >> 16U);
  data[2] = static_cast<uint8_t>(value >> 8U);
  data[3] = static_cast<uint8_t>(value);
}

bool equal_ascii_case_insensitive(const char *actual, size_t length, const char *expected) {
  const size_t expected_length = std::strlen(expected);
  if (length != expected_length) return false;
  for (size_t index = 0U; index < length; ++index) {
    if (std::tolower(static_cast<unsigned char>(actual[index])) !=
        std::tolower(static_cast<unsigned char>(expected[index]))) {
      return false;
    }
  }
  return true;
}

bool valid_bootstrap_label(const char *label, size_t length) {
  constexpr size_t prefix_length = sizeof(BOOTSTRAP_PREFIX) - 1U;
  if (length != prefix_length + espectre::MdnsBootstrapResponder::NONCE_HEX_LENGTH ||
      !equal_ascii_case_insensitive(label, prefix_length, BOOTSTRAP_PREFIX)) {
    return false;
  }
  for (size_t index = prefix_length; index < length; ++index) {
    const unsigned char character = static_cast<unsigned char>(label[index]);
    if (!std::isxdigit(character)) return false;
  }
  return true;
}

struct ParsedQuestion {
  std::array<char, 64U> host{};
  size_t host_length{0U};
  uint16_t type{0U};
  uint16_t clas{0U};
  size_t next_offset{0U};
  bool matchable_name{false};
};

bool parse_question(const uint8_t *packet,
                    size_t length,
                    size_t offset,
                    ParsedQuestion *question) {
  if (packet == nullptr || question == nullptr || offset >= length) return false;
  *question = {};
  bool uncompressed = true;
  bool exact_name = true;
  size_t label_count = 0U;
  while (true) {
    if (offset >= length) return false;
    const uint8_t label_length = packet[offset++];
    if (label_length == 0U) break;
    if ((label_length & 0xc0U) == 0xc0U) {
      if (offset >= length) return false;
      const size_t pointer =
          (static_cast<size_t>(label_length & 0x3fU) << 8U) | packet[offset++];
      if (pointer >= length) return false;
      uncompressed = false;
      break;
    }
    if ((label_length & 0xc0U) != 0U || offset + label_length > length) return false;
    if (label_count == 0U) {
      std::memcpy(question->host.data(), packet + offset, label_length);
      question->host_length = label_length;
    } else if (label_count == 1U) {
      exact_name = equal_ascii_case_insensitive(
          reinterpret_cast<const char *>(packet + offset), label_length, LOCAL_LABEL);
    } else {
      exact_name = false;
    }
    offset += label_length;
    label_count += 1U;
  }
  if (offset + 4U > length) return false;
  question->type = read_u16(packet + offset);
  question->clas = read_u16(packet + offset + 2U);
  question->next_offset = offset + 4U;
  question->matchable_name =
      uncompressed && exact_name && label_count == 2U && question->host_length != 0U;
  return true;
}

size_t append_name(uint8_t *destination,
                   size_t capacity,
                   size_t offset,
                   const char *host,
                   size_t host_length) {
  constexpr size_t local_length = sizeof(LOCAL_LABEL) - 1U;
  if (destination == nullptr || host == nullptr || host_length > 63U ||
      offset + 1U + host_length + 1U + local_length + 1U > capacity) {
    return 0U;
  }
  destination[offset++] = static_cast<uint8_t>(host_length);
  std::memcpy(destination + offset, host, host_length);
  offset += host_length;
  destination[offset++] = static_cast<uint8_t>(local_length);
  std::memcpy(destination + offset, LOCAL_LABEL, local_length);
  offset += local_length;
  destination[offset++] = 0U;
  return offset;
}

size_t append_nsec_record(const ParsedQuestion &question,
                          uint8_t *destination,
                          size_t capacity,
                          size_t offset) {
  offset = append_name(destination, capacity, offset, question.host.data(), question.host_length);
  if (offset == 0U || offset + 10U > capacity) return 0U;
  write_u16(destination + offset, DNS_TYPE_NSEC);
  write_u16(destination + offset + 2U, DNS_CLASS_IN);
  write_u32(destination + offset + 4U,
            espectre::MdnsBootstrapResponder::RESPONSE_TTL_SECONDS);
  const size_t length_offset = offset + 8U;
  const size_t rdata_offset = offset + 10U;
  offset = append_name(
      destination, capacity, rdata_offset, question.host.data(), question.host_length);
  if (offset == 0U || offset + 3U > capacity) return 0U;
  destination[offset++] = 0U;     // Type bitmap window 0.
  destination[offset++] = 1U;     // One bitmap byte covers types 0-7.
  destination[offset++] = 0x40U;  // Type A exists; type AAAA does not.
  write_u16(destination + length_offset, static_cast<uint16_t>(offset - rdata_offset));
  return offset;
}

size_t build_response(const ParsedQuestion &question,
                      uint16_t query_id,
                      uint32_t ipv4_address,
                      bool legacy_unicast,
                      uint8_t *destination,
                      size_t capacity) {
  if (destination == nullptr || capacity < DNS_HEADER_SIZE) return 0U;
  std::memset(destination, 0, capacity);
  const bool address_answer = question.type == DNS_TYPE_A;
  write_u16(destination, legacy_unicast ? query_id : 0U);
  write_u16(destination + 2U, DNS_FLAG_RESPONSE_AUTHORITATIVE);
  write_u16(destination + 4U, legacy_unicast ? 1U : 0U);
  write_u16(destination + 6U, 1U);
  write_u16(destination + 10U, address_answer ? 1U : 0U);
  size_t offset = DNS_HEADER_SIZE;
  if (legacy_unicast) {
    offset = append_name(
        destination, capacity, offset, question.host.data(), question.host_length);
    if (offset == 0U || offset + 4U > capacity) return 0U;
    write_u16(destination + offset, question.type);
    write_u16(destination + offset + 2U, DNS_CLASS_IN);
    offset += 4U;
  }
  if (address_answer) {
    offset = append_name(destination, capacity, offset, question.host.data(), question.host_length);
    if (offset == 0U || offset + 14U > capacity) return 0U;
    write_u16(destination + offset, DNS_TYPE_A);
    write_u16(destination + offset + 2U, DNS_CLASS_IN);
    write_u32(destination + offset + 4U,
              espectre::MdnsBootstrapResponder::RESPONSE_TTL_SECONDS);
    write_u16(destination + offset + 8U, 4U);
    std::memcpy(destination + offset + 10U, &ipv4_address, sizeof(ipv4_address));
    offset += 14U;
  }
  return append_nsec_record(question, destination, capacity, offset);
}

}  // namespace

extern "C" void __real_mdns_priv_receive_action(mdns_action_t *action,
                                                  mdns_action_subtype_t type);

extern "C" void __wrap_mdns_priv_receive_action(mdns_action_t *action,
                                                  mdns_action_subtype_t type) {
  espectre::MdnsBootstrapResponder *responder = g_bootstrap_responder.load();
  if (responder != nullptr && action != nullptr && type == ACTION_RUN &&
      action->type == ACTION_RX_HANDLE && action->data.rx_handle.packet != nullptr) {
    mdns_rx_packet_t *packet = action->data.rx_handle.packet;
    if (packet->ip_protocol == MDNS_IP_PROTOCOL_V4) {
      responder->ingest_query(
          static_cast<const uint8_t *>(mdns_priv_get_packet_data(packet)),
          mdns_priv_get_packet_len(packet),
          packet->tcpip_if,
          packet->src.u_addr.ip4.addr,
          packet->src_port);
    }
  }
  __real_mdns_priv_receive_action(action, type);
}

namespace espectre {

MdnsBootstrapResponder::~MdnsBootstrapResponder() {
  shutdown();
  if (mutex_ != nullptr) {
    vSemaphoreDelete(static_cast<SemaphoreHandle_t>(mutex_));
    mutex_ = nullptr;
  }
}

bool MdnsBootstrapResponder::setup() {
  shutdown();
  if (mutex_ == nullptr) {
    mutex_ = xSemaphoreCreateMutex();
    if (mutex_ == nullptr) {
      ESP_LOGE(TAG, "Failed to allocate bootstrap responder mutex");
      return false;
    }
  }
  MdnsBootstrapResponder *owner = nullptr;
  if (!g_bootstrap_responder.compare_exchange_strong(owner, this) && owner != this) {
    ESP_LOGE(TAG, "Another bootstrap responder is already active");
    return false;
  }
  configured_ = true;
  return true;
}

bool MdnsBootstrapResponder::update(uint32_t ipv4_address) {
  if (!configured_ || mutex_ == nullptr) return false;
  xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
  if (ipv4_address_.load() != ipv4_address) {
    clear_pending_();
    ipv4_address_ = ipv4_address;
    response_time_count_ = 0U;
  }
  xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
  return true;
}

void MdnsBootstrapResponder::ingest_query(const uint8_t *packet,
                                                 size_t length,
                                                 size_t interface,
                                                 uint32_t source_ipv4,
                                                 uint16_t source_port) {
  if (mutex_ == nullptr || packet == nullptr || length < DNS_HEADER_SIZE) return;
  const uint16_t flags = read_u16(packet + 2U);
  if ((flags & (DNS_FLAG_RESPONSE | DNS_FLAG_TRUNCATED | DNS_OPCODE_MASK)) != 0U) return;
  const uint16_t question_count = read_u16(packet + 4U);
  if (question_count == 0U) return;

  ParsedQuestion match;
  bool found = false;
  size_t offset = DNS_HEADER_SIZE;
  for (uint16_t index = 0U; index < question_count; ++index) {
    ParsedQuestion question;
    if (!parse_question(packet, length, offset, &question)) return;
    offset = question.next_offset;
    if (question.matchable_name &&
        (question.type == DNS_TYPE_A || question.type == DNS_TYPE_AAAA) &&
        (question.clas & ~DNS_CLASS_UNICAST_RESPONSE) == DNS_CLASS_IN &&
        valid_bootstrap_label(question.host.data(), question.host_length)) {
      if (!found || question.type == DNS_TYPE_A) {
        match = question;
        found = true;
      }
    }
  }
  if (!found) return;

  const bool legacy_unicast = source_port != MDNS_PORT;
  const bool unicast_response = legacy_unicast || (match.clas & DNS_CLASS_UNICAST_RESPONSE) != 0U;
  const int64_t now_us = esp_timer_get_time();
  xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
  if (!active()) {
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  size_t retained_response_count = 0U;
  for (size_t index = 0U; index < response_time_count_; ++index) {
    if (now_us - response_times_[index] < RATE_WINDOW_US) {
      response_times_[retained_response_count++] = response_times_[index];
    }
  }
  response_time_count_ = retained_response_count;
  if (response_time_count_ >= MAX_RESPONSES_PER_SECOND) {
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  if (unicast_response) {
    std::array<uint8_t, MAX_RESPONSE_BYTES> response{};
    const size_t response_length = build_response(match,
                                                  read_u16(packet),
                                                  ipv4_address_.load(),
                                                  legacy_unicast,
                                                  response.data(),
                                                  response.size());
    if (response_length != 0U) {
      esp_ip_addr_t destination{};
      destination.type = ESP_IPADDR_TYPE_V4;
      destination.u_addr.ip4.addr = source_ipv4;
      (void) mdns_priv_if_write(static_cast<mdns_if_t>(interface),
                                MDNS_IP_PROTOCOL_V4,
                                &destination,
                                source_port,
                                response.data(),
                                response_length);
      response_times_[response_time_count_++] = now_us;
    }
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  const auto available = std::find_if(pending_.begin(), pending_.end(), [](const auto &response) {
    return !response.used;
  });
  if (available == pending_.end()) {
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  const size_t slot = static_cast<size_t>(available - pending_.begin());
  available->length = build_response(match,
                                     read_u16(packet),
                                     ipv4_address_.load(),
                                     legacy_unicast,
                                     available->bytes.data(),
                                     available->bytes.size());
  if (available->length == 0U) {
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  available->interface = interface;
  available->destination_ipv4 = MDNS_MULTICAST_IPV4;
  available->destination_port = MDNS_PORT;
  available->due_us = now_us + (static_cast<int64_t>(slot) + 1) * RESPONSE_DELAY_STEP_US;
  available->used = true;
  response_times_[response_time_count_++] = now_us;
  xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
}

void MdnsBootstrapResponder::loop() {
  if (mutex_ == nullptr) return;
  const int64_t now_us = esp_timer_get_time();
  xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
  if (!active()) {
    clear_pending_();
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  for (auto &response : pending_) {
    if (!response.used || response.due_us > now_us) continue;
    esp_ip_addr_t destination{};
    destination.type = ESP_IPADDR_TYPE_V4;
    destination.u_addr.ip4.addr = response.destination_ipv4;
    (void) mdns_priv_if_write(static_cast<mdns_if_t>(response.interface),
                              MDNS_IP_PROTOCOL_V4,
                              &destination,
                              response.destination_port,
                              response.bytes.data(),
                              response.length);
    response = {};
  }
  xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
}

void MdnsBootstrapResponder::shutdown() {
  if (mutex_ != nullptr) {
    xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
    clear_pending_();
    ipv4_address_ = 0U;
    response_time_count_ = 0U;
    configured_ = false;
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
  } else {
    configured_ = false;
    ipv4_address_ = 0U;
  }
  MdnsBootstrapResponder *owner = this;
  (void) g_bootstrap_responder.compare_exchange_strong(owner, nullptr);
}

void MdnsBootstrapResponder::clear_pending_() {
  for (auto &response : pending_) response = {};
}

}  // namespace espectre
