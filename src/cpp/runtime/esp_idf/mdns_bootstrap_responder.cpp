/*
 * ESPectre - Shared mDNS Bootstrap Responder
 *
 * This extension observes bootstrap mDNS questions before the Espressif responder
 * filters unregistered hostnames. Matching one-shot bootstrap questions are
 * answered through the responder's existing socket without registering or
 * retaining the queried nonce beyond bounded pending work and send history.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mdns_bootstrap_responder.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "core/espectre_log.h"
#include <esp_timer.h>
#include <esp_random.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
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
constexpr uint16_t DNS_RCODE_MASK = 0x000fU;
constexpr uint16_t DNS_CLASS_ANY = 0x00ffU;
constexpr uint16_t DNS_TYPE_ANY = 0x00ffU;
constexpr uint8_t RECORD_A = 1U;
constexpr uint8_t RECORD_NSEC = 2U;
constexpr size_t MAX_QUERY_BYTES = 9000U;
constexpr size_t MAX_PARSE_WORK = 4096U;
constexpr int64_t PENDING_LIFETIME_US = 3000000;
constexpr uint16_t DNS_CLASS_IN = 0x0001U;
constexpr uint16_t DNS_CLASS_UNICAST_RESPONSE = 0x8000U;
constexpr uint16_t DNS_TYPE_A = 0x0001U;
constexpr uint16_t DNS_TYPE_AAAA = 0x001cU;
constexpr uint16_t DNS_TYPE_NSEC = 0x002fU;
constexpr uint16_t MDNS_PORT = 5353U;
constexpr int64_t RATE_WINDOW_US = 1000000;
constexpr uint32_t MDNS_MULTICAST_IPV4 =
    static_cast<uint32_t>(224U) | (static_cast<uint32_t>(251U) << 24U);

std::atomic<espectre::MdnsBootstrapResponder *> g_bootstrap_responder{nullptr};
std::atomic<uint32_t> g_bootstrap_callbacks_in_flight{0U};

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

uint32_t read_u32(const uint8_t *data) {
  return (static_cast<uint32_t>(read_u16(data)) << 16U) | read_u16(data + 2U);
}

unsigned char ascii_lower(unsigned char c) {
  return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
}

bool equal_ascii_case_insensitive(const char *actual, size_t length, const char *expected) {
  if (length != std::strlen(expected)) return false;
  for (size_t index = 0U; index < length; ++index) {
    if (ascii_lower(actual[index]) != ascii_lower(expected[index])) return false;
  }
  return true;
}

bool same_host(const char *left, const char *right) {
  return equal_ascii_case_insensitive(left, std::strlen(left), right);
}

bool valid_bootstrap_label(const char *label, size_t length) {
  constexpr size_t prefix_length = sizeof(BOOTSTRAP_PREFIX) - 1U;
  if (length != prefix_length + espectre::MdnsBootstrapResponder::NONCE_HEX_LENGTH ||
      !equal_ascii_case_insensitive(label, prefix_length, BOOTSTRAP_PREFIX)) return false;
  for (size_t index = prefix_length; index < length; ++index) {
    const unsigned char c = ascii_lower(label[index]);
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

// Scale a uniformly distributed 32-bit sample in constant time. The interval includes both endpoints.
int64_t response_delay(bool truncated) {
  constexpr uint64_t range_us = 100001U;
  return (truncated ? 400000 : 20000) +
         static_cast<int64_t>((static_cast<uint64_t>(esp_random()) * range_us) >> 32U);
}

struct ParsedName {
  std::array<char, 64U> host{};
  std::array<char, 6U> domain{};
  size_t next_offset{0U};
  bool matchable{false};
};

// Decode only the two labels we serve, but validate the entire expanded name.
// A packet-wide work budget also bounds repeated compression chains.
bool parse_name(const uint8_t *packet, size_t length, size_t offset,
                size_t *work, ParsedName *name) {
  *name = {};
  size_t labels = 0U;
  size_t expanded = 1U;
  bool exact = true;
  bool jumped = false;
  while (offset < length && *work != 0U) {
    --*work;
    const size_t label_offset = offset;
    const uint8_t count = packet[offset++];
    if ((count & 0xc0U) == 0xc0U) {
      if (offset >= length) return false;
      const size_t pointer = (static_cast<size_t>(count & 0x3fU) << 8U) | packet[offset++];
      // RFC 1035 compression refers to a prior occurrence, never the header.
      if (pointer < DNS_HEADER_SIZE || pointer >= label_offset) return false;
      if (!jumped) name->next_offset = offset;
      jumped = true;
      offset = pointer;
      continue;
    }
    if ((count & 0xc0U) != 0U || offset + count > length) return false;
    if (count == 0U) {
      if (!jumped) name->next_offset = offset;
      name->matchable = exact && labels == 2U;
      return true;
    }
    expanded += count + 1U;
    if (expanded > 255U) return false;
    if (labels == 0U) {
      std::memcpy(name->host.data(), packet + offset, count);
      exact = valid_bootstrap_label(name->host.data(), count);
    } else if (labels == 1U) {
      exact = exact && equal_ascii_case_insensitive(
          reinterpret_cast<const char *>(packet + offset), count, LOCAL_LABEL);
      if (count == 5U) std::memcpy(name->domain.data(), packet + offset, count);
    } else {
      exact = false;
    }
    offset += count;
    ++labels;
  }
  return false;
}

struct ParsedQuestion {
  ParsedName name{};
  uint16_t type{0U};
  uint16_t clas{0U};
  uint8_t requested{0U};
};

bool parse_question(const uint8_t *packet, size_t length, size_t *offset,
                    size_t *work, ParsedQuestion *question) {
  *question = {};
  if (!parse_name(packet, length, *offset, work, &question->name)) return false;
  *offset = question->name.next_offset;
  if (*offset + 4U > length) return false;
  question->type = read_u16(packet + *offset);
  question->clas = read_u16(packet + *offset + 2U);
  *offset += 4U;
  const uint16_t clas = question->clas & ~DNS_CLASS_UNICAST_RESPONSE;
  if (question->name.matchable && (clas == DNS_CLASS_IN || clas == DNS_CLASS_ANY)) {
    if (question->type == DNS_TYPE_A) question->requested = RECORD_A;
    if (question->type == DNS_TYPE_AAAA) question->requested = RECORD_NSEC;
    if (question->type == DNS_TYPE_ANY) question->requested = RECORD_A | RECORD_NSEC;
  }
  return true;
}

struct ParsedRecord {
  ParsedName name{};
  uint16_t type{0U};
  uint16_t clas{0U};
  uint32_t ttl{0U};
  size_t rdata{0U};
  size_t end{0U};
  bool same_nsec{false};
};

bool parse_record(const uint8_t *packet, size_t length, size_t *offset,
                  size_t *work, ParsedRecord *record) {
  *record = {};
  if (!parse_name(packet, length, *offset, work, &record->name)) return false;
  *offset = record->name.next_offset;
  if (*offset + 10U > length) return false;
  record->type = read_u16(packet + *offset);
  record->clas = read_u16(packet + *offset + 2U) & ~DNS_CLASS_UNICAST_RESPONSE;
  record->ttl = read_u32(packet + *offset + 4U);
  record->rdata = *offset + 10U;
  record->end = record->rdata + read_u16(packet + *offset + 8U);
  if (record->end > length) return false;
  if (record->type == DNS_TYPE_NSEC) {
    ParsedName next;
    if (!parse_name(packet, length, record->rdata, work, &next) ||
        next.next_offset > record->end) return false;
    size_t bitmap = next.next_offset;
    // Our NSEC contains only A in window zero. Trailing zero bytes do not
    // change the record's meaning, although we emit the shortest bitmap.
    if (bitmap + 3U <= record->end && packet[bitmap] == 0U &&
        packet[bitmap + 1U] >= 1U && packet[bitmap + 1U] <= 32U &&
        bitmap + 2U + packet[bitmap + 1U] == record->end && packet[bitmap + 2U] == 0x40U) {
      record->same_nsec = next.matchable && record->name.matchable &&
                          same_host(next.host.data(), record->name.host.data());
      for (bitmap += 3U; bitmap < record->end; ++bitmap) {
        if (packet[bitmap] != 0U) record->same_nsec = false;
      }
    }
  }
  *offset = record->end;
  return true;
}

uint8_t known_record(const uint8_t *packet, const ParsedRecord &record, uint32_t ipv4_address) {
  if (!record.name.matchable || record.clas != DNS_CLASS_IN ||
      record.ttl < (espectre::MdnsBootstrapResponder::RESPONSE_TTL_SECONDS + 1U) / 2U) return 0U;
  if (record.type == DNS_TYPE_A && record.end - record.rdata == 4U &&
      std::memcmp(packet + record.rdata, &ipv4_address, 4U) == 0) return RECORD_A;
  return record.type == DNS_TYPE_NSEC && record.same_nsec ? RECORD_NSEC : 0U;
}

size_t append_name(uint8_t *destination, size_t capacity, size_t offset,
                   const char *host, const char *domain) {
  const size_t host_length = std::strlen(host);
  if (offset + host_length + 8U > capacity) return 0U;
  destination[offset++] = static_cast<uint8_t>(host_length);
  std::memcpy(destination + offset, host, host_length);
  offset += host_length;
  destination[offset++] = 5U;
  std::memcpy(destination + offset, domain, 5U);
  offset += 5U;
  destination[offset++] = 0U;
  return offset;
}

size_t build_response(const char *host, const char *domain, uint16_t query_type,
                      uint16_t query_class, uint16_t query_id, uint32_t ipv4_address,
                      bool legacy_unicast, uint8_t records, uint8_t *destination, size_t capacity) {
  if (capacity < DNS_HEADER_SIZE || records == 0U) return 0U;
  std::memset(destination, 0, capacity);
  const bool address_answer = (records & RECORD_A) != 0U;
  const bool nsec_answer = (records & RECORD_NSEC) != 0U;
  const bool additional_nsec = address_answer && nsec_answer && query_type != DNS_TYPE_ANY;
  write_u16(destination, legacy_unicast ? query_id : 0U);
  write_u16(destination + 2U, DNS_FLAG_RESPONSE_AUTHORITATIVE);
  write_u16(destination + 4U, legacy_unicast ? 1U : 0U);
  write_u16(destination + 6U, (address_answer ? 1U : 0U) + (nsec_answer && !additional_nsec ? 1U : 0U));
  write_u16(destination + 10U, additional_nsec ? 1U : 0U);
  size_t offset = DNS_HEADER_SIZE;
  if (legacy_unicast) {
    offset = append_name(destination, capacity, offset, host, domain);
    if (offset == 0U || offset + 4U > capacity) return 0U;
    write_u16(destination + offset, query_type);
    write_u16(destination + offset + 2U, query_class);
    offset += 4U;
  }
  if (address_answer) {
    offset = append_name(destination, capacity, offset, host, domain);
    if (offset == 0U || offset + 14U > capacity) return 0U;
    write_u16(destination + offset, DNS_TYPE_A);
    write_u16(destination + offset + 2U, DNS_CLASS_IN);
    write_u32(destination + offset + 4U, espectre::MdnsBootstrapResponder::RESPONSE_TTL_SECONDS);
    write_u16(destination + offset + 8U, 4U);
    std::memcpy(destination + offset + 10U, &ipv4_address, 4U);
    offset += 14U;
  }
  if (nsec_answer) {
    offset = append_name(destination, capacity, offset, host, domain);
    if (offset == 0U || offset + 10U > capacity) return 0U;
    write_u16(destination + offset, DNS_TYPE_NSEC);
    write_u16(destination + offset + 2U, DNS_CLASS_IN);
    write_u32(destination + offset + 4U, espectre::MdnsBootstrapResponder::RESPONSE_TTL_SECONDS);
    const size_t length_offset = offset + 8U;
    const size_t rdata = offset + 10U;
    offset = append_name(destination, capacity, rdata, host, domain);
    if (offset == 0U || offset + 3U > capacity) return 0U;
    destination[offset++] = 0U;
    destination[offset++] = 1U;
    destination[offset++] = 0x40U;
    write_u16(destination + length_offset, static_cast<uint16_t>(offset - rdata));
  }
  return offset;
}

}  // namespace

extern "C" void __real_mdns_priv_receive_action(mdns_action_t *action,
                                                  mdns_action_subtype_t type);

extern "C" void __wrap_mdns_priv_receive_action(mdns_action_t *action,
                                                  mdns_action_subtype_t type) {
  g_bootstrap_callbacks_in_flight.fetch_add(1U, std::memory_order_acquire);
  espectre::MdnsBootstrapResponder *responder =
      g_bootstrap_responder.load(std::memory_order_acquire);
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
  g_bootstrap_callbacks_in_flight.fetch_sub(1U, std::memory_order_release);
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
      ESPECTRE_LOGE(TAG, "Failed to allocate bootstrap responder mutex");
      return false;
    }
  }
  MdnsBootstrapResponder *owner = nullptr;
  if (!g_bootstrap_responder.compare_exchange_strong(owner, this) && owner != this) {
    ESPECTRE_LOGE(TAG, "Another bootstrap responder is already active");
    return false;
  }
  configured_ = true;
  return true;
}

bool MdnsBootstrapResponder::update(uint32_t ipv4_address) {
  if (!configured_ || mutex_ == nullptr) return false;
  if (ipv4_address_.load(std::memory_order_acquire) == ipv4_address) return true;
  bool changed = false;
  xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
  if (ipv4_address_.load(std::memory_order_relaxed) != ipv4_address) {
    clear_pending_();
    generation_.fetch_add(1U, std::memory_order_acq_rel);
    ipv4_address_.store(ipv4_address, std::memory_order_release);
    response_time_count_ = 0U;
    changed = true;
  }
  xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
  if (changed) wait_for_sends_();
  return true;
}

void MdnsBootstrapResponder::ingest_query(const uint8_t *packet,
                                         size_t length,
                                         size_t interface,
                                         uint32_t source_ipv4,
                                         uint16_t source_port) {
  if (mutex_ == nullptr || !active() || packet == nullptr ||
      length < DNS_HEADER_SIZE || length > MAX_QUERY_BYTES) return;
  const uint16_t flags = read_u16(packet + 2U);
  if ((flags & (DNS_FLAG_RESPONSE | DNS_OPCODE_MASK | DNS_RCODE_MASK)) != 0U) return;
  const bool truncated = (flags & DNS_FLAG_TRUNCATED) != 0U;
  const bool legacy = source_port != MDNS_PORT;
  const uint16_t question_count = read_u16(packet + 4U);
  const uint16_t answer_count = read_u16(packet + 6U);
  std::array<ParsedQuestion, MAX_PENDING_RESPONSES> questions{};
  size_t count = 0U;
  size_t offset = DNS_HEADER_SIZE;
  size_t work = MAX_PARSE_WORK;
  for (size_t index = 0U; index < question_count; ++index) {
    ParsedQuestion question;
    if (!parse_question(packet, length, &offset, &work, &question)) return;
    if (question.requested == 0U) continue;
    size_t slot = 0U;
    for (; slot < count; ++slot) {
      if (same_host(questions[slot].name.host.data(), question.name.host.data()) &&
          questions[slot].clas == question.clas && (!legacy || questions[slot].type == question.type)) break;
    }
    if (slot < count) {
      questions[slot].requested |= question.requested;
      if (question.type == DNS_TYPE_A || question.type == DNS_TYPE_ANY) questions[slot].type = question.type;
    } else if (count < questions.size()) {
      questions[count++] = question;
    }
  }
  const size_t answers_offset = offset;
  // Validate every section before publishing any work. Unknown records never
  // become Known Answers, and malformed tails cannot trigger partial replies.
  const size_t records = static_cast<size_t>(answer_count) + read_u16(packet + 8U) + read_u16(packet + 10U);
  for (size_t index = 0U; index < records; ++index) {
    ParsedRecord record;
    if (!parse_record(packet, length, &offset, &work, &record)) return;
  }
  if (offset != length) return;

  if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), 0U) != pdTRUE) {
    if (question_count == 0U) {
      defer_known_answers_(packet, length, answers_offset, answer_count, interface,
                           source_ipv4, source_port, truncated);
    }
    return;
  }
  if (!active()) {
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    return;
  }
  const int64_t now_us = esp_timer_get_time();
  apply_deferred_known_answers_();
  std::array<bool, MAX_PENDING_RESPONSES> affected{};
  for (size_t index = 0U; index < pending_.size(); ++index) {
    auto &response = pending_[index];
    if (response.used && response.expires_us <= now_us) response = {};
    // Questionless continuations belong only to this source's TC sequence.
    affected[index] = response.used && response.truncated && response.interface == interface &&
                      response.source_ipv4 == source_ipv4 && response.source_port == source_port;
  }
  for (size_t index = 0U; index < count; ++index) {
    const auto &question = questions[index];
    const bool unicast = legacy || (question.clas & DNS_CLASS_UNICAST_RESPONSE) != 0U;
    auto slot = std::find_if(pending_.begin(), pending_.end(), [&](const auto &response) {
      return response.used && response.interface == interface && response.source_ipv4 == source_ipv4 &&
             response.source_port == source_port && response.unicast == unicast &&
             same_host(response.host.data(), question.name.host.data()) &&
             (!legacy || (response.query_id == read_u16(packet) && response.query_type == question.type &&
                          response.query_class == question.clas));
    });
    if (slot == pending_.end()) {
      slot = std::find_if(pending_.begin(), pending_.end(), [](const auto &response) { return !response.used; });
      if (slot == pending_.end()) continue;
      slot->host = question.name.host;
      slot->domain = question.name.domain;
      slot->interface = interface;
      slot->source_ipv4 = source_ipv4;
      slot->source_port = source_port;
      slot->query_id = read_u16(packet);
      slot->query_type = question.type;
      slot->query_class = question.clas;
      slot->unicast = unicast;
      slot->due_us = now_us + (unicast && !truncated ? 0 : response_delay(truncated));
      slot->expires_us = now_us + PENDING_LIFETIME_US;
      slot->used = true;
    } else if (!slot->truncated) {
      // A fresh query can require data that this client previously knew.
      slot->known = 0U;
    }
    slot->requested |= question.requested;
    if (question.type == DNS_TYPE_A || question.type == DNS_TYPE_ANY) slot->query_type = question.type;
    slot->truncated = slot->truncated || truncated;
    affected[static_cast<size_t>(slot - pending_.begin())] = true;
  }
  for (size_t index = 0U; index < pending_.size(); ++index) {
    if (affected[index] && truncated) {
      pending_[index].due_us = std::min(pending_[index].expires_us, now_us + response_delay(true));
    }
  }
  offset = answers_offset;
  work = MAX_PARSE_WORK;
  for (size_t index = 0U; index < answer_count; ++index) {
    ParsedRecord record;
    // Already validated above, using the same bounded decoder.
    if (!parse_record(packet, length, &offset, &work, &record)) break;
    const uint8_t known = known_record(packet, record, ipv4_address_.load());
    for (size_t slot = 0U; slot < pending_.size(); ++slot) {
      if (affected[slot] && same_host(pending_[slot].host.data(), record.name.host.data())) {
        pending_[slot].known |= known;
      }
    }
  }
  for (auto &response : pending_) {
    if (response.used && !response.truncated && (response.requested & ~response.known) == 0U) response = {};
  }
  xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
  // QU and legacy replies remain immediate; multicast work waits for its due time.
  loop();
}

void MdnsBootstrapResponder::defer_known_answers_(const uint8_t *packet, size_t length,
                                                 size_t offset, uint16_t answer_count,
                                                 size_t interface, uint32_t source_ipv4,
                                                 uint16_t source_port, bool truncated) {
  const uint32_t head = deferred_head_.load(std::memory_order_relaxed);
  const uint32_t available = deferred_.size() -
                            (head - deferred_tail_.load(std::memory_order_acquire));
  if (available == 0U) return;
  DeferredKnownAnswer entry{};
  entry.received_us = esp_timer_get_time();
  entry.interface = interface;
  entry.source_ipv4 = source_ipv4;
  entry.source_port = source_port;
  entry.generation = generation_.load(std::memory_order_acquire);
  entry.ipv4_address = ipv4_address_.load(std::memory_order_acquire);
  uint32_t count = 0U;
  if (truncated) {
    entry.truncated = true;
    deferred_[head % deferred_.size()] = entry;
    entry.truncated = false;
    ++count;
  }
  size_t work = MAX_PARSE_WORK;
  for (size_t index = 0U; index < answer_count; ++index) {
    ParsedRecord record;
    if (!parse_record(packet, length, &offset, &work, &record)) break;
    entry.records = known_record(packet, record, entry.ipv4_address);
    if (entry.records == 0U) continue;
    uint32_t slot = 0U;
    for (; slot < count; ++slot) {
      auto &queued = deferred_[(head + slot) % deferred_.size()];
      if (same_host(queued.host.data(), record.name.host.data())) {
        queued.records |= entry.records;
        break;
      }
    }
    if (slot != count || count == available) continue;
    entry.host = record.name.host;
    deferred_[(head + count++) % deferred_.size()] = entry;
  }
  deferred_head_.store(head + count, std::memory_order_release);
}

void MdnsBootstrapResponder::apply_deferred_known_answers_() {
  uint32_t tail = deferred_tail_.load(std::memory_order_relaxed);
  const uint32_t head = deferred_head_.load(std::memory_order_acquire);
  for (; tail != head; ++tail) {
    const auto &entry = deferred_[tail % deferred_.size()];
    if (entry.generation != generation_.load() || entry.ipv4_address != ipv4_address_.load()) continue;
    for (auto &response : pending_) {
      if (!response.used || !response.truncated || response.interface != entry.interface ||
          response.source_ipv4 != entry.source_ipv4 || response.source_port != entry.source_port ||
          response.expires_us <= entry.received_us) continue;
      if (entry.truncated) {
        response.due_us = std::min(response.expires_us, entry.received_us + response_delay(true));
      } else if (same_host(response.host.data(), entry.host.data())) {
        response.known |= entry.records;
      }
    }
  }
  deferred_tail_.store(tail, std::memory_order_release);
}

// Called under the mutex after a completed send. The completion slot lets the
// receive callback and loop use only nonblocking lock attempts, even for I/O
// completion. No other send starts before its history has been committed.
void MdnsBootstrapResponder::finish_send_() {
  if (!sending_ || !send_complete_.load(std::memory_order_acquire)) return;
  if (sent_success_ && sent_generation_ == generation_.load() && active()) {
    size_t retained = 0U;
    for (size_t index = 0U; index < response_time_count_; ++index) {
      if (sent_time_us_ - response_times_[index] < RATE_WINDOW_US) response_times_[retained++] = response_times_[index];
    }
    response_time_count_ = retained;
    response_times_[response_time_count_++] = sent_time_us_;
    if (!sent_.unicast) {
      auto history = std::find_if(recent_.begin(), recent_.end(), [&](const auto &entry) {
        return entry.used && entry.interface == sent_.interface && same_host(entry.host.data(), sent_.host.data());
      });
      if (history == recent_.end()) {
        history = std::find_if(recent_.begin(), recent_.end(), [&](const auto &entry) {
          return !entry.used || (sent_time_us_ - entry.sent_us[0] >= RATE_WINDOW_US &&
                                 sent_time_us_ - entry.sent_us[1] >= RATE_WINDOW_US);
        });
        // A slot was checked before sending; a serialized sender cannot exhaust it.
        if (history != recent_.end()) {
          *history = {};
          history->host = sent_.host;
          history->interface = sent_.interface;
          history->used = true;
        }
      }
      if (history != recent_.end()) {
        if ((sent_records_ & RECORD_A) != 0U) history->sent_us[0] = sent_time_us_;
        if ((sent_records_ & RECORD_NSEC) != 0U) history->sent_us[1] = sent_time_us_;
      }
    }
    for (auto &response : pending_) {
      if (!response.used || response.interface != sent_.interface ||
          !same_host(response.host.data(), sent_.host.data())) continue;
      // Legacy clients need a reply at their own port, with their query ID.
      if (!sent_.unicast && response.unicast) continue;
      if (sent_.unicast && (!response.unicast || response.source_ipv4 != sent_.source_ipv4 ||
          response.source_port != sent_.source_port || response.query_id != sent_.query_id ||
          response.query_type != sent_.query_type || response.query_class != sent_.query_class)) continue;
      // A multicast answer satisfies multicast requesters, including those still
      // collecting Known Answers. One requester's suppression never affects another.
      response.requested &= ~sent_records_;
      if ((response.requested & ~response.known) == 0U) response = {};
    }
  } else if (sent_generation_ == generation_.load()) {
    for (auto &response : pending_) {
      if (response.used && response.interface == sent_.interface &&
          same_host(response.host.data(), sent_.host.data())) {
        response.due_us = std::max(response.due_us, sent_time_us_ + 20000);
      }
    }
  }
  send_complete_.store(false, std::memory_order_relaxed);
  sending_ = false;
}

void MdnsBootstrapResponder::loop() {
  if (mutex_ == nullptr) return;
  // Bound both sends and scans per invocation, including partial ANY answers.
  for (size_t attempt = 0U; attempt <= MAX_PENDING_RESPONSES; ++attempt) {
    if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), 0U) != pdTRUE) return;
    finish_send_();
    apply_deferred_known_answers_();
    if (!active() || sending_ || attempt == MAX_PENDING_RESPONSES) {
      xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
      return;
    }
    const int64_t now_us = esp_timer_get_time();
    size_t retained = 0U;
    for (size_t index = 0U; index < response_time_count_; ++index) {
      if (now_us - response_times_[index] < RATE_WINDOW_US) response_times_[retained++] = response_times_[index];
    }
    response_time_count_ = retained;
    for (auto &entry : recent_) {
      if (entry.used && now_us - entry.sent_us[0] >= RATE_WINDOW_US &&
          now_us - entry.sent_us[1] >= RATE_WINDOW_US) entry = {};
    }
    for (auto &response : pending_) {
      if (response.used && (response.expires_us <= now_us || (response.requested & ~response.known) == 0U)) response = {};
    }
    auto due = std::find_if(pending_.begin(), pending_.end(), [&](const auto &response) {
      return response.used && response.due_us <= now_us;
    });
    if (due == pending_.end() || response_time_count_ >= MAX_RESPONSES_PER_SECOND) {
      xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
      return;
    }
    uint8_t records = due->requested & ~due->known;
    if ((records & RECORD_A) != 0U && (due->known & RECORD_NSEC) == 0U) records |= RECORD_NSEC;
    if (!due->unicast) {
      for (const auto &response : pending_) {
        if (response.used && !response.unicast && response.due_us <= now_us &&
            response.interface == due->interface && same_host(response.host.data(), due->host.data())) {
          const uint8_t needed = response.requested & ~response.known;
          records |= needed;
          if ((needed & RECORD_A) != 0U && (response.known & RECORD_NSEC) == 0U) records |= RECORD_NSEC;
        }
      }
      bool have_history_slot = false;
      int64_t next_due = now_us + RATE_WINDOW_US;
      for (const auto &entry : recent_) {
        if (!entry.used || (now_us - entry.sent_us[0] >= RATE_WINDOW_US &&
                            now_us - entry.sent_us[1] >= RATE_WINDOW_US)) have_history_slot = true;
        if (!entry.used || entry.interface != due->interface || !same_host(entry.host.data(), due->host.data())) continue;
        have_history_slot = true;
        for (size_t bit = 0U; bit < 2U; ++bit) {
          if ((records & (1U << bit)) != 0U && now_us - entry.sent_us[bit] < RATE_WINDOW_US) {
            records &= ~(1U << bit);
            next_due = std::min(next_due, entry.sent_us[bit] + RATE_WINDOW_US);
          }
        }
      }
      // Do not send an optional NSEC alone when the requested A is rate-limited.
      uint8_t needed = due->requested & ~due->known;
      for (const auto &response : pending_) {
        if (response.used && !response.unicast && response.due_us <= now_us &&
            response.interface == due->interface && same_host(response.host.data(), due->host.data())) {
          needed |= response.requested & ~response.known;
        }
      }
      if (!have_history_slot || (records & needed) == 0U) {
        due->due_us = next_due;
        xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
        continue;
      }
    }
    std::array<uint8_t, MAX_RESPONSE_BYTES> bytes{};
    const size_t length = build_response(due->host.data(), due->domain.data(), due->query_type,
                                         due->query_class, due->query_id, ipv4_address_.load(),
                                         due->source_port != MDNS_PORT, records, bytes.data(), bytes.size());
    if (length == 0U) {
      *due = {};
      xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
      continue;
    }
    sent_ = *due;
    sent_records_ = records;
    sent_generation_ = generation_.load();
    sending_ = true;
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    sent_success_ = false;
    if (begin_send_(sent_generation_)) {
      esp_ip_addr_t destination{};
      destination.type = ESP_IPADDR_TYPE_V4;
      destination.u_addr.ip4.addr = sent_.unicast ? sent_.source_ipv4 : MDNS_MULTICAST_IPV4;
      sent_success_ = mdns_priv_if_write(static_cast<mdns_if_t>(sent_.interface), MDNS_IP_PROTOCOL_V4,
                                         &destination, sent_.unicast ? sent_.source_port : MDNS_PORT,
                                         bytes.data(), length) == length;
      sent_time_us_ = esp_timer_get_time();
      end_send_();
    } else {
      sent_time_us_ = esp_timer_get_time();
    }
    send_complete_.store(true, std::memory_order_release);
  }
}

void MdnsBootstrapResponder::shutdown() {
  MdnsBootstrapResponder *owner = this;
  const bool was_published = g_bootstrap_responder.compare_exchange_strong(
      owner, nullptr, std::memory_order_acq_rel);
  if (was_published) {
    while (g_bootstrap_callbacks_in_flight.load(std::memory_order_acquire) != 0U) {
      vTaskDelay(1U);
    }
  }

  if (mutex_ != nullptr) {
    xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), portMAX_DELAY);
    generation_.fetch_add(1U, std::memory_order_acq_rel);
    clear_pending_();
    ipv4_address_ = 0U;
    response_time_count_ = 0U;
    configured_ = false;
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
    wait_for_sends_();
  } else {
    configured_ = false;
    ipv4_address_ = 0U;
  }
}

bool MdnsBootstrapResponder::begin_send_(uint32_t generation) {
  sends_in_flight_.fetch_add(1U, std::memory_order_acq_rel);
  if (!active() || generation_.load(std::memory_order_acquire) != generation) {
    end_send_();
    return false;
  }
  return true;
}

void MdnsBootstrapResponder::end_send_() {
  sends_in_flight_.fetch_sub(1U, std::memory_order_release);
}

void MdnsBootstrapResponder::wait_for_sends_() {
  while (sends_in_flight_.load(std::memory_order_acquire) != 0U) {
    vTaskDelay(1U);
  }
}

void MdnsBootstrapResponder::clear_pending_() {
  for (auto &response : pending_) response = {};
  for (auto &entry : recent_) entry = {};
  response_time_count_ = 0U;
  deferred_tail_.store(deferred_head_.load(std::memory_order_acquire), std::memory_order_release);
}

}  // namespace espectre
