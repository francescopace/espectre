/*
 * ESPectre - Shared mDNS Discovery Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "esp_netif.h"
#include "esp_timer.h"
#include "direct_http_protocol.h"
#include "espectre_protocol.h"
#include "mdns.h"
#include "mdns_private.h"
#include "mdns_discovery_service.h"
#include "mdns_bootstrap_responder.h"
#include "peer_discovery.h"
#include "peer_discovery_service_esp_idf.h"

using namespace espectre;

namespace {

extern "C" void __wrap_mdns_priv_receive_action(mdns_action_t *action,
                                                 mdns_action_subtype_t type);

MdnsDiscoveryServiceConfig direct_config() {
  return {
      "espectre-0123456789abcdef",
      "Kitchen sensor",
      "_espectre",
      "_tcp",
      80U,
      {{"device_id", "0123456789abcdef"},
       {"transport", ESPECTRE_DIRECT_HTTP_TRANSPORT},
       {"path", ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT},
       {"events", ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT},
       {"protovers", "1.0"}},
  };
}

void reset_mocks() {
  mdns_mock_reset();
  esp_netif_mock_reset();
}

uint32_t ipv4(uint8_t first, uint8_t second, uint8_t third, uint8_t fourth);

void test_registers_identity_service_and_txt() {
  reset_mocks();
  MdnsDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(direct_config()));
  TEST_ASSERT_EQUAL_STRING("espectre-0123456789abcdef", g_mdns_mock.hostname);
  TEST_ASSERT_EQUAL_STRING("Kitchen sensor", g_mdns_mock.instance_name);
  TEST_ASSERT_EQUAL_STRING("_espectre", g_mdns_mock.service_type);
  TEST_ASSERT_EQUAL_STRING("_tcp", g_mdns_mock.service_proto);
  TEST_ASSERT_EQUAL(80U, g_mdns_mock.service_port);
  TEST_ASSERT_EQUAL(5U, g_mdns_mock.txt_count);
  TEST_ASSERT_EQUAL_STRING("device_id", g_mdns_mock.txt_keys[0]);
  TEST_ASSERT_EQUAL_STRING("0123456789abcdef", g_mdns_mock.txt_values[0]);
  TEST_ASSERT_EQUAL_STRING("transport", g_mdns_mock.txt_keys[1]);
  TEST_ASSERT_EQUAL_STRING("http", g_mdns_mock.txt_values[1]);
  TEST_ASSERT_EQUAL_STRING("path", g_mdns_mock.txt_keys[2]);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT, g_mdns_mock.txt_values[2]);
  service.shutdown();
  TEST_ASSERT_EQUAL_INT(1, g_mdns_mock.service_remove_call_count);
  TEST_ASSERT_EQUAL_INT(1, g_mdns_mock.free_call_count);
}

void test_follows_wifi_lifecycle_and_updates_txt_atomically() {
  reset_mocks();
  MdnsDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(direct_config()));
  service.on_wifi_connected();
  TEST_ASSERT_TRUE(service.service_enabled());
  TEST_ASSERT_EQUAL_INT(MDNS_EVENT_ENABLE_IP4, g_mdns_mock.last_netif_action);
  service.on_wifi_connected();
  TEST_ASSERT_EQUAL_INT(MDNS_EVENT_ANNOUNCE_IP4, g_mdns_mock.last_netif_action);
  TEST_ASSERT_TRUE(service.update_txt({{"name", "Office"}, {"protocol", "1"}}));
  TEST_ASSERT_EQUAL(2U, g_mdns_mock.txt_count);
  TEST_ASSERT_EQUAL_STRING("name", g_mdns_mock.txt_keys[0]);
  TEST_ASSERT_EQUAL_STRING("Office", g_mdns_mock.txt_values[0]);
  service.on_wifi_disconnected();
  TEST_ASSERT_FALSE(service.service_enabled());
  TEST_ASSERT_EQUAL_INT(MDNS_EVENT_DISABLE_IP4, g_mdns_mock.last_netif_action);
}

void test_does_not_free_mdns_owned_by_another_component() {
  reset_mocks();
  g_mdns_mock.init_result = ESP_ERR_INVALID_STATE;
  MdnsDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(direct_config()));
  service.shutdown();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.free_call_count);
}

void test_attaches_without_mutating_existing_responder_identity() {
  reset_mocks();
  MdnsDiscoveryServiceConfig config = direct_config();
  config.hostname.clear();
  config.responder_mode = MdnsResponderMode::USE_EXISTING_RESPONDER;
  MdnsDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(config));
  TEST_ASSERT_EQUAL_INT(1, g_mdns_mock.init_call_count);
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.hostname_set_call_count);
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.instance_name_set_call_count);
  TEST_ASSERT_TRUE(service.service_enabled());
  service.on_wifi_connected();
  service.on_wifi_disconnected();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.netif_action_call_count);
  service.shutdown();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.free_call_count);
}

void test_initializes_missing_shared_responder_with_fallback_hostname() {
  reset_mocks();
  MdnsDiscoveryServiceConfig config = direct_config();
  config.responder_mode = MdnsResponderMode::USE_EXISTING_RESPONDER;
  MdnsDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(config));
  TEST_ASSERT_EQUAL_INT(1, g_mdns_mock.init_call_count);
  TEST_ASSERT_EQUAL_INT(1, g_mdns_mock.hostname_set_call_count);
  TEST_ASSERT_EQUAL_STRING("espectre-0123456789abcdef", g_mdns_mock.hostname);
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.instance_name_set_call_count);
  service.shutdown();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.free_call_count);
}

constexpr char BOOTSTRAP_HOST[] =
    "espectre-devices-0123456789abcdef01234567";
constexpr uint16_t DNS_TYPE_NSEC = 47U;

uint16_t packet_u16(const uint8_t *data) {
  return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8U) | data[1]);
}

uint32_t packet_u32(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24U) |
         (static_cast<uint32_t>(data[1]) << 16U) |
         (static_cast<uint32_t>(data[2]) << 8U) |
         static_cast<uint32_t>(data[3]);
}

void append_dns_name(std::vector<uint8_t> *packet, const char *host) {
  const size_t host_length = std::strlen(host);
  packet->push_back(static_cast<uint8_t>(host_length));
  packet->insert(packet->end(), host, host + host_length);
  packet->push_back(5U);
  packet->insert(packet->end(), {'l', 'o', 'c', 'a', 'l', 0U});
}

std::vector<uint8_t> bootstrap_query(const char *host = BOOTSTRAP_HOST,
                                     uint16_t type = MDNS_TYPE_A,
                                     uint16_t clas = 1U,
                                     uint16_t id = 0U,
                                     uint16_t flags = 0U) {
  std::vector<uint8_t> packet(12U, 0U);
  packet[0] = static_cast<uint8_t>(id >> 8U);
  packet[1] = static_cast<uint8_t>(id);
  packet[2] = static_cast<uint8_t>(flags >> 8U);
  packet[3] = static_cast<uint8_t>(flags);
  packet[5] = 1U;
  append_dns_name(&packet, host);
  packet.push_back(static_cast<uint8_t>(type >> 8U));
  packet.push_back(static_cast<uint8_t>(type));
  packet.push_back(static_cast<uint8_t>(clas >> 8U));
  packet.push_back(static_cast<uint8_t>(clas));
  return packet;
}

std::vector<uint8_t> chrome_bootstrap_query() {
  std::vector<uint8_t> packet =
      bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_A, 0x8001U);
  packet[5] = 2U;
  packet.insert(packet.end(), {0xc0U, 0x0cU, 0x00U, MDNS_TYPE_AAAA, 0x80U, 0x01U});
  return packet;
}

size_t skip_dns_name(const uint8_t *packet, size_t length, size_t offset) {
  while (offset < length && packet[offset] != 0U) {
    const size_t label_length = packet[offset++];
    if (label_length > 63U || offset + label_length > length) return length;
    offset += label_length;
  }
  return offset < length ? offset + 1U : length;
}

std::string dns_name_at(const uint8_t *packet, size_t length, size_t offset) {
  std::string name;
  while (offset < length && packet[offset] != 0U) {
    const size_t label_length = packet[offset++];
    if (label_length > 63U || offset + label_length > length) return {};
    if (!name.empty()) name.push_back('.');
    name.append(reinterpret_cast<const char *>(packet + offset), label_length);
    offset += label_length;
  }
  return name;
}

size_t answer_offset(const uint8_t *packet, size_t length) {
  size_t offset = 12U;
  if (packet_u16(packet + 4U) != 0U) {
    offset = skip_dns_name(packet, length, offset);
    offset += 4U;
  }
  return offset;
}

void assert_bootstrap_nsec(const uint8_t *packet,
                           size_t length,
                           size_t offset,
                           const char *expected_host) {
  const std::string expected_owner = std::string(expected_host) + ".local";
  const std::string actual_owner = dns_name_at(packet, length, offset);
  TEST_ASSERT_EQUAL_STRING(expected_owner.c_str(), actual_owner.c_str());
  const size_t fields = skip_dns_name(packet, length, offset);
  TEST_ASSERT_TRUE(fields + 10U <= length);
  TEST_ASSERT_EQUAL(DNS_TYPE_NSEC, packet_u16(packet + fields));
  TEST_ASSERT_EQUAL(1U, packet_u16(packet + fields + 2U));
  TEST_ASSERT_EQUAL(MdnsBootstrapResponder::RESPONSE_TTL_SECONDS,
                    packet_u32(packet + fields + 4U));
  const size_t rdata_length = packet_u16(packet + fields + 8U);
  const size_t rdata = fields + 10U;
  TEST_ASSERT_TRUE(rdata + rdata_length <= length);
  const std::string next_domain = dns_name_at(packet, length, rdata);
  TEST_ASSERT_EQUAL_STRING(expected_owner.c_str(), next_domain.c_str());
  const size_t bitmap = skip_dns_name(packet, length, rdata);
  TEST_ASSERT_EQUAL(rdata + rdata_length, bitmap + 3U);
  TEST_ASSERT_EQUAL(0U, packet[bitmap]);
  TEST_ASSERT_EQUAL(1U, packet[bitmap + 1U]);
  TEST_ASSERT_EQUAL(0x40U, packet[bitmap + 2U]);
}

void assert_bootstrap_answer(const char *expected_host,
                             uint32_t expected_address,
                             bool legacy_unicast,
                             uint16_t expected_id) {
  const uint8_t *packet = g_mdns_mock.last_write_packet;
  const size_t length = g_mdns_mock.last_write_len;
  TEST_ASSERT_TRUE(length > 12U);
  TEST_ASSERT_EQUAL(expected_id, packet_u16(packet));
  TEST_ASSERT_EQUAL(0x8400U, packet_u16(packet + 2U));
  TEST_ASSERT_EQUAL(legacy_unicast ? 1U : 0U, packet_u16(packet + 4U));
  TEST_ASSERT_EQUAL(1U, packet_u16(packet + 6U));
  TEST_ASSERT_EQUAL(1U, packet_u16(packet + 10U));
  const size_t offset = answer_offset(packet, length);
  const std::string expected_owner = std::string(expected_host) + ".local";
  const std::string actual_owner = dns_name_at(packet, length, offset);
  TEST_ASSERT_EQUAL_STRING(expected_owner.c_str(), actual_owner.c_str());
  const size_t fields = skip_dns_name(packet, length, offset);
  TEST_ASSERT_TRUE(fields + 14U <= length);
  TEST_ASSERT_EQUAL(MDNS_TYPE_A, packet_u16(packet + fields));
  TEST_ASSERT_EQUAL(1U, packet_u16(packet + fields + 2U));
  TEST_ASSERT_EQUAL(MdnsBootstrapResponder::RESPONSE_TTL_SECONDS,
                    packet_u32(packet + fields + 4U));
  TEST_ASSERT_EQUAL(4U, packet_u16(packet + fields + 8U));
  uint32_t address = 0U;
  std::memcpy(&address, packet + fields + 10U, sizeof(address));
  TEST_ASSERT_EQUAL(expected_address, address);
  assert_bootstrap_nsec(packet, length, fields + 14U, expected_host);
}

void assert_bootstrap_negative_aaaa(const char *expected_host) {
  const uint8_t *packet = g_mdns_mock.last_write_packet;
  const size_t length = g_mdns_mock.last_write_len;
  TEST_ASSERT_TRUE(length > 12U);
  TEST_ASSERT_EQUAL(0U, packet_u16(packet));
  TEST_ASSERT_EQUAL(0x8400U, packet_u16(packet + 2U));
  TEST_ASSERT_EQUAL(0U, packet_u16(packet + 4U));
  TEST_ASSERT_EQUAL(1U, packet_u16(packet + 6U));
  TEST_ASSERT_EQUAL(0U, packet_u16(packet + 10U));
  assert_bootstrap_nsec(packet, length, answer_offset(packet, length), expected_host);
}

void test_bootstrap_answers_multicast_a_after_bounded_delay() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  const uint32_t address = ipv4(192U, 168U, 1U, 42U);
  TEST_ASSERT_TRUE(responder.update(address));
  const std::vector<uint8_t> query = bootstrap_query();
  responder.ingest_query(query.data(), query.size(), 1U, ipv4(192U, 168U, 1U, 9U), 5353U);
  responder.loop();
  TEST_ASSERT_EQUAL(0, g_mdns_mock.real_write_call_count);
  esp_timer_mock::advance(24999);
  responder.loop();
  TEST_ASSERT_EQUAL(0, g_mdns_mock.real_write_call_count);
  esp_timer_mock::advance(1);
  responder.loop();
  TEST_ASSERT_EQUAL(1, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(1U, g_mdns_mock.last_write_interface);
  TEST_ASSERT_EQUAL(MDNS_IP_PROTOCOL_V4, g_mdns_mock.last_write_protocol);
  TEST_ASSERT_EQUAL(ipv4(224U, 0U, 0U, 251U),
                    g_mdns_mock.last_write_destination_ipv4);
  TEST_ASSERT_EQUAL(5353U, g_mdns_mock.last_write_destination_port);
  assert_bootstrap_answer(BOOTSTRAP_HOST, address, false, 0U);
}

void test_bootstrap_handles_qu_and_legacy_unicast() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  const uint32_t address = ipv4(10U, 0U, 0U, 7U);
  const uint32_t source = ipv4(10U, 0U, 0U, 2U);
  constexpr char uppercase_host[] =
      "ESPECTRE-DEVICES-ABCDEF0123456789ABCDEF01";
  TEST_ASSERT_TRUE(responder.update(address));

  std::vector<uint8_t> query = bootstrap_query(uppercase_host, MDNS_TYPE_A, 0x8001U, 0x4567U);
  responder.ingest_query(query.data(), query.size(), 0U, source, 5353U);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(source, g_mdns_mock.last_write_destination_ipv4);
  TEST_ASSERT_EQUAL(5353U, g_mdns_mock.last_write_destination_port);
  assert_bootstrap_answer(uppercase_host, address, false, 0U);

  query = bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_A, 1U, 0x1234U);
  responder.ingest_query(query.data(), query.size(), 0U, source, 9999U);
  TEST_ASSERT_EQUAL(2, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(source, g_mdns_mock.last_write_destination_ipv4);
  TEST_ASSERT_EQUAL(9999U, g_mdns_mock.last_write_destination_port);
  assert_bootstrap_answer(BOOTSTRAP_HOST, address, true, 0x1234U);
}

void test_bootstrap_answers_chrome_a_with_compressed_aaaa_question() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  const uint32_t address = ipv4(192U, 168U, 1U, 42U);
  const uint32_t source = ipv4(192U, 168U, 1U, 22U);
  TEST_ASSERT_TRUE(responder.update(address));

  const std::vector<uint8_t> query = chrome_bootstrap_query();
  responder.ingest_query(query.data(), query.size(), 0U, source, 5353U);

  TEST_ASSERT_EQUAL(1, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(source, g_mdns_mock.last_write_destination_ipv4);
  TEST_ASSERT_EQUAL(5353U, g_mdns_mock.last_write_destination_port);
  assert_bootstrap_answer(BOOTSTRAP_HOST, address, false, 0U);
}

void test_bootstrap_negates_aaaa_without_advertising_ipv6() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  const uint32_t source = ipv4(192U, 168U, 1U, 22U);
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 42U)));

  const std::vector<uint8_t> query =
      bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_AAAA, 0x8001U);
  responder.ingest_query(query.data(), query.size(), 0U, source, 5353U);

  TEST_ASSERT_EQUAL(1, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(source, g_mdns_mock.last_write_destination_ipv4);
  TEST_ASSERT_EQUAL(5353U, g_mdns_mock.last_write_destination_port);
  assert_bootstrap_negative_aaaa(BOOTSTRAP_HOST);
}

void test_bootstrap_rejects_static_invalid_and_unsupported_queries() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 42U)));
  const char *invalid_hosts[] = {
      "espectre-devices",
      "espectre-devices-0123456789abcdef0123456",
      "espectre-devices-0123456789abcdef012345678",
      "espectre-devices-0123456789abcdef0123456g",
  };
  for (const char *host : invalid_hosts) {
    const std::vector<uint8_t> query = bootstrap_query(host);
    responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  }
  std::vector<uint8_t> query = bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_A, 3U);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  query = bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_A, 1U, 0U, 0x8000U);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  query = bootstrap_query(BOOTSTRAP_HOST, MDNS_TYPE_A, 1U, 0U, 0x0200U);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  query = bootstrap_query();
  query[12] = 0xc0U;
  query[13] = 0x0cU;
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  query = bootstrap_query();
  query.pop_back();
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  esp_timer_mock::advance(100000);
  responder.loop();
  TEST_ASSERT_EQUAL(0, g_mdns_mock.real_write_call_count);
}

void test_bootstrap_requires_ipv4_and_cancels_pending_responses() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  const std::vector<uint8_t> query = bootstrap_query();
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 42U)));
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 43U)));
  esp_timer_mock::advance(100000);
  responder.loop();
  TEST_ASSERT_EQUAL(0, g_mdns_mock.real_write_call_count);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  responder.shutdown();
  esp_timer_mock::advance(100000);
  responder.loop();
  TEST_ASSERT_EQUAL(0, g_mdns_mock.real_write_call_count);
}

void test_bootstrap_bounds_pending_pool_and_global_rate() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 42U)));
  const std::vector<uint8_t> query = bootstrap_query();
  for (size_t index = 0U; index < 5U; ++index) {
    responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  }
  for (int expected = 1; expected <= 4; ++expected) {
    esp_timer_mock::advance(25000);
    responder.loop();
    TEST_ASSERT_EQUAL(expected, g_mdns_mock.real_write_call_count);
  }
  TEST_ASSERT_EQUAL(4, g_mdns_mock.real_write_call_count);
  for (size_t index = 0U; index < 4U; ++index) {
    responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
    esp_timer_mock::advance(25000);
    responder.loop();
  }
  TEST_ASSERT_EQUAL(8, g_mdns_mock.real_write_call_count);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  esp_timer_mock::advance(25000);
  responder.loop();
  TEST_ASSERT_EQUAL(8, g_mdns_mock.real_write_call_count);
  esp_timer_mock::advance(775000);
  responder.ingest_query(query.data(), query.size(), 0U, 1U, 5353U);
  esp_timer_mock::advance(25000);
  responder.loop();
  TEST_ASSERT_EQUAL(9, g_mdns_mock.real_write_call_count);
}

void test_bootstrap_wrapper_always_forwards_to_espressif() {
  reset_mocks();
  esp_timer_mock::reset(100000, 0);
  MdnsBootstrapResponder responder;
  TEST_ASSERT_TRUE(responder.setup());
  TEST_ASSERT_TRUE(responder.update(ipv4(192U, 168U, 1U, 42U)));
  std::vector<uint8_t> query = bootstrap_query();
  mdns_rx_packet_t rx{};
  rx.tcpip_if = 1U;
  rx.ip_protocol = MDNS_IP_PROTOCOL_V4;
  rx.src.type = ESP_IPADDR_TYPE_V4;
  rx.src.u_addr.ip4.addr = ipv4(192U, 168U, 1U, 9U);
  rx.src_port = 5353U;
  rx.mock_data = query.data();
  rx.mock_length = query.size();
  mdns_action_t action{};
  action.type = ACTION_RX_HANDLE;
  action.data.rx_handle.packet = &rx;

  __wrap_mdns_priv_receive_action(&action, ACTION_RUN);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.receive_real_call_count);
  query = bootstrap_query("espectre-devices");
  rx.mock_data = query.data();
  rx.mock_length = query.size();
  __wrap_mdns_priv_receive_action(&action, ACTION_RUN);
  TEST_ASSERT_EQUAL(2, g_mdns_mock.receive_real_call_count);
  __wrap_mdns_priv_receive_action(&action, ACTION_CLEANUP);
  TEST_ASSERT_EQUAL(3, g_mdns_mock.receive_real_call_count);
  rx.ip_protocol = MDNS_IP_PROTOCOL_V6;
  __wrap_mdns_priv_receive_action(&action, ACTION_RUN);
  TEST_ASSERT_EQUAL(4, g_mdns_mock.receive_real_call_count);
}

PeerDiscoveryCandidate peer(const char *device_id,
                            const char *hostname,
                            uint32_t address) {
  PeerDiscoveryCandidate candidate;
  candidate.instance = std::string("ESPectre ") + device_id;
  candidate.hostname = hostname;
  candidate.device_id = device_id;
  candidate.name = "Kitchen sensor";
  candidate.frontend = "native";
  candidate.txt_version = ESPECTRE_DNS_SD_TXT_SCHEMA_VERSION;
  candidate.protocol_version = ESPECTRE_PROTOCOL_VERSION;
  candidate.transport = ESPECTRE_DIRECT_HTTP_TRANSPORT;
  candidate.path = ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT;
  candidate.events = ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT;
  candidate.firmware = "3.0.0-rc1";
  candidate.chip = "esp32c3";
  candidate.capabilities = "monitor,config,peer_discovery";
  candidate.port = 80U;
  candidate.ipv4_addresses = {address};
  return candidate;
}

uint32_t ipv4(uint8_t first, uint8_t second, uint8_t third, uint8_t fourth) {
  return static_cast<uint32_t>(first) | (static_cast<uint32_t>(second) << 8U) |
         (static_cast<uint32_t>(third) << 16U) | (static_cast<uint32_t>(fourth) << 24U);
}

struct PeerMdnsFixture {
  mdns_txt_item_t txt[11] = {
      {"device_id", "2222222222222222"},
      {"name", "Office sensor"},
      {"frontend", "native"},
      {"txtvers", "1"},
      {"protovers", "1.0"},
      {"transport", "http"},
      {"path", "/espectre/v1/request"},
      {"events", "/espectre/v1/events"},
      {"firmware", "3.0.0-rc1"},
      {"chip", "esp32c3"},
      {"capabilities", "config,monitor,peer_discovery"},
  };
  mdns_ip_addr_t address{};
  mdns_result_t result{};

  PeerMdnsFixture() {
    address.addr.type = ESP_IPADDR_TYPE_V4;
    address.addr.u_addr.ip4.addr = ipv4(192U, 168U, 1U, 42U);
    result.instance_name = const_cast<char *>("ESPectre 2222222222222222");
    result.service_type = const_cast<char *>("_espectre");
    result.proto = const_cast<char *>("_tcp");
    result.hostname = const_cast<char *>("espectre-2222222222222222");
    result.port = 80U;
    result.txt = txt;
    result.txt_count = sizeof(txt) / sizeof(txt[0]);
    result.addr = &address;
  }
};

void test_peer_results_are_bounded_validated_sorted_and_serializable() {
  const uint32_t station = ipv4(192U, 168U, 1U, 100U);
  const uint32_t netmask = ipv4(255U, 255U, 255U, 0U);
  PeerDiscoveryCandidate second = peer("2222222222222222", "espectre-2", station + (2U << 24U));
  PeerDiscoveryCandidate micro = peer(
      "3333333333333333", "espectre-3333333333333333", station + (3U << 24U));
  micro.frontend = "micro";
  micro.capabilities = "monitor";
  PeerDiscoveryCandidate first = peer("1111111111111111", "espectre-1", station + (1U << 24U));
  first.ipv4_addresses.push_back(8U | (8U << 8U) | (8U << 16U) | (8U << 24U));
  PeerDiscoveryCandidate malformed = peer("not-an-identity", "bad", station + (3U << 24U));
  PeerDiscoveryCandidate conflict = first;
  conflict.hostname = "conflicting-host";

  const PeerDiscoverySnapshot snapshot = validate_peer_discovery_candidates(
      {second, malformed, first, conflict, micro}, station, netmask, 3000U, false);
  TEST_ASSERT_EQUAL(2U, snapshot.devices.size());
  TEST_ASSERT_EQUAL_STRING("2222222222222222", snapshot.devices[0].device_id.c_str());
  TEST_ASSERT_TRUE(snapshot.rejected_results >= 2U);
  const std::string payload = peer_discovery_snapshot_json(snapshot);
  TEST_ASSERT_TRUE(payload.size() <= ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE);
  TEST_ASSERT_TRUE(payload.find("\"schema_version\":2") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"dns_sd_schema_version\":1") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"protocol_version\":\"1.0\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("192.168.1.102") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("8.8.8.8") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("peer_discovery") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"frontend\":\"micro\"") != std::string::npos);
}

void test_peer_results_reject_every_malformed_and_nonlocal_boundary() {
  const uint32_t station = ipv4(192U, 168U, 1U, 100U);
  const uint32_t netmask = ipv4(255U, 255U, 255U, 0U);
  const PeerDiscoveryCandidate valid = peer("1111111111111111", "espectre-valid", ipv4(192U, 168U, 1U, 42U));
  std::vector<PeerDiscoveryCandidate> candidates{valid};
  auto reject = [&candidates, &valid](auto mutate) {
    PeerDiscoveryCandidate candidate = valid;
    mutate(candidate);
    candidates.push_back(std::move(candidate));
  };
  reject([](auto &candidate) { candidate.instance.assign(64U, 'x'); });
  reject([](auto &candidate) { candidate.hostname = "bad.host"; });
  reject([](auto &candidate) { candidate.device_id = "ABCDEF0123456789"; });
  reject([](auto &candidate) { candidate.name = "bad\nname"; });
  reject([](auto &candidate) { candidate.frontend = "unknown"; });
  reject([](auto &candidate) { candidate.txt_version = "2"; });
  reject([](auto &candidate) { candidate.protocol_version = "2"; });
  reject([](auto &candidate) { candidate.transport = "websocket"; });
  reject([](auto &candidate) { candidate.path = "/espectre/v1/ws"; });
  reject([](auto &candidate) { candidate.events = "/events"; });
  reject([](auto &candidate) { candidate.firmware.assign(49U, 'x'); });
  reject([](auto &candidate) { candidate.chip = "esp32.c3"; });
  reject([](auto &candidate) { candidate.capabilities = "monitor,monitor"; });
  reject([](auto &candidate) { candidate.port = 0U; });
  reject([](auto &candidate) { candidate.ipv4_addresses = {ipv4(192U, 168U, 2U, 42U)}; });
  reject([](auto &candidate) { candidate.ipv4_addresses = {ipv4(127U, 0U, 0U, 1U)}; });
  reject([](auto &candidate) { candidate.ipv4_addresses = {ipv4(224U, 0U, 0U, 1U)}; });
  reject([](auto &candidate) { candidate.ipv4_addresses = {ipv4(192U, 168U, 1U, 0U)}; });
  reject([](auto &candidate) { candidate.ipv4_addresses = {ipv4(192U, 168U, 1U, 255U)}; });

  const PeerDiscoverySnapshot snapshot =
      validate_peer_discovery_candidates(candidates, station, netmask, 1U, false);
  TEST_ASSERT_EQUAL(1U, snapshot.devices.size());
  TEST_ASSERT_EQUAL(candidates.size() - 1U, snapshot.rejected_results);
}

void test_peer_results_enforce_address_device_and_serialized_size_limits() {
  const uint32_t station = ipv4(192U, 168U, 1U, 100U);
  const uint32_t netmask = ipv4(255U, 255U, 255U, 0U);
  std::vector<PeerDiscoveryCandidate> candidates;
  for (size_t index = 0U; index < ESPECTRE_PEER_DISCOVERY_MAX_DEVICES + 2U; ++index) {
    char identity[17]{};
    std::snprintf(identity, sizeof(identity), "%016zx", index + 1U);
    PeerDiscoveryCandidate candidate = peer(identity, "espectre-peer", ipv4(192U, 168U, 1U, 10U + index));
    candidate.instance.assign(63U, 'I');
    candidate.hostname = std::string(62U, 'h') + std::to_string(index % 10U);
    candidate.name.assign(63U, 'N');
    candidate.firmware.assign(48U, 'F');
    candidate.chip.assign(16U, 'c');
    candidate.capabilities =
        "aaaaaaaaaa,bbbbbbbbbb,cccccccccc,dddddddddd,eeeeeeeeee,ffffffffff,gggggggggg,hhhhhhhhhh";
    candidate.ipv4_addresses = {
        ipv4(192U, 168U, 1U, 10U + index),
        ipv4(192U, 168U, 1U, 30U + index),
        ipv4(192U, 168U, 1U, 50U + index),
    };
    candidates.push_back(std::move(candidate));
  }

  const PeerDiscoverySnapshot snapshot =
      validate_peer_discovery_candidates(candidates, station, netmask, 3000U, true);
  TEST_ASSERT_EQUAL(ESPECTRE_PEER_DISCOVERY_MAX_DEVICES, snapshot.devices.size());
  TEST_ASSERT_EQUAL(ESPECTRE_PEER_DISCOVERY_MAX_ADDRESSES,
                    snapshot.devices.front().ipv4_addresses.size());
  TEST_ASSERT_TRUE(snapshot.truncated);
  const std::string payload = peer_discovery_snapshot_json(snapshot);
  TEST_ASSERT_TRUE(payload.size() <= ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE);
  TEST_ASSERT_TRUE(payload.find("\"status\":\"timeout\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"truncated\":true") != std::string::npos);
}

void test_peer_query_uses_fixed_bounds_and_frees_results_exactly_once() {
  reset_mocks();
  esp_timer_mock::reset(0, 1000);
  PeerMdnsFixture fixture;
  g_mdns_mock.async_results = &fixture.result;
  EspIdfPeerDiscoveryService service;
  service.set_local_candidate(peer("1111111111111111", "espectre-local", ipv4(192U, 168U, 1U, 100U)));
  service.set_wifi_ready(true);
  size_t completion_count = 0U;
  PeerDiscoverySnapshot delivered;
  TEST_ASSERT_TRUE(service.start([&](PeerDiscoverySnapshot snapshot) {
    completion_count += 1U;
    delivered = std::move(snapshot);
  }));
  TEST_ASSERT_EQUAL(ESPECTRE_PEER_DISCOVERY_TIMEOUT_MS, g_mdns_mock.last_query_timeout_ms);
  TEST_ASSERT_EQUAL(ESPECTRE_PEER_DISCOVERY_MAX_DEVICES * 2U, g_mdns_mock.last_query_max_results);
  TEST_ASSERT_EQUAL_STRING("_espectre", g_mdns_mock.service_type);
  TEST_ASSERT_EQUAL_STRING("_tcp", g_mdns_mock.service_proto);

  service.loop();
  TEST_ASSERT_EQUAL(0U, completion_count);
  TEST_ASSERT_EQUAL(0, g_mdns_mock.query_results_free_call_count);
  g_mdns_mock.async_get_results_finished = true;
  service.loop();
  TEST_ASSERT_EQUAL(1U, completion_count);
  TEST_ASSERT_EQUAL(2U, delivered.devices.size());
  TEST_ASSERT_EQUAL(1, g_mdns_mock.query_results_free_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.async_delete_call_count);
  TEST_ASSERT_TRUE(&fixture.result == g_mdns_mock.last_freed_results);
  service.loop();
  TEST_ASSERT_EQUAL(1, g_mdns_mock.query_results_free_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.async_delete_call_count);
}

void test_peer_query_rejects_allocation_failure_without_leaking_state() {
  reset_mocks();
  EspIdfPeerDiscoveryService service;
  service.set_wifi_ready(true);
  g_mdns_mock.async_new_succeeds = false;
  TEST_ASSERT_FALSE(service.start([](PeerDiscoverySnapshot) {}));
  TEST_ASSERT_FALSE(service.active());
  TEST_ASSERT_EQUAL(0, g_mdns_mock.async_delete_call_count);
  TEST_ASSERT_EQUAL(0, g_mdns_mock.query_results_free_call_count);

  g_mdns_mock.async_new_succeeds = true;
  TEST_ASSERT_TRUE(service.start([](PeerDiscoverySnapshot) {}));
  service.shutdown();
  TEST_ASSERT_FALSE(service.active());
  TEST_ASSERT_EQUAL(1, g_mdns_mock.async_delete_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.query_results_free_call_count);
}

void test_peer_query_suppresses_delivery_after_wifi_loss() {
  reset_mocks();
  PeerMdnsFixture fixture;
  g_mdns_mock.async_results = &fixture.result;
  EspIdfPeerDiscoveryService service;
  service.set_wifi_ready(true);
  size_t completion_count = 0U;
  TEST_ASSERT_TRUE(service.start([&](PeerDiscoverySnapshot) { completion_count += 1U; }));
  service.set_wifi_ready(false);
  g_mdns_mock.async_get_results_finished = true;
  service.loop();
  TEST_ASSERT_EQUAL(0U, completion_count);
  TEST_ASSERT_FALSE(service.active());
  TEST_ASSERT_EQUAL(1, g_mdns_mock.query_results_free_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.async_delete_call_count);
}

void test_peer_query_repeated_cancellation_releases_each_operation_once() {
  reset_mocks();
  EspIdfPeerDiscoveryService service;
  size_t completion_count = 0U;
  for (size_t iteration = 0U; iteration < 4U; ++iteration) {
    service.set_wifi_ready(true);
    TEST_ASSERT_TRUE(service.start([&](PeerDiscoverySnapshot) { completion_count += 1U; }));
    TEST_ASSERT_TRUE(service.active());
    service.shutdown();
    service.shutdown();
    TEST_ASSERT_FALSE(service.active());
    TEST_ASSERT_EQUAL(iteration + 1U,
                      static_cast<size_t>(g_mdns_mock.async_delete_call_count));
    TEST_ASSERT_EQUAL(iteration + 1U,
                      static_cast<size_t>(g_mdns_mock.query_results_free_call_count));
  }
  TEST_ASSERT_EQUAL(0U, completion_count);
}

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_registers_identity_service_and_txt);
  RUN_TEST(test_follows_wifi_lifecycle_and_updates_txt_atomically);
  RUN_TEST(test_does_not_free_mdns_owned_by_another_component);
  RUN_TEST(test_attaches_without_mutating_existing_responder_identity);
  RUN_TEST(test_initializes_missing_shared_responder_with_fallback_hostname);
  RUN_TEST(test_bootstrap_answers_multicast_a_after_bounded_delay);
  RUN_TEST(test_bootstrap_handles_qu_and_legacy_unicast);
  RUN_TEST(test_bootstrap_answers_chrome_a_with_compressed_aaaa_question);
  RUN_TEST(test_bootstrap_negates_aaaa_without_advertising_ipv6);
  RUN_TEST(test_bootstrap_rejects_static_invalid_and_unsupported_queries);
  RUN_TEST(test_bootstrap_requires_ipv4_and_cancels_pending_responses);
  RUN_TEST(test_bootstrap_bounds_pending_pool_and_global_rate);
  RUN_TEST(test_bootstrap_wrapper_always_forwards_to_espressif);
  RUN_TEST(test_peer_results_are_bounded_validated_sorted_and_serializable);
  RUN_TEST(test_peer_results_reject_every_malformed_and_nonlocal_boundary);
  RUN_TEST(test_peer_results_enforce_address_device_and_serialized_size_limits);
  RUN_TEST(test_peer_query_uses_fixed_bounds_and_frees_results_exactly_once);
  RUN_TEST(test_peer_query_rejects_allocation_failure_without_leaking_state);
  RUN_TEST(test_peer_query_suppresses_delivery_after_wifi_loss);
  RUN_TEST(test_peer_query_repeated_cancellation_releases_each_operation_once);
  return espectre::test::end_suite();
}
