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
#include "direct_websocket_protocol.h"
#include "mdns.h"
#include "mdns_discovery_service.h"
#include "native_shared_mdns_alias.h"
#include "peer_discovery.h"
#include "peer_discovery_service_esp_idf.h"
#include "streamer_discovery_service.h"

using namespace espectre;

namespace {

extern "C" size_t __wrap_mdns_priv_if_write(mdns_if_t tcpip_if,
                                             mdns_ip_protocol_t ip_protocol,
                                             const esp_ip_addr_t *ip,
                                             uint16_t port,
                                             uint8_t *data,
                                             size_t len);

MdnsDiscoveryServiceConfig direct_config() {
  return {
      "espectre-0123456789abcdef",
      "Kitchen sensor",
      "_espectre",
      "_tcp",
      80U,
      {{"device_id", "0123456789abcdef"}, {"path", "/espectre/v1/ws"}, {"protovers", "1"}},
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
  TEST_ASSERT_EQUAL(3U, g_mdns_mock.txt_count);
  TEST_ASSERT_EQUAL_STRING("device_id", g_mdns_mock.txt_keys[0]);
  TEST_ASSERT_EQUAL_STRING("0123456789abcdef", g_mdns_mock.txt_values[0]);
  TEST_ASSERT_EQUAL_STRING("path", g_mdns_mock.txt_keys[1]);
  TEST_ASSERT_EQUAL_STRING("/espectre/v1/ws", g_mdns_mock.txt_values[1]);
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
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.init_call_count);
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.hostname_set_call_count);
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.instance_name_set_call_count);
  TEST_ASSERT_TRUE(service.service_enabled());
  service.on_wifi_connected();
  service.on_wifi_disconnected();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.netif_action_call_count);
  service.shutdown();
  TEST_ASSERT_EQUAL_INT(0, g_mdns_mock.free_call_count);
}

void test_streamer_advertises_canonical_direct_service() {
  reset_mocks();
  StreamerDiscoveryService service;
  TEST_ASSERT_TRUE(service.setup(StreamerDiscoveryServiceConfig{0x1234U, "esp32c3", 80U, 5501U}));
  TEST_ASSERT_EQUAL_STRING("_espectre", g_mdns_mock.service_type);
  TEST_ASSERT_EQUAL_STRING("_tcp", g_mdns_mock.service_proto);
  TEST_ASSERT_EQUAL(80U, g_mdns_mock.service_port);
  TEST_ASSERT_EQUAL_STRING("device_id", g_mdns_mock.txt_keys[0]);
  TEST_ASSERT_EQUAL_STRING("0000000000001234", g_mdns_mock.txt_values[0]);
  TEST_ASSERT_EQUAL_STRING("frontend", g_mdns_mock.txt_keys[2]);
  TEST_ASSERT_EQUAL_STRING("streamer", g_mdns_mock.txt_values[2]);
}

struct SharedAliasPacket {
  std::vector<uint8_t> bytes;
  size_t class_offset{0U};
  size_t ttl_offset{0U};
};

SharedAliasPacket shared_alias_answer_packet() {
  SharedAliasPacket packet;
  packet.bytes.resize(12U, 0U);
  packet.bytes[7] = 1U;
  const auto append_label = [&packet](const char *label) {
    const size_t length = std::strlen(label);
    packet.bytes.push_back(static_cast<uint8_t>(length));
    packet.bytes.insert(packet.bytes.end(), label, label + length);
  };
  append_label("espectre-devices");
  append_label("local");
  packet.bytes.push_back(0U);
  packet.bytes.push_back(0U);
  packet.bytes.push_back(MDNS_TYPE_A);
  packet.class_offset = packet.bytes.size();
  packet.bytes.push_back(0x80U);
  packet.bytes.push_back(0x01U);
  packet.ttl_offset = packet.bytes.size();
  packet.bytes.insert(packet.bytes.end(), {0U, 0U, 0U, 120U, 0U, 4U, 192U, 168U, 1U, 42U});
  return packet;
}

void test_shared_alias_owns_add_update_goodbye_remove_and_packet_class() {
  reset_mocks();
  NativeSharedMdnsAlias alias;
  TEST_ASSERT_FALSE(alias.setup(""));
  TEST_ASSERT_FALSE(alias.setup("ESPectre-devices"));
  TEST_ASSERT_TRUE(alias.setup("espectre-devices"));

  const uint32_t first = ipv4(192U, 168U, 1U, 42U);
  const uint32_t second = ipv4(192U, 168U, 1U, 43U);
  TEST_ASSERT_TRUE(alias.update(first));
  TEST_ASSERT_TRUE(alias.published());
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_add_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.private_announce_count);
  TEST_ASSERT_EQUAL(MDNS_FLAGS_QR_AUTHORITATIVE, g_mdns_mock.last_private_packet_flags);
  TEST_ASSERT_EQUAL(MDNS_TYPE_A, g_mdns_mock.last_private_answer_type);
  TEST_ASSERT_FALSE(g_mdns_mock.last_private_answer_flush);
  TEST_ASSERT_FALSE(g_mdns_mock.last_private_answer_goodbye);

  TEST_ASSERT_TRUE(alias.update(first));
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_add_call_count);
  TEST_ASSERT_EQUAL(0, g_mdns_mock.delegate_set_address_call_count);
  TEST_ASSERT_EQUAL(2, g_mdns_mock.private_announce_count);

  TEST_ASSERT_TRUE(alias.update(second));
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_set_address_call_count);
  TEST_ASSERT_EQUAL(second, g_mdns_mock.delegated_ipv4_address);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.private_goodbye_count);
  TEST_ASSERT_EQUAL(3, g_mdns_mock.private_announce_count);

  SharedAliasPacket packet = shared_alias_answer_packet();
  TEST_ASSERT_EQUAL(0x80U, packet.bytes[packet.class_offset]);
  TEST_ASSERT_EQUAL(packet.bytes.size(),
                    __wrap_mdns_priv_if_write(0U,
                                              MDNS_IP_PROTOCOL_V4,
                                              nullptr,
                                              5353U,
                                              packet.bytes.data(),
                                              packet.bytes.size()));
  TEST_ASSERT_EQUAL(1, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(0U, g_mdns_mock.last_write_packet[packet.class_offset] & 0x80U);
  TEST_ASSERT_EQUAL(10U, g_mdns_mock.last_write_packet[packet.ttl_offset + 3U]);

  SharedAliasPacket goodbye = shared_alias_answer_packet();
  goodbye.bytes[goodbye.ttl_offset + 3U] = 0U;
  TEST_ASSERT_EQUAL(goodbye.bytes.size(),
                    __wrap_mdns_priv_if_write(0U,
                                              MDNS_IP_PROTOCOL_V4,
                                              nullptr,
                                              5353U,
                                              goodbye.bytes.data(),
                                              goodbye.bytes.size()));
  TEST_ASSERT_EQUAL(2, g_mdns_mock.real_write_call_count);
  TEST_ASSERT_EQUAL(0U, g_mdns_mock.last_write_packet[goodbye.ttl_offset + 3U]);

  alias.shutdown();
  TEST_ASSERT_FALSE(alias.published());
  TEST_ASSERT_EQUAL(2, g_mdns_mock.private_goodbye_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_remove_call_count);
  alias.shutdown();
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_remove_call_count);
}

void test_shared_alias_rolls_back_partial_publication_and_can_retry() {
  reset_mocks();
  NativeSharedMdnsAlias alias;
  TEST_ASSERT_TRUE(alias.setup("espectre-devices"));
  g_mdns_mock.private_alloc_succeeds = false;
  TEST_ASSERT_FALSE(alias.update(ipv4(192U, 168U, 1U, 42U)));
  TEST_ASSERT_FALSE(alias.published());
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_add_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.delegate_remove_call_count);
  TEST_ASSERT_EQUAL(0, g_mdns_mock.private_dispatch_call_count);

  g_mdns_mock.private_alloc_succeeds = true;
  g_mdns_mock.private_create_answer_succeeds = false;
  TEST_ASSERT_FALSE(alias.update(ipv4(192U, 168U, 1U, 42U)));
  TEST_ASSERT_FALSE(alias.published());
  TEST_ASSERT_EQUAL(2, g_mdns_mock.delegate_add_call_count);
  TEST_ASSERT_EQUAL(2, g_mdns_mock.delegate_remove_call_count);
  TEST_ASSERT_EQUAL(1, g_mdns_mock.private_free_packet_call_count);

  g_mdns_mock.private_create_answer_succeeds = true;
  TEST_ASSERT_TRUE(alias.update(ipv4(192U, 168U, 1U, 42U)));
  TEST_ASSERT_TRUE(alias.published());
  alias.shutdown();
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
  candidate.txt_version = "1";
  candidate.protocol_version = "1";
  candidate.path = ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT;
  candidate.firmware = "3.0.0-rc1";
  candidate.chip = "esp32c3";
  candidate.tls = "0";
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
  mdns_txt_item_t txt[10] = {
      {"device_id", "2222222222222222"},
      {"name", "Office sensor"},
      {"frontend", "native"},
      {"txtvers", "1"},
      {"protovers", "1"},
      {"path", "/espectre/v1/ws"},
      {"firmware", "3.0.0-rc1"},
      {"chip", "esp32c3"},
      {"tls", "0"},
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
  PeerDiscoveryCandidate first = peer("1111111111111111", "espectre-1", station + (1U << 24U));
  first.ipv4_addresses.push_back(8U | (8U << 8U) | (8U << 16U) | (8U << 24U));
  PeerDiscoveryCandidate malformed = peer("not-an-identity", "bad", station + (3U << 24U));
  PeerDiscoveryCandidate conflict = first;
  conflict.hostname = "conflicting-host";

  const PeerDiscoverySnapshot snapshot = validate_peer_discovery_candidates(
      {second, malformed, first, conflict}, station, netmask, 3000U, false);
  TEST_ASSERT_EQUAL(1U, snapshot.devices.size());
  TEST_ASSERT_EQUAL_STRING("2222222222222222", snapshot.devices[0].device_id.c_str());
  TEST_ASSERT_TRUE(snapshot.rejected_results >= 2U);
  const std::string payload = peer_discovery_snapshot_json(snapshot);
  TEST_ASSERT_TRUE(payload.size() <= ESPECTRE_PEER_DISCOVERY_MAX_RESULT_SIZE);
  TEST_ASSERT_TRUE(payload.find("\"schema_version\":1") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("192.168.1.102") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("8.8.8.8") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("peer_discovery") != std::string::npos);
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
  reject([](auto &candidate) { candidate.path = "/espectre/v1/ws?token=x"; });
  reject([](auto &candidate) { candidate.firmware.assign(49U, 'x'); });
  reject([](auto &candidate) { candidate.chip = "esp32.c3"; });
  reject([](auto &candidate) { candidate.tls = "1"; });
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
  RUN_TEST(test_streamer_advertises_canonical_direct_service);
  RUN_TEST(test_shared_alias_owns_add_update_goodbye_remove_and_packet_class);
  RUN_TEST(test_shared_alias_rolls_back_partial_publication_and_can_retry);
  RUN_TEST(test_peer_results_are_bounded_validated_sorted_and_serializable);
  RUN_TEST(test_peer_results_reject_every_malformed_and_nonlocal_boundary);
  RUN_TEST(test_peer_results_enforce_address_device_and_serialized_size_limits);
  RUN_TEST(test_peer_query_uses_fixed_bounds_and_frees_results_exactly_once);
  RUN_TEST(test_peer_query_rejects_allocation_failure_without_leaking_state);
  RUN_TEST(test_peer_query_suppresses_delivery_after_wifi_loss);
  RUN_TEST(test_peer_query_repeated_cancellation_releases_each_operation_once);
  return espectre::test::end_suite();
}
