/*
 * ESPectre - Shared mDNS Discovery Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <string>

#include "esp_netif.h"
#include "mdns.h"
#include "mdns_discovery_service.h"
#include "streamer_discovery_service.h"

using namespace espectre;

namespace {

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

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_registers_identity_service_and_txt);
  RUN_TEST(test_follows_wifi_lifecycle_and_updates_txt_atomically);
  RUN_TEST(test_does_not_free_mdns_owned_by_another_component);
  RUN_TEST(test_attaches_without_mutating_existing_responder_identity);
  RUN_TEST(test_streamer_advertises_canonical_direct_service);
  return espectre::test::end_suite();
}
