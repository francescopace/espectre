/*
 * ESPectre - WiFi Lifecycle Unit Tests
 *
 * Unit tests for WiFi Lifecycle.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "esp_event.h"
#include "esp_wifi.h"
#include "standalone_wifi_service.h"
#include "wifi_lifecycle.h"

using namespace espectre;

namespace espectre {

struct StandaloneWifiServiceTestAccess {
  static bool deferred_connect_fallback_pending(const StandaloneWifiService &service) {
    return service.deferred_connect_fallback_pending_;
  }

  static void expire_deferred_connect_fallback(StandaloneWifiService &service) {
    service.deferred_connect_fallback_deadline_us_ = 0U;
  }
};

}  // namespace espectre

void setUp(void) {
  esp_event_mock_reset();
  esp_netif_mock_reset();
  esp_wifi_mock_reset();
}

void tearDown(void) {}

void test_wifi_lifecycle_init_configures_protocol_bandwidth_and_promiscuous(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_EQUAL(ESP_OK, manager.process_pending_events());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidth == WIFI_BW_HT20);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_promiscuous_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(WIFI_PS_NONE, g_esp_wifi_mock.last_set_ps_type);
}

void test_wifi_lifecycle_init_reports_bgn_configuration_failure(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.set_protocol_results[0] = ESP_FAIL;
  g_esp_wifi_mock.set_protocol_result_count = 1;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_EQUAL(ESP_FAIL, manager.process_pending_events());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
}

void test_wifi_lifecycle_rejects_dual_band_policies_on_single_band_targets(void) {
  WiFiLifecycleManager manager;
  TEST_ASSERT_EQUAL(ESP_ERR_NOT_SUPPORTED,
                    manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                              WifiBandPolicy::BAND_5G));
  TEST_ASSERT_EQUAL(0, g_esp_event_mock.register_call_count);

  TEST_ASSERT_EQUAL(ESP_ERR_NOT_SUPPORTED,
                    manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                              WifiBandPolicy::AUTO));
  TEST_ASSERT_EQUAL(0, g_esp_event_mock.register_call_count);
}

// STA_START can fire before the handlers are registered, for instance when a
// host frontend brings the station up first. The policy is then never
// attempted, and failing at GOT_IP consumed the event and left CSI off with
// nothing to retry until the next reconnect. Applying it late is valid because
// the station is already up.
void test_wifi_lifecycle_applies_policy_late_when_sta_start_was_missed(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  // Deliberately no WIFI_EVENT_STA_START.
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocol_call_count);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);

  TEST_ASSERT_EQUAL(ESP_OK, manager.process_pending_events());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidth == WIFI_BW_HT20);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_ps_call_count);
}

// A policy that was attempted and failed is a radio failure, not an ordering
// accident, so it must still propagate rather than be retried at GOT_IP.
void test_wifi_lifecycle_does_not_retry_a_policy_that_actually_failed(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.set_protocol_results[0] = ESP_FAIL;
  g_esp_wifi_mock.set_protocol_result_count = 1;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);

  TEST_ASSERT_EQUAL(ESP_FAIL, manager.process_pending_events());
  // One attempt only: the mock would have succeeded on a second call.
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
}

void test_wifi_lifecycle_started_policy_skips_matching_radio_settings(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.protocol_bitmap = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT20;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
}

void test_wifi_lifecycle_register_handlers_dispatches_and_unregisters(void) {
  WiFiLifecycleManager manager;
  int connected_calls = 0;
  int disconnected_calls = 0;
  esp_netif_ip_info_t observed_ip_info{};

  TEST_ASSERT_EQUAL(
      ESP_OK, manager.register_handlers([&](const esp_netif_ip_info_t &ip_info) {
                                          connected_calls++;
                                          observed_ip_info = ip_info;
                                        },
                                        [&disconnected_calls]() { disconnected_calls++; }));
  TEST_ASSERT_EQUAL(3, g_esp_event_mock.register_call_count);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  event.ip_info.gw.addr = 0x0101A8C0U;
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);

  // Handlers only record events; callbacks fire from process_pending_events().
  TEST_ASSERT_EQUAL(0, connected_calls);
  TEST_ASSERT_EQUAL(0, disconnected_calls);

  manager.process_pending_events();
  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, disconnected_calls);
  TEST_ASSERT_EQUAL(event.ip_info.gw.addr, observed_ip_info.gw.addr);

  manager.unregister_handlers();
  TEST_ASSERT_EQUAL(3, g_esp_event_mock.unregister_call_count);

  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);

  manager.process_pending_events();
  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, disconnected_calls);
}

void test_wifi_lifecycle_register_handlers_cleans_up_when_second_registration_fails(void) {
  WiFiLifecycleManager manager;
  g_esp_event_mock.register_results[0] = ESP_OK;
  g_esp_event_mock.register_results[1] = ESP_FAIL;
  g_esp_event_mock.register_result_count = 2;

  TEST_ASSERT_EQUAL(
      ESP_FAIL, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  TEST_ASSERT_EQUAL(2, g_esp_event_mock.register_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_event_mock.unregister_call_count);
}

void test_standalone_wifi_service_configures_fast_scan_bssid_and_channel(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";
  config.bssid = "aa:bb:cc:dd:ee:ff";
  config.channel = 10;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.init_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_storage_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_mode_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_EQUAL(WIFI_FAST_SCAN, g_esp_wifi_mock.last_config.sta.scan_method);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_config.sta.bssid_set);
  TEST_ASSERT_EQUAL_UINT8(10, g_esp_wifi_mock.last_config.sta.channel);
  TEST_ASSERT_EQUAL_UINT8(0xaa, g_esp_wifi_mock.last_config.sta.bssid[0]);
  TEST_ASSERT_EQUAL_UINT8(0xff, g_esp_wifi_mock.last_config.sta.bssid[5]);
}

void test_standalone_wifi_service_applies_policy_and_connects_on_start(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";
  config.manage_csi_lifecycle = true;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.start());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.start_call_count);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
}

void test_standalone_wifi_service_unmanaged_applies_policy_before_connect(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";
  config.manage_csi_lifecycle = false;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.start());

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);
}

void test_standalone_wifi_service_reconnects_after_sta_stop(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.start());

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  // A later STA_START without STOP must not double-connect.
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  // Protocol/coexistence restarts clear the latch so association can resume.
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_STOP, nullptr);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
}

void test_standalone_wifi_service_runs_deferred_connect_fallback_once(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.start());

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);
  TEST_ASSERT_TRUE(StandaloneWifiServiceTestAccess::deferred_connect_fallback_pending(service));

  StandaloneWifiServiceTestAccess::expire_deferred_connect_fallback(service);
  service.loop();
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
  TEST_ASSERT_FALSE(StandaloneWifiServiceTestAccess::deferred_connect_fallback_pending(service));

  service.loop();
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
}

void test_standalone_wifi_service_managed_lifecycle_dispatches_after_csi_init(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.manage_csi_lifecycle = true;
  int connected_calls = 0;
  int disconnected_calls = 0;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config,
                                         [&connected_calls]() { connected_calls++; },
                                         [&disconnected_calls]() { disconnected_calls++; }));

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  event.ip_info.gw.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_EQUAL(0, connected_calls);

  service.loop();
  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(WIFI_PS_NONE, g_esp_wifi_mock.last_set_ps_type);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_promiscuous_call_count);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_promiscuous);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);
  service.loop();
  TEST_ASSERT_EQUAL(1, disconnected_calls);
}

void test_standalone_wifi_service_get_info_reports_station_details(void) {
  StandaloneWifiService service;
  StandaloneWifiInfo info{};
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";

  TEST_ASSERT_FALSE(service.get_info(nullptr));
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = g_esp_netif_mock.ip_addr;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_TRUE(service.get_info(&info));
  TEST_ASSERT_TRUE(info.connected);
  TEST_ASSERT_EQUAL_UINT8(6, info.channel);
  TEST_ASSERT_TRUE(std::string(info.ip_address).find('.') != std::string::npos);
  TEST_ASSERT_EQUAL_STRING("7C:2C:67:42:BB:AC", info.mac_address);
}

void test_standalone_wifi_service_get_info_uses_cached_ip_from_got_ip_event(void) {
  StandaloneWifiService service;
  StandaloneWifiInfo info{};
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));

  g_esp_netif_mock.ip_addr = 0U;

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr =
      ((uint32_t)192U << 0U) | ((uint32_t)168U << 8U) | ((uint32_t)1U << 16U) | ((uint32_t)55U << 24U);
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);

  TEST_ASSERT_TRUE(service.get_info(&info));
  TEST_ASSERT_EQUAL_STRING("192.168.1.55", info.ip_address);

  wifi_event_sta_disconnected_t disconnect_event{};
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &disconnect_event);

  StandaloneWifiInfo after_disconnect{};
  TEST_ASSERT_TRUE(service.get_info(&after_disconnect));
  TEST_ASSERT_EQUAL_STRING("", after_disconnect.ip_address);
}

void test_standalone_wifi_service_update_station_config_handles_setup_and_reconnect_paths(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "InitialSSID";
  config.password = "secret";

  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, service.update_station_config(config));

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.update_station_config(config));
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  TEST_ASSERT_EQUAL(ESP_OK, service.start());
  StandaloneWifiConfig empty = config;
  empty.ssid = "";
  TEST_ASSERT_EQUAL(ESP_OK, service.update_station_config(empty));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  TEST_ASSERT_EQUAL(ESP_OK, service.update_station_config(config));
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
}

void test_standalone_wifi_service_update_station_config_rejects_invalid_bssid(void) {
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "SSID";

  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  config.bssid = "not-a-bssid";
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, service.update_station_config(config));
}

void test_standalone_wifi_service_apply_started_policy_and_reconnect_logic(void) {
  WiFiLifecycleManager lifecycle;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;
  TEST_ASSERT_EQUAL(ESP_OK,
                    lifecycle.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);

  lifecycle.unregister_handlers();
  esp_wifi_mock_reset();
  g_esp_wifi_mock.set_protocol_results[0] = ESP_FAIL;
  g_esp_wifi_mock.set_protocol_result_count = 1;
  WiFiLifecycleManager failing_lifecycle;
  TEST_ASSERT_EQUAL(ESP_OK,
                    failing_lifecycle.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  ip_event_got_ip_t got_ip{};
  got_ip.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &got_ip);
  TEST_ASSERT_EQUAL(ESP_FAIL, failing_lifecycle.process_pending_events());

  failing_lifecycle.unregister_handlers();
  esp_wifi_mock_reset();
  StandaloneWifiService service;
  StandaloneWifiConfig config;
  config.ssid = "SSID";
  config.max_retry = 2;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, service.start());

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  wifi_event_sta_disconnected_t event{};
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.connect_call_count);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_lifecycle_init_configures_protocol_bandwidth_and_promiscuous);
  RUN_TEST(test_wifi_lifecycle_init_reports_bgn_configuration_failure);
  RUN_TEST(test_wifi_lifecycle_rejects_dual_band_policies_on_single_band_targets);
  RUN_TEST(test_wifi_lifecycle_applies_policy_late_when_sta_start_was_missed);
  RUN_TEST(test_wifi_lifecycle_does_not_retry_a_policy_that_actually_failed);
  RUN_TEST(test_wifi_lifecycle_started_policy_skips_matching_radio_settings);
  RUN_TEST(test_wifi_lifecycle_register_handlers_dispatches_and_unregisters);
  RUN_TEST(test_wifi_lifecycle_register_handlers_cleans_up_when_second_registration_fails);
  RUN_TEST(test_standalone_wifi_service_configures_fast_scan_bssid_and_channel);
  RUN_TEST(test_standalone_wifi_service_applies_policy_and_connects_on_start);
  RUN_TEST(test_standalone_wifi_service_unmanaged_applies_policy_before_connect);
  RUN_TEST(test_standalone_wifi_service_reconnects_after_sta_stop);
  RUN_TEST(test_standalone_wifi_service_runs_deferred_connect_fallback_once);
  RUN_TEST(test_standalone_wifi_service_managed_lifecycle_dispatches_after_csi_init);
  RUN_TEST(test_standalone_wifi_service_get_info_reports_station_details);
  RUN_TEST(test_standalone_wifi_service_get_info_uses_cached_ip_from_got_ip_event);
  RUN_TEST(test_standalone_wifi_service_update_station_config_handles_setup_and_reconnect_paths);
  RUN_TEST(test_standalone_wifi_service_update_station_config_rejects_invalid_bssid);
  RUN_TEST(test_standalone_wifi_service_apply_started_policy_and_reconnect_logic);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
