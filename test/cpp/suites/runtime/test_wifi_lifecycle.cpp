#include "test_harness.h"

#include "esp_event.h"
#include "esp_wifi.h"
#include "standalone_wifi_manager.h"
#include "wifi_lifecycle.h"

using namespace esphome::espectre;

void setUp(void) {
  esp_event_mock_reset();
  esp_wifi_mock_reset();
}

void tearDown(void) {}

void test_wifi_lifecycle_init_configures_protocol_bandwidth_and_promiscuous(void) {
  WiFiLifecycleManager manager;

  TEST_ASSERT_EQUAL(ESP_OK, manager.init());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11N, g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidth == WIFI_BW_HT20);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_promiscuous_call_count);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_promiscuous);
}

void test_wifi_lifecycle_init_falls_back_when_11n_only_is_rejected(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.set_protocol_results[0] = ESP_ERR_INVALID_ARG;
  g_esp_wifi_mock.set_protocol_results[1] = ESP_OK;
  g_esp_wifi_mock.set_protocol_result_count = 2;

  TEST_ASSERT_EQUAL(ESP_OK, manager.init());
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N,
                    g_esp_wifi_mock.last_protocol_bitmap);
}

void test_wifi_lifecycle_register_handlers_dispatches_and_unregisters(void) {
  WiFiLifecycleManager manager;
  int connected_calls = 0;
  int disconnected_calls = 0;

  TEST_ASSERT_EQUAL(
      ESP_OK, manager.register_handlers([&connected_calls]() { connected_calls++; },
                                        [&disconnected_calls]() { disconnected_calls++; }));
  TEST_ASSERT_EQUAL(2, g_esp_event_mock.register_call_count);

  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, nullptr);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);

  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, disconnected_calls);

  manager.unregister_handlers();
  TEST_ASSERT_EQUAL(2, g_esp_event_mock.unregister_call_count);

  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, nullptr);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);

  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, disconnected_calls);
}

void test_wifi_lifecycle_register_handlers_cleans_up_when_second_registration_fails(void) {
  WiFiLifecycleManager manager;
  g_esp_event_mock.register_results[0] = ESP_OK;
  g_esp_event_mock.register_results[1] = ESP_FAIL;
  g_esp_event_mock.register_result_count = 2;

  TEST_ASSERT_EQUAL(
      ESP_FAIL, manager.register_handlers([]() {}, []() {}));
  TEST_ASSERT_EQUAL(2, g_esp_event_mock.register_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_event_mock.unregister_call_count);
}

void test_standalone_wifi_manager_configures_fast_scan_bssid_and_channel(void) {
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";
  config.bssid = "aa:bb:cc:dd:ee:ff";
  config.channel = 10;

  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.init_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_storage_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_mode_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(WIFI_PS_MIN_MODEM, g_esp_wifi_mock.last_set_ps_type);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_EQUAL(WIFI_FAST_SCAN, g_esp_wifi_mock.last_config.sta.scan_method);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_config.sta.bssid_set);
  TEST_ASSERT_EQUAL_UINT8(10, g_esp_wifi_mock.last_config.sta.channel);
  TEST_ASSERT_EQUAL_UINT8(0xaa, g_esp_wifi_mock.last_config.sta.bssid[0]);
  TEST_ASSERT_EQUAL_UINT8(0xff, g_esp_wifi_mock.last_config.sta.bssid[5]);
}

void test_standalone_wifi_manager_applies_policy_and_connects_on_start(void) {
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.password = "secret";

  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, manager.start());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.start_call_count);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(WIFI_PS_MIN_MODEM, g_esp_wifi_mock.last_set_ps_type);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(WIFI_PROTOCOL_11N, g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
}

void test_standalone_wifi_manager_managed_lifecycle_dispatches_after_csi_init(void) {
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "TestSSID";
  config.manage_csi_lifecycle = true;
  int connected_calls = 0;
  int disconnected_calls = 0;

  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config,
                                         [&connected_calls]() { connected_calls++; },
                                         [&disconnected_calls]() { disconnected_calls++; }));

  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, nullptr);
  TEST_ASSERT_EQUAL(1, connected_calls);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_promiscuous_call_count);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_promiscuous);

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, nullptr);
  TEST_ASSERT_EQUAL(1, disconnected_calls);
}

void test_standalone_wifi_manager_get_info_reports_station_details(void) {
  StandaloneWifiManager manager;
  StandaloneWifiInfo info{};

  TEST_ASSERT_FALSE(manager.get_info(nullptr));
  TEST_ASSERT_TRUE(manager.get_info(&info));
  TEST_ASSERT_TRUE(info.connected);
  TEST_ASSERT_EQUAL_UINT8(6, info.channel);
  TEST_ASSERT_TRUE(std::string(info.ip_address).find('.') != std::string::npos);
  TEST_ASSERT_EQUAL_STRING("7C:2C:67:42:BB:AC", info.mac_address);
}

void test_standalone_wifi_manager_update_station_config_handles_setup_and_reconnect_paths(void) {
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "InitialSSID";
  config.password = "secret";

  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, manager.update_station_config(config));

  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, manager.update_station_config(config));
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  TEST_ASSERT_EQUAL(ESP_OK, manager.start());
  StandaloneWifiConfig empty = config;
  empty.ssid = "";
  TEST_ASSERT_EQUAL(ESP_OK, manager.update_station_config(empty));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.connect_call_count);

  TEST_ASSERT_EQUAL(ESP_OK, manager.update_station_config(config));
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.set_ps_call_count);
}

void test_standalone_wifi_manager_update_station_config_rejects_invalid_bssid(void) {
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "SSID";

  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config));
  config.bssid = "not-a-bssid";
  TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, manager.update_station_config(config));
}

void test_standalone_wifi_manager_apply_started_policy_and_retry_logic(void) {
  TEST_ASSERT_EQUAL(ESP_OK, StandaloneWifiManager::apply_started_csi_policy());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_ps_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);

  esp_wifi_mock_reset();
  g_esp_wifi_mock.set_protocol_results[0] = ESP_FAIL;
  g_esp_wifi_mock.set_protocol_results[1] = ESP_FAIL;
  g_esp_wifi_mock.set_protocol_result_count = 2;
  TEST_ASSERT_EQUAL(ESP_FAIL, StandaloneWifiManager::apply_started_csi_policy());

  esp_wifi_mock_reset();
  StandaloneWifiManager manager;
  StandaloneWifiConfig config;
  config.ssid = "SSID";
  config.max_retry = 2;
  TEST_ASSERT_EQUAL(ESP_OK, manager.setup(config));
  TEST_ASSERT_EQUAL(ESP_OK, manager.start());

  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.connect_call_count);

  wifi_event_sta_disconnected_t event{};
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &event);
  TEST_ASSERT_EQUAL(3, g_esp_wifi_mock.connect_call_count);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_lifecycle_init_configures_protocol_bandwidth_and_promiscuous);
  RUN_TEST(test_wifi_lifecycle_init_falls_back_when_11n_only_is_rejected);
  RUN_TEST(test_wifi_lifecycle_register_handlers_dispatches_and_unregisters);
  RUN_TEST(test_wifi_lifecycle_register_handlers_cleans_up_when_second_registration_fails);
  RUN_TEST(test_standalone_wifi_manager_configures_fast_scan_bssid_and_channel);
  RUN_TEST(test_standalone_wifi_manager_applies_policy_and_connects_on_start);
  RUN_TEST(test_standalone_wifi_manager_managed_lifecycle_dispatches_after_csi_init);
  RUN_TEST(test_standalone_wifi_manager_get_info_reports_station_details);
  RUN_TEST(test_standalone_wifi_manager_update_station_config_handles_setup_and_reconnect_paths);
  RUN_TEST(test_standalone_wifi_manager_update_station_config_rejects_invalid_bssid);
  RUN_TEST(test_standalone_wifi_manager_apply_started_policy_and_retry_logic);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
