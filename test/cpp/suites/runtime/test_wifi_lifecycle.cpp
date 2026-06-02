#include "test_harness.h"

#include "esp_event.h"
#include "esp_wifi.h"
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

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_lifecycle_init_configures_protocol_bandwidth_and_promiscuous);
  RUN_TEST(test_wifi_lifecycle_init_falls_back_when_11n_only_is_rejected);
  RUN_TEST(test_wifi_lifecycle_register_handlers_dispatches_and_unregisters);
  RUN_TEST(test_wifi_lifecycle_register_handlers_cleans_up_when_second_registration_fails);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
