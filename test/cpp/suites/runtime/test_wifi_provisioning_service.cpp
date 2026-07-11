#include "test_harness.h"

#include "device_config_store.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "nvs.h"
#include "standalone_wifi_service.h"
#include "wifi_provisioning_service.h"

using namespace esphome::espectre;

namespace {

WifiProvisioningDefaults make_defaults() {
  WifiProvisioningDefaults defaults;
  defaults.ssid = "DefaultSSID";
  defaults.password = "default-secret";
  defaults.bssid = "11:22:33:44:55:66";
  defaults.channel = 6;
  defaults.max_retry = 4;
  defaults.manage_csi_lifecycle = false;
  return defaults;
}

}  // namespace

void setUp(void) {
  esp_event_mock_reset();
  esp_wifi_mock_reset();
  nvs_mock_reset();
}

void tearDown(void) {}

void test_wifi_provisioning_loads_defaults_when_no_saved_config(void) {
  WifiProvisioningService service(nullptr);

  TEST_ASSERT_EQUAL(ESP_OK, service.load_or_set_defaults(make_defaults()));

  TEST_ASSERT_EQUAL(ESP_OK, service.last_load_result());
  TEST_ASSERT_FALSE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("DefaultSSID", service.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("default-secret", service.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("11:22:33:44:55:66", service.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(6, service.config().channel);
  TEST_ASSERT_TRUE(service.password_set());
}

void test_wifi_provisioning_loads_saved_config(void) {
  StoredWifiConfig saved;
  saved.ssid = "SavedSSID";
  saved.password = "saved-secret";
  saved.bssid = "aa:bb:cc:dd:ee:ff";
  saved.channel = 11;
  saved.has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(saved));

  WifiProvisioningService service(nullptr);
  TEST_ASSERT_EQUAL(ESP_OK, service.load_or_set_defaults(make_defaults()));

  TEST_ASSERT_TRUE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("SavedSSID", service.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("saved-secret", service.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", service.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(11, service.config().channel);
}

void test_wifi_provisioning_records_load_error_and_falls_back_to_defaults(void) {
  nvs_mock_set_open_result(ESP_FAIL);
  WifiProvisioningService service(nullptr);

  TEST_ASSERT_EQUAL(ESP_OK, service.load_or_set_defaults(make_defaults()));

  TEST_ASSERT_EQUAL(ESP_FAIL, service.last_load_result());
  TEST_ASSERT_FALSE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("DefaultSSID", service.config().ssid.c_str());
}

void test_wifi_provisioning_commands_validate_and_persist_config(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));

  TEST_ASSERT_FALSE(service.handle_command("SET_WIFI_CONFIG:ssid=&password=secret&channel=9", &message));
  TEST_ASSERT_EQUAL_STRING("SSID must be 1..32 bytes", message.c_str());
  TEST_ASSERT_FALSE(service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&channel=15", &message));
  TEST_ASSERT_EQUAL_STRING("channel must be 0..14", message.c_str());

  TEST_ASSERT_TRUE(
      service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=9", &message));

  WifiProvisioningService reloaded(nullptr);
  TEST_ASSERT_EQUAL(ESP_OK, reloaded.load_or_set_defaults(make_defaults()));
  TEST_ASSERT_TRUE(reloaded.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("Lab", reloaded.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("secret", reloaded.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", reloaded.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(9, reloaded.config().channel);
}

void test_wifi_provisioning_apply_updates_wifi_manager_live(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_TRUE(service.handle_command("SET_WIFI_CONFIG:ssid=Applied&password=applied-secret&channel=3", &message));

  TEST_ASSERT_EQUAL_STRING("Wi-Fi config applied", message.c_str());
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_EQUAL_STRING("Applied", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL_STRING("applied-secret", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.password));
  TEST_ASSERT_EQUAL_UINT8(3, g_esp_wifi_mock.last_config.sta.channel);
}

void test_wifi_provisioning_batch_command_persists_and_applies_config(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_TRUE(service.handle_command(
      "SET_WIFI_CONFIG:ssid=Lab%20Net&password=top%20secret&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=9",
      &message));

  TEST_ASSERT_EQUAL_STRING("Wi-Fi config applied", message.c_str());
  TEST_ASSERT_TRUE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("Lab Net", service.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("top secret", service.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", service.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(9, service.config().channel);
  TEST_ASSERT_EQUAL_STRING("Lab Net", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL_STRING("top secret", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.password));
  TEST_ASSERT_EQUAL_UINT8(9, g_esp_wifi_mock.last_config.sta.channel);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_provisioning_loads_defaults_when_no_saved_config);
  RUN_TEST(test_wifi_provisioning_loads_saved_config);
  RUN_TEST(test_wifi_provisioning_records_load_error_and_falls_back_to_defaults);
  RUN_TEST(test_wifi_provisioning_commands_validate_and_persist_config);
  RUN_TEST(test_wifi_provisioning_apply_updates_wifi_manager_live);
  RUN_TEST(test_wifi_provisioning_batch_command_persists_and_applies_config);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
