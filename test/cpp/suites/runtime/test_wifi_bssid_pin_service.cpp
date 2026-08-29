/*
 * ESPectre - Wi-Fi BSSID Pin Service Tests
 *
 * Covers SSID-bound persistence, verified commit, rollback, and Matter-owned
 * network changes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <string>
#include <vector>

#include "esp_timer.h"
#include "nvs.h"
#include "wifi_bssid_pin_service.h"

using namespace espectre;

namespace {

WifiBssidPinStationState g_station;
std::vector<std::string> g_applied_pins;
int g_prepare_calls = 0;
int g_resume_calls = 0;

WifiBssidPinServiceConfig make_config() {
  WifiBssidPinServiceConfig config;
  config.station_state_getter = []() { return g_station; };
  config.apply_callback = [](const std::string &bssid, std::string *) {
    g_applied_pins.push_back(bssid);
    return true;
  };
  config.prepare_callback = []() { ++g_prepare_calls; };
  config.resume_callback = []() { ++g_resume_calls; };
  config.candidate_timeout_ms = 30000U;
  return config;
}

void set_connected_station(const char *ssid, const char *bssid) {
  g_station.configured = true;
  g_station.connected = true;
  g_station.has_ipv4 = true;
  g_station.ssid = ssid;
  g_station.bssid = bssid;
}

void seed_pin(const char *ssid, const char *bssid) {
  set_connected_station(ssid, bssid);
  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));
  std::string message;
  TEST_ASSERT_TRUE(service.request_update(bssid, &message));
  TEST_ASSERT_EQUAL_STRING(bssid, service.stored_bssid().c_str());
  g_applied_pins.clear();
  g_prepare_calls = 0;
  g_resume_calls = 0;
}

}  // namespace

void setUp(void) {
  nvs_mock_reset();
  esp_timer_mock::reset(0, 0);
  g_station = {};
  g_applied_pins.clear();
  g_prepare_calls = 0;
  g_resume_calls = 0;
}

void tearDown(void) {}

void test_wifi_bssid_pin_service_commits_only_after_verified_ipv4(void) {
  set_connected_station("MatterLab", "11:22:33:44:55:66");
  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));

  std::string message;
  TEST_ASSERT_TRUE(service.request_update("aa:bb:cc:dd:ee:ff", &message));
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::VERIFYING);
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", g_applied_pins.back().c_str());
  TEST_ASSERT_TRUE(service.stored_bssid().empty());
  TEST_ASSERT_EQUAL(1, g_prepare_calls);
  TEST_ASSERT_EQUAL(0, g_resume_calls);

  g_station.connected = false;
  g_station.has_ipv4 = false;
  g_station.bssid.clear();
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.stored_bssid().empty());

  set_connected_station("MatterLab", "AA:BB:CC:DD:EE:FF");
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::APPLIED);
  TEST_ASSERT_EQUAL_STRING("MatterLab", service.stored_ssid().c_str());
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", service.stored_bssid().c_str());
  TEST_ASSERT_EQUAL(1, g_resume_calls);

  WifiBssidPinService reloaded;
  TEST_ASSERT_EQUAL(ESP_OK, reloaded.setup(make_config()));
  TEST_ASSERT_EQUAL_STRING("MatterLab", reloaded.stored_ssid().c_str());
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", reloaded.stored_bssid().c_str());
}

void test_wifi_bssid_pin_service_restores_the_previous_pin_after_timeout(void) {
  seed_pin("MatterLab", "11:22:33:44:55:66");
  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));

  std::string message;
  TEST_ASSERT_TRUE(service.request_update("AA:BB:CC:DD:EE:FF", &message));
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", g_applied_pins.back().c_str());

  esp_timer_mock::advance(30000000);
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::ROLLING_BACK);
  TEST_ASSERT_EQUAL_STRING("11:22:33:44:55:66", g_applied_pins.back().c_str());

  set_connected_station("MatterLab", "11:22:33:44:55:66");
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::ROLLED_BACK);
  TEST_ASSERT_EQUAL_STRING("11:22:33:44:55:66", service.stored_bssid().c_str());
  TEST_ASSERT_EQUAL(1, g_prepare_calls);
  TEST_ASSERT_EQUAL(1, g_resume_calls);
}

void test_wifi_bssid_pin_service_reapplies_a_stored_pin_after_boot(void) {
  seed_pin("MatterLab", "AA:BB:CC:DD:EE:FF");
  set_connected_station("MatterLab", "11:22:33:44:55:66");

  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::VERIFYING);
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", g_applied_pins.back().c_str());

  set_connected_station("MatterLab", "AA:BB:CC:DD:EE:FF");
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::APPLIED);
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", service.stored_bssid().c_str());
  TEST_ASSERT_EQUAL(1, g_resume_calls);
}

void test_wifi_bssid_pin_service_keeps_a_pin_dormant_when_matter_changes_ssid(void) {
  seed_pin("MatterLab", "AA:BB:CC:DD:EE:FF");
  set_connected_station("NewMatterNetwork", "22:33:44:55:66:77");

  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_EQUAL_STRING("MatterLab", service.stored_ssid().c_str());
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", service.stored_bssid().c_str());
  TEST_ASSERT_TRUE(g_applied_pins.empty());

  WifiBssidPinService reloaded;
  TEST_ASSERT_EQUAL(ESP_OK, reloaded.setup(make_config()));
  TEST_ASSERT_EQUAL_STRING("MatterLab", reloaded.stored_ssid().c_str());
  TEST_ASSERT_EQUAL_STRING("AA:BB:CC:DD:EE:FF", reloaded.stored_bssid().c_str());
}

void test_wifi_bssid_pin_service_persists_an_explicit_clear(void) {
  seed_pin("MatterLab", "AA:BB:CC:DD:EE:FF");
  WifiBssidPinService service;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup(make_config()));

  std::string message;
  TEST_ASSERT_TRUE(service.request_update("", &message));
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::VERIFYING);
  TEST_ASSERT_TRUE(g_applied_pins.back().empty());

  set_connected_station("MatterLab", "11:22:33:44:55:66");
  service.notify_station_changed();
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiBssidPinApplyState::APPLIED);
  TEST_ASSERT_TRUE(service.stored_ssid().empty());
  TEST_ASSERT_TRUE(service.stored_bssid().empty());

  WifiBssidPinService reloaded;
  TEST_ASSERT_EQUAL(ESP_OK, reloaded.setup(make_config()));
  TEST_ASSERT_TRUE(reloaded.stored_ssid().empty());
  TEST_ASSERT_TRUE(reloaded.stored_bssid().empty());
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_bssid_pin_service_commits_only_after_verified_ipv4);
  RUN_TEST(test_wifi_bssid_pin_service_restores_the_previous_pin_after_timeout);
  RUN_TEST(test_wifi_bssid_pin_service_reapplies_a_stored_pin_after_boot);
  RUN_TEST(test_wifi_bssid_pin_service_keeps_a_pin_dormant_when_matter_changes_ssid);
  RUN_TEST(test_wifi_bssid_pin_service_persists_an_explicit_clear);
  return UNITY_END();
}
