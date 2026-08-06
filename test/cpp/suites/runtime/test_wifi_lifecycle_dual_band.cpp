/*
 * ESPectre - Dual-Band Wi-Fi Lifecycle Unit Tests
 *
 * Unit tests for the CSI radio policy on parts that can associate on either
 * band. Built with CONFIG_SOC_WIFI_SUPPORT_5G forced on, so this suite covers
 * the per-band ESP-IDF APIs the single-band suite never reaches.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include "esp_event.h"
#include "esp_wifi.h"
#include "wifi_band_helpers.h"
#include "wifi_lifecycle.h"

using namespace espectre;

namespace {

constexpr uint16_t EXPECTED_PROTOCOL_2G =
    WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
constexpr uint16_t EXPECTED_PROTOCOL_5G = WIFI_PROTOCOL_11A | WIFI_PROTOCOL_11N;

}  // namespace

void setUp(void) {
  esp_event_mock_reset();
  esp_netif_mock_reset();
  esp_wifi_mock_reset();
}

void tearDown(void) {}

// The single-band esp_wifi_set_protocol and esp_wifi_set_bandwidth return
// ESP_ERR_NOT_SUPPORTED while the radio may use either band, so the policy must
// go through the per-band APIs and pin both bands in one call.
void test_dual_band_policy_pins_ht20_on_both_bands(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.bandwidths.ghz_2g = WIFI_BW_HT40;
  g_esp_wifi_mock.bandwidths.ghz_5g = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(EXPECTED_PROTOCOL_2G, g_esp_wifi_mock.last_protocols.ghz_2g);
  TEST_ASSERT_EQUAL(EXPECTED_PROTOCOL_5G, g_esp_wifi_mock.last_protocols.ghz_5g);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidths_call_count);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidths.ghz_2g == WIFI_BW_HT20);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidths.ghz_5g == WIFI_BW_HT20);
}

// Both bands are opened explicitly rather than left to the SDK default, so the
// station and the AP decide the link between themselves on every build.
void test_dual_band_policy_opens_both_bands(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_2G_ONLY;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_band_mode_call_count);
  TEST_ASSERT_EQUAL(WIFI_BAND_MODE_AUTO, g_esp_wifi_mock.last_band_mode);
}

// Ordering is load-bearing, not cosmetic: ESP-IDF ignores the inactive band's
// entry in esp_wifi_set_protocols, so pinning while the radio is still on one
// band would leave 5 GHz at its 11ax default and yield no HT20 CSI there.
void test_dual_band_policy_opens_the_bands_before_pinning_protocols(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_2G_ONLY;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_TRUE(g_esp_wifi_mock.set_band_mode_sequence > 0);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.set_protocols_sequence > 0);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.set_band_mode_sequence < g_esp_wifi_mock.set_protocols_sequence);
}

// A radio already on both bands needs no write, so a reconnect does not churn
// the band mode.
void test_dual_band_policy_skips_a_band_mode_already_open(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_AUTO;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_band_mode_call_count);
}

// An explicit band choice must fail closed instead of associating on another
// band and silently changing the detector's physical operating conditions.
void test_dual_band_policy_reports_band_mode_failure(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_2G_ONLY;
  g_esp_wifi_mock.set_band_mode_result = ESP_FAIL;
  g_esp_wifi_mock.bandwidths.ghz_2g = WIFI_BW_HT40;
  g_esp_wifi_mock.bandwidths.ghz_5g = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_EQUAL(ESP_FAIL, manager.process_pending_events());
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidths_call_count);
}

void test_dual_band_policy_skips_matching_radio_settings(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.protocols.ghz_2g = EXPECTED_PROTOCOL_2G;
  g_esp_wifi_mock.protocols.ghz_5g = EXPECTED_PROTOCOL_5G;
  g_esp_wifi_mock.bandwidths.ghz_2g = WIFI_BW_HT20;
  g_esp_wifi_mock.bandwidths.ghz_5g = WIFI_BW_HT20;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidths_call_count);
}

// A 5 GHz-only protocol mismatch still has to be corrected: leaving 11ac or
// 11ax active there would replace the HT20 training field the detectors read.
void test_dual_band_policy_repins_when_only_the_5ghz_band_drifts(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.protocols.ghz_2g = EXPECTED_PROTOCOL_2G;
  g_esp_wifi_mock.protocols.ghz_5g = EXPECTED_PROTOCOL_5G | WIFI_PROTOCOL_11AX;
  g_esp_wifi_mock.bandwidths.ghz_2g = WIFI_BW_HT20;
  g_esp_wifi_mock.bandwidths.ghz_5g = WIFI_BW_HT20;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(EXPECTED_PROTOCOL_5G, g_esp_wifi_mock.last_protocols.ghz_5g);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidths_call_count);
}

void test_dual_band_policy_reports_protocol_failure(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.set_protocols_result = ESP_FAIL;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::AUTO));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  TEST_ASSERT_EQUAL(ESP_FAIL, manager.process_pending_events());
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_bandwidths_call_count);
}

void test_dual_band_policy_defaults_to_2g_ht20(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_AUTO;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {}));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_band_mode_call_count);
  TEST_ASSERT_EQUAL(WIFI_BAND_MODE_2G_ONLY, g_esp_wifi_mock.last_band_mode);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(EXPECTED_PROTOCOL_2G, g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidth == WIFI_BW_HT20);
}

void test_dual_band_policy_honors_explicit_5g_ht20(void) {
  WiFiLifecycleManager manager;
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_2G_ONLY;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT40;

  TEST_ASSERT_EQUAL(ESP_OK, manager.register_handlers([](const esp_netif_ip_info_t &) {}, []() {},
                                                       WifiBandPolicy::BAND_5G));
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);

  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_band_mode_call_count);
  TEST_ASSERT_EQUAL(WIFI_BAND_MODE_5G_ONLY, g_esp_wifi_mock.last_band_mode);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_protocol_call_count);
  TEST_ASSERT_EQUAL(EXPECTED_PROTOCOL_5G, g_esp_wifi_mock.last_protocol_bitmap);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.set_protocols_call_count);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_bandwidth_call_count);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.last_bandwidth == WIFI_BW_HT20);
}

void test_dual_band_channel_lock_accepts_both_bands(void) {
  TEST_ASSERT_TRUE(wifi_channel_is_supported(0));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(14));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(36));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(64));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(100));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(144));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(149));
  TEST_ASSERT_TRUE(wifi_channel_is_supported(177));

  // Numbers between the 20 MHz centers are not channels on any band plan.
  TEST_ASSERT_FALSE(wifi_channel_is_supported(-1));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(15));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(35));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(37));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(148));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(150));
  TEST_ASSERT_FALSE(wifi_channel_is_supported(181));

  TEST_ASSERT_TRUE(wifi_channel_matches_band_policy(6, WifiBandPolicy::BAND_2G));
  TEST_ASSERT_FALSE(wifi_channel_matches_band_policy(36, WifiBandPolicy::BAND_2G));
  TEST_ASSERT_TRUE(wifi_channel_matches_band_policy(36, WifiBandPolicy::BAND_5G));
  TEST_ASSERT_FALSE(wifi_channel_matches_band_policy(6, WifiBandPolicy::BAND_5G));
  TEST_ASSERT_TRUE(wifi_channel_matches_band_policy(6, WifiBandPolicy::AUTO));
  TEST_ASSERT_TRUE(wifi_channel_matches_band_policy(36, WifiBandPolicy::AUTO));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_dual_band_policy_pins_ht20_on_both_bands);
  RUN_TEST(test_dual_band_policy_opens_both_bands);
  RUN_TEST(test_dual_band_policy_opens_the_bands_before_pinning_protocols);
  RUN_TEST(test_dual_band_policy_skips_a_band_mode_already_open);
  RUN_TEST(test_dual_band_policy_reports_band_mode_failure);
  RUN_TEST(test_dual_band_policy_skips_matching_radio_settings);
  RUN_TEST(test_dual_band_policy_repins_when_only_the_5ghz_band_drifts);
  RUN_TEST(test_dual_band_policy_reports_protocol_failure);
  RUN_TEST(test_dual_band_policy_defaults_to_2g_ht20);
  RUN_TEST(test_dual_band_policy_honors_explicit_5g_ht20);
  RUN_TEST(test_dual_band_channel_lock_accepts_both_bands);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
