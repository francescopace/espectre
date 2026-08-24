/*
 * ESPectre - WiFi Provisioning Service Unit Tests
 *
 * Unit tests for WiFi Provisioning Service.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "device_config_store.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "improv.h"
#include "improv_serial_service.h"
#include "nvs.h"
#include "standalone_wifi_service.h"
#include "wifi_provisioning_service.h"

using namespace espectre;

namespace {

struct CapturedImprovFrame {
  uint8_t type{0U};
  std::vector<uint8_t> payload;
};

std::vector<uint8_t> make_improv_rpc_frame(uint8_t command, const std::vector<uint8_t> &command_data = {}) {
  std::vector<uint8_t> frame{'I', 'M', 'P', 'R', 'O', 'V', 1U, improv::TYPE_RPC};
  const size_t payload_length = command_data.size() + 2U;
  frame.push_back(static_cast<uint8_t>(payload_length));
  frame.push_back(command);
  frame.push_back(static_cast<uint8_t>(command_data.size()));
  frame.insert(frame.end(), command_data.begin(), command_data.end());
  uint8_t checksum = 0U;
  for (uint8_t byte : frame) {
    checksum = static_cast<uint8_t>(checksum + byte);
  }
  frame.push_back(checksum);
  return frame;
}

std::vector<CapturedImprovFrame> parse_improv_frames(const std::vector<uint8_t> &bytes) {
  constexpr uint8_t header[] = {'I', 'M', 'P', 'R', 'O', 'V'};
  std::vector<CapturedImprovFrame> frames;
  size_t position = 0U;
  while (position + 10U <= bytes.size()) {
    if (!std::equal(std::begin(header), std::end(header), bytes.begin() + position)) {
      ++position;
      continue;
    }
    const size_t payload_length = bytes[position + 8U];
    const size_t frame_length = 10U + payload_length;
    if (position + frame_length > bytes.size()) {
      break;
    }
    uint8_t checksum = 0U;
    for (size_t index = position; index < position + frame_length - 1U; ++index) {
      checksum = static_cast<uint8_t>(checksum + bytes[index]);
    }
    TEST_ASSERT_EQUAL_UINT8(checksum, bytes[position + frame_length - 1U]);
    frames.push_back(CapturedImprovFrame{
        bytes[position + 7U],
        std::vector<uint8_t>(bytes.begin() + position + 9U, bytes.begin() + position + 9U + payload_length),
    });
    position += frame_length;
  }
  return frames;
}

std::vector<std::string> parse_improv_rpc_strings(const CapturedImprovFrame &frame, uint8_t expected_command) {
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_RPC_RESPONSE, frame.type);
  TEST_ASSERT_TRUE(frame.payload.size() >= 2U);
  TEST_ASSERT_EQUAL_UINT8(expected_command, frame.payload[0]);
  TEST_ASSERT_EQUAL_UINT8(frame.payload.size() - 2U, frame.payload[1]);
  std::vector<std::string> values;
  size_t position = 2U;
  while (position < frame.payload.size()) {
    const size_t length = frame.payload[position++];
    TEST_ASSERT_TRUE(position + length <= frame.payload.size());
    values.emplace_back(reinterpret_cast<const char *>(frame.payload.data() + position), length);
    position += length;
  }
  return values;
}

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

void emit_got_ip(StandaloneWifiService *manager) {
  ip_event_got_ip_t event{};
  event.ip_info.ip.addr = 0x0101A8C0U;
  esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &event);
  manager->loop();
}

}  // namespace

void setUp(void) {
  esp_event_mock_reset();
  esp_wifi_mock_reset();
  nvs_mock_reset();
  esp_timer_mock::reset(0, 0);
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
  TEST_ASSERT_TRUE(service.config().band_policy == WifiBandPolicy::BAND_2G);
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
  TEST_ASSERT_TRUE(service.config().band_policy == WifiBandPolicy::BAND_2G);
}

void test_wifi_provisioning_normalizes_an_incompatible_stored_channel(void) {
  StoredWifiConfig saved;
  saved.ssid = "SavedSSID";
  saved.channel = 36;
  saved.has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(saved));

  WifiProvisioningService service(nullptr);
  TEST_ASSERT_EQUAL(ESP_OK, service.load_or_set_defaults(make_defaults()));
  TEST_ASSERT_TRUE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_UINT8(0, service.config().channel);

  StoredWifiConfig normalized;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&normalized));
  TEST_ASSERT_EQUAL_UINT8(0, normalized.channel);
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
  // A 2.4 GHz-only radio must reject a 5 GHz channel it cannot tune, rather
  // than accept a lock that would silently prevent association.
  TEST_ASSERT_FALSE(service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&channel=36", &message));
  TEST_ASSERT_EQUAL_STRING("channel must be 0..14", message.c_str());
  TEST_ASSERT_FALSE(
      service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&band_policy=5g&channel=36", &message));
  TEST_ASSERT_EQUAL_STRING("5 GHz Wi-Fi is not supported by this device", message.c_str());
  TEST_ASSERT_FALSE(
      service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&band_policy=wide&channel=0", &message));
  TEST_ASSERT_EQUAL_STRING("band_policy must be 2g, 5g, or auto", message.c_str());

  TEST_ASSERT_TRUE(
      service.handle_command("SET_WIFI_CONFIG:ssid=Lab&password=secret&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=9&band_policy=2g", &message));
  TEST_ASSERT_TRUE(service.apply_pending());
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::VERIFYING);
  service.loop();

  StoredWifiConfig before_verification;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&before_verification));
  TEST_ASSERT_FALSE(before_verification.has_saved_config);

  emit_got_ip(&manager);
  TEST_ASSERT_FALSE(service.apply_pending());
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::APPLIED);

  WifiProvisioningService reloaded(nullptr);
  TEST_ASSERT_EQUAL(ESP_OK, reloaded.load_or_set_defaults(make_defaults()));
  TEST_ASSERT_TRUE(reloaded.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("Lab", reloaded.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("secret", reloaded.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", reloaded.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(9, reloaded.config().channel);
  TEST_ASSERT_TRUE(reloaded.config().has_saved_band_policy);
  TEST_ASSERT_TRUE(reloaded.config().band_policy == WifiBandPolicy::BAND_2G);
}

void test_wifi_provisioning_apply_updates_wifi_manager_live(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_TRUE(service.handle_command("SET_WIFI_CONFIG:ssid=Applied&password=applied-secret&channel=3", &message));

  TEST_ASSERT_EQUAL_STRING("Wi-Fi candidate accepted; reconnect after address verification", message.c_str());
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_config_call_count);
  service.loop();
  TEST_ASSERT_EQUAL(2, g_esp_wifi_mock.set_config_call_count);
  TEST_ASSERT_EQUAL_STRING("Applied", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL_STRING("applied-secret", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.password));
  TEST_ASSERT_EQUAL_UINT8(3, g_esp_wifi_mock.last_config.sta.channel);
  TEST_ASSERT_EQUAL_STRING("DefaultSSID", service.config().ssid.c_str());

  emit_got_ip(&manager);
  TEST_ASSERT_EQUAL_STRING("Applied", service.config().ssid.c_str());
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::APPLIED);
}

void test_wifi_provisioning_stages_standard_improv_credentials_without_radio_lock(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_TRUE(service.begin_serial_provisioning("Improv Network", "serial-secret", &message));

  TEST_ASSERT_EQUAL_STRING("Wi-Fi candidate accepted; reconnect after address verification", message.c_str());
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::VERIFYING);
  service.loop();
  TEST_ASSERT_EQUAL_STRING("Improv Network", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL_STRING("serial-secret", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.password));
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_config.sta.bssid_set);
  TEST_ASSERT_EQUAL_UINT8(0U, g_esp_wifi_mock.last_config.sta.channel);

  StoredWifiConfig before_verification;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&before_verification));
  TEST_ASSERT_FALSE(before_verification.has_saved_config);

  emit_got_ip(&manager);
  TEST_ASSERT_EQUAL_STRING("Improv Network", service.config().ssid.c_str());
  TEST_ASSERT_TRUE(service.config().has_saved_config);
}

void test_wifi_provisioning_rejects_overlapping_direct_and_improv_candidates(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_TRUE(service.begin_serial_provisioning("First", "first-secret", &message));
  service.loop();
  TEST_ASSERT_FALSE(service.handle_command("SET_WIFI_CONFIG:ssid=Second&password=second-secret", &message));
  TEST_ASSERT_EQUAL_STRING("Wi-Fi configuration change already in progress", message.c_str());
  TEST_ASSERT_EQUAL_STRING("First", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
}

void test_improv_serial_reports_info_and_completes_verified_provisioning(void) {
  StandaloneWifiService manager;
  WifiProvisioningService wifi_provisioning(&manager);
  TEST_ASSERT_EQUAL(ESP_OK, wifi_provisioning.setup_station(make_defaults()));

  std::vector<uint8_t> input;
  size_t input_position = 0U;
  std::vector<uint8_t> output;
  ImprovSerialService service(
      &wifi_provisioning,
      &manager,
      [&input, &input_position](uint8_t *data, size_t capacity) {
        const size_t available = input.size() - input_position;
        const size_t count = std::min(capacity, available);
        if (count > 0U) {
          std::memcpy(data, input.data() + input_position, count);
          input_position += count;
        }
        return static_cast<int>(count);
      },
      [&output](const uint8_t *data, size_t length) {
        output.insert(output.end(), data, data + length);
        return static_cast<int>(length);
      });
  TEST_ASSERT_TRUE(service.setup(ImprovSerialServiceConfig{
      "ESPectre Native",
      "3.0.0-test",
      "esp32c3",
      "ESPectre test device",
      []() { return "https://espectre.dev/tools/configure/?target=192.168.1.42"; },
  }));

  std::vector<CapturedImprovFrame> frames = parse_improv_frames(output);
  TEST_ASSERT_EQUAL(2U, frames.size());
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_ERROR_STATE, frames[0].type);
  TEST_ASSERT_EQUAL_UINT8(improv::ERROR_NONE, frames[0].payload[0]);
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_CURRENT_STATE, frames[1].type);
  TEST_ASSERT_EQUAL_UINT8(improv::STATE_AUTHORIZED, frames[1].payload[0]);

  output.clear();
  input = make_improv_rpc_frame(improv::GET_DEVICE_INFO);
  input_position = 0U;
  service.loop();
  frames = parse_improv_frames(output);
  TEST_ASSERT_EQUAL(2U, frames.size());
  const std::vector<std::string> info = parse_improv_rpc_strings(frames[1], improv::GET_DEVICE_INFO);
  TEST_ASSERT_EQUAL(6U, info.size());
  TEST_ASSERT_EQUAL_STRING("ESPectre Native", info[0].c_str());
  TEST_ASSERT_EQUAL_STRING("3.0.0-test", info[1].c_str());
  TEST_ASSERT_EQUAL_STRING("esp32c3", info[2].c_str());
  TEST_ASSERT_EQUAL_STRING("ESPectre test device", info[3].c_str());
  TEST_ASSERT_EQUAL_STRING("ESP-IDF", info[4].c_str());

  const std::string ssid = "Improv Network";
  const std::string password = "serial-secret";
  std::vector<uint8_t> wifi_data{static_cast<uint8_t>(ssid.size())};
  wifi_data.insert(wifi_data.end(), ssid.begin(), ssid.end());
  wifi_data.push_back(static_cast<uint8_t>(password.size()));
  wifi_data.insert(wifi_data.end(), password.begin(), password.end());
  output.clear();
  input = make_improv_rpc_frame(improv::WIFI_SETTINGS, wifi_data);
  input_position = 0U;
  service.loop();
  wifi_provisioning.loop();

  frames = parse_improv_frames(output);
  TEST_ASSERT_EQUAL(2U, frames.size());
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_CURRENT_STATE, frames[1].type);
  TEST_ASSERT_EQUAL_UINT8(improv::STATE_PROVISIONING, frames[1].payload[0]);
  TEST_ASSERT_EQUAL_STRING(ssid.c_str(), reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_config.sta.bssid_set);
  TEST_ASSERT_TRUE(std::search(output.begin(), output.end(), password.begin(), password.end()) == output.end());

  output.clear();
  emit_got_ip(&manager);
  service.loop();
  frames = parse_improv_frames(output);
  TEST_ASSERT_EQUAL(2U, frames.size());
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_CURRENT_STATE, frames[0].type);
  TEST_ASSERT_EQUAL_UINT8(improv::STATE_PROVISIONED, frames[0].payload[0]);
  const std::vector<std::string> result = parse_improv_rpc_strings(frames[1], improv::WIFI_SETTINGS);
  TEST_ASSERT_EQUAL(1U, result.size());
  TEST_ASSERT_EQUAL_STRING("https://espectre.dev/tools/configure/?target=192.168.1.42", result[0].c_str());
  TEST_ASSERT_EQUAL_STRING(ssid.c_str(), wifi_provisioning.config().ssid.c_str());
}

void test_improv_serial_flushes_partial_writes_without_blocking(void) {
  StandaloneWifiService manager;
  WifiProvisioningService wifi_provisioning(&manager);
  TEST_ASSERT_EQUAL(ESP_OK, wifi_provisioning.setup_station(make_defaults()));

  std::vector<uint8_t> output;
  ImprovSerialService service(
      &wifi_provisioning,
      &manager,
      [](uint8_t *, size_t) { return 0; },
      [&output](const uint8_t *data, size_t length) {
        const size_t written = std::min<size_t>(3U, length);
        output.insert(output.end(), data, data + written);
        return static_cast<int>(written);
      });
  TEST_ASSERT_TRUE(service.setup(ImprovSerialServiceConfig{
      "ESPectre Native", "3.0.0-test", "esp32s2", "ESPectre test device", {}}));

  TEST_ASSERT_TRUE(output.size() < 20U);
  for (int iteration = 0; iteration < 8; ++iteration) {
    service.loop();
  }
  const std::vector<CapturedImprovFrame> frames = parse_improv_frames(output);
  TEST_ASSERT_EQUAL(2U, frames.size());
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_ERROR_STATE, frames[0].type);
  TEST_ASSERT_EQUAL_UINT8(improv::TYPE_CURRENT_STATE, frames[1].type);
}

void test_wifi_provisioning_batch_command_persists_and_applies_config(void) {
  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;
  int prepare_calls = 0;
  int resume_calls = 0;
  int apply_completed_calls = 0;
  service.set_reconfigure_callbacks([&prepare_calls]() { ++prepare_calls; },
                                    [&resume_calls]() { ++resume_calls; });
  service.set_apply_completed_callback([&apply_completed_calls]() { ++apply_completed_calls; });

  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_TRUE(service.handle_command(
      "SET_WIFI_CONFIG:ssid=Lab%20Net&password=top%20secret&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=9",
      &message));

  TEST_ASSERT_EQUAL_STRING("Wi-Fi candidate accepted; reconnect after address verification", message.c_str());
  TEST_ASSERT_TRUE(service.apply_pending());
  service.loop();
  TEST_ASSERT_EQUAL(1, prepare_calls);
  TEST_ASSERT_EQUAL(0, resume_calls);
  emit_got_ip(&manager);
  TEST_ASSERT_EQUAL(1, prepare_calls);
  TEST_ASSERT_EQUAL(1, resume_calls);
  TEST_ASSERT_EQUAL(1, apply_completed_calls);
  TEST_ASSERT_TRUE(service.config().has_saved_config);
  TEST_ASSERT_EQUAL_STRING("Lab Net", service.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("top secret", service.config().password.c_str());
  TEST_ASSERT_EQUAL_STRING("aa:bb:cc:dd:ee:ff", service.config().bssid.c_str());
  TEST_ASSERT_EQUAL_UINT8(9, service.config().channel);
  TEST_ASSERT_EQUAL_STRING("Lab Net", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL_STRING("top secret", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.password));
  TEST_ASSERT_EQUAL_UINT8(9, g_esp_wifi_mock.last_config.sta.channel);
}

void test_wifi_provisioning_clear_command_erases_and_disconnects(void) {
  StoredWifiConfig saved;
  saved.ssid = "SavedSSID";
  saved.password = "saved-secret";
  saved.bssid = "AA:BB:CC:DD:EE:FF";
  saved.channel = 6U;
  saved.has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(saved));

  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_EQUAL(ESP_OK, manager.start());
  emit_got_ip(&manager);
  const int disconnect_calls = g_esp_wifi_mock.disconnect_call_count;
  const int connect_calls = g_esp_wifi_mock.connect_call_count;

  TEST_ASSERT_TRUE(service.handle_command("CLEAR_WIFI", &message));
  TEST_ASSERT_EQUAL_STRING("Wi-Fi config applied", message.c_str());
  TEST_ASSERT_FALSE(service.config().has_saved_config);
  TEST_ASSERT_TRUE(service.config().ssid.empty());
  TEST_ASSERT_TRUE(service.config().password.empty());
  TEST_ASSERT_TRUE(service.config().bssid.empty());
  TEST_ASSERT_EQUAL_UINT8(0U, service.config().channel);
  TEST_ASSERT_EQUAL_STRING("", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));
  TEST_ASSERT_EQUAL(disconnect_calls + 1, g_esp_wifi_mock.disconnect_call_count);
  TEST_ASSERT_EQUAL(connect_calls, g_esp_wifi_mock.connect_call_count);

  StoredWifiConfig after_clear;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&after_clear));
  TEST_ASSERT_FALSE(after_clear.has_saved_config);
  TEST_ASSERT_TRUE(after_clear.ssid.empty());
}

void test_wifi_provisioning_falls_back_from_bssid_then_rolls_back_to_last_good(void) {
  StoredWifiConfig saved;
  saved.ssid = "KnownGood";
  saved.password = "known-good-secret";
  saved.channel = 1U;
  saved.has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(saved));

  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  WifiProvisioningDefaults defaults = make_defaults();
  defaults.candidate_timeout_ms = 1000U;
  std::string message;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(defaults));

  TEST_ASSERT_TRUE(service.handle_command(
      "SET_WIFI_CONFIG:ssid=Candidate&password=bad&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=9",
      &message));
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::VERIFYING);
  service.loop();

  esp_timer_mock::advance(1000000);
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::VERIFYING_WITHOUT_BSSID);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_config.sta.bssid_set);
  TEST_ASSERT_EQUAL_UINT8(0U, g_esp_wifi_mock.last_config.sta.channel);

  esp_timer_mock::advance(1000000);
  service.loop();
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::ROLLING_BACK);
  TEST_ASSERT_EQUAL_STRING("KnownGood", reinterpret_cast<const char *>(g_esp_wifi_mock.last_config.sta.ssid));

  StoredWifiConfig still_saved;
  TEST_ASSERT_EQUAL(ESP_OK, load_stored_wifi_config(&still_saved));
  TEST_ASSERT_EQUAL_STRING("KnownGood", still_saved.ssid.c_str());

  emit_got_ip(&manager);
  TEST_ASSERT_TRUE(service.apply_state() == WifiProvisioningApplyState::ROLLED_BACK);
  TEST_ASSERT_EQUAL_STRING("KnownGood", service.config().ssid.c_str());
}

void test_wifi_provisioning_reboot_during_apply_keeps_last_good(void) {
  StoredWifiConfig saved;
  saved.ssid = "KnownGood";
  saved.password = "known-good-secret";
  saved.has_saved_config = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_stored_wifi_config(saved));

  StandaloneWifiService manager;
  WifiProvisioningService service(&manager);
  std::string message;
  TEST_ASSERT_EQUAL(ESP_OK, service.setup_station(make_defaults()));
  TEST_ASSERT_TRUE(service.handle_command(
      "SET_WIFI_CONFIG:ssid=Unverified&password=unverified&bssid=&channel=0", &message));

  WifiProvisioningService after_reboot(nullptr);
  TEST_ASSERT_EQUAL(ESP_OK, after_reboot.load_or_set_defaults(make_defaults()));
  TEST_ASSERT_EQUAL_STRING("KnownGood", after_reboot.config().ssid.c_str());
  TEST_ASSERT_EQUAL_STRING("known-good-secret", after_reboot.config().password.c_str());
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_wifi_provisioning_loads_defaults_when_no_saved_config);
  RUN_TEST(test_wifi_provisioning_loads_saved_config);
  RUN_TEST(test_wifi_provisioning_normalizes_an_incompatible_stored_channel);
  RUN_TEST(test_wifi_provisioning_records_load_error_and_falls_back_to_defaults);
  RUN_TEST(test_wifi_provisioning_commands_validate_and_persist_config);
  RUN_TEST(test_wifi_provisioning_apply_updates_wifi_manager_live);
  RUN_TEST(test_wifi_provisioning_stages_standard_improv_credentials_without_radio_lock);
  RUN_TEST(test_wifi_provisioning_rejects_overlapping_direct_and_improv_candidates);
  RUN_TEST(test_improv_serial_reports_info_and_completes_verified_provisioning);
  RUN_TEST(test_improv_serial_flushes_partial_writes_without_blocking);
  RUN_TEST(test_wifi_provisioning_batch_command_persists_and_applies_config);
  RUN_TEST(test_wifi_provisioning_clear_command_erases_and_disconnects);
  RUN_TEST(test_wifi_provisioning_falls_back_from_bssid_then_rolls_back_to_last_good);
  RUN_TEST(test_wifi_provisioning_reboot_during_apply_keeps_last_good);
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
