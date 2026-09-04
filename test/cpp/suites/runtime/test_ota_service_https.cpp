/*
 * ESPectre - HTTPS OTA Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <atomic>
#include <string>

#define private public
#include "ota_service_https.h"
#undef private

#include "esp_http_client.h"

using namespace espectre;

void setUp(void) { esp_http_client_mock_reset(); }
void tearDown(void) {}

void test_https_ota_manifest_parser_accepts_canonical_and_legacy_url(void) {
  HttpsOtaService service("native", "esp32-s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestInfo manifest;
  std::string error;

  TEST_ASSERT_TRUE(service.parse_manifest_(
      R"({"version":"3.1.0","image_url":"https://example.invalid/fw.bin"})",
      &manifest, &error));
  TEST_ASSERT_EQUAL_STRING("3.1.0", manifest.version.c_str());
  TEST_ASSERT_EQUAL_STRING("https://example.invalid/fw.bin", manifest.image_url.c_str());

  TEST_ASSERT_TRUE(service.parse_manifest_(
      R"({"version":"3.2.0","url":"https://example.invalid/legacy.bin"})",
      &manifest, &error));
  TEST_ASSERT_EQUAL_STRING("https://example.invalid/legacy.bin", manifest.image_url.c_str());
  TEST_ASSERT_FALSE(service.parse_manifest_(R"({"version":"3.2.0"})", &manifest, &error));
  TEST_ASSERT_EQUAL_STRING("invalid manifest", error.c_str());
  TEST_ASSERT_FALSE(service.parse_manifest_("{}", nullptr, &error));
}

void test_https_ota_fetch_enforces_status_and_manifest_size(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::RELEASE);
  std::string body;
  std::string error;

  g_esp_http_client_mock.response_body = "ok";
  TEST_ASSERT_TRUE(service.fetch_https_text_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL_STRING("ok", body.c_str());
  TEST_ASSERT_EQUAL(30000, g_esp_http_client_mock.last_config.timeout_ms);
  TEST_ASSERT_EQUAL(8192, g_esp_http_client_mock.last_config.buffer_size);
  TEST_ASSERT_EQUAL(1024, g_esp_http_client_mock.last_config.buffer_size_tx);

  esp_http_client_mock_reset();
  g_esp_http_client_mock.status_code = 503;
  TEST_ASSERT_FALSE(service.fetch_https_text_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_TRUE(error.find("503") != std::string::npos);

  esp_http_client_mock_reset();
  g_esp_http_client_mock.response_body.assign(4097U, 'x');
  TEST_ASSERT_FALSE(service.fetch_https_text_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL_STRING("manifest too large", error.c_str());
  TEST_ASSERT_FALSE(service.fetch_https_text_("", &body, &error));
  TEST_ASSERT_FALSE(service.fetch_https_text_("https://example.invalid", nullptr, &error));
}

void test_https_ota_check_updates_status_and_delivers_callback(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::PREVIEW);
  g_esp_http_client_mock.response_body =
      R"({"version":"99.0.0","image_url":"https://example.invalid/fw.bin"})";
  int callback_count = 0;
  EspectreOtaStatus delivered;
  service.set_status_callback([&](const EspectreOtaStatus& status) {
    callback_count++;
    delivered = status;
  });

  TEST_ASSERT_TRUE(service.start_check("3.0.0"));
  service.loop();

  TEST_ASSERT_EQUAL(1, callback_count);
  TEST_ASSERT_TRUE(delivered.state == EspectreOtaState::UPDATE_AVAILABLE);
  TEST_ASSERT_TRUE(delivered.update_available);
  TEST_ASSERT_EQUAL_STRING("preview", delivered.default_channel.c_str());
  TEST_ASSERT_FALSE(service.start_check("3.0.0", "invalid"));
  service.shutdown();
  TEST_ASSERT_FALSE(service.start_check("3.0.0"));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_https_ota_manifest_parser_accepts_canonical_and_legacy_url);
  RUN_TEST(test_https_ota_fetch_enforces_status_and_manifest_size);
  RUN_TEST(test_https_ota_check_updates_status_and_delivers_callback);
  return UNITY_END();
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return process();
}
