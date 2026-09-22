/*
 * ESPectre - HTTPS OTA Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <atomic>
#include <array>
#include <memory>
#include "esp_http_client.h"
#include <new>
#include <string>

// Load shared dependencies before exposing HttpsOtaService internals.
#include "frontend/ota_service.h"

#define private public
#include "frontend/ota_service_https.h"
#undef private

#include "esp_http_client.h"
#include "esp_https_ota.h"

using namespace espectre;

static bool fail_buffer_allocation = false;
static size_t largest_buffer_allocation = 0U;

void *operator new[](size_t size, const std::nothrow_t &) noexcept {
  largest_buffer_allocation = std::max(largest_buffer_allocation, size);
  return fail_buffer_allocation ? nullptr : ::operator new[](size);
}

class LegacyOtaService : public IOtaService {
 public:
  void loop() override {}
  void shutdown() override {}
  bool start_check(const std::string &) override {
    check_calls++;
    return true;
  }
  bool start_update(const std::string &) override {
    update_calls++;
    return true;
  }
  EspectreOtaStatus status() const override { return {}; }
  void set_status_callback(StatusCallback) override {}
  void set_prepare_for_update_callback(PrepareForUpdateCallback) override {}

  int check_calls{0};
  int update_calls{0};
};

void test_legacy_ota_service_rejects_channels_it_cannot_honor(void) {
  LegacyOtaService service;
  IOtaService &api = service;

  TEST_ASSERT_TRUE(api.start_check("3.0.0", ""));
  TEST_ASSERT_TRUE(api.start_update("3.0.0", ""));
  TEST_ASSERT_FALSE(api.start_check("3.0.0", ESPECTRE_OTA_CHANNEL_PREVIEW));
  TEST_ASSERT_FALSE(api.start_update("3.0.0", "invalid"));
  TEST_ASSERT_EQUAL(1, service.check_calls);
  TEST_ASSERT_EQUAL(1, service.update_calls);
}

void setUp(void) {
  fail_buffer_allocation = false;
  largest_buffer_allocation = 0U;
  esp_http_client_mock_reset();
  g_esp_https_ota_calls = 0;
  g_esp_https_ota_result = ESP_OK;
}
void tearDown(void) {}

namespace {

std::string buffer_text(const HttpsOtaService::ManifestBuffer &body) {
  std::string text;
  for (size_t i = 0U; i < body.size(); ++i) text.push_back(body[i]);
  return text;
}

bool parse_manifest_text(HttpsOtaService &service, const std::string &text, const std::string &channel,
                         HttpsOtaService::ManifestInfo *manifest, std::string *error) {
  g_esp_http_client_mock.response_body = text;
  HttpsOtaService::ManifestBuffer body;
  return service.fetch_manifest_("https://example.invalid/manifest.json", &body, error) &&
      service.parse_manifest_(body, channel, manifest, error);
}


std::string firmware_catalog(const std::string &artifacts, const char *channel = "develop",
                             const char *version = "3.1.0") {
  return std::string(R"({"schema_version":1,"channel":")") + channel +
      R"(","version":")" + version + R"(","frontends":{
        "esphome":{"artifacts":[{"chip":"esp32s2","build_type":"ota","url":"https://example.invalid/esphome.bin"}]},
        "native":{"artifacts":)" + artifacts + "}}}";
}

const char *const kOtaArtifact =
    R"({"chip":"esp32s2","build_type":"ota","url":"https://example.invalid/fw.bin"})";

}  // namespace

void test_https_ota_manifest_parser_selects_frontend_chip_and_ota_image(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestInfo manifest;
  std::string error;
  const std::string artifacts = std::string(R"([
      {"chip":"esp32s2","build_type":"factory","url":"https://example.invalid/factory.bin"},
      {"chip":"esp32c3","build_type":"ota","url":"https://example.invalid/c3.bin"},
      {"chip":"esp32s2","build_type":"ota",
       "compliance":[{"url":"https://example.invalid/license.zip"}],
       "url":"https://example.invalid/fw.bin"}])");

  for (const char *channel : {"release", "preview", "develop"}) {
    TEST_ASSERT_TRUE(parse_manifest_text(service, firmware_catalog(artifacts, channel), channel, &manifest, &error));
    TEST_ASSERT_EQUAL_STRING("3.1.0", manifest.version.c_str());
    TEST_ASSERT_EQUAL_STRING("https://example.invalid/fw.bin", manifest.image_url.c_str());
  }
}

void test_https_ota_manifest_parser_rejects_missing_or_ambiguous_targets(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestInfo manifest;
  std::string error;
  for (const std::string &artifacts : {
      std::string("[]"),
      std::string(R"([{"chip":"esp32c3","build_type":"ota","url":"https://example.invalid/c3.bin"}])"),
      std::string(R"([{"chip":"esp32s2","build_type":"factory","url":"https://example.invalid/factory.bin"}])"),
      std::string("[") + kOtaArtifact + "," + kOtaArtifact + "]",
      std::string(R"([{"chip":"esp32s2","build_type":"ota","url":"http://example.invalid/fw.bin"}])"),
      std::string(R"([{"chip":"esp32s2","build_type":"ota"}])"),
      std::string(R"([{"chip":"esp32s2","build_type":"ota","url":42}])"),
      std::string(R"([{"chip":"esp32s2","chip":"esp32s2","build_type":"ota","url":"https://example.invalid/fw.bin"}])"),
      std::string("[null]"), std::string("[{} ,]"), std::string("{}")}) {
    TEST_ASSERT_FALSE(parse_manifest_text(service, firmware_catalog(artifacts), "develop", &manifest, &error));
    TEST_ASSERT_FALSE(error.empty());
    TEST_ASSERT_TRUE(manifest.image_url.empty());
  }
  HttpsOtaService absent_frontend("matter", "esp32s2", OtaReleaseChannel::DEVELOP);
  TEST_ASSERT_FALSE(parse_manifest_text(absent_frontend, firmware_catalog("[]"), "develop", &manifest, &error));
}

void test_https_ota_manifest_parser_validates_catalog_metadata(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestInfo manifest;
  std::string error;
  const std::string valid = firmware_catalog(std::string("[") + kOtaArtifact + "]");
  TEST_ASSERT_FALSE(parse_manifest_text(service, valid, "preview", &manifest, &error));
  TEST_ASSERT_FALSE(parse_manifest_text(service, firmware_catalog("[]", "develop", ""), "develop", &manifest, &error));
  for (const char *schema : {"2", "null", "\"1\""}) {
    std::string invalid = valid;
    invalid.replace(invalid.find("\"schema_version\":1"), 18, std::string("\"schema_version\":") + schema);
    TEST_ASSERT_FALSE(parse_manifest_text(service, invalid, "develop", &manifest, &error));
  }
  TEST_ASSERT_FALSE(parse_manifest_text(service, "{}", "develop", &manifest, &error));
  TEST_ASSERT_FALSE(parse_manifest_text(service, valid + "garbage", "develop", &manifest, &error));
  TEST_ASSERT_FALSE(parse_manifest_text(service, valid, "develop", nullptr, &error));
}

void test_https_ota_fetch_enforces_status_and_manifest_size(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::RELEASE);
  HttpsOtaService::ManifestBuffer body;
  std::string error;

  g_esp_http_client_mock.response_body = "ok";
  TEST_ASSERT_TRUE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL_STRING("ok", buffer_text(body).c_str());
  TEST_ASSERT_EQUAL(30000, g_esp_http_client_mock.last_config.timeout_ms);
  TEST_ASSERT_EQUAL(8192, g_esp_http_client_mock.last_config.buffer_size);
  TEST_ASSERT_EQUAL(1024, g_esp_http_client_mock.last_config.buffer_size_tx);

  esp_http_client_mock_reset();
  g_esp_http_client_mock.response_body = firmware_catalog(std::string("[") + kOtaArtifact + "]");
  g_esp_http_client_mock.response_body.insert(1, "\"notes\":\"" + std::string(40000, 'x') + "\",");
  TEST_ASSERT_TRUE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  HttpsOtaService target("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestInfo manifest;
  TEST_ASSERT_TRUE(target.parse_manifest_(body, "develop", &manifest, &error));
  TEST_ASSERT_EQUAL_STRING("https://example.invalid/fw.bin", manifest.image_url.c_str());

  esp_http_client_mock_reset();
  g_esp_http_client_mock.status_code = 503;
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_TRUE(error.find("503") != std::string::npos);

  esp_http_client_mock_reset();
  g_esp_http_client_mock.response_body.assign(64U * 1024U + 1U, 'x');
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL_STRING("manifest too large", error.c_str());
  TEST_ASSERT_FALSE(service.fetch_manifest_("", &body, &error));
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid", nullptr, &error));
}

void test_https_ota_fetch_preserves_fragmented_manifest_bytes(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::RELEASE);
  for (size_t chunk_size : {1U, 1371U, 8192U, 65536U}) {
    esp_http_client_mock_reset();
    HttpsOtaService::ManifestBuffer body;
    std::string error;
    g_esp_http_client_mock.response_body.assign(65536U, 'x');
    const std::string boundary = "across-chunks";
    g_esp_http_client_mock.response_body.replace(1020U, boundary.size(), boundary);
    g_esp_http_client_mock.response_chunk_size = chunk_size;
    TEST_ASSERT_TRUE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
    TEST_ASSERT_TRUE(largest_buffer_allocation <= HttpsOtaService::ManifestBuffer::kChunkBytes);
    TEST_ASSERT_EQUAL_STRING(g_esp_http_client_mock.response_body.c_str(), buffer_text(body).c_str());
    TEST_ASSERT_EQUAL(1, g_esp_http_client_mock.cleanup_calls);
  }
}

void test_https_ota_fetch_handles_allocation_failure_and_retries(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::RELEASE);
  HttpsOtaService::ManifestBuffer body;
  std::string error;
  g_esp_http_client_mock.response_body.assign(8192U, 'x');
  int callbacks = 0;
  g_esp_http_client_mock.after_data = [&]() {
    ++callbacks;
    fail_buffer_allocation = true;
  };
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_TRUE(callbacks > 1);
  TEST_ASSERT_TRUE(body.size() == 0U);
  TEST_ASSERT_FALSE(error.empty());
  TEST_ASSERT_EQUAL(1, g_esp_http_client_mock.cleanup_calls);
  fail_buffer_allocation = false;
  g_esp_http_client_mock.after_data = {};
  TEST_ASSERT_TRUE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL_STRING(g_esp_http_client_mock.response_body.c_str(), buffer_text(body).c_str());
  TEST_ASSERT_TRUE(error.empty());
}

void test_https_ota_check_updates_status_and_delivers_callback(void) {
  HttpsOtaService service("native", "esp32", OtaReleaseChannel::PREVIEW);
  g_esp_http_client_mock.response_body =
      firmware_catalog(R"([{"chip":"esp32","build_type":"ota","url":"https://example.invalid/fw.bin"}])",
                       "preview", "99.0.0");
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

void test_https_ota_update_applies_newer_image_and_delivers_completion_once(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  g_esp_http_client_mock.response_body = firmware_catalog(std::string("[") + kOtaArtifact + "]");
  int preparations = 0;
  int completions = 0;
  service.set_prepare_for_update_callback([&]() { preparations++; });
  service.set_status_callback([&](const EspectreOtaStatus &status) {
    completions++;
    TEST_ASSERT_TRUE(status.state == EspectreOtaState::REBOOT_SCHEDULED);
    TEST_ASSERT_FALSE(status.busy);
  });

  TEST_ASSERT_TRUE(service.start_update("3.0.0"));
  TEST_ASSERT_EQUAL(1, g_esp_https_ota_calls);
  TEST_ASSERT_EQUAL(1, g_esp_http_client_mock.cleanup_calls);
  const auto status = service.status();
  TEST_ASSERT_TRUE(status.state == EspectreOtaState::REBOOT_SCHEDULED);
  TEST_ASSERT_EQUAL_STRING("3.1.0", status.target_version.c_str());
  TEST_ASSERT_EQUAL_STRING("https://example.invalid/fw.bin", status.image_url.c_str());
  service.loop();
  service.loop();
  TEST_ASSERT_EQUAL(1, preparations);
  TEST_ASSERT_EQUAL(1, completions);
  service.shutdown();
  service.loop();
  TEST_ASSERT_FALSE(service.start_update("3.0.0"));
  TEST_ASSERT_TRUE(service.status().state == EspectreOtaState::REBOOT_SCHEDULED);
}

void test_https_ota_update_never_installs_same_older_or_unordered_versions(void) {
  for (const char *current : {"3.1.0", "4.0.0", "3.0.0"}) {
    HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
    const bool unordered = std::string(current) == "3.0.0";
    g_esp_http_client_mock.response_body = firmware_catalog(
        std::string("[") + kOtaArtifact + "]", "develop", unordered ? "snapshot" : "3.1.0");
    TEST_ASSERT_TRUE(service.start_update(current, "develop"));
    const auto status = service.status();
    const auto expected = unordered ? EspectreOtaState::ERROR : EspectreOtaState::UP_TO_DATE;
    TEST_ASSERT_TRUE(status.state == expected);
    TEST_ASSERT_FALSE(status.busy);
    TEST_ASSERT_FALSE(status.update_available);
    TEST_ASSERT_EQUAL(0, g_esp_https_ota_calls);
  }
}

void test_https_ota_failure_retains_target_and_allows_retry(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  g_esp_http_client_mock.response_body = firmware_catalog(std::string("[") + kOtaArtifact + "]");
  g_esp_https_ota_result = ESP_FAIL;
  TEST_ASSERT_TRUE(service.start_update("3.0.0"));
  const auto failure = service.status();
  TEST_ASSERT_TRUE(failure.state == EspectreOtaState::ERROR);
  TEST_ASSERT_FALSE(failure.busy);
  TEST_ASSERT_EQUAL_STRING("3.1.0", failure.target_version.c_str());
  TEST_ASSERT_EQUAL_STRING("https://example.invalid/fw.bin", failure.image_url.c_str());
  g_esp_https_ota_result = ESP_OK;
  TEST_ASSERT_TRUE(service.start_update("3.0.0"));
  TEST_ASSERT_TRUE(service.status().state == EspectreOtaState::REBOOT_SCHEDULED);
  TEST_ASSERT_EQUAL(2, g_esp_https_ota_calls);
}

void test_https_ota_check_rejects_unordered_version_and_bad_manifest(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  g_esp_http_client_mock.response_body = firmware_catalog(
      std::string("[") + kOtaArtifact + "]", "develop", "snapshot");
  TEST_ASSERT_TRUE(service.start_check("3.0.0"));
  TEST_ASSERT_TRUE(service.status().state == EspectreOtaState::ERROR);
  g_esp_http_client_mock.response_body = "invalid";
  TEST_ASSERT_TRUE(service.start_check("3.0.0"));
  TEST_ASSERT_TRUE(service.status().state == EspectreOtaState::ERROR);
  TEST_ASSERT_FALSE(service.status().busy);
  TEST_ASSERT_EQUAL(0, g_esp_https_ota_calls);
}

void test_https_ota_fetch_releases_client_after_transport_error(void) {
  HttpsOtaService service("native", "esp32s2", OtaReleaseChannel::DEVELOP);
  HttpsOtaService::ManifestBuffer body;
  std::string error;
  g_esp_http_client_mock.init_succeeds = false;
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_TRUE(body.size() == 0U);
  TEST_ASSERT_FALSE(error.empty());
  TEST_ASSERT_EQUAL(0, g_esp_http_client_mock.cleanup_calls);
  g_esp_http_client_mock.init_succeeds = true;
  g_esp_http_client_mock.perform_result = ESP_FAIL;
  TEST_ASSERT_FALSE(service.fetch_manifest_("https://example.invalid/manifest.json", &body, &error));
  TEST_ASSERT_EQUAL(1, g_esp_http_client_mock.cleanup_calls);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_legacy_ota_service_rejects_channels_it_cannot_honor);
  RUN_TEST(test_https_ota_manifest_parser_selects_frontend_chip_and_ota_image);
  RUN_TEST(test_https_ota_manifest_parser_rejects_missing_or_ambiguous_targets);
  RUN_TEST(test_https_ota_manifest_parser_validates_catalog_metadata);
  RUN_TEST(test_https_ota_fetch_enforces_status_and_manifest_size);
  RUN_TEST(test_https_ota_fetch_preserves_fragmented_manifest_bytes);
  RUN_TEST(test_https_ota_fetch_handles_allocation_failure_and_retries);
  RUN_TEST(test_https_ota_check_updates_status_and_delivers_callback);
  RUN_TEST(test_https_ota_update_applies_newer_image_and_delivers_completion_once);
  RUN_TEST(test_https_ota_update_never_installs_same_older_or_unordered_versions);
  RUN_TEST(test_https_ota_failure_retains_target_and_allows_retry);
  RUN_TEST(test_https_ota_check_rejects_unordered_version_and_bad_manifest);
  RUN_TEST(test_https_ota_fetch_releases_client_after_transport_error);
  return UNITY_END();
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return process();
}
