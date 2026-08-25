/*
 * ESPectre - ESP-IDF Direct HTTP Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cstring>
#include <string>

#include "direct_http_service_esp_idf.h"
#include "esp_http_server.h"
#include "esp_timer.h"

using namespace espectre;

namespace {

DirectHttpServiceConfig config() {
  DirectHttpServiceConfig value = DirectHttpServiceConfig::for_first_party_portals();
  value.max_event_clients = 2U;
  value.max_pending_requests = 2U;
  value.outbound_queue_depth = 2U;
  value.max_mutations_per_minute = 2U;
  value.allow_http_loopback_origins = true;
  return value;
}

httpd_req_t request_for(size_t registered_index, int fd = 7) {
  return httpd_req_t{g_httpd_mock.registered_uris[registered_index].user_ctx,
                     fd,
                     g_httpd_mock.incoming_length,
                     0U,
                     false};
}

void prepare_json(const char *payload, const char *origin = "https://espectre.dev") {
  httpd_mock_set_incoming(payload);
  httpd_mock_set_header("Origin", origin);
  httpd_mock_set_header("Content-Type", "application/json; charset=utf-8");
}

std::string sent_payload(int index) {
  return std::string(reinterpret_cast<const char *>(g_httpd_mock.sent_payloads[index]),
                     g_httpd_mock.sent_lengths[index]);
}

void test_setup_registers_http_post_sse_raw_and_preflight() {
  httpd_mock_reset();
  EspIdfDirectHttpService service;
  DirectHttpServiceConfig invalid;
  TEST_ASSERT_FALSE(service.setup(invalid, [](const auto &) { return std::string{"{}"}; }, {}));
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  TEST_ASSERT_TRUE(service.running());
  TEST_ASSERT_EQUAL(6, g_httpd_mock.register_calls);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT,
                           g_httpd_mock.registered_uris[0].uri);
  TEST_ASSERT_EQUAL(HTTP_POST, g_httpd_mock.registered_uris[0].method);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT,
                           g_httpd_mock.registered_uris[1].uri);
  TEST_ASSERT_EQUAL(HTTP_GET, g_httpd_mock.registered_uris[1].method);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_RAW_CSI_ENDPOINT, g_httpd_mock.registered_uris[2].uri);
  TEST_ASSERT_EQUAL(HTTP_GET, g_httpd_mock.registered_uris[2].method);
  TEST_ASSERT_EQUAL(HTTP_OPTIONS, g_httpd_mock.registered_uris[3].method);
  TEST_ASSERT_EQUAL(7U, g_httpd_mock.last_config.max_open_sockets);
  TEST_ASSERT_EQUAL(8U, g_httpd_mock.last_config.max_uri_handlers);
  TEST_ASSERT_EQUAL(1U, g_httpd_mock.last_config.recv_wait_timeout);
  TEST_ASSERT_EQUAL(1U, g_httpd_mock.last_config.send_wait_timeout);
  service.shutdown();
  TEST_ASSERT_EQUAL(1, g_httpd_mock.stop_calls);
}

void test_post_validates_origin_content_type_size_and_dispatches_on_loop() {
  httpd_mock_reset();
  EspIdfDirectHttpService service;
  std::string method;
  TEST_ASSERT_TRUE(service.setup(
      config(),
      [&method](const DirectRequest &request) {
        method = request.method;
        return direct_http_success_response(request.id, "{\"methods\":[]}");
      },
      {}));

  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"r1\",\"method\":\"capabilities\",\"params\":{}}",
               "https://evil.example");
  httpd_req_t rejected = request_for(0U);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&rejected));
  TEST_ASSERT_EQUAL_STRING(HTTPD_403_FORBIDDEN, g_httpd_mock.response_status);

  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"r1\",\"method\":\"capabilities\",\"params\":{}}");
  httpd_req_t request = request_for(0U);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[0].handler(&request));
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
  service.loop();
  TEST_ASSERT_EQUAL_STRING("capabilities", method.c_str());
  TEST_ASSERT_EQUAL(2, g_httpd_mock.send_calls);
  TEST_ASSERT_TRUE(sent_payload(1).find("\"id\":\"r1\"") != std::string::npos);
  TEST_ASSERT_EQUAL_STRING("application/json; charset=utf-8", g_httpd_mock.response_type);
  TEST_ASSERT_EQUAL_STRING("no-store", g_httpd_mock.cache_control);
  TEST_ASSERT_EQUAL_STRING("https://espectre.dev", g_httpd_mock.allow_origin);
  TEST_ASSERT_EQUAL(1, g_httpd_mock.async_complete_calls);

  prepare_json("{}");
  httpd_mock_set_header("Content-Type", "text/plain");
  httpd_req_t wrong_type = request_for(0U);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&wrong_type));
  TEST_ASSERT_EQUAL_STRING(HTTPD_415_UNSUPPORTED_MEDIA_TYPE, g_httpd_mock.response_status);

  prepare_json("not-json");
  httpd_req_t malformed = request_for(0U);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&malformed));
  TEST_ASSERT_EQUAL_STRING(HTTPD_400_BAD_REQUEST, g_httpd_mock.response_status);
  TEST_ASSERT_EQUAL(1U, service.diagnostics().malformed_requests);

  prepare_json("{}");
  httpd_req_t oversized = request_for(0U);
  oversized.content_len = ESPECTRE_DIRECT_MAX_REQUEST_SIZE + 1U;
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&oversized));
  TEST_ASSERT_EQUAL_STRING(HTTPD_413_CONTENT_TOO_LARGE, g_httpd_mock.response_status);
  TEST_ASSERT_EQUAL(1U, service.diagnostics().oversized_requests);
}

void test_options_returns_private_network_cors_headers() {
  httpd_mock_reset();
  EspIdfDirectHttpService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  httpd_mock_set_header("Origin", "https://test.espectre.dev");
  httpd_mock_set_header("Access-Control-Request-Private-Network", "true");
  httpd_req_t request = request_for(3U);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[3].handler(&request));
  TEST_ASSERT_EQUAL_STRING("204 No Content", g_httpd_mock.response_status);
  TEST_ASSERT_EQUAL_STRING("https://test.espectre.dev", g_httpd_mock.allow_origin);
  TEST_ASSERT_EQUAL_STRING("true", g_httpd_mock.allow_private_network);
  TEST_ASSERT_EQUAL_STRING("no-store", g_httpd_mock.cache_control);
}

void test_post_distinguishes_queue_saturation_from_mutation_rate_limit() {
  httpd_mock_reset();
  EspIdfDirectHttpService service;
  DirectHttpServiceConfig limits = config();
  limits.max_pending_requests = 1U;
  limits.max_mutations_per_minute = 1U;
  TEST_ASSERT_TRUE(service.setup(
      limits, [](const DirectRequest &request) {
        return direct_http_success_response(request.id, "{}");
      }, {}));

  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"q1\",\"method\":\"capabilities\",\"params\":{}}");
  httpd_req_t queued = request_for(0U, 20);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[0].handler(&queued));
  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"q2\",\"method\":\"status\",\"params\":{}}");
  httpd_req_t full = request_for(0U, 21);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&full));
  TEST_ASSERT_EQUAL_STRING(HTTPD_503_SERVICE_UNAVAILABLE, g_httpd_mock.response_status);
  service.loop();

  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"m1\",\"method\":\"set_threshold\",\"params\":{}}");
  httpd_req_t first_mutation = request_for(0U, 22);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[0].handler(&first_mutation));
  service.loop();
  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"m2\",\"method\":\"recalibrate\",\"params\":{}}");
  httpd_req_t limited = request_for(0U, 23);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[0].handler(&limited));
  TEST_ASSERT_EQUAL_STRING(HTTPD_429_TOO_MANY_REQUESTS, g_httpd_mock.response_status);
  TEST_ASSERT_EQUAL(1U, service.diagnostics().rate_limited_requests);
}

void test_sse_limits_clients_frames_events_coalesces_and_heartbeats() {
  httpd_mock_reset();
  esp_timer_mock::reset(100000U, 0U);
  EspIdfDirectHttpService service;
  size_t reported_clients = 0U;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; },
                                 [&reported_clients](size_t count) { reported_clients = count; }));
  httpd_mock_set_header("Origin", "https://espectre.dev");
  httpd_req_t first = request_for(1U, 11);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[1].handler(&first));
  TEST_ASSERT_EQUAL_STRING("text/event-stream; charset=utf-8", g_httpd_mock.response_type);
  TEST_ASSERT_EQUAL(1U, service.event_client_count());
  TEST_ASSERT_EQUAL(1U, reported_clients);
  TEST_ASSERT_TRUE(sent_payload(0).find("retry: 3000") != std::string::npos);

  TEST_ASSERT_TRUE(service.publish_event("telemetry", "{\"movement\":0.1}", true));
  TEST_ASSERT_TRUE(service.publish_event("telemetry", "{\"movement\":0.9}", true));
  service.loop();
  TEST_ASSERT_EQUAL(2, g_httpd_mock.send_calls);
  TEST_ASSERT_TRUE(sent_payload(1).find("event: telemetry") != std::string::npos);
  TEST_ASSERT_TRUE(sent_payload(1).find("0.9") != std::string::npos);
  TEST_ASSERT_TRUE(sent_payload(1).find("0.1") == std::string::npos);

  esp_timer_mock::advance(10000000U);
  service.loop();
  TEST_ASSERT_EQUAL_STRING(": ping\n\n", sent_payload(2).c_str());

  httpd_req_t second = request_for(1U, 12);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[1].handler(&second));
  httpd_req_t extra = request_for(1U, 13);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[1].handler(&extra));
  TEST_ASSERT_EQUAL_STRING(HTTPD_503_SERVICE_UNAVAILABLE, g_httpd_mock.response_status);
  TEST_ASSERT_EQUAL(1U, service.diagnostics().rejected_connections);
}

void test_deferred_post_completes_only_once() {
  httpd_mock_reset();
  EspIdfDirectHttpService service;
  uint64_t token = 0U;
  std::string request_id;
  TEST_ASSERT_TRUE(service.setup_deferred(
      config(),
      [&token, &request_id](uint64_t current, const DirectRequest &request) {
        token = current;
        request_id = request.id;
        return IDirectHttpService::DeferredRequestResult{true, {}};
      },
      {}));
  prepare_json("{\"v\":1,\"type\":\"request\",\"id\":\"peers\",\"method\":\"discover_peers\",\"params\":{}}");
  httpd_req_t request = request_for(0U);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[0].handler(&request));
  service.loop();
  TEST_ASSERT_TRUE(token != 0U);
  TEST_ASSERT_EQUAL_STRING("peers", request_id.c_str());
  TEST_ASSERT_TRUE(service.complete_deferred_response(
      token, direct_http_success_response(request_id, "{\"devices\":[]}")));
  TEST_ASSERT_FALSE(service.complete_deferred_response(
      token, direct_http_success_response(request_id, "{\"devices\":[]}")));
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
}

void test_raw_get_requires_bearer_and_emits_paced_v8_frame() {
  httpd_mock_reset();
  esp_timer_mock::reset(100000U, 0U);
  EspIdfDirectHttpService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  RawCsiSessionConfig session{};
  session.device_id = 0x112233445566ULL;
  session.chip = StreamChipType::C3;
  session.target_pps = 100U;
  session.max_sample_age_us = 20000U;
  for (size_t index = 0U; index < sizeof(session.session_id); ++index) {
    session.session_id[index] = static_cast<uint8_t>(index);
  }
  TEST_ASSERT_TRUE(service.start_raw_session(session, {}));

  httpd_mock_set_header("Origin", "https://espectre.dev");
  httpd_req_t missing = request_for(2U, 9);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uris[2].handler(&missing));
  TEST_ASSERT_EQUAL_STRING(HTTPD_401_UNAUTHORIZED, g_httpd_mock.response_status);

  httpd_mock_set_header("Authorization",
                        "Bearer 000102030405060708090a0b0c0d0e0f");
  httpd_req_t raw_request = request_for(2U, 9);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[2].handler(&raw_request));
  TEST_ASSERT_TRUE(service.raw_diagnostics().binary_bound);
  TEST_ASSERT_EQUAL_STRING("application/octet-stream", g_httpd_mock.response_type);

  const int8_t csi[] = {1, -2, 3, -4};
  RawCsiPacketView packet{};
  packet.csi = csi;
  packet.csi_len = sizeof(csi);
  packet.captured_at_us = 100000U;
  packet.stream_flags = STREAM_FLAG_CSI_FRESH;
  packet.channel = 6U;
  packet.rssi_dbm = -45;
  packet.noise_floor_dbm = -96;
  packet.phy_mode = StreamPhyMode::HT;
  packet.ltf_type = StreamLtfType::HT_LTF;
  packet.channel_width = StreamChannelWidth::MHZ_20;
  TEST_ASSERT_TRUE(service.offer_raw_packet(packet));
  service.loop();
  TEST_ASSERT_EQUAL(2, g_httpd_mock.send_calls);
  TEST_ASSERT_EQUAL(9, g_httpd_mock.sent_fds[1]);
  TEST_ASSERT_EQUAL(sizeof(RawCsiHttpFramePrefixV1) + sizeof(CsiStreamHeaderV8) + sizeof(csi),
                    g_httpd_mock.sent_lengths[1]);
  const auto *prefix = reinterpret_cast<const RawCsiHttpFramePrefixV1 *>(
      g_httpd_mock.sent_payloads[1]);
  TEST_ASSERT_EQUAL(ESPECTRE_RAW_CSI_RESPONSE_MAGIC, prefix->magic);
  TEST_ASSERT_EQUAL(static_cast<uint8_t>(RawCsiResponseStatus::FRESH), prefix->status);
  TEST_ASSERT_EQUAL(1U, prefix->stream_sequence);
  TEST_ASSERT_EQUAL(sizeof(CsiStreamHeaderV8) + sizeof(csi), prefix->record_len);
  const auto *header = reinterpret_cast<const CsiStreamHeaderV8 *>(
      g_httpd_mock.sent_payloads[1] + sizeof(RawCsiHttpFramePrefixV1));
  TEST_ASSERT_EQUAL(STREAM_VERSION_V8, header->version);
  TEST_ASSERT_EQUAL(100000U, header->device_ticks_us);
  TEST_ASSERT_EQUAL(1U, header->fresh_record_total);
  TEST_ASSERT_TRUE(service.stop_raw_session(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_FALSE(service.raw_diagnostics().active);
}

void test_raw_pacing_uses_absolute_deadlines_without_accumulating_worker_jitter() {
  httpd_mock_reset();
  esp_timer_mock::reset(100000U, 0U);
  EspIdfDirectHttpService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  RawCsiSessionConfig session{};
  session.device_id = 0x112233445566ULL;
  session.chip = StreamChipType::C3;
  session.target_pps = 100U;
  session.max_sample_age_us = 20000U;
  session.session_id[0] = 1U;
  TEST_ASSERT_TRUE(service.start_raw_session(session, {}));

  httpd_mock_set_header("Origin", "https://espectre.dev");
  httpd_mock_set_header("Authorization", "Bearer 01000000000000000000000000000000");
  httpd_req_t raw_request = request_for(2U, 9);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uris[2].handler(&raw_request));

  const int8_t csi[] = {1, -2, 3, -4};
  RawCsiPacketView packet{};
  packet.csi = csi;
  packet.csi_len = sizeof(csi);
  packet.captured_at_us = 100000U;
  TEST_ASSERT_TRUE(service.offer_raw_packet(packet));
  service.loop();
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);

  esp_timer_mock::advance(11000U);
  packet.captured_at_us = 111000U;
  TEST_ASSERT_TRUE(service.offer_raw_packet(packet));
  service.loop();
  TEST_ASSERT_EQUAL(2, g_httpd_mock.send_calls);

  // The next absolute deadline remains 120000 us. The old now+interval
  // implementation drifted to 121000 us and missed this worker iteration.
  esp_timer_mock::advance(9000U);
  packet.captured_at_us = 120000U;
  TEST_ASSERT_TRUE(service.offer_raw_packet(packet));
  service.loop();
  TEST_ASSERT_EQUAL(3, g_httpd_mock.send_calls);
}

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_setup_registers_http_post_sse_raw_and_preflight);
  RUN_TEST(test_post_validates_origin_content_type_size_and_dispatches_on_loop);
  RUN_TEST(test_options_returns_private_network_cors_headers);
  RUN_TEST(test_post_distinguishes_queue_saturation_from_mutation_rate_limit);
  RUN_TEST(test_sse_limits_clients_frames_events_coalesces_and_heartbeats);
  RUN_TEST(test_deferred_post_completes_only_once);
  RUN_TEST(test_raw_get_requires_bearer_and_emits_paced_v8_frame);
  RUN_TEST(test_raw_pacing_uses_absolute_deadlines_without_accumulating_worker_jitter);
  return espectre::test::end_suite();
}
