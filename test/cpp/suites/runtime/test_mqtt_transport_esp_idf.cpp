/*
 * ESPectre - ESP-IDF MQTT Transport Tests
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <memory>
#include <string>
#include <new>

#include "sdkconfig.h"
#if CONFIG_MBEDTLS_CERTIFICATE_BUNDLE
#include "esp_crt_bundle.h"
#endif
#include "mqtt_client.h"
#include "mqtt_transport_esp_idf.h"

namespace {
bool reject_nothrow_allocation = false;
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  if (reject_nothrow_allocation) return nullptr;
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

using namespace espectre;

namespace {

EspectreDeviceConfig config() {
  EspectreDeviceConfig value;
  value.device_id = 1U;
  value.mqtt_scheme = "mqtt";
  value.mqtt_host = "broker.local";
  value.mqtt_port = 1883U;
  return value;
}

void connect(EspIdfMqttTransport &transport) {
  mqtt_client_mock_emit(MQTT_EVENT_CONNECTED, nullptr, nullptr, 0, 0);
  transport.loop();
  TEST_ASSERT_TRUE(transport.connected());
}

void test_receive_storage_failure_shutdown_and_fragmented_recovery() {
  mqtt_client_mock_reset();
  TEST_ASSERT_TRUE(sizeof(EspIdfMqttTransport) < 2048U);
  EspIdfMqttTransport transport;
  reject_nothrow_allocation = true;
  const bool started_without_storage = transport.setup(config());
  reject_nothrow_allocation = false;
  TEST_ASSERT_FALSE(started_without_storage);
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.init_calls);
  size_t received = 0U;
  transport.set_command_callback([&received](const std::string &payload) {
    TEST_ASSERT_EQUAL_STRING("abcdef", payload.c_str());
    ++received;
  });
  for (uint8_t cycle = 0U; cycle < 3U; ++cycle) {
    TEST_ASSERT_TRUE(transport.setup(config()));
    connect(transport);
    const std::string topic = espectre_topic(config(), "commands/request");
    mqtt_client_mock_emit(MQTT_EVENT_DATA, topic.c_str(), "abc", 6, 0);
    mqtt_client_mock_emit(MQTT_EVENT_DATA, cycle == 0 ? topic.c_str() : nullptr, "def", 6, 3);
    transport.loop();
    TEST_ASSERT_EQUAL(cycle + 1U, received);
    mqtt_client_mock_emit(MQTT_EVENT_DATA, topic.c_str(), "abc", 6, 0);
    transport.shutdown();
    transport.loop();
    TEST_ASSERT_FALSE(transport.connected());
  }
}

void test_topicless_fragments_require_an_uninterrupted_command() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);
  int received = 0;
  transport.set_command_callback([&](const std::string &) { ++received; });
  const std::string topic = espectre_topic(config(), "commands/request");
  mqtt_client_mock_emit(MQTT_EVENT_DATA, topic.c_str(), "abc", 6, 0);
  mqtt_client_mock_emit(MQTT_EVENT_DATA, "other", "uvw", 6, 0);
  mqtt_client_mock_emit(MQTT_EVENT_DATA, nullptr, "xyz", 6, 3);
  transport.loop();
  TEST_ASSERT_EQUAL(0, received);
}

void test_client_is_destroyed_once_on_scope_exit() {
  for (bool explicit_shutdown : {false, true}) {
    mqtt_client_mock_reset();
    {
      EspIdfMqttTransport transport;
      TEST_ASSERT_TRUE(transport.setup(config()));
      if (explicit_shutdown) transport.shutdown();
    }
    TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.stop_calls);
    TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.destroy_calls);
  }
}

void test_setup_bounds_the_esp_mqtt_outbox() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_FALSE(transport.setup(EspectreDeviceConfig{}));
  TEST_ASSERT_TRUE(transport.setup(config()));
  TEST_ASSERT_EQUAL_STRING("", g_mqtt_client_mock.broker_uri);
  TEST_ASSERT_EQUAL_STRING("broker.local", g_mqtt_client_mock.broker_hostname);
  TEST_ASSERT_EQUAL(1883U, g_mqtt_client_mock.broker_port);
  TEST_ASSERT_EQUAL(MQTT_TRANSPORT_OVER_TCP, g_mqtt_client_mock.broker_transport);
  TEST_ASSERT_TRUE(g_mqtt_client_mock.crt_bundle_attach == nullptr);
  TEST_ASSERT_EQUAL(8192U, g_mqtt_client_mock.outbox_limit);
  TEST_ASSERT_EQUAL(16U, transport.diagnostics().queue_capacity);
  TEST_ASSERT_EQUAL(8192U, transport.diagnostics().outbox_capacity_bytes);
}

void test_mqtts_requires_the_certificate_bundle_for_broker_verification() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  EspectreDeviceConfig secure = config();
  secure.mqtt_scheme = "mqtts";
  secure.mqtt_port = 8883U;
#if CONFIG_MBEDTLS_CERTIFICATE_BUNDLE
  TEST_ASSERT_TRUE(transport.setup(secure));
  TEST_ASSERT_EQUAL_STRING("broker.local", g_mqtt_client_mock.broker_hostname);
  TEST_ASSERT_EQUAL(8883U, g_mqtt_client_mock.broker_port);
  TEST_ASSERT_EQUAL(MQTT_TRANSPORT_OVER_SSL, g_mqtt_client_mock.broker_transport);
  TEST_ASSERT_TRUE(g_mqtt_client_mock.crt_bundle_attach == esp_crt_bundle_attach);
  TEST_ASSERT_FALSE(g_mqtt_client_mock.skip_cert_common_name_check);
#else
  TEST_ASSERT_FALSE(transport.setup(secure));
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.init_calls);

  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);
  TEST_ASSERT_FALSE(transport.setup(secure));
  TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.init_calls);
  TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.destroy_calls);
  TEST_ASSERT_FALSE(transport.connected());
  TEST_ASSERT_FALSE(transport.publish_suffix("telemetry", "{}", false));
#endif
}

void test_invalid_endpoint_is_rejected_before_client_initialization() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  EspectreDeviceConfig invalid = config();
  invalid.mqtt_scheme.clear();
  TEST_ASSERT_FALSE(transport.setup(invalid));
  invalid = config();
  invalid.mqtt_host = "mqtt://broker.local";
  TEST_ASSERT_FALSE(transport.setup(invalid));
  invalid = config();
  invalid.mqtt_port = 0U;
  TEST_ASSERT_FALSE(transport.setup(invalid));
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.init_calls);
}

void test_publish_is_deferred_and_latest_snapshot_replaces_stale_data() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);

  TEST_ASSERT_TRUE(transport.publish_suffix("telemetry", "{\"sequence\":1}", false));
  TEST_ASSERT_TRUE(transport.publish_suffix("telemetry", "{\"sequence\":2}", false));
  TEST_ASSERT_TRUE(transport.publish("homeassistant/sensor/espectre/config", "{\"sequence\":1}", true));
  TEST_ASSERT_TRUE(transport.publish("homeassistant/sensor/espectre/config", "{\"sequence\":2}", true));
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.enqueue_calls);
  TEST_ASSERT_EQUAL(2U, transport.diagnostics().queued_publishes);

  transport.loop();
  TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.enqueue_calls);
  TEST_ASSERT_TRUE(std::string(g_mqtt_client_mock.enqueued_payloads[0]).find("\"sequence\":2") !=
                   std::string::npos);
  transport.loop();
  TEST_ASSERT_EQUAL(2, g_mqtt_client_mock.enqueue_calls);
  TEST_ASSERT_TRUE(std::string(g_mqtt_client_mock.enqueued_payloads[1]).find("\"sequence\":2") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL(0U, transport.diagnostics().queued_publishes);
}

void test_command_results_overtake_replaceable_snapshots() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);

  TEST_ASSERT_TRUE(transport.publish_suffix("telemetry", "{\"sequence\":1}", false));
  g_mqtt_client_mock.enqueue_result = -2;
  transport.loop();
  TEST_ASSERT_TRUE(transport.publish_suffix("commands/result", "{\"id\":\"cmd-1\"}", false));

  g_mqtt_client_mock.enqueue_result = 7;
  transport.loop();
  TEST_ASSERT_TRUE(std::string(g_mqtt_client_mock.enqueued_topics[1]).find("commands/result") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL(1U, transport.diagnostics().queued_publishes);
}

void test_replaceable_publishes_use_available_outbox_capacity() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);

  g_mqtt_client_mock.outbox_size = 1;
  TEST_ASSERT_TRUE(transport.publish_suffix("telemetry", "{\"sequence\":1}", false));
  transport.loop();
  TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.enqueue_calls);
  TEST_ASSERT_TRUE(std::string(g_mqtt_client_mock.enqueued_topics[0]).find("telemetry") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL(0U, transport.diagnostics().queued_publishes);

  g_mqtt_client_mock.outbox_size = 2048;
  TEST_ASSERT_TRUE(transport.publish_suffix("telemetry", "{\"sequence\":2}", false));
  transport.loop();
  TEST_ASSERT_EQUAL(1, g_mqtt_client_mock.enqueue_calls);
  TEST_ASSERT_EQUAL(1U, transport.diagnostics().queued_publishes);
}

void test_queue_rejects_overflow_without_discarding_critical_messages() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);

  for (int index = 0; index < 16; ++index) {
    TEST_ASSERT_TRUE(transport.publish("critical/" + std::to_string(index), "{}", false));
  }
  TEST_ASSERT_FALSE(transport.publish("critical/overflow", "{}", false));
  const auto diagnostics = transport.diagnostics();
  TEST_ASSERT_EQUAL(16U, diagnostics.queued_publishes);
  TEST_ASSERT_EQUAL(1U, diagnostics.dropped_publishes);
}

void test_full_outbox_retries_without_growing_the_frontend_queue() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);
  TEST_ASSERT_TRUE(transport.publish("critical/state", "{\"ready\":true}", false));

  g_mqtt_client_mock.enqueue_result = -2;
  transport.loop();
  TEST_ASSERT_EQUAL(1U, transport.diagnostics().queued_publishes);
  TEST_ASSERT_EQUAL(1U, transport.diagnostics().publish_failures);

  g_mqtt_client_mock.enqueue_result = 7;
  transport.loop();
  TEST_ASSERT_EQUAL(0U, transport.diagnostics().queued_publishes);
  TEST_ASSERT_EQUAL(2, g_mqtt_client_mock.enqueue_calls);
}

void test_reconnects_are_observable() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);
  mqtt_client_mock_emit(MQTT_EVENT_DISCONNECTED, nullptr, nullptr, 0, 0);
  transport.loop();
  connect(transport);
  TEST_ASSERT_EQUAL(1U, transport.diagnostics().reconnects);
}

void test_subscribe_rejects_unusable_requests_and_defers_until_connected() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  const EspIdfMqttTransport::MessageCallback callback = [](const std::string &, const std::string &) {};
  // Without a client the registration would be dropped by setup().
  TEST_ASSERT_FALSE(transport.subscribe("espectre/topic", callback));
  TEST_ASSERT_TRUE(transport.setup(config()));
  TEST_ASSERT_FALSE(transport.subscribe("", callback));
  TEST_ASSERT_FALSE(transport.subscribe("espectre/topic", nullptr));

  TEST_ASSERT_TRUE(transport.subscribe("espectre/topic", callback));
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.subscribe_calls);
  connect(transport);
  const int after_connect = g_mqtt_client_mock.subscribe_calls;
  TEST_ASSERT_TRUE(after_connect >= 1);

  // Subscribing again replaces the handler and, while connected, renews the broker subscription.
  TEST_ASSERT_TRUE(transport.subscribe("espectre/topic", callback));
  TEST_ASSERT_EQUAL(after_connect + 1, g_mqtt_client_mock.subscribe_calls);
  TEST_ASSERT_TRUE(transport.subscribe("espectre/other", callback));
  TEST_ASSERT_EQUAL(after_connect + 2, g_mqtt_client_mock.subscribe_calls);
}

void test_rejected_setup_preserves_the_previous_client() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_TRUE(transport.setup(config()));
  connect(transport);
  TEST_ASSERT_TRUE(transport.connected());
  int received = 0;
  TEST_ASSERT_TRUE(transport.subscribe("test/topic", [&](const auto &, const auto &) { ++received; }));
  TEST_ASSERT_FALSE(transport.setup(EspectreDeviceConfig{}));
  TEST_ASSERT_EQUAL(0, g_mqtt_client_mock.destroy_calls);
  TEST_ASSERT_TRUE(transport.connected());
  mqtt_client_mock_emit(MQTT_EVENT_DATA, "test/topic", "data", 4, 0);
  transport.loop();
  TEST_ASSERT_EQUAL(1, received);
  TEST_ASSERT_TRUE(transport.publish_suffix("motion", "{}", false));
}

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_receive_storage_failure_shutdown_and_fragmented_recovery);
  RUN_TEST(test_topicless_fragments_require_an_uninterrupted_command);
  RUN_TEST(test_client_is_destroyed_once_on_scope_exit);
  RUN_TEST(test_setup_bounds_the_esp_mqtt_outbox);
  RUN_TEST(test_rejected_setup_preserves_the_previous_client);
  RUN_TEST(test_mqtts_requires_the_certificate_bundle_for_broker_verification);
  RUN_TEST(test_invalid_endpoint_is_rejected_before_client_initialization);
  RUN_TEST(test_publish_is_deferred_and_latest_snapshot_replaces_stale_data);
  RUN_TEST(test_command_results_overtake_replaceable_snapshots);
  RUN_TEST(test_replaceable_publishes_use_available_outbox_capacity);
  RUN_TEST(test_queue_rejects_overflow_without_discarding_critical_messages);
  RUN_TEST(test_full_outbox_retries_without_growing_the_frontend_queue);
  RUN_TEST(test_reconnects_are_observable);
  RUN_TEST(test_subscribe_rejects_unusable_requests_and_defers_until_connected);
  return espectre::test::end_suite();
}
