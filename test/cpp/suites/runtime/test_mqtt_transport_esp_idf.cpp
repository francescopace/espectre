/*
 * ESPectre - ESP-IDF MQTT Transport Tests
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <string>

#include "mqtt_client.h"
#include "mqtt_transport_esp_idf.h"

using namespace espectre;

namespace {

EspectreDeviceConfig config() {
  EspectreDeviceConfig value;
  value.device_id = 1U;
  value.mqtt_host = "broker.local";
  value.mqtt_port = 1883U;
  return value;
}

void connect(EspIdfMqttTransport &transport) {
  mqtt_client_mock_emit(MQTT_EVENT_CONNECTED, nullptr, nullptr, 0, 0);
  transport.loop();
  TEST_ASSERT_TRUE(transport.connected());
}

void test_setup_bounds_the_esp_mqtt_outbox() {
  mqtt_client_mock_reset();
  EspIdfMqttTransport transport;
  TEST_ASSERT_FALSE(transport.setup(EspectreDeviceConfig{}));
  TEST_ASSERT_TRUE(transport.setup(config()));
  TEST_ASSERT_EQUAL_STRING("mqtt://broker.local:1883", g_mqtt_client_mock.broker_uri);
  TEST_ASSERT_EQUAL(8192U, g_mqtt_client_mock.outbox_limit);
  TEST_ASSERT_EQUAL(16U, transport.diagnostics().queue_capacity);
  TEST_ASSERT_EQUAL(8192U, transport.diagnostics().outbox_capacity_bytes);
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

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_setup_bounds_the_esp_mqtt_outbox);
  RUN_TEST(test_publish_is_deferred_and_latest_snapshot_replaces_stale_data);
  RUN_TEST(test_command_results_overtake_replaceable_snapshots);
  RUN_TEST(test_replaceable_publishes_use_available_outbox_capacity);
  RUN_TEST(test_queue_rejects_overflow_without_discarding_critical_messages);
  RUN_TEST(test_full_outbox_retries_without_growing_the_frontend_queue);
  RUN_TEST(test_reconnects_are_observable);
  return espectre::test::end_suite();
}
