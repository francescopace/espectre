/*
 * ESPectre - Runtime Direct HTTP Bridge Tests
 *
 * Covers the bridge when it serves a runtime that another component owns.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "direct_http_service_mock.h"
#include "frontend_runtime_shim.h"
#include "runtime_direct_http_bridge.h"
#include "runtime_frontend_controller.h"
#include "runtime_guest_access.h"

#include <string>

using namespace espectre;

namespace {

namespace service_mock = direct_http_service_mock;

RuntimeDirectHttpBridgeConfig bridge_config() {
  RuntimeDirectHttpBridgeConfig config;
  config.frontend = "esphome";
  config.device_name = "Guest";
  config.chip = "esp32c6";
  config.device_id = 0x1234U;
  return config;
}

// What the owner of the runtime was asked to change.
struct OwnerRequests {
  int threshold_calls{0};
  float threshold{0.0f};
  int recalibrate_calls{0};
};

RuntimeControlSink queued_sink(OwnerRequests *requests) {
  RuntimeControlSink sink;
  sink.threshold = [requests](float value, std::string *) {
    requests->threshold_calls++;
    requests->threshold = value;
    return true;
  };
  sink.recalibrate = [requests](std::string *) {
    requests->recalibrate_calls++;
    return true;
  };
  sink.queued = true;
  return sink;
}

size_t sensing_events() {
  size_t count = 0U;
  for (const auto &event : service_mock::state.published_events) {
    if (event.event_name == "sensing") count++;
  }
  return count;
}

bool contains(const std::string &text, const char *needle) { return text.find(needle) != std::string::npos; }

const DirectRequest THRESHOLD_REQUEST{"", "update_sensing", R"({"threshold":0.4})", "/espectre/v1/sensing", "PATCH"};

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  service_mock::reset();
}

void tearDown(void) {}

void test_guest_bridge_needs_a_set_up_runtime(void) {
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  TEST_ASSERT_FALSE(bridge.setup_guest(&service, bridge_config(), RuntimeControlSink{}));

  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), RuntimeControlSink{}));
  TEST_ASSERT_TRUE(bridge.running());
}

void test_guest_bridge_leaves_changes_to_the_owner_and_answers_after_it(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  OwnerRequests requests;
  int config_changed = 0;
  TEST_ASSERT_TRUE(
      bridge.setup_guest(&service, bridge_config(), queued_sink(&requests), [&config_changed] { config_changed++; }));
  service.emit_client_count(1U);

  const auto result = service.emit_deferred_request(7U, THRESHOLD_REQUEST);
  TEST_ASSERT_TRUE(result.deferred);
  TEST_ASSERT_EQUAL(1, requests.threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.4f, requests.threshold);
  // The bridge neither wrote to the runtime nor reported a state the owner has not reached.
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(0U, sensing_events());
  TEST_ASSERT_EQUAL(0, config_changed);
  TEST_ASSERT_EQUAL(0U, service_mock::state.last_completed_token);

  // The owner applies the queued value in its own loop, then the bridge loops again.
  TEST_ASSERT_TRUE(owner.set_threshold(requests.threshold));
  bridge.loop();
  TEST_ASSERT_EQUAL(1U, sensing_events());
  TEST_ASSERT_EQUAL(1, config_changed);
  TEST_ASSERT_EQUAL(7U, service_mock::state.last_completed_token);
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, "\"accepted\":true"));

  bridge.loop();
  TEST_ASSERT_EQUAL(1U, sensing_events());
  TEST_ASSERT_EQUAL(1, config_changed);
}

void test_guest_bridge_reports_a_change_the_owner_did_not_apply(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  OwnerRequests requests;
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), queued_sink(&requests)));
  service.emit_client_count(1U);

  // The owner took the value, but its runtime refused it; the owner can only log that.
  TEST_ASSERT_TRUE(service.emit_deferred_request(11U, THRESHOLD_REQUEST).deferred);
  bridge.loop();
  TEST_ASSERT_EQUAL(11U, service_mock::state.last_completed_token);
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, "\"accepted\":false"));
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, "\"code\":\"unavailable\""));

  // A recalibration that did not start is busy, as it is when the bridge writes to the runtime.
  TEST_ASSERT_TRUE(service.emit_deferred_request(12U, DirectRequest{"", "recalibrate", "{}"}).deferred);
  bridge.loop();
  TEST_ASSERT_EQUAL(12U, service_mock::state.last_completed_token);
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, "\"code\":\"busy\""));

  TEST_ASSERT_TRUE(service.emit_deferred_request(13U, DirectRequest{"", "recalibrate", "{}"}).deferred);
  TEST_ASSERT_TRUE(owner.trigger_recalibration());
  bridge.loop();
  TEST_ASSERT_EQUAL(13U, service_mock::state.last_completed_token);
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, "\"accepted\":true"));

  // A calibration is running now, so the next one is refused before the owner is asked.
  const auto busy = service.emit_deferred_request(14U, DirectRequest{"", "recalibrate", "{}"});
  TEST_ASSERT_FALSE(busy.deferred);
  TEST_ASSERT_TRUE(contains(busy.response, "\"code\":\"busy\""));
  TEST_ASSERT_EQUAL(2, requests.recalibrate_calls);
}

void test_guest_bridge_with_an_immediate_sink_answers_at_once(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  RuntimeControlSink sink;
  sink.threshold = [&owner](float value, std::string *) { return owner.set_threshold(value); };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), sink));
  service.emit_client_count(1U);

  const auto result = service.emit_deferred_request(9U, THRESHOLD_REQUEST);
  TEST_ASSERT_FALSE(result.deferred);
  TEST_ASSERT_TRUE(contains(result.response, "\"accepted\":true"));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(1U, sensing_events());
}

void test_guest_bridge_offers_only_the_controls_the_owner_takes(void) {
  frontend_runtime_shim::state.capabilities.supports_runtime_detector_selection = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  OwnerRequests requests;
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), queued_sink(&requests)));

  const std::string capabilities = service.emit_request(DirectRequest{"", "capabilities", "{}"});
  TEST_ASSERT_TRUE(contains(capabilities, "\"recalibrate\""));
  // The owner takes the threshold, so the sensing route is offered without an on/off callback.
  TEST_ASSERT_TRUE(contains(capabilities, "\"update_sensing\""));
  TEST_ASSERT_TRUE(contains(capabilities, "\"fields\":[\"threshold\"]"));
  const std::string enabled = service.emit_request(
      DirectRequest{"", "update_sensing", R"({"enabled":false})", "/espectre/v1/sensing", "PATCH"});
  TEST_ASSERT_TRUE(contains(enabled, "\"code\":\"unsupported\""));
  // No callback for the detector, and the owner manages the station.
  TEST_ASSERT_FALSE(contains(capabilities, "\"set_wifi_bssid\""));
  TEST_ASSERT_FALSE(contains(capabilities, "\"scan_wifi\""));

  const std::string detector = service.emit_request(
      DirectRequest{"", "update_sensing", R"({"detector":"high_accuracy"})", "/espectre/v1/sensing", "PATCH"});
  TEST_ASSERT_TRUE(contains(detector, "\"code\":\"unsupported\""));
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_detector_calls);

  const auto pin = service.emit_deferred_request(
      3U, DirectRequest{"", "set_wifi_bssid", R"({"bssid":"E6:FA:C4:20:19:DE"})"});
  TEST_ASSERT_FALSE(pin.deferred);
  TEST_ASSERT_TRUE(contains(pin.response, "\"code\":\"unsupported\""));

  const auto recalibrate = service.emit_deferred_request(4U, DirectRequest{"", "recalibrate", "{}"});
  TEST_ASSERT_TRUE(recalibrate.deferred);
  TEST_ASSERT_EQUAL(1, requests.recalibrate_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.trigger_recalibration_calls);
}

void test_guest_bridge_offers_no_sensing_route_without_a_field(void) {
  // The owner takes the threshold, but this runtime cannot change it.
  frontend_runtime_shim::state.capabilities.supports_runtime_threshold_updates = false;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  OwnerRequests requests;
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), queued_sink(&requests)));

  const std::string capabilities = service.emit_request(DirectRequest{"", "capabilities", "{}"});
  TEST_ASSERT_TRUE(contains(capabilities, "\"recalibrate\""));
  TEST_ASSERT_FALSE(contains(capabilities, "\"update_sensing\""));
}

void test_guest_bridge_refuses_changes_while_the_owner_runtime_is_stopped(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  OwnerRequests requests;
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), queued_sink(&requests)));

  owner.shutdown();
  const auto stopped = service.emit_deferred_request(5U, THRESHOLD_REQUEST);
  TEST_ASSERT_FALSE(stopped.deferred);
  TEST_ASSERT_TRUE(contains(stopped.response, "\"code\":\"unavailable\""));
  TEST_ASSERT_EQUAL(0, requests.threshold_calls);
  const std::string sensing =
      service.emit_request(DirectRequest{"", "sensing", "{}", "/espectre/v1/sensing", "GET"});
  TEST_ASSERT_TRUE(contains(sensing, "\"threshold\""));

  TEST_ASSERT_TRUE(owner.setup(nullptr));
  TEST_ASSERT_TRUE(service.emit_deferred_request(6U, THRESHOLD_REQUEST).deferred);
  TEST_ASSERT_EQUAL(1, requests.threshold_calls);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_guest_bridge_needs_a_set_up_runtime);
  RUN_TEST(test_guest_bridge_leaves_changes_to_the_owner_and_answers_after_it);
  RUN_TEST(test_guest_bridge_reports_a_change_the_owner_did_not_apply);
  RUN_TEST(test_guest_bridge_with_an_immediate_sink_answers_at_once);
  RUN_TEST(test_guest_bridge_offers_only_the_controls_the_owner_takes);
  RUN_TEST(test_guest_bridge_offers_no_sensing_route_without_a_field);
  RUN_TEST(test_guest_bridge_refuses_changes_while_the_owner_runtime_is_stopped);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
