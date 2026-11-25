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
#include "runtime/direct_http_protocol.h"
#include "runtime_frontend_controller.h"
#include "runtime_guest_access.h"

#include <cstdio>
#include <string>
#include <type_traits>

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

void test_guest_can_recover_wifi_after_sensing_stops(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  bool scanning = false;
  config.wifi_scan_started = [&]() { scanning = true; };
  config.wifi_scan_results_getter = [&](const std::string &, std::vector<StandaloneWifiAccessPoint> &results) {
    if (!scanning) {
      results.push_back({"mesh", "02:00:00:00:00:01", -45, 6});
      results.push_back({"other", "02:00:00:00:00:02", -50, 11});
    }
    return scanning;
  };
  config.wifi_snapshot_getter = []() {
    DirectWifiSnapshot wifi;
    wifi.ssid = "mesh";
    wifi.bssid = "02:00:00:00:00:01";
    wifi.apply_state = "applied";
    return wifi;
  };
  std::string applied;
  config.wifi_bssid_pin_setter = [&applied](const std::string &pin, bool, std::string *) {
    applied = pin;
    return true;
  };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
  const DirectRequest pin{"", "set_wifi_bssid", R"({"bssid":"02:00:00:00:00:01"})"};
  owner.shutdown();
  TEST_ASSERT_TRUE(contains(service.emit_deferred_request(1U, {"", "scan_wifi", "{}"}).response, "\"accepted\":true"));
  const DirectRequest aps{"", "wifi_access_points", "{}", "/espectre/v1/wifi/access-points", "GET"};
  TEST_ASSERT_TRUE(contains(service.emit_deferred_request(2U, aps).response, "\"scanning\":true"));
  scanning = false;
  const auto scan = service.emit_deferred_request(3U, aps);
  TEST_ASSERT_TRUE(contains(scan.response, "02:00:00:00:00:01"));
  TEST_ASSERT_FALSE(contains(scan.response, "02:00:00:00:00:02"));
  const auto wifi = service.emit_deferred_request(4U, {"", "wifi", "{}", "/espectre/v1/wifi", "GET"});
  TEST_ASSERT_TRUE(contains(wifi.response, "\"bssid\":\"02:00:00:00:00:01\""));
  TEST_ASSERT_TRUE(contains(wifi.response, "\"pinned_bssid\":\"\""));
  TEST_ASSERT_TRUE(contains(wifi.response, "\"apply_state\":\"applied\""));
  auto result = service.emit_deferred_request(7U, pin);
  TEST_ASSERT_TRUE(contains(result.response, "\"accepted\":true"));
  TEST_ASSERT_TRUE(applied.empty());
  TEST_ASSERT_TRUE(static_cast<bool>(result.response_sent_callback));
  result.response_sent_callback(false);
  TEST_ASSERT_TRUE(applied.empty());
  result = service.emit_deferred_request(8U, pin);
  TEST_ASSERT_TRUE(static_cast<bool>(result.response_sent_callback));
  result.response_sent_callback(true);
  TEST_ASSERT_EQUAL_STRING("02:00:00:00:00:01", applied.c_str());
}

void test_guest_refuses_wifi_pin_before_acknowledging_during_raw_collection(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  config.wifi_bssid_pin_setter = [](const std::string &, bool, std::string *) { return true; };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
  TEST_ASSERT_TRUE(owner.start_raw_collection([](void *, const RawCsiPacketView &) { return true; }, nullptr));
  const auto scan = service.emit_deferred_request(8U, DirectRequest{"", "scan_wifi", "{}"});
  TEST_ASSERT_TRUE(contains(scan.response, "\"code\":\"busy_raw_collection\""));
  const auto result = service.emit_deferred_request(9U, DirectRequest{"", "clear_wifi_bssid", "{}"});
  TEST_ASSERT_TRUE(contains(result.response, "\"code\":\"busy_raw_collection\""));
  TEST_ASSERT_FALSE(static_cast<bool>(result.response_sent_callback));
}

void test_pending_pin_holds_off_raw_capture_until_response_completion(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  config.raw_csi = true;
  int apply_calls = 0;
  config.wifi_bssid_pin_setter = [&](const std::string &, bool, std::string *) {
    ++apply_calls;
    return true;
  };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
  const DirectRequest clear{"", "clear_wifi_bssid", "{}"};
  const auto capture = [](void *, const RawCsiPacketView &) { return true; };
  for (const bool sent : {false, true}) {
    auto pin = service.emit_deferred_request(1U, clear);
    TEST_ASSERT_TRUE(contains(pin.response, "\"accepted\":true"));
    TEST_ASSERT_FALSE(service.emit_raw_session_request());
    TEST_ASSERT_FALSE(owner.start_raw_collection(capture, nullptr));
    const auto second = service.emit_deferred_request(2U, clear);
    TEST_ASSERT_TRUE(contains(second.response, "\"accepted\":false"));
    TEST_ASSERT_FALSE(static_cast<bool>(second.response_sent_callback));
    TEST_ASSERT_TRUE(static_cast<bool>(pin.response_sent_callback));
    pin.response_sent_callback(sent);
    TEST_ASSERT_EQUAL(sent ? 1 : 0, apply_calls);
    TEST_ASSERT_TRUE(service.emit_raw_session_request());
    TEST_ASSERT_TRUE(service.stop_raw_session(RawCsiStopReason::REQUESTED));
  }
  // Dropping the pending response also releases the station.
  TEST_ASSERT_TRUE(contains(service.emit_deferred_request(3U, clear).response, "\"accepted\":true"));
  TEST_ASSERT_TRUE(owner.start_raw_collection(capture, nullptr));
  TEST_ASSERT_EQUAL(1, apply_calls);
}

void test_raw_capture_waits_for_reported_wifi_verification_and_rollback(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  config.raw_csi = true;
  DirectWifiSnapshot wifi;
  wifi.connected = true;
  config.wifi_snapshot_getter = [&wifi]() { return wifi; };
  config.wifi_bssid_pin_setter = [&wifi](const std::string &, bool, std::string *) {
    wifi.apply_state = "verifying";
    return true;
  };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
  auto pin = service.emit_deferred_request(1U, {"", "clear_wifi_bssid", "{}"});
  TEST_ASSERT_TRUE(static_cast<bool>(pin.response_sent_callback));
  pin.response_sent_callback(true);
  TEST_ASSERT_FALSE(service.emit_raw_session_request());
  wifi.apply_state = "rolling_back";
  TEST_ASSERT_FALSE(service.emit_raw_session_request());
  TEST_ASSERT_FALSE(service.raw_diagnostics().active);

  TEST_ASSERT_TRUE(espectre_wifi_apply_in_progress(ESPECTRE_WIFI_APPLY_STATE_VERIFYING));
  TEST_ASSERT_TRUE(espectre_wifi_apply_in_progress(ESPECTRE_WIFI_APPLY_STATE_ROLLING_BACK));
  for (const char *state : {ESPECTRE_WIFI_APPLY_STATE_ROLLED_BACK, ESPECTRE_WIFI_APPLY_STATE_APPLIED,
                            ESPECTRE_WIFI_APPLY_STATE_RECOVERY_REQUIRED, ESPECTRE_WIFI_APPLY_STATE_IDLE, ""}) {
    TEST_ASSERT_FALSE(espectre_wifi_apply_in_progress(state));
    wifi.apply_state = state;
    TEST_ASSERT_TRUE(service.emit_raw_session_request());
    TEST_ASSERT_TRUE(service.stop_raw_session(RawCsiStopReason::REQUESTED));
  }

  wifi.apply_state = "verifying";
  config.wifi_snapshot_getter = {};
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
  TEST_ASSERT_TRUE(service.emit_raw_session_request());
  TEST_ASSERT_TRUE(service.stop_raw_session(RawCsiStopReason::REQUESTED));
}

void test_frontend_scan_caps_matching_access_points(void) {
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  config.wifi_snapshot_getter = []() {
    DirectWifiSnapshot wifi;
    wifi.ssid = "mesh";
    return wifi;
  };
  config.wifi_scan_results_getter = [](const std::string &ssid, std::vector<StandaloneWifiAccessPoint> &results) {
    results.push_back({"other", "02:00:00:00:01:00", -40, 1});
    for (unsigned i = 0; i < 160; ++i) {
      char bssid[18];
      std::snprintf(bssid, sizeof(bssid), "02:00:00:00:00:%02X", i);
      results.push_back({ssid, bssid, -50, 6});
    }
    return false;
  };
  TEST_ASSERT_TRUE(bridge.setup(&service, &owner, config));
  const auto scan = service.emit_request({"", "wifi_access_points", "{}", "/espectre/v1/wifi/access-points", "GET"});
  TEST_ASSERT_TRUE(scan.size() <= ESPECTRE_DIRECT_MAX_RESPONSE_SIZE);
  TEST_ASSERT_TRUE(contains(scan, "02:00:00:00:00:1F"));
  TEST_ASSERT_FALSE(contains(scan, "02:00:00:00:00:20"));
  TEST_ASSERT_FALSE(contains(scan, "02:00:00:00:01:00"));
}

void test_bridge_exposes_raw_capture_only_when_enabled(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  for (bool enabled : {true, false}) {
    config.raw_csi = enabled;
    TEST_ASSERT_TRUE(bridge.setup_guest(&service, config, {}));
    const auto capabilities = service.emit_request({"", "capabilities", "{}", "", "GET"});
    TEST_ASSERT_TRUE(contains(capabilities, enabled ? R"("csi":true)" : R"("csi":false)"));
    TEST_ASSERT_EQUAL(enabled, service.emit_raw_session_request());
    TEST_ASSERT_EQUAL(enabled, service.raw_diagnostics().active);
    bridge.shutdown();
    TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());
  }
}

void test_raw_controller_detaches_capture_and_pending_notifications(void) {
  using Controller = RawCsiSessionController;
  using Configure = void (Controller::*)(IDirectHttpService *, RuntimeFrontendController *, uint64_t,
                                         std::string, Controller::StoppedCallback, Controller::StartedCallback);
  static_assert(std::is_same_v<decltype(&Controller::configure), Configure>);
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  IDirectHttpService::RawSessionStoppedCallback pending;
  std::vector<RawCsiStopReason> reasons;
  {
    Controller raw;
    raw.configure(&service, &owner, 1U, "esp32", [&](RawCsiStopReason reason) { reasons.push_back(reason); });
    TEST_ASSERT_TRUE(service.emit_raw_session_request());
    raw.shutdown(RawCsiStopReason::REQUESTED);
    TEST_ASSERT_FALSE(raw.active());
    TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());
    TEST_ASSERT_EQUAL(1U, reasons.size());
    TEST_ASSERT_EQUAL(RawCsiStopReason::REQUESTED, reasons.front());
    TEST_ASSERT_TRUE(service.emit_raw_session_request());

    // The worker already stopped the transport, but its notification is queued.
    pending = std::move(service_mock::state.raw_stopped_callback);
    TEST_ASSERT_TRUE(service.stop_raw_session(RawCsiStopReason::SLOW_CLIENT));
    raw.detach();
    TEST_ASSERT_EQUAL(1U, reasons.size());
    TEST_ASSERT_FALSE(service.emit_raw_session_request());
    TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());

    raw.configure(&service, &owner, 1U, "esp32",
                  [&](RawCsiStopReason reason) { reasons.push_back(reason); });
    TEST_ASSERT_TRUE(service.emit_raw_session_request());
    pending(RawCsiStopReason::SLOW_CLIENT);
    pending(RawCsiStopReason::SLOW_CLIENT);
    TEST_ASSERT_TRUE(raw.active());
    TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, owner.operation_state());
    TEST_ASSERT_EQUAL(2U, reasons.size());
    TEST_ASSERT_EQUAL(RawCsiStopReason::SLOW_CLIENT, reasons.back());
    pending = std::move(service_mock::state.raw_stopped_callback);
  }
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());
  TEST_ASSERT_FALSE(service.emit_raw_session_request());
  pending(RawCsiStopReason::SHUTDOWN);
  TEST_ASSERT_EQUAL(2U, reasons.size());
}

void test_raw_detach_refuses_a_queued_open_from_its_stop_callback(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RawCsiSessionController raw;
  IDirectHttpService::RawSessionRequestedCallback queued_open;
  bool reopened = false;
  raw.configure(&service, &owner, 1U, "esp32", [&](RawCsiStopReason) { reopened = queued_open(nullptr); });
  queued_open = service_mock::state.raw_session_requested_callback;
  TEST_ASSERT_TRUE(service.emit_raw_session_request());
  raw.detach();
  TEST_ASSERT_FALSE(reopened);
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());
}

void test_bridge_reports_and_confirms_only_effective_sensing_settings(void) {
  RuntimeFrontendController owner;
  owner.config().motion_on_hits = 3;
  owner.config().motion_off_hits = 5;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  RuntimeControlSink sink;
  sink.queued = true;
  sink.motion_hits = [](uint8_t, uint8_t, std::string *) { return true; };
  TEST_ASSERT_TRUE(bridge.setup_guest(&service, bridge_config(), sink));
  owner.config().motion_on_hits = 7;
  const DirectRequest read{"", "sensing", "{}", "", "GET"};
  TEST_ASSERT_TRUE(contains(service.emit_request(read), R"("motion_on_hits":3)"));

  const DirectRequest update{"", "update_sensing", R"({"motion_on_hits":7,"motion_off_hits":5})"};
  TEST_ASSERT_TRUE(service.emit_deferred_request(1U, update).deferred);
  bridge.loop();
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, R"("accepted":false)"));

  owner.config().motion_on_hits = 8;
  TEST_ASSERT_TRUE(service.emit_deferred_request(2U, update).deferred);
  TEST_ASSERT_TRUE(owner.set_motion_hits(7, 5));
  bridge.loop();
  TEST_ASSERT_TRUE(contains(service_mock::state.last_deferred_response, R"("accepted":true)"));
  TEST_ASSERT_TRUE(contains(service.emit_request(read), R"("motion_on_hits":7)"));
  TEST_ASSERT_EQUAL(8, owner.config().motion_on_hits);
}

void test_reused_bridge_releases_previous_peer_discovery(void) {
  class Discovery : public IPeerDiscoveryService {
   public:
    void set_wifi_ready(bool) override {}
    bool ready() const override { return false; }
    bool active() const override { return false; }
    bool start(Completion) override { return false; }
    void loop() override {}
    void shutdown() override { ++stops; }
    int stops{0};
  } discovery;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RuntimeDirectHttpBridge bridge;
  auto config = bridge_config();
  config.peer_discovery = &discovery;
  TEST_ASSERT_TRUE(bridge.setup(&service, &owner, config));
  bridge.shutdown();
  TEST_ASSERT_EQUAL(1, discovery.stops);
  service_mock::state.setup_result = false;
  TEST_ASSERT_FALSE(bridge.setup(&service, &owner, config));
  service_mock::state.setup_result = true;
  TEST_ASSERT_TRUE(bridge.setup(&service, &owner, bridge_config()));
  TEST_ASSERT_EQUAL(1, discovery.stops);
}

void test_destroyed_bridge_leaves_no_callback_in_the_service(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  {
    RuntimeDirectHttpBridge bridge;
    auto config = bridge_config();
    config.raw_csi = true;
    TEST_ASSERT_TRUE(bridge.setup(&service, &owner, config));
    TEST_ASSERT_TRUE(service.emit_raw_session_request());
  }
  TEST_ASSERT_TRUE(service_mock::state.shutdown_called);
  TEST_ASSERT_FALSE(service.emit_raw_session_request());
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, owner.operation_state());
}

void test_raw_capture_reports_paused_sensing(void) {
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  RuntimeFrontendController owner;
  TEST_ASSERT_TRUE(owner.setup(nullptr));
  service_mock::MockDirectHttpService service;
  RawCsiSessionController raw;
  raw.configure(&service, &owner, 1U, "esp32");
  owner.set_services_armed(false);
  std::string message;
  TEST_ASSERT_FALSE(service.emit_raw_session_request(&message));
  TEST_ASSERT_EQUAL_STRING("sensing is paused", message.c_str());
  TEST_ASSERT_FALSE(service.raw_diagnostics().active);
  owner.set_services_armed(true);
  TEST_ASSERT_TRUE(service.emit_raw_session_request(&message));
  raw.shutdown();
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_bridge_exposes_raw_capture_only_when_enabled);
  RUN_TEST(test_raw_controller_detaches_capture_and_pending_notifications);
  RUN_TEST(test_raw_detach_refuses_a_queued_open_from_its_stop_callback);
  RUN_TEST(test_bridge_reports_and_confirms_only_effective_sensing_settings);
  RUN_TEST(test_reused_bridge_releases_previous_peer_discovery);
  RUN_TEST(test_destroyed_bridge_leaves_no_callback_in_the_service);
  RUN_TEST(test_raw_capture_reports_paused_sensing);
  RUN_TEST(test_guest_bridge_needs_a_set_up_runtime);
  RUN_TEST(test_guest_bridge_leaves_changes_to_the_owner_and_answers_after_it);
  RUN_TEST(test_guest_bridge_reports_a_change_the_owner_did_not_apply);
  RUN_TEST(test_guest_bridge_with_an_immediate_sink_answers_at_once);
  RUN_TEST(test_guest_bridge_offers_only_the_controls_the_owner_takes);
  RUN_TEST(test_guest_bridge_offers_no_sensing_route_without_a_field);
  RUN_TEST(test_guest_bridge_refuses_changes_while_the_owner_runtime_is_stopped);
  RUN_TEST(test_guest_can_recover_wifi_after_sensing_stops);
  RUN_TEST(test_guest_refuses_wifi_pin_before_acknowledging_during_raw_collection);
  RUN_TEST(test_pending_pin_holds_off_raw_capture_until_response_completion);
  RUN_TEST(test_raw_capture_waits_for_reported_wifi_verification_and_rollback);
  RUN_TEST(test_frontend_scan_caps_matching_access_points);
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
