/*
 * ESPectre - Native Frontend Unit Tests
 *
 * Unit tests for Native Frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "test_harness.h"

#include <algorithm>
#include <vector>

#define private public
#define protected public
#include "native_frontend.h"
#include "recovery_button_service.h"
#undef protected
#undef private

#include "direct_http_service_mock.h"
#include "frontend_runtime_shim.h"
#include "mqtt_transport_mock.h"
#include "ota_service_mock.h"

using namespace espectre;
using espectre::direct_http_service_mock::MockDirectHttpService;
using espectre::mqtt_transport_mock::MockMqttTransport;
using espectre::ota_service_mock::MockOtaService;

namespace {

class MockPeerDiscoveryService final : public IPeerDiscoveryService {
 public:
  void set_local_candidate(PeerDiscoveryCandidate candidate) override {
    local_candidate = std::move(candidate);
  }
  void set_wifi_ready(bool ready) override { wifi_ready = ready; }
  bool ready() const override { return wifi_ready && !query_active; }
  bool active() const override { return query_active; }
  bool start(Completion next_completion) override {
    start_calls += 1;
    if (!start_result || !ready() || !next_completion) return false;
    completion = std::move(next_completion);
    query_active = true;
    return true;
  }
  void loop() override {}
  void shutdown() override {
    shutdown_calls += 1;
    query_active = false;
    completion = {};
  }
  void finish(PeerDiscoverySnapshot snapshot) {
    query_active = false;
    Completion current = std::move(completion);
    completion = {};
    if (current) current(std::move(snapshot));
  }

  bool wifi_ready{false};
  bool query_active{false};
  bool start_result{true};
  size_t start_calls{0U};
  size_t shutdown_calls{0U};
  PeerDiscoveryCandidate local_candidate{};
  Completion completion{};
};

[[maybe_unused]] RuntimeSnapshot make_ready_snapshot() {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.startup_threshold = 0.42f;
  snapshot.detector_name = "lightweight";
  return snapshot;
}

[[maybe_unused]] bool has_mqtt_publish(const std::string &topic, const char *payload = nullptr) {
  return std::any_of(mqtt_transport_mock::state.publishes.begin(),
                     mqtt_transport_mock::state.publishes.end(),
                     [&](const mqtt_transport_mock::Publish &publish) {
                       if (publish.topic != topic) {
                         return false;
                       }
                       return payload == nullptr || publish.payload == payload;
                     });
}

[[maybe_unused]] bool has_mqtt_publish_containing(const std::string &topic, const char *fragment) {
  return std::any_of(mqtt_transport_mock::state.publishes.begin(),
                     mqtt_transport_mock::state.publishes.end(),
                     [&](const mqtt_transport_mock::Publish &publish) {
                       return publish.topic == topic && publish.payload.find(fragment) != std::string::npos;
                     });
}

[[maybe_unused]] int mqtt_publish_index(const std::string &topic) {
  const auto &publishes = mqtt_transport_mock::state.publishes;
  for (size_t i = 0; i < publishes.size(); ++i) {
    if (publishes[i].topic == topic) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  direct_http_service_mock::reset();
  mqtt_transport_mock::reset();
  ota_service_mock::reset();
}

void tearDown(void) {}

