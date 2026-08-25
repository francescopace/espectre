/*
 * ESPectre - Runtime Frontend Controller Tests
 *
 * Covers the lightweight controller that wraps EspIdfRuntime for the host-side
 * frontend tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "frontend_runtime_shim.h"
#include "runtime_frontend_controller.h"

#include <string>

using namespace espectre;

namespace {

class DummyRuntimeListener : public IRuntimeListener {
 public:
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override {
    threshold_count++;
    last_threshold = snapshot.threshold;
    if (controller != nullptr) {
      cached_threshold_during_callback = controller->snapshot().threshold;
      configured_threshold_during_callback = controller->config().segmentation_threshold;
      if (shutdown_on_threshold) {
        controller->shutdown();
      }
    }
  }

  void on_live_telemetry(float movement, float threshold) override {
    telemetry_count++;
    last_movement = movement;
    last_threshold = threshold;
  }

  void on_runtime_fault(const char *message) override {
    fault_count++;
    last_fault = message != nullptr ? message : "";
  }

  int fault_count{0};
  int threshold_count{0};
  int telemetry_count{0};
  float last_threshold{0.0f};
  float last_movement{0.0f};
  float cached_threshold_during_callback{0.0f};
  float configured_threshold_during_callback{0.0f};
  RuntimeFrontendController *controller{nullptr};
  bool shutdown_on_threshold{false};
  std::string last_fault;
};

}  // namespace

void setUp(void) { frontend_runtime_shim::reset(); }

void tearDown(void) {}

void test_runtime_frontend_controller_preserves_pre_setup_config_and_snapshot(void) {
  RuntimeFrontendController controller;
  RuntimeConfig config;
  config.segmentation_threshold = 0.85f;

  controller.set_config(config);

  TEST_ASSERT_EQUAL_FLOAT(0.85f, controller.config().segmentation_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.85f, controller.snapshot().threshold);

  frontend_runtime_shim::state.snapshot.threshold = 0.7f;
  DummyRuntimeListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  RuntimeConfig updated = controller.config();
  updated.segmentation_threshold = 0.5f;
  controller.set_config(updated);
  TEST_ASSERT_EQUAL_FLOAT(0.85f, controller.config().segmentation_threshold);
}

void test_runtime_frontend_controller_rejects_invalid_config_before_backend_setup(void) {
  RuntimeFrontendController controller;
  RuntimeConfig config;
  config.publish_interval_ms = 0U;
  controller.set_config(config);
  DummyRuntimeListener listener;

  TEST_ASSERT_FALSE(controller.setup(&listener));
  TEST_ASSERT_FALSE(controller.is_setup_complete());
  TEST_ASSERT_NULL(frontend_runtime_shim::state.last_instance);
  TEST_ASSERT_EQUAL(1, listener.fault_count);
  TEST_ASSERT_EQUAL_STRING("invalid publish interval", listener.last_fault.c_str());
}

void test_runtime_frontend_controller_setup_propagates_state_and_handles_failure(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  frontend_runtime_shim::state.snapshot.threshold = 3.0f;
  frontend_runtime_shim::state.capabilities.supports_manual_recalibration = false;

  controller.set_services_armed(false);
  controller.set_live_telemetry_enabled(false);

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.is_setup_complete());
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_listener);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener != &listener);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
  TEST_ASSERT_EQUAL_FLOAT(3.0f, controller.snapshot().threshold);
  TEST_ASSERT_FALSE(controller.capabilities().supports_manual_recalibration);

  TEST_ASSERT_TRUE(controller.setup(&listener));

  RuntimeFrontendController failing;
  frontend_runtime_shim::reset();
  frontend_runtime_shim::state.setup_result = false;
  TEST_ASSERT_FALSE(failing.setup(&listener));
  TEST_ASSERT_FALSE(failing.is_setup_complete());
}

void test_runtime_frontend_controller_loop_shutdown_and_runtime_toggles_forward(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;

  controller.loop();
  controller.set_services_armed(false);
  controller.set_live_telemetry_enabled(false);

  TEST_ASSERT_TRUE(controller.setup(&listener));
  controller.loop();
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);

  controller.set_services_armed(true);
  controller.set_live_telemetry_enabled(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  controller.shutdown();
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
  TEST_ASSERT_FALSE(controller.is_setup_complete());
}

void test_runtime_frontend_controller_reads_diagnostics_from_backend(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  frontend_runtime_shim::state.diagnostics.wifi_channel = 10U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 123U;
  const RuntimeDiagnosticsSnapshot diagnostics = controller.diagnostics();

  TEST_ASSERT_EQUAL(10U, diagnostics.wifi_channel);
  TEST_ASSERT_EQUAL(123U, diagnostics.csi_callbacks_total);
}

void test_runtime_frontend_controller_threshold_runtime_updates_config_and_snapshot(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;

  TEST_ASSERT_FALSE(controller.set_threshold_runtime(-0.5f));
  TEST_ASSERT_TRUE(controller.set_threshold_runtime(0.75f));
  TEST_ASSERT_EQUAL_FLOAT(0.75f, controller.snapshot().threshold);

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.set_threshold_runtime(0.5f));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.5f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.5f, controller.snapshot().threshold);
}

void test_runtime_frontend_controller_threshold_requires_capability(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  frontend_runtime_shim::state.capabilities.supports_runtime_threshold_updates = false;

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_FALSE(controller.set_threshold_runtime(0.5f));
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT,
                          controller.config().segmentation_threshold);
}

void test_runtime_frontend_controller_motion_hits_runtime_updates_config(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;

  TEST_ASSERT_FALSE(controller.set_motion_hits_runtime(0U, 3U));
  TEST_ASSERT_TRUE(controller.set_motion_hits_runtime(6U, 4U));
  TEST_ASSERT_EQUAL_UINT8(6U, controller.config().motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, controller.config().motion_off_hits);

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.set_motion_hits_runtime(5U, 2U));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(5U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(2U, frontend_runtime_shim::state.last_motion_off_hits);
}

void test_runtime_frontend_controller_traffic_runtime_updates_config(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;

  TEST_ASSERT_TRUE(controller.set_csi_traffic_mode_runtime(CsiTrafficMode::EXTERNAL));
  TEST_ASSERT_TRUE(controller.config().csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_TRUE(controller.set_traffic_generator_mode_runtime(RuntimeTrafficMode::DNS));
  TEST_ASSERT_TRUE(controller.config().traffic_generator_mode == RuntimeTrafficMode::DNS);

  TEST_ASSERT_FALSE(controller.set_csi_traffic_mode_runtime(CsiTrafficMode::PACING));
  TEST_ASSERT_FALSE(controller.set_csi_traffic_mode_runtime(static_cast<CsiTrafficMode>(0x7f)));
  TEST_ASSERT_FALSE(controller.set_traffic_generator_mode_runtime(static_cast<RuntimeTrafficMode>(0x7f)));

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.set_csi_traffic_mode_runtime(CsiTrafficMode::DISABLED));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::DISABLED);

  TEST_ASSERT_TRUE(controller.set_traffic_generator_mode_runtime(RuntimeTrafficMode::PING));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_traffic_generator_mode == RuntimeTrafficMode::PING);
}

void test_runtime_frontend_controller_recalibration_requires_capability_and_runtime(void) {
  RuntimeFrontendController controller;

  TEST_ASSERT_FALSE(controller.trigger_recalibration());

  DummyRuntimeListener listener;
  frontend_runtime_shim::state.capabilities.supports_manual_recalibration = true;
  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_FALSE(controller.is_calibrating());

  frontend_runtime_shim::state.calibrating = true;
  TEST_ASSERT_TRUE(controller.is_calibrating());
  TEST_ASSERT_TRUE(controller.trigger_recalibration());
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);

  controller.shutdown();
  TEST_ASSERT_FALSE(controller.is_calibrating());
}

void test_runtime_frontend_controller_refreshes_snapshot_across_raw_collection(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  frontend_runtime_shim::state.snapshot.ready_to_publish = true;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  TEST_ASSERT_TRUE(controller.start_raw_collection(
      [](void *, const RawCsiPacketView &) { return true; }, nullptr));
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, controller.operation_state());
  TEST_ASSERT_FALSE(controller.snapshot().ready_to_publish);

  TEST_ASSERT_TRUE(controller.stop_raw_collection());
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, controller.operation_state());
  TEST_ASSERT_TRUE(controller.snapshot().ready_to_publish);
}

void test_runtime_frontend_controller_caches_and_forwards_listener_events(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  listener.controller = &controller;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  RuntimeSnapshot snapshot = controller.snapshot();
  snapshot.threshold = 0.65f;
  frontend_runtime_shim::state.last_listener->on_threshold_changed(snapshot);

  TEST_ASSERT_EQUAL(1, listener.threshold_count);
  TEST_ASSERT_EQUAL_FLOAT(0.65f, listener.last_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.65f, listener.cached_threshold_during_callback);
  TEST_ASSERT_EQUAL_FLOAT(0.65f, listener.configured_threshold_during_callback);

  frontend_runtime_shim::state.last_listener->on_live_telemetry(0.4f, 0.6f);
  TEST_ASSERT_EQUAL(1, listener.telemetry_count);
  TEST_ASSERT_EQUAL_FLOAT(0.4f, listener.last_movement);
  TEST_ASSERT_EQUAL_FLOAT(0.4f, controller.snapshot().movement_metric);
  TEST_ASSERT_EQUAL_FLOAT(0.6f, controller.snapshot().threshold);
}

void test_runtime_frontend_controller_defers_shutdown_requested_by_listener(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  listener.controller = &controller;
  listener.shutdown_on_threshold = true;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  frontend_runtime_shim::state.snapshot.threshold = 0.55f;
  frontend_runtime_shim::state.emit_threshold_on_next_loop = true;
  controller.loop();

  TEST_ASSERT_EQUAL(1, listener.threshold_count);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
  TEST_ASSERT_FALSE(controller.is_setup_complete());
}

void test_runtime_frontend_controller_switches_detector_and_resets_threshold(void) {
  RuntimeFrontendController controller;
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.segmentation_threshold = 0.4f;
  controller.set_config(config);

  TEST_ASSERT_TRUE(controller.set_detection_algorithm_runtime(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_TRUE(controller.config().detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY);
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, controller.snapshot().threshold);

  frontend_runtime_shim::state.capabilities.supports_runtime_detector_selection = true;
  DummyRuntimeListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.set_detection_algorithm_runtime(DetectionAlgorithm::LIGHTWEIGHT));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_detector == DetectionAlgorithm::LIGHTWEIGHT);
  TEST_ASSERT_EQUAL_FLOAT(LIGHTWEIGHT_DEFAULT_THRESHOLD, controller.snapshot().threshold);
}

void test_runtime_frontend_controller_can_select_stream_runtime_profile(void) {
  RuntimeFrontendController controller;
  DummyRuntimeListener listener;
  RuntimeConfig config;
  config.runtime_profile = RuntimeProfile::STREAM;
  config.csi_traffic_mode = CsiTrafficMode::DISABLED;
  config.device_id = 0x1234U;

  frontend_runtime_shim::reset();
  controller.set_config(config);

  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.is_setup_complete());
  TEST_ASSERT_NULL(frontend_runtime_shim::state.last_listener);

  controller.shutdown();
  TEST_ASSERT_FALSE(controller.is_setup_complete());
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_runtime_frontend_controller_preserves_pre_setup_config_and_snapshot);
  RUN_TEST(test_runtime_frontend_controller_rejects_invalid_config_before_backend_setup);
  RUN_TEST(test_runtime_frontend_controller_setup_propagates_state_and_handles_failure);
  RUN_TEST(test_runtime_frontend_controller_loop_shutdown_and_runtime_toggles_forward);
  RUN_TEST(test_runtime_frontend_controller_reads_diagnostics_from_backend);
  RUN_TEST(test_runtime_frontend_controller_threshold_runtime_updates_config_and_snapshot);
  RUN_TEST(test_runtime_frontend_controller_threshold_requires_capability);
  RUN_TEST(test_runtime_frontend_controller_motion_hits_runtime_updates_config);
  RUN_TEST(test_runtime_frontend_controller_traffic_runtime_updates_config);
  RUN_TEST(test_runtime_frontend_controller_recalibration_requires_capability_and_runtime);
  RUN_TEST(test_runtime_frontend_controller_refreshes_snapshot_across_raw_collection);
  RUN_TEST(test_runtime_frontend_controller_caches_and_forwards_listener_events);
  RUN_TEST(test_runtime_frontend_controller_defers_shutdown_requested_by_listener);
  RUN_TEST(test_runtime_frontend_controller_switches_detector_and_resets_threshold);
  RUN_TEST(test_runtime_frontend_controller_can_select_stream_runtime_profile);
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
