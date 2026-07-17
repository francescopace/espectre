/*
 * ESPectre - Runtime Frontend Controller Tests
 *
 * Covers the lightweight controller that wraps EspIdfRuntime for the host-side
 * frontend tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include "frontend_runtime_shim.h"
#include "runtime_frontend_controller.h"

using namespace espectre;

namespace {

class DummyRuntimeListener : public IRuntimeListener {};

}  // namespace

void setUp(void) { frontend_runtime_shim::reset(); }

void tearDown(void) {}

void test_runtime_frontend_controller_preserves_pre_setup_config_and_snapshot(void) {
  RuntimeFrontendController controller;
  RuntimeConfig config;
  config.segmentation_threshold = 4.5f;

  controller.set_config(config);

  TEST_ASSERT_EQUAL_FLOAT(4.5f, controller.config().segmentation_threshold);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, controller.snapshot().threshold);

  frontend_runtime_shim::state.snapshot.threshold = 2.25f;
  DummyRuntimeListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));

  RuntimeConfig updated = controller.config();
  updated.segmentation_threshold = 1.5f;
  controller.set_config(updated);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, controller.config().segmentation_threshold);
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
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == &listener);
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

void test_runtime_frontend_controller_recalibration_requires_capability_and_runtime(void) {
  RuntimeFrontendController controller;
  RuntimeSnapshot snapshot;
  snapshot.threshold = 1.25f;

  TEST_ASSERT_FALSE(controller.trigger_recalibration());
  controller.record_snapshot(snapshot);
  TEST_ASSERT_EQUAL_FLOAT(1.25f, controller.snapshot().threshold);

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

void test_runtime_frontend_controller_switches_detector_and_resets_threshold(void) {
  RuntimeFrontendController controller;
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.segmentation_threshold = 0.4f;
  controller.set_config(config);

  TEST_ASSERT_TRUE(controller.set_detection_algorithm_runtime(DetectionAlgorithm::ML));
  TEST_ASSERT_TRUE(controller.config().detection_algorithm == DetectionAlgorithm::ML);
  TEST_ASSERT_EQUAL_FLOAT(ML_DEFAULT_THRESHOLD, controller.snapshot().threshold);

  frontend_runtime_shim::state.capabilities.supports_runtime_detector_selection = true;
  DummyRuntimeListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));
  TEST_ASSERT_TRUE(controller.set_detection_algorithm_runtime(DetectionAlgorithm::CLASSIC));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_detector == DetectionAlgorithm::CLASSIC);
  TEST_ASSERT_EQUAL_FLOAT(CLASSIC_DEFAULT_THRESHOLD, controller.snapshot().threshold);
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
  RUN_TEST(test_runtime_frontend_controller_setup_propagates_state_and_handles_failure);
  RUN_TEST(test_runtime_frontend_controller_loop_shutdown_and_runtime_toggles_forward);
  RUN_TEST(test_runtime_frontend_controller_threshold_runtime_updates_config_and_snapshot);
  RUN_TEST(test_runtime_frontend_controller_recalibration_requires_capability_and_runtime);
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
