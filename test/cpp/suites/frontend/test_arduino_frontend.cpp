/*
 * ESPectre - Arduino Frontend Unit Tests
 *
 * Unit tests for the Arduino sensor adapter.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include "esp_timer.h"

#include <string>
#include <vector>

#include "ESPectre.h"
#include "frontend_runtime_shim.h"

using namespace espectre;

namespace {

RuntimeSnapshot make_snapshot(bool ready, bool motion) {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = ready;
  snapshot.motion_state = motion ? MotionState::MOTION : MotionState::IDLE;
  snapshot.movement_metric = 0.75f;
  snapshot.threshold = 0.5f;
  return snapshot;
}

}  // namespace

void setUp(void) {
  esp_timer_mock::reset();
  frontend_runtime_shim::reset();
}

void tearDown(void) {}

void test_arduino_sensor_starts_from_sdk_defaults_without_saved_controls(void) {
  const RuntimeConfig defaults = make_runtime_sensing_config_from_kconfig();
  ESPectre sensor;

  TEST_ASSERT_FALSE(sensor.config().persist_runtime_overrides);
  TEST_ASSERT_TRUE(sensor.config().detection_algorithm == defaults.detection_algorithm);
  TEST_ASSERT_EQUAL_FLOAT(defaults.threshold, sensor.config().threshold);
  TEST_ASSERT_TRUE(sensor.config().traffic_generator_mode == defaults.traffic_generator_mode);
}

void test_arduino_sensor_begin_starts_the_runtime_once(void) {
  ESPectre sensor;
  sensor.config().csi_target_pps = 50U;

  TEST_ASSERT_TRUE(sensor.begin());
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_listener);
  TEST_ASSERT_EQUAL(50U, sensor.runtime().config().csi_target_pps);
  IRuntimeListener *listener = frontend_runtime_shim::state.last_listener;

  TEST_ASSERT_TRUE(sensor.begin());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == listener);

  sensor.loop();
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
}

void test_arduino_sensor_begin_can_retry_after_a_failed_setup(void) {
  ESPectre sensor;
  frontend_runtime_shim::state.setup_result = false;
  TEST_ASSERT_FALSE(sensor.begin());
  TEST_ASSERT_FALSE(sensor.runtime().is_setup_complete());

  frontend_runtime_shim::state.setup_result = true;
  TEST_ASSERT_TRUE(sensor.begin());
  TEST_ASSERT_TRUE(sensor.runtime().is_setup_complete());
}

void test_arduino_sensor_reports_motion_only_while_ready(void) {
  ESPectre sensor;
  std::vector<bool> motions;
  sensor.onMotion([&motions](bool motion) { motions.push_back(motion); });
  TEST_ASSERT_TRUE(sensor.begin());
  IRuntimeListener *listener = frontend_runtime_shim::state.last_listener;

  listener->on_motion_state_changed(make_snapshot(false, true));
  TEST_ASSERT_TRUE(motions.empty());
  TEST_ASSERT_FALSE(sensor.motion());

  listener->on_motion_state_changed(make_snapshot(true, true));
  listener->on_motion_state_changed(make_snapshot(true, false));
  TEST_ASSERT_EQUAL(2U, motions.size());
  TEST_ASSERT_TRUE(motions[0]);
  TEST_ASSERT_FALSE(motions[1]);
  TEST_ASSERT_FALSE(sensor.motion());
  TEST_ASSERT_EQUAL_FLOAT(0.75f, sensor.movement());
  TEST_ASSERT_EQUAL_FLOAT(0.5f, sensor.threshold());

  listener->on_motion_state_changed(make_snapshot(true, true));
  TEST_ASSERT_TRUE(sensor.motion());
}

void test_arduino_sensor_forwards_readiness_and_faults(void) {
  ESPectre sensor;
  std::vector<bool> readiness;
  std::string fault;
  sensor.onReady([&readiness](bool ready) { readiness.push_back(ready); });
  sensor.onFault([&fault](const char *message) { fault = message; });
  TEST_ASSERT_TRUE(sensor.begin());
  IRuntimeListener *listener = frontend_runtime_shim::state.last_listener;

  // The controller derives readiness from the backend snapshot in loop().
  frontend_runtime_shim::state.snapshot = make_snapshot(true, false);
  frontend_runtime_shim::state.emit_threshold_on_next_loop = true;
  sensor.loop();
  TEST_ASSERT_TRUE(sensor.ready());
  listener->on_runtime_fault("wifi disconnected");
  TEST_ASSERT_EQUAL_STRING("wifi disconnected", fault.c_str());

  sensor.end();
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
  TEST_ASSERT_FALSE(sensor.ready());
  TEST_ASSERT_EQUAL(2U, readiness.size());
  TEST_ASSERT_TRUE(readiness[0]);
  TEST_ASSERT_FALSE(readiness[1]);
}

void test_arduino_sensor_callbacks_are_optional(void) {
  ESPectre sensor;
  TEST_ASSERT_TRUE(sensor.begin());
  IRuntimeListener *listener = frontend_runtime_shim::state.last_listener;

  frontend_runtime_shim::state.snapshot = make_snapshot(true, false);
  frontend_runtime_shim::state.emit_threshold_on_next_loop = true;
  sensor.loop();
  listener->on_motion_state_changed(make_snapshot(true, true));
  listener->on_runtime_fault("ignored");
  TEST_ASSERT_TRUE(sensor.motion());
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_arduino_sensor_starts_from_sdk_defaults_without_saved_controls);
  RUN_TEST(test_arduino_sensor_begin_starts_the_runtime_once);
  RUN_TEST(test_arduino_sensor_begin_can_retry_after_a_failed_setup);
  RUN_TEST(test_arduino_sensor_reports_motion_only_while_ready);
  RUN_TEST(test_arduino_sensor_forwards_readiness_and_faults);
  RUN_TEST(test_arduino_sensor_callbacks_are_optional);
  return UNITY_END();
}
