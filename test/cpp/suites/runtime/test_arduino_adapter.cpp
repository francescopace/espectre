/*
 * ESPectre - Arduino Adapter Unit Tests
 *
 * Unit tests for the `ESPectre` sensor adapter.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include "esp32-hal-log.h"
#include "esp_timer.h"

#include <string>
#include <vector>

#include "core/espectre_log.h"
#include "runtime/esp_idf/espectre_sensor.h"
#include "runtime/esp_idf/runtime_sensing_kconfig.h"
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

void make_ready(ESPectre &sensor) {
  // The controller derives readiness from the backend snapshot in loop().
  frontend_runtime_shim::state.snapshot = make_snapshot(true, false);
  frontend_runtime_shim::state.emit_threshold_on_next_loop = true;
  sensor.loop();
}

int sketch_sink_writes = 0;

bool sketch_sink_enabled(void *, LogLevel, const char *) { return true; }

void sketch_sink_write(void *, LogLevel, const char *, int, const char *, va_list) { ++sketch_sink_writes; }

}  // namespace

void setUp(void) {
  esp_timer_mock::reset();
  frontend_runtime_shim::reset();
  arduino_log_mock::reset();
  clear_log_sink();
  sketch_sink_writes = 0;
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

void test_arduino_sensor_reports_idle_when_readiness_drops_during_motion(void) {
  ESPectre sensor;
  std::vector<std::string> events;
  sensor.onMotion([&events](bool motion) { events.push_back(motion ? "motion" : "idle"); });
  sensor.onReady([&events](bool ready) { events.push_back(ready ? "ready" : "not ready"); });
  TEST_ASSERT_TRUE(sensor.begin());
  IRuntimeListener *listener = frontend_runtime_shim::state.last_listener;
  make_ready(sensor);
  listener->on_motion_state_changed(make_snapshot(true, true));

  // A capture stop resets motion while sensing is no longer ready.
  listener->on_motion_state_changed(make_snapshot(false, false));
  TEST_ASSERT_EQUAL(3U, events.size());
  TEST_ASSERT_EQUAL_STRING("idle", events[2].c_str());
  TEST_ASSERT_FALSE(sensor.motion());

  // Stopping the sensor during motion also reports idle before readiness.
  listener->on_motion_state_changed(make_snapshot(true, true));
  sensor.end();
  const std::vector<std::string> expected = {"ready", "motion", "idle", "motion", "idle", "not ready"};
  TEST_ASSERT_EQUAL(expected.size(), events.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
  }
}

void test_arduino_sensor_sends_logs_to_the_core_without_a_sink(void) {
  ESPectre sensor;
  TEST_ASSERT_TRUE(sensor.begin());
  TEST_ASSERT_TRUE(log_sink_registered());
  arduino_log_mock::reset();

  ESPECTRE_LOGI("espectre.test", "core %d", 1);
  ESPECTRE_LOGD("espectre.test", "below the core level");
  TEST_ASSERT_TRUE(arduino_log_mock::output.find("core 1") != std::string::npos);
  TEST_ASSERT_TRUE(arduino_log_mock::output.find("below the core level") == std::string::npos);
}

void test_arduino_sensor_keeps_a_sink_registered_by_the_sketch(void) {
  TEST_ASSERT_TRUE(set_log_sink({nullptr, sketch_sink_enabled, sketch_sink_write}));
  ESPectre sensor;
  TEST_ASSERT_TRUE(sensor.begin());
  arduino_log_mock::reset();
  sketch_sink_writes = 0;

  ESPECTRE_LOGI("espectre.test", "sketch");
  TEST_ASSERT_EQUAL(1, sketch_sink_writes);
  TEST_ASSERT_TRUE(arduino_log_mock::output.empty());
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
  RUN_TEST(test_arduino_sensor_reports_idle_when_readiness_drops_during_motion);
  RUN_TEST(test_arduino_sensor_sends_logs_to_the_core_without_a_sink);
  RUN_TEST(test_arduino_sensor_keeps_a_sink_registered_by_the_sketch);
  RUN_TEST(test_arduino_sensor_forwards_readiness_and_faults);
  RUN_TEST(test_arduino_sensor_callbacks_are_optional);
  return UNITY_END();
}
