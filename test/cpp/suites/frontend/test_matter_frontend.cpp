/*
 * ESPectre - Matter Frontend Unit Tests
 *
 * Unit tests for Matter Frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#define private public
#define protected public
#include "matter_frontend.h"
#undef protected
#undef private

#include "frontend_runtime_shim.h"
#include "matter_bindings_mock.h"
#include "matter_surface.h"

using namespace espectre;
using espectre::matter_bindings_mock::MockMatterBindings;

namespace {

RuntimeSnapshot make_ready_snapshot(bool motion) {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = motion ? MotionState::MOTION : MotionState::IDLE;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.startup_threshold = 0.42f;
  snapshot.detector_name = "lightweight";
  return snapshot;
}

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  matter_bindings_mock::reset();
}

void tearDown(void) {}

void test_matter_frontend_setup_registers_runtime_listener(void) {
  frontend_runtime_shim::state.snapshot.threshold = 3.25f;

  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 7);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.is_setup_complete());
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_listener);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener != &frontend);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, frontend.snapshot().threshold);
}

void test_matter_frontend_setup_fails_without_bindings(void) {
  MatterFrontend frontend(nullptr, 1);
  TEST_ASSERT_FALSE(frontend.setup());
}

void test_matter_frontend_setup_fails_when_runtime_setup_fails(void) {
  frontend_runtime_shim::state.setup_result = false;

  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 1);
  TEST_ASSERT_FALSE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.is_setup_complete());
}

void test_matter_frontend_loop_and_shutdown_forward_to_runtime(void) {
  MockMatterBindings bindings;
  {
    MatterFrontend frontend(&bindings, 2);
    TEST_ASSERT_TRUE(frontend.setup());
    frontend.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
}

void test_matter_frontend_motion_and_periodic_callbacks_publish_bindings(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 3);
  TEST_ASSERT_TRUE(frontend.setup());

  RuntimeSnapshot not_ready{};
  not_ready.ready_to_publish = false;
  frontend.on_motion_state_changed(not_ready);
  TEST_ASSERT_EQUAL(0, matter_bindings_mock::state.motion_events.size());

  RuntimeSnapshot ready_motion = make_ready_snapshot(true);
  frontend.on_motion_state_changed(ready_motion);
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.motion_events.size());
  TEST_ASSERT_TRUE(matter_bindings_mock::state.motion_events[0].motion_detected);

  RuntimeSnapshot ready_idle = make_ready_snapshot(false);
  frontend.on_periodic_update(ready_idle, 128);
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.motion_events.size());
}

void test_matter_frontend_threshold_and_calibration_callbacks_update_runtime_snapshot(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 4);
  TEST_ASSERT_TRUE(frontend.setup());

  RuntimeSnapshot threshold_snapshot = make_ready_snapshot(false);
  threshold_snapshot.threshold = 4.25f;
  frontend_runtime_shim::state.last_listener->on_threshold_changed(threshold_snapshot);
  TEST_ASSERT_EQUAL_FLOAT(4.25f, frontend.snapshot().threshold);

  RuntimeSnapshot calibrating = make_ready_snapshot(false);
  calibrating.calibrating = true;
  frontend_runtime_shim::state.last_listener->on_calibration_started(calibrating);
  TEST_ASSERT_TRUE(frontend.snapshot().calibrating);

  RuntimeSnapshot finished = make_ready_snapshot(false);
  finished.calibrating = false;
  frontend_runtime_shim::state.last_listener->on_calibration_finished(finished, false);
  TEST_ASSERT_FALSE(frontend.snapshot().calibrating);
}

void test_matter_frontend_runtime_fault_is_reported(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 8);
  TEST_ASSERT_TRUE(frontend.setup());

  frontend.on_runtime_fault("wifi disconnected");
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.faults.size());
  TEST_ASSERT_EQUAL_STRING("wifi disconnected", matter_bindings_mock::state.faults[0].c_str());
}

void test_matter_surface_mapping_helpers(void) {
  RuntimeSnapshot snapshot = make_ready_snapshot(true);

  TEST_ASSERT_TRUE(snapshot_to_motion_detected(snapshot));
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_matter_frontend_setup_registers_runtime_listener);
  RUN_TEST(test_matter_frontend_setup_fails_without_bindings);
  RUN_TEST(test_matter_frontend_setup_fails_when_runtime_setup_fails);
  RUN_TEST(test_matter_frontend_loop_and_shutdown_forward_to_runtime);
  RUN_TEST(test_matter_frontend_motion_and_periodic_callbacks_publish_bindings);
  RUN_TEST(test_matter_frontend_threshold_and_calibration_callbacks_update_runtime_snapshot);
  RUN_TEST(test_matter_frontend_runtime_fault_is_reported);
  RUN_TEST(test_matter_surface_mapping_helpers);
  return UNITY_END();
}
