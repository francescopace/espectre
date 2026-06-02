#include "test_harness.h"

#define private public
#define protected public
#include "matter_frontend.h"
#undef protected
#undef private

#include "frontend_runtime_shim.h"
#include "matter_bindings_mock.h"
#include "matter_surface.h"

using namespace esphome::espectre;
using esphome::espectre::matter_bindings_mock::MockMatterBindings;

namespace {

RuntimeSnapshot make_ready_snapshot(bool motion) {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = motion ? MotionState::MOTION : MotionState::IDLE;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.best_pxx = 0.42f;
  snapshot.gain_locked = true;
  snapshot.detector_name = "mvs";
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
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == &frontend);
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
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.threshold_events.size());
  TEST_ASSERT_EQUAL_FLOAT(1.5f, matter_bindings_mock::state.threshold_events[0].threshold);
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.periodic_events.size());
  TEST_ASSERT_EQUAL_FLOAT(2.75f, matter_bindings_mock::state.periodic_events[0].state.movement_metric);
  TEST_ASSERT_EQUAL(128, matter_bindings_mock::state.periodic_events[0].state.packets_received);
}

void test_matter_frontend_threshold_and_calibration_callbacks_publish_bindings(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 4);
  TEST_ASSERT_TRUE(frontend.setup());

  RuntimeSnapshot threshold_snapshot = make_ready_snapshot(false);
  threshold_snapshot.threshold = 4.25f;
  frontend.on_threshold_changed(threshold_snapshot);
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.threshold_events.size());
  TEST_ASSERT_EQUAL_FLOAT(4.25f, matter_bindings_mock::state.threshold_events[0].threshold);

  RuntimeSnapshot calibrating = make_ready_snapshot(false);
  calibrating.calibrating = true;
  frontend.on_calibration_started(calibrating);
  TEST_ASSERT_EQUAL(1, matter_bindings_mock::state.calibrating_events.size());
  TEST_ASSERT_TRUE(matter_bindings_mock::state.calibrating_events[0].calibrating);

  RuntimeSnapshot finished = make_ready_snapshot(false);
  finished.calibrating = false;
  frontend.on_calibration_finished(finished, false);
  TEST_ASSERT_EQUAL(2, matter_bindings_mock::state.calibrating_events.size());
  TEST_ASSERT_FALSE(matter_bindings_mock::state.calibrating_events[1].calibrating);
}

void test_matter_frontend_handle_threshold_write_updates_runtime(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 5);
  TEST_ASSERT_TRUE(frontend.setup());

  TEST_ASSERT_TRUE(frontend.handle_threshold_write(6.0f));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(6.0f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(frontend.runtime_config_.threshold_mode == ThresholdMode::MANUAL);
  TEST_ASSERT_FALSE(frontend.handle_threshold_write(11.0f));
}

void test_matter_frontend_handle_recalibrate_respects_capabilities(void) {
  MockMatterBindings bindings;
  MatterFrontend frontend(&bindings, 6);
  TEST_ASSERT_TRUE(frontend.setup());

  TEST_ASSERT_TRUE(frontend.handle_recalibrate_request());
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);

  frontend.runtime_capabilities_.supports_manual_recalibration = false;
  TEST_ASSERT_FALSE(frontend.handle_recalibrate_request());
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
  snapshot.calibrating = true;

  TEST_ASSERT_TRUE(snapshot_to_motion_detected(snapshot));
  MatterPeriodicState periodic = snapshot_to_periodic_state(snapshot, 64);
  TEST_ASSERT_EQUAL_FLOAT(2.75f, periodic.movement_metric);
  TEST_ASSERT_TRUE(periodic.calibrating);
  TEST_ASSERT_EQUAL(64, periodic.packets_received);
  TEST_ASSERT_TRUE(validate_matter_threshold(0.0f));
  TEST_ASSERT_TRUE(validate_matter_threshold(10.0f));
  TEST_ASSERT_FALSE(validate_matter_threshold(10.1f));
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
  RUN_TEST(test_matter_frontend_threshold_and_calibration_callbacks_publish_bindings);
  RUN_TEST(test_matter_frontend_handle_threshold_write_updates_runtime);
  RUN_TEST(test_matter_frontend_handle_recalibrate_respects_capabilities);
  RUN_TEST(test_matter_frontend_runtime_fault_is_reported);
  RUN_TEST(test_matter_surface_mapping_helpers);
  return UNITY_END();
}
