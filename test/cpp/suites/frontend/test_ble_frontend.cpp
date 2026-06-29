#include "test_harness.h"

#include <cstring>

#define private public
#define protected public
#include "ble_frontend.h"
#undef protected
#undef private

#include "ble_bindings_mock.h"
#include "frontend_runtime_shim.h"

using namespace esphome::espectre;
using esphome::espectre::ble_bindings_mock::MockBleBindings;

namespace {

RuntimeSnapshot make_ready_snapshot() {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.best_pxx = 0.42f;
  snapshot.gain_locked = true;
  snapshot.detector_name = "mvs";
  return snapshot;
}

float read_float_at(const std::vector<uint8_t> &payload, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, payload.data() + offset, sizeof(float));
  return value;
}

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  ble_bindings_mock::reset();
}

void tearDown(void) {}

void test_ble_frontend_setup_registers_runtime_listener_and_bindings_callbacks(void) {
  frontend_runtime_shim::state.snapshot.threshold = 3.25f;

  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.is_setup_complete());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == &frontend);
  TEST_ASSERT_TRUE(static_cast<bool>(ble_bindings_mock::state.connection_callback));
  TEST_ASSERT_TRUE(static_cast<bool>(ble_bindings_mock::state.control_callback));
  TEST_ASSERT_EQUAL_FLOAT(3.25f, frontend.snapshot().threshold);
}

void test_ble_frontend_setup_fails_without_bindings_or_when_transport_fails(void) {
  BleFrontend without_bindings(nullptr);
  TEST_ASSERT_FALSE(without_bindings.setup());

  ble_bindings_mock::state.setup_result = false;
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_FALSE(frontend.setup());
}

void test_ble_frontend_loop_and_shutdown_forward_to_runtime(void) {
  MockBleBindings bindings;
  {
    BleFrontend frontend(&bindings);
    TEST_ASSERT_TRUE(frontend.setup());
    frontend.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
  TEST_ASSERT_TRUE(ble_bindings_mock::state.shutdown_called);
}

void test_ble_frontend_connection_and_sysinfo_paths(void) {
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  bindings.emit_connection(true);
  TEST_ASSERT_TRUE(frontend.client_connected());
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
  TEST_ASSERT_EQUAL_STRING("proto_version=1", ble_bindings_mock::state.sysinfo_lines.front().c_str());
  TEST_ASSERT_EQUAL_STRING("END", ble_bindings_mock::state.sysinfo_lines.back().c_str());

  bindings.emit_connection(false);
  TEST_ASSERT_FALSE(frontend.client_connected());
}

void test_ble_frontend_control_commands_validate_and_update_runtime(void) {
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  bindings.emit_control("REQ_SYSINFO");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  bindings.emit_control("SET_THRESHOLD:invalid");
  bindings.emit_control("SET_THRESHOLD:42");
  bindings.emit_control("UNKNOWN");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.sysinfo_lines.size()));

  bindings.emit_control("SET_THRESHOLD:4.25");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(4.25f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(frontend.runtime_.config().threshold_mode == ThresholdMode::MANUAL);
}

void test_ble_frontend_telemetry_is_throttled_and_encoded_as_two_floats(void) {
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  frontend.telemetry_interval_ms_ = 150;

  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  bindings.emit_connection(true);
  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(1, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
  const auto &first_payload = ble_bindings_mock::state.telemetry_events[0].payload;
  TEST_ASSERT_EQUAL(sizeof(float) * 2, static_cast<int>(first_payload.size()));
  TEST_ASSERT_EQUAL_FLOAT(2.5f, read_float_at(first_payload, 0));
  TEST_ASSERT_EQUAL_FLOAT(1.5f, read_float_at(first_payload, sizeof(float)));

  frontend.on_live_telemetry(3.0f, 2.0f);
  TEST_ASSERT_EQUAL(1, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  frontend.on_live_telemetry(4.0f, 2.5f);
  TEST_ASSERT_EQUAL(2, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
}

void test_ble_frontend_threshold_and_calibration_callbacks_publish_sysinfo(void) {
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.threshold = 4.5f;
  frontend.on_threshold_changed(snapshot);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, frontend.runtime_.config().segmentation_threshold);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  frontend.on_calibration_started(snapshot);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  frontend.on_calibration_finished(snapshot, false);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
}

void test_ble_frontend_runtime_fault_is_reported_to_bindings(void) {
  MockBleBindings bindings;
  BleFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  frontend.on_runtime_fault("wifi disconnected");
  TEST_ASSERT_EQUAL(1, static_cast<int>(ble_bindings_mock::state.faults.size()));
  TEST_ASSERT_EQUAL_STRING("wifi disconnected", ble_bindings_mock::state.faults[0].c_str());
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_ble_frontend_setup_registers_runtime_listener_and_bindings_callbacks);
  RUN_TEST(test_ble_frontend_setup_fails_without_bindings_or_when_transport_fails);
  RUN_TEST(test_ble_frontend_loop_and_shutdown_forward_to_runtime);
  RUN_TEST(test_ble_frontend_connection_and_sysinfo_paths);
  RUN_TEST(test_ble_frontend_control_commands_validate_and_update_runtime);
  RUN_TEST(test_ble_frontend_telemetry_is_throttled_and_encoded_as_two_floats);
  RUN_TEST(test_ble_frontend_threshold_and_calibration_callbacks_publish_sysinfo);
  RUN_TEST(test_ble_frontend_runtime_fault_is_reported_to_bindings);
  return UNITY_END();
}
