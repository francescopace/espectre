#include "test_harness.h"

#include <memory>
#define private public
#define protected public
#include "calibrate_switch.h"
#include "espectre.h"
#include "threshold_number.h"
#undef protected
#undef private

#include "esphome/core/hal.h"
#include "frontend_runtime_shim.h"

using namespace esphome::espectre_component;

namespace {

class ESpectreComponentProbe : public ESpectreComponent {
 public:
  using ESpectreComponent::on_calibration_finished;
  using ESpectreComponent::on_calibration_started;
  using ESpectreComponent::on_motion_state_changed;
  using ESpectreComponent::on_periodic_update;
  using ESpectreComponent::on_runtime_fault;
  using ESpectreComponent::on_threshold_changed;
};

class ThresholdNumberProbe : public ESpectreThresholdNumber {
 public:
  using ESpectreThresholdNumber::control;
};

class CalibrateSwitchProbe : public ESpectreCalibrateSwitch {
 public:
  using ESpectreCalibrateSwitch::write_state;
};

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  esphome::reset_mock_millis();
}

void tearDown(void) {}

void test_espectre_component_setup_uses_mock_runtime_snapshot(void) {
  frontend_runtime_shim::state.snapshot.threshold = 4.5f;

  ESpectreComponentProbe component;
  component.setup();

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == &component);
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_instance);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, component.get_threshold());
}

void test_espectre_component_setup_marks_failed_when_runtime_setup_fails(void) {
  frontend_runtime_shim::state.setup_result = false;

  ESpectreComponentProbe component;
  component.setup();

  TEST_ASSERT_TRUE(component.is_failed());
}

void test_espectre_component_loop_and_destructor_forward_to_runtime(void) {
  {
    ESpectreComponentProbe component;
    component.setup();
    component.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
}

void test_espectre_component_configuration_setters_update_runtime_config(void) {
  ESpectreComponentProbe component;
  esphome::sensor::Sensor movement_sensor;
  esphome::binary_sensor::BinarySensor binary_sensor;
  ThresholdNumberProbe threshold_number;
  CalibrateSwitchProbe calibrate_switch;

  component.set_segmentation_threshold(2.5f);
  component.set_threshold_mode("min");
  TEST_ASSERT_TRUE(component.runtime_.config().threshold_mode == ThresholdMode::MIN);
  component.set_threshold_mode("auto");
  TEST_ASSERT_TRUE(component.runtime_.config().threshold_mode == ThresholdMode::AUTO);
  component.set_segmentation_window_size(64);
  component.set_traffic_generator_rate(0);
  component.set_traffic_generator_mode("dns");
  TEST_ASSERT_TRUE(component.runtime_.config().traffic_generator_mode == RuntimeTrafficMode::DNS);
  component.set_traffic_generator_mode("ping");
  TEST_ASSERT_TRUE(component.runtime_.config().traffic_generator_mode == RuntimeTrafficMode::PING);
  component.set_detection_algorithm("ml");
  TEST_ASSERT_TRUE(component.runtime_.config().detection_algorithm == DetectionAlgorithm::ML);
  component.set_detection_algorithm("classic");
  TEST_ASSERT_TRUE(component.runtime_.config().detection_algorithm == DetectionAlgorithm::CLASSIC);
  component.set_publish_interval(200);
  component.set_evaluation_interval(50);
  component.set_motion_on_hits(4);
  component.set_motion_off_hits(5);
  component.set_lowpass_enabled(true);
  component.set_lowpass_cutoff(8.5f);
  component.set_hampel_enabled(false);
  component.set_hampel_window(9);
  component.set_hampel_threshold(4.5f);
  component.set_movement_sensor(&movement_sensor);
  component.set_motion_binary_sensor(&binary_sensor);
  component.set_threshold_number(&threshold_number);
  component.set_calibrate_switch(&calibrate_switch);

  component.set_threshold_mode("min");
  component.set_traffic_generator_mode("dns");
  component.set_detection_algorithm("ml");

  TEST_ASSERT_TRUE(component.runtime_.config().threshold_mode == ThresholdMode::MIN);
  TEST_ASSERT_EQUAL_FLOAT(2.5f, component.runtime_.config().segmentation_threshold);
  TEST_ASSERT_EQUAL(64, component.runtime_.config().segmentation_window_size);
  TEST_ASSERT_EQUAL(0, component.runtime_.config().traffic_generator_rate);
  TEST_ASSERT_TRUE(component.runtime_.config().traffic_generator_mode == RuntimeTrafficMode::DNS);
  TEST_ASSERT_TRUE(component.runtime_.config().detection_algorithm == DetectionAlgorithm::ML);
  TEST_ASSERT_EQUAL(200, component.runtime_.config().publish_interval);
  TEST_ASSERT_EQUAL(50, component.runtime_.config().evaluation_interval);
  TEST_ASSERT_EQUAL(4, component.runtime_.config().motion_on_hits);
  TEST_ASSERT_EQUAL(5, component.runtime_.config().motion_off_hits);
  TEST_ASSERT_TRUE(component.runtime_.config().lowpass_enabled);
  TEST_ASSERT_EQUAL_FLOAT(8.5f, component.runtime_.config().lowpass_cutoff);
  TEST_ASSERT_FALSE(component.runtime_.config().hampel_enabled);
  TEST_ASSERT_EQUAL(9, component.runtime_.config().hampel_window);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, component.runtime_.config().hampel_threshold);
  TEST_ASSERT_TRUE(component.sensor_publisher_.has_movement_sensor());
  TEST_ASSERT_TRUE(component.sensor_publisher_.has_motion_binary_sensor());
  TEST_ASSERT_EQUAL(0.0f, component.get_setup_priority() - esphome::setup_priority::AFTER_WIFI);
}

void test_threshold_number_behaviors_cover_parent_and_no_parent_paths(void) {
  ESpectreComponentProbe component;
  ThresholdNumberProbe number;

  number.setup();
  number.dump_config();
  number.control(1.25f);
  number.republish_state();
  TEST_ASSERT_FALSE(number.has_state());

  number.set_parent(&component);
  number.control(6.25f);
  TEST_ASSERT_EQUAL_FLOAT(6.25f, component.get_threshold());
  component.set_threshold_runtime(3.75f);
  number.republish_state();

  TEST_ASSERT_EQUAL_FLOAT(3.75f, component.get_threshold());
  TEST_ASSERT_TRUE(number.has_state());
  TEST_ASSERT_EQUAL_FLOAT(3.75f, number.get_state());
}

void test_calibrate_switch_behaviors_cover_all_user_paths(void) {
  ESpectreComponentProbe component;
  frontend_runtime_shim::state.capabilities.supports_manual_recalibration = true;
  component.setup();

  CalibrateSwitchProbe calibrate_switch;
  calibrate_switch.setup();
  calibrate_switch.dump_config();
  calibrate_switch.write_state(true);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.trigger_recalibration_calls);

  calibrate_switch.set_parent(&component);
  calibrate_switch.set_calibrating(true);
  TEST_ASSERT_TRUE(calibrate_switch.state);

  frontend_runtime_shim::state.calibrating = true;
  calibrate_switch.write_state(false);
  TEST_ASSERT_TRUE(calibrate_switch.state);

  frontend_runtime_shim::state.calibrating = false;
  calibrate_switch.write_state(true);
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);

  frontend_runtime_shim::state.calibrating = false;
  calibrate_switch.write_state(false);
  TEST_ASSERT_FALSE(calibrate_switch.state);
}

void test_motion_threshold_and_calibration_callbacks_publish_expected_state(void) {
  ESpectreComponentProbe component;
  esphome::sensor::Sensor movement_sensor;
  esphome::binary_sensor::BinarySensor binary_sensor;
  ThresholdNumberProbe threshold_number;
  CalibrateSwitchProbe calibrate_switch;

  threshold_number.set_parent(&component);
  calibrate_switch.set_parent(&component);
  component.set_threshold_number(&threshold_number);
  component.set_calibrate_switch(&calibrate_switch);
  component.set_movement_sensor(&movement_sensor);
  component.set_motion_binary_sensor(&binary_sensor);
  component.set_threshold_runtime(5.5f);

  component.threshold_republished_ = true;

  RuntimeSnapshot idle_snapshot{};
  idle_snapshot.ready_to_publish = false;
  idle_snapshot.motion_state = MotionState::IDLE;
  component.on_motion_state_changed(idle_snapshot);
  TEST_ASSERT_FALSE(component.threshold_republished_);
  TEST_ASSERT_FALSE(binary_sensor.has_state());

  RuntimeSnapshot motion_snapshot{};
  motion_snapshot.ready_to_publish = true;
  motion_snapshot.motion_state = MotionState::MOTION;
  motion_snapshot.threshold = 5.5f;
  motion_snapshot.movement_metric = 7.25f;
  component.on_motion_state_changed(motion_snapshot);
  TEST_ASSERT_TRUE(binary_sensor.get_state());

  component.on_periodic_update(idle_snapshot, 42);
  TEST_ASSERT_EQUAL(0, movement_sensor.get_publish_count());

  component.on_periodic_update(motion_snapshot, 42);
  component.on_periodic_update(motion_snapshot, 42);
  TEST_ASSERT_TRUE(threshold_number.has_state());
  TEST_ASSERT_EQUAL_FLOAT(5.5f, threshold_number.get_state());
  TEST_ASSERT_EQUAL(1, threshold_number.get_publish_count());
  TEST_ASSERT_EQUAL(2, movement_sensor.get_publish_count());

  RuntimeSnapshot threshold_snapshot = motion_snapshot;
  threshold_snapshot.threshold = 6.75f;
  component.on_threshold_changed(threshold_snapshot);
  TEST_ASSERT_EQUAL_FLOAT(6.75f, component.runtime_.config().segmentation_threshold);
  TEST_ASSERT_EQUAL_FLOAT(6.75f, threshold_number.get_state());

  component.on_calibration_started(motion_snapshot);
  TEST_ASSERT_TRUE(calibrate_switch.state);
  component.sensor_publisher_.log_status("frontend", motion_snapshot, 25);
  TEST_ASSERT_TRUE(component.sensor_publisher_.has_motion_binary_sensor());
  TEST_ASSERT_TRUE(component.sensor_publisher_.has_movement_sensor());
  component.on_calibration_finished(motion_snapshot, false);
  TEST_ASSERT_FALSE(calibrate_switch.state);
}

void test_runtime_fault_callback_handles_null_and_message_paths(void) {
  ESpectreComponentProbe component;
  component.setup();

  component.on_runtime_fault(nullptr);
  component.on_runtime_fault("fault");

  TEST_ASSERT_TRUE(true);
}

void test_dump_config_covers_configuration_branches(void) {
  ESpectreComponentProbe component;
  esphome::sensor::Sensor movement_sensor;
  esphome::binary_sensor::BinarySensor binary_sensor;

  component.set_movement_sensor(&movement_sensor);
  component.set_motion_binary_sensor(&binary_sensor);
  RuntimeSnapshot snapshot{};
  snapshot.detector_name = "ml";
  snapshot.threshold = 4.2f;
  snapshot.startup_threshold = 0.42f;
  snapshot.ready_to_publish = true;
  snapshot.subcarrier_source = RuntimeSubcarrierSource::FIXED_DEFAULT;
  component.runtime_.record_snapshot(snapshot);
  component.runtime_.config().threshold_mode = ThresholdMode::MANUAL;
  component.runtime_.config().traffic_generator_rate = 25;
  component.runtime_.config().traffic_generator_mode = RuntimeTrafficMode::DNS;
  component.runtime_.config().lowpass_enabled = true;
  component.runtime_.config().lowpass_cutoff = 7.5f;
  component.runtime_.config().hampel_enabled = true;
  component.runtime_.config().hampel_window = 9;
  component.runtime_.config().hampel_threshold = 4.0f;
  component.dump_config();

  component.runtime_.config().threshold_mode = ThresholdMode::AUTO;
  component.runtime_.config().traffic_generator_rate = 0;
  component.runtime_.config().lowpass_enabled = false;
  component.runtime_.config().hampel_enabled = false;
  snapshot.subcarrier_source = RuntimeSubcarrierSource::FIXED_DEFAULT;
  component.runtime_.record_snapshot(snapshot);
  component.dump_config();

  TEST_ASSERT_TRUE(true);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_espectre_component_setup_uses_mock_runtime_snapshot);
  RUN_TEST(test_espectre_component_setup_marks_failed_when_runtime_setup_fails);
  RUN_TEST(test_espectre_component_loop_and_destructor_forward_to_runtime);
  RUN_TEST(test_espectre_component_configuration_setters_update_runtime_config);
  RUN_TEST(test_threshold_number_behaviors_cover_parent_and_no_parent_paths);
  RUN_TEST(test_calibrate_switch_behaviors_cover_all_user_paths);
  RUN_TEST(test_motion_threshold_and_calibration_callbacks_publish_expected_state);
  RUN_TEST(test_runtime_fault_callback_handles_null_and_message_paths);
  RUN_TEST(test_dump_config_covers_configuration_branches);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
