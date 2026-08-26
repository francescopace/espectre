/*
 * ESPectre - Runtime Detector Switch Unit Tests
 *
 * Unit tests for Runtime Detector Switch.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <memory>
#include <mutex>
#include <string>

#define private public
#define protected public
#include "esp_idf_runtime.h"
#undef protected
#undef private

#include "nvs.h"
#include "runtime_detector_store.h"
#include "runtime_motion_hits_store.h"
#include "runtime_traffic_mode_store.h"

using namespace espectre;

namespace {

class DetectorListener : public IRuntimeListener {
 public:
  void on_detector_changed(const RuntimeSnapshot &snapshot) override {
    detector_changes++;
    last_detector = snapshot.detector_name;
  }
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override {
    threshold_changes++;
    last_threshold = snapshot.threshold;
  }
  void on_calibration_started(const RuntimeSnapshot &) override { calibration_starts++; }
  void on_calibration_finished(const RuntimeSnapshot &, bool success) override {
    calibration_finishes++;
    last_calibration_success = success;
  }

  int detector_changes{0};
  int threshold_changes{0};
  int calibration_starts{0};
  int calibration_finishes{0};
  std::string last_detector;
  float last_threshold{0.0f};
  bool last_calibration_success{true};
};

bool accept_raw_packet(void *, const RawCsiPacketView &) { return true; }

}  // namespace

void setUp(void) { nvs_mock_reset(); }
void tearDown(void) {}

void test_runtime_detector_switch_updates_pipeline_threshold_and_calibration(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL_STRING("high_accuracy", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.detector_changes);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);

  runtime.csi_pipeline_.enabled_ = true;
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::LIGHTWEIGHT));
  TEST_ASSERT_EQUAL_STRING("lightweight", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(LIGHTWEIGHT_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  TEST_ASSERT_FALSE(listener.last_calibration_success);

  TEST_ASSERT_TRUE(runtime.set_threshold_runtime(0.75f));
  TEST_ASSERT_EQUAL_FLOAT(0.75f, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
}

void test_runtime_detector_configuration_preserves_the_requested_threshold(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  config.segmentation_threshold = 0.73f;
  EspIdfRuntime runtime(config);

  TEST_ASSERT_TRUE(runtime.configure_detector_());
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.config_.segmentation_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.detector_->get_threshold());
}

void test_runtime_traffic_updates_roll_back_when_persistence_fails(void) {
  RuntimeConfig config;
  config.csi_traffic_mode = CsiTrafficMode::INTERNAL;
  EspIdfRuntime runtime(config);
  nvs_mock_set_open_result(ESP_FAIL);

  TEST_ASSERT_FALSE(runtime.set_csi_traffic_mode_runtime(CsiTrafficMode::EXTERNAL));
  TEST_ASSERT_TRUE(runtime.config_.csi_traffic_mode == CsiTrafficMode::INTERNAL);
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode_runtime(RuntimeTrafficMode::DNS));
  TEST_ASSERT_TRUE(runtime.config_.traffic_generator_mode == RuntimeTrafficMode::PING);
}

void test_runtime_detector_adaptation_emits_threshold_changed(void) {
  RuntimeConfig config;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  runtime.snapshot_.threshold = 0.80f;
  runtime.config_.segmentation_threshold = 0.80f;

  runtime.notify_threshold_if_changed_(0.80f);
  TEST_ASSERT_EQUAL(0, listener.threshold_changes);

  runtime.notify_threshold_if_changed_(0.42f);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, listener.last_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, runtime.config_.segmentation_threshold);

  runtime.notify_threshold_if_changed_(0.42f);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);
}

void test_runtime_motion_hits_runtime_updates_pipeline_and_persists(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  EspIdfRuntime runtime(config);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);

  TEST_ASSERT_TRUE(runtime.set_motion_hits_runtime(8U, 6U));
  TEST_ASSERT_EQUAL_UINT8(8U, runtime.csi_pipeline_.motion_on_hits_);
  TEST_ASSERT_EQUAL_UINT8(6U, runtime.csi_pipeline_.motion_off_hits_);

  uint8_t saved_motion_on_hits = 0U;
  uint8_t saved_motion_off_hits = 0U;
  bool has_saved_value = false;
  TEST_ASSERT_EQUAL(ESP_OK,
                    load_runtime_motion_hits(&saved_motion_on_hits, &saved_motion_off_hits, &has_saved_value));
  TEST_ASSERT_TRUE(has_saved_value);
  TEST_ASSERT_EQUAL_UINT8(8U, saved_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(6U, saved_motion_off_hits);
}

void test_runtime_setup_loads_all_persisted_runtime_controls(void) {
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_motion_hits(8U, 6U));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_csi_traffic_mode(CsiTrafficMode::EXTERNAL));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_traffic_generator_mode(RuntimeTrafficMode::DNS));

  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.motion_on_hits = 4U;
  config.motion_off_hits = 3U;
  config.csi_traffic_mode = CsiTrafficMode::INTERNAL;
  config.traffic_generator_mode = RuntimeTrafficMode::PING;
  EspIdfRuntime runtime(config);

  TEST_ASSERT_TRUE(runtime.setup());
  const RuntimeConfig &effective = runtime.effective_config();
  TEST_ASSERT_TRUE(effective.detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY);
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, effective.segmentation_threshold);
  TEST_ASSERT_EQUAL_UINT8(8U, effective.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(6U, effective.motion_off_hits);
  TEST_ASSERT_TRUE(effective.csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_TRUE(effective.traffic_generator_mode == RuntimeTrafficMode::DNS);
  TEST_ASSERT_TRUE(runtime.csi_traffic_service_.mode_ == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_TRUE(runtime.csi_traffic_service_.traffic_generator_.mode_ == TrafficGeneratorMode::DNS);
  runtime.shutdown();
}

void test_runtime_diagnostics_read_current_wifi_association(void) {
  RuntimeConfig config;
  EspIdfRuntime runtime(config);

  const RuntimeDiagnosticsSnapshot diagnostics = runtime.get_diagnostics();

  TEST_ASSERT_EQUAL_UINT8(6U, diagnostics.wifi_channel);
  TEST_ASSERT_EQUAL_INT8(-55, diagnostics.wifi_rssi_dbm);
}

void test_runtime_channel_change_rearms_csi_and_restarts_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());

  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  runtime.on_csi_channel_changed_(8U, 10U);

  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(MotionState::IDLE, runtime.get_snapshot().motion_state);
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);
  runtime.csi_traffic_service_.stop();
}

void test_runtime_services_armed_preserves_wifi_ip_and_restarts_capture(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());
  runtime.setup_complete_ = true;
  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  runtime.snapshot_.ready_to_publish = true;

  runtime.set_services_armed(false);
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.wifi_ready_);
  TEST_ASSERT_EQUAL(0x0101A8C0U, runtime.wifi_ip_info_.ip.addr);
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);

  runtime.set_services_armed(true);
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.wifi_ready_);
  TEST_ASSERT_EQUAL(0x0101A8C0U, runtime.wifi_ip_info_.ip.addr);
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);

  runtime.on_wifi_disconnected_();
  TEST_ASSERT_FALSE(runtime.wifi_ready_);
  TEST_ASSERT_EQUAL(0U, runtime.wifi_ip_info_.ip.addr);
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  runtime.csi_traffic_service_.stop();
}

void test_runtime_raw_collection_restores_armed_and_disarmed_sensing(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  runtime.capabilities_.supports_raw_csi = true;
  runtime.setup_complete_ = true;
  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());
  runtime.snapshot_.calibrating = true;
  runtime.snapshot_.ready_to_publish = true;

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, runtime.operation_state());
  TEST_ASSERT_EQUAL(CsiTrafficMode::EXTERNAL, runtime.csi_traffic_service_.mode_);
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.snapshot_.calibrating);
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);
  TEST_ASSERT_FALSE(runtime.set_threshold_runtime(0.5f));

  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_EQUAL(CsiTrafficMode::EXTERNAL, runtime.csi_traffic_service_.mode_);
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.snapshot_.calibrating);
  TEST_ASSERT_TRUE(runtime.snapshot_.ready_to_publish);

  runtime.set_services_armed(false);
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);
  runtime.csi_traffic_service_.stop();
}

void test_runtime_raw_collection_terminates_on_wifi_loss_and_channel_change(void) {
  RuntimeConfig config;
  config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  EspIdfRuntime runtime(config);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  runtime.capabilities_.supports_raw_csi = true;
  runtime.setup_complete_ = true;
  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  runtime.on_csi_channel_changed_(6U, 11U);
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  runtime.on_wifi_disconnected_();
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_FALSE(runtime.wifi_ready_);
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  runtime.csi_traffic_service_.stop();
}

void test_runtime_channel_change_cold_resets_ml_without_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  EspIdfRuntime runtime(config);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());

  int8_t csi_data[HT20_CSI_LEN] = {};
  runtime.detector_->process_packet(csi_data, sizeof(csi_data), DEFAULT_SUBCARRIERS,
                                    HT20_SELECTED_BAND_SIZE, -50);
  TEST_ASSERT_TRUE(runtime.detector_->get_buffer_count() > 0U);

  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  runtime.on_csi_channel_changed_(8U, 10U);

  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(0U, runtime.detector_->get_buffer_count());
  runtime.csi_traffic_service_.stop();
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  UNITY_BEGIN();
  RUN_TEST(test_runtime_detector_switch_updates_pipeline_threshold_and_calibration);
  RUN_TEST(test_runtime_detector_configuration_preserves_the_requested_threshold);
  RUN_TEST(test_runtime_traffic_updates_roll_back_when_persistence_fails);
  RUN_TEST(test_runtime_detector_adaptation_emits_threshold_changed);
  RUN_TEST(test_runtime_motion_hits_runtime_updates_pipeline_and_persists);
  RUN_TEST(test_runtime_setup_loads_all_persisted_runtime_controls);
  RUN_TEST(test_runtime_diagnostics_read_current_wifi_association);
  RUN_TEST(test_runtime_channel_change_rearms_csi_and_restarts_calibration);
  RUN_TEST(test_runtime_services_armed_preserves_wifi_ip_and_restarts_capture);
  RUN_TEST(test_runtime_raw_collection_restores_armed_and_disarmed_sensing);
  RUN_TEST(test_runtime_raw_collection_terminates_on_wifi_loss_and_channel_change);
  RUN_TEST(test_runtime_channel_change_cold_resets_ml_without_calibration);
  return UNITY_END();
}
