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
#include "stream_esp_idf_runtime.h"
#undef protected
#undef private

#include "nvs.h"
#include "runtime_motion_hits_store.h"

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

}  // namespace

void setUp(void) { nvs_mock_reset(); }
void tearDown(void) {}

void test_runtime_detector_switch_updates_pipeline_threshold_and_calibration(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::CLASSIC;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::ML));
  TEST_ASSERT_EQUAL_STRING("ml", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(ML_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.detector_changes);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);

  runtime.csi_pipeline_.enabled_ = true;
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::CLASSIC));
  TEST_ASSERT_EQUAL_STRING("classic", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(CLASSIC_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::ML));
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  TEST_ASSERT_FALSE(listener.last_calibration_success);

  TEST_ASSERT_TRUE(runtime.set_threshold_runtime(0.75f));
  TEST_ASSERT_EQUAL_FLOAT(0.75f, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
  TEST_ASSERT_EQUAL_FLOAT(ML_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
}

void test_runtime_motion_hits_runtime_updates_pipeline_and_persists(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::CLASSIC;
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

void test_runtime_diagnostics_read_current_wifi_association(void) {
  RuntimeConfig config;
  EspIdfRuntime runtime(config);

  const RuntimeDiagnosticsSnapshot diagnostics = runtime.get_diagnostics();

  TEST_ASSERT_EQUAL_UINT8(6U, diagnostics.wifi_channel);
  TEST_ASSERT_EQUAL_INT8(-55, diagnostics.wifi_rssi_dbm);
}

void test_runtime_channel_change_rearms_csi_and_restarts_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::CLASSIC;
  config.csi_traffic_mode = CsiTrafficMode::DISABLED;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config, CsiTrafficMode::EXTERNAL));
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
}

void test_runtime_channel_change_cold_resets_ml_without_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::ML;
  config.csi_traffic_mode = CsiTrafficMode::DISABLED;
  EspIdfRuntime runtime(config);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval_ms);
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config, CsiTrafficMode::EXTERNAL));
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
}

void test_stream_runtime_channel_change_resets_capture_and_stream_session(void) {
  RuntimeConfig config;
  config.csi_traffic_mode = CsiTrafficMode::DISABLED;
  StreamEspIdfRuntime runtime(config);
  runtime.capture_service_.init();
  TEST_ASSERT_EQUAL(ESP_OK, runtime.capture_service_.enable());
  runtime.wifi_connected_.store(true, std::memory_order_relaxed);
  runtime.state_.store(StreamEspIdfRuntime::WorkflowState::STREAMING, std::memory_order_relaxed);
  runtime.snapshot_.ready_to_publish = true;

  runtime.on_csi_channel_changed_(8U, 10U);

  TEST_ASSERT_FALSE(runtime.capture_service_.is_enabled());
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(StreamEspIdfRuntime::WorkflowState::WAIT_WIFI,
                    runtime.state_.load(std::memory_order_relaxed));
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  UNITY_BEGIN();
  RUN_TEST(test_runtime_detector_switch_updates_pipeline_threshold_and_calibration);
  RUN_TEST(test_runtime_motion_hits_runtime_updates_pipeline_and_persists);
  RUN_TEST(test_runtime_diagnostics_read_current_wifi_association);
  RUN_TEST(test_runtime_channel_change_rearms_csi_and_restarts_calibration);
  RUN_TEST(test_runtime_channel_change_cold_resets_ml_without_calibration);
  RUN_TEST(test_stream_runtime_channel_change_resets_capture_and_stream_session);
  return UNITY_END();
}
