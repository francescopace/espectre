#include "test_harness.h"

#include <memory>
#include <string>

#define private public
#include "esp_idf_runtime.h"
#undef private

#include "nvs.h"

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
  runtime.csi_pipeline_.init(runtime.detector_.get(), config.publish_interval);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::ML));
  TEST_ASSERT_EQUAL_STRING("ml", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(ML_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.detector_changes);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);

  runtime.csi_pipeline_.enabled_ = true;
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::CLASSIC));
  TEST_ASSERT_EQUAL_STRING("classic", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(SEGMENTATION_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm_runtime(DetectionAlgorithm::ML));
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  TEST_ASSERT_FALSE(listener.last_calibration_success);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  UNITY_BEGIN();
  RUN_TEST(test_runtime_detector_switch_updates_pipeline_threshold_and_calibration);
  return UNITY_END();
}
