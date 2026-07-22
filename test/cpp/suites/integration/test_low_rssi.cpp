/*
 * ESPectre - Low-RSSI Integration Test
 *
 * Validates the production Classic startup and replay path on the real
 * low-RSSI C3 pair.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include "classic_detector.h"
#include "csi_test_data.h"
#include "runtime_sensing_schema.h"

#include <algorithm>

using namespace espectre;

namespace {

bool calibrate(ClassicDetector& detector, const csi_test_data::CsiData& baseline) {
  StartupThresholdCalibrator calibrator;
  const int calibration_packets = std::min(
      baseline.num_packets, static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
  calibrator.begin(static_cast<uint16_t>(calibration_packets),
                   detector.startup_gate_enabled());

  uint16_t packets_since_evaluation = 0U;
  for (int i = 0; i < calibration_packets; i++) {
    detector.process_packet(baseline.packets[static_cast<size_t>(i)].data(),
                            baseline.packet_size, DEFAULT_SUBCARRIERS, 12U);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    calibrator.observe(detector.is_ready(), detector.get_motion_metric(),
                       detector.get_startup_floor_metric(), packets_since_evaluation);
    packets_since_evaluation = 0U;
    if (calibrator.is_complete()) {
      break;
    }
  }
  if (!calibrator.is_successful()) {
    return false;
  }
  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(calibrator.threshold_metric());
  detector.clear_buffer();
  return true;
}

void replay_phase(ClassicDetector& detector, const csi_test_data::CsiData& data,
                  int& evaluations, int& motion_evaluations) {
  uint16_t packets_since_evaluation = 0U;
  for (int i = 0; i < data.num_packets; i++) {
    detector.process_packet(data.packets[static_cast<size_t>(i)].data(),
                            data.packet_size, DEFAULT_SUBCARRIERS, 12U);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    packets_since_evaluation = 0U;
    if (i < DETECTOR_DEFAULT_WINDOW_SIZE) {
      continue;
    }
    evaluations++;
    if (detector.get_state() == MotionState::MOTION) {
      motion_evaluations++;
    }
  }
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_classic_handles_real_low_rssi_pair(void) {
  const csi_test_data::LowRssiDatasetSelection* pair_paths =
      csi_test_data::real_low_rssi_pair_for_chip(csi_test_data::ChipType::C3);
  TEST_ASSERT_TRUE(pair_paths != nullptr);
  const csi_test_data::CsiData baseline =
      csi_test_data::load_npz(pair_paths->static_presence_path);
  const csi_test_data::CsiData motion =
      csi_test_data::load_npz(pair_paths->motion_path);
  ClassicDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, CLASSIC_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);

  TEST_ASSERT_TRUE(calibrate(detector, baseline));

  int baseline_evaluations = 0;
  int baseline_motion = 0;
  replay_phase(detector, baseline, baseline_evaluations, baseline_motion);
  int motion_evaluations = 0;
  int motion_detected = 0;
  replay_phase(detector, motion, motion_evaluations, motion_detected);

  const float fp_rate = baseline_evaluations > 0
                            ? 100.0f * baseline_motion / baseline_evaluations
                            : 100.0f;
  const float recall = motion_evaluations > 0
                           ? 100.0f * motion_detected / motion_evaluations
                           : 0.0f;
  printf("Low-RSSI Classic: recall=%.2f%%, fp=%.2f%%\n", recall, fp_rate);
  TEST_ASSERT_TRUE(recall >= 85.0f);
  TEST_ASSERT_TRUE(fp_rate <= 5.0f);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_classic_handles_real_low_rssi_pair);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) { return process(); }
#endif
