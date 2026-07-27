/*
 * ESPectre - Low-RSSI Integration Test
 *
 * Validates the production Classic startup and replay path on every real
 * low-RSSI pair exposed by dataset metadata.
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
  detector.on_startup_calibration_begin();
  calibrator.begin(static_cast<uint16_t>(calibration_packets),
                   detector.startup_gate_enabled());

  uint16_t packets_since_evaluation = 0U;
  for (int i = 0; i < calibration_packets; i++) {
    detector.process_packet(baseline.packets[static_cast<size_t>(i)].data(),
                            baseline.packet_size, DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE,
                            baseline.rssi_dbm.empty() ? INT8_MIN
                                                      : baseline.rssi_dbm[static_cast<size_t>(i)]);
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
  detector.reset();
  detector.clear_buffer();
  return true;
}

int count_effective_alarms(const std::vector<bool>& raw_motion_states) {
  MotionState effective_state = MotionState::IDLE;
  MotionState pending_state = MotionState::IDLE;
  uint8_t pending_hits = 0U;
  int effective_alarms = 0;

  for (bool raw_motion : raw_motion_states) {
    const MotionState detector_state = raw_motion ? MotionState::MOTION : MotionState::IDLE;
    const MotionState previous_state = effective_state;

    if (detector_state == effective_state) {
      pending_state = effective_state;
      pending_hits = 0U;
    } else {
      if (detector_state != pending_state) {
        pending_state = detector_state;
        pending_hits = 1U;
      } else if (pending_hits < UINT8_MAX) {
        pending_hits++;
      }

      const uint8_t required_hits =
          pending_state == MotionState::MOTION ? RUNTIME_MOTION_ON_HITS_DEFAULT
                                               : RUNTIME_MOTION_OFF_HITS_DEFAULT;
      if (pending_hits >= required_hits) {
        effective_state = pending_state;
        pending_hits = 0U;
      }
    }

    if (effective_state != previous_state && effective_state == MotionState::MOTION) {
      effective_alarms++;
    }
  }

  return effective_alarms;
}

void replay_phase(ClassicDetector& detector, const csi_test_data::CsiData& data,
                  int& evaluations, int& motion_evaluations,
                  std::vector<bool>* raw_motion_states = nullptr) {
  uint16_t packets_since_evaluation = 0U;
  for (int i = 0; i < data.num_packets; i++) {
    detector.process_packet(data.packets[static_cast<size_t>(i)].data(),
                            data.packet_size, DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE,
                            data.rssi_dbm.empty() ? INT8_MIN
                                                  : data.rssi_dbm[static_cast<size_t>(i)]);
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
    const bool raw_motion = detector.get_state() == MotionState::MOTION;
    if (raw_motion_states != nullptr) {
      raw_motion_states->push_back(raw_motion);
    }
    if (raw_motion) {
      motion_evaluations++;
    }
  }
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_classic_handles_loaded_low_rssi_pair(void) {
  const csi_test_data::CsiData baseline = csi_test_data::g_static_presence_data;
  const csi_test_data::CsiData motion = csi_test_data::g_motion_data;
  ClassicDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, CLASSIC_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);

  TEST_ASSERT_TRUE(calibrate(detector, baseline));

  int baseline_evaluations = 0;
  int baseline_motion = 0;
  std::vector<bool> baseline_motion_states;
  replay_phase(detector, baseline, baseline_evaluations, baseline_motion,
               &baseline_motion_states);
  int motion_evaluations = 0;
  int motion_detected = 0;
  replay_phase(detector, motion, motion_evaluations, motion_detected);

  const float fp_rate = baseline_evaluations > 0
                            ? 100.0f * baseline_motion / baseline_evaluations
                            : 100.0f;
  const float recall = motion_evaluations > 0
                           ? 100.0f * motion_detected / motion_evaluations
                           : 0.0f;
  const int effective_alarms = count_effective_alarms(baseline_motion_states);
  printf("Low-RSSI Classic: recall=%.2f%%, fp=%.2f%%, effective_alarms=%d\n",
         recall, fp_rate, effective_alarms);
  // Every weak-link pair clears this recall floor, but the margin on the
  // weakest C3 pair is under a point, so 85 is the level the corpus actually
  // supports today. Raise it to 90 only once the Classic feature work lands,
  // and never lower it to accommodate a regression.
  TEST_ASSERT_TRUE(recall >= 85.0f);
  // Sanity bound, not a false-positive gate. These baselines hold a stationary
  // person, whose breathing and small shifts are real channel motion, so a
  // share of these evaluations is the detector working rather than failing.
  // Zero alarms is asserted in test_empty_rooms, on the only recordings in the
  // corpus with nobody in the room. Corpus maximum is 10.6%.
  TEST_ASSERT_TRUE(fp_rate <= 12.0f);
}

int process(void) {
  int failures = 0;
  const int pair_count = csi_test_data::get_available_low_rssi_pair_count();
  if (pair_count <= 0) {
    printf("ERROR: No real low-RSSI dataset pairs available\n");
    return 1;
  }

  for (int pair_index = 0; pair_index < pair_count; pair_index++) {
    const csi_test_data::ChipType chip = csi_test_data::low_rssi_pair_chip(pair_index);
    printf("\n========================================\n");
    printf("Running low-RSSI Classic with %s dataset pair\n",
           csi_test_data::chip_name(chip));
    printf("Pair: %s\n", csi_test_data::low_rssi_pair_label(pair_index));
    printf("========================================\n");

    if (!csi_test_data::switch_low_rssi_dataset_pair(pair_index)) {
      printf("ERROR: Failed to load %s low-RSSI dataset pair\n",
             csi_test_data::chip_name(chip));
      failures++;
      continue;
    }

    UNITY_BEGIN();
    RUN_TEST(test_classic_handles_loaded_low_rssi_pair);
    failures += UNITY_END();
  }

  return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) { return process(); }
#endif
