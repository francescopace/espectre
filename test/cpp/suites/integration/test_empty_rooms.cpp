/*
 * ESPectre - Empty-Room Integration Test
 *
 * Validates that the production Lightweight path stays inside the empty-room
 * alarm budget on recordings made with nobody in the room.
 *
 * Empty rooms are the corpus ground truth for "nothing is moving". The
 * static-presence baselines cannot serve that role: a stationary person still
 * breathes and shifts, and the detector sees it, so a share of their
 * evaluations is the detector working rather than failing. High Accuracy still
 * requires zero empty-room alarms. Lightweight may raise at most one effective
 * alarm per recording. See the host-side validation-gates ADR.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include "dataset_test_cli.h"

#include "high_accuracy_detector.h"
#include "lightweight_detector.h"
#include "csi_replay_metrics.h"
#include "csi_test_data.h"
#include "runtime_sensing_schema.h"

#include <algorithm>

using namespace espectre;

namespace {

constexpr int kLightweightEmptyMaxEffectiveAlarms = 1;
constexpr float kLightweightEmptyMaxFpRate = 6.0f;

espectre::test::replay::ReplayPacketMetadata empty_metadata() {
  espectre::test::replay::ReplayPacketMetadata metadata;
  metadata.stream_seq_num = csi_test_data::g_empty_data.stream_seq_num.empty()
                                ? nullptr
                                : csi_test_data::g_empty_data.stream_seq_num.data();
  metadata.device_ticks_us = csi_test_data::g_empty_data.device_ticks_us.empty()
                                 ? nullptr
                                 : csi_test_data::g_empty_data.device_ticks_us.data();
  metadata.wifi_rx_ts_us = csi_test_data::g_empty_data.wifi_rx_ts_us.empty()
                               ? nullptr
                               : csi_test_data::g_empty_data.wifi_rx_ts_us.data();
  metadata.csi_target_pps = csi_test_data::g_empty_data.csi_target_pps;
  return metadata;
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_classic_empty_room_stays_within_alarm_budget(void) {
  const csi_test_data::CsiData& empty = csi_test_data::g_empty_data;
  const int8_t* const* packets = csi_test_data::g_empty_ptrs.data();
  const int8_t* rssi =
      empty.rssi_dbm.empty() ? nullptr : empty.rssi_dbm.data();
  const espectre::test::replay::ReplayPacketMetadata metadata = empty_metadata();

  const uint16_t window_size = espectre::test::replay::detector_window_packets(
      metadata, empty.num_packets);
  LightweightDetector detector(window_size, LIGHTWEIGHT_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);

  const int calibration_packets = std::min(
      empty.num_packets,
      static_cast<int>(espectre::test::replay::calibration_packet_count(
          metadata, empty.num_packets)));
  float adaptive_threshold = 0.0f;
  const bool calibrated = espectre::test::replay::calibrate_lightweight_detector(
      detector, calibration_packets, packets, empty.num_packets, rssi, metadata,
      empty.packet_size, DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE,
      adaptive_threshold);
  TEST_ASSERT_TRUE(calibrated);

  // Replaying with no motion stream leaves the recall half of the metrics
  // empty, which is exactly what an idle-only recording can report.
  const espectre::test::replay::ReplayMetrics metrics =
      espectre::test::replay::evaluate_detector(
          detector, packets, empty.num_packets, rssi, metadata,
          nullptr, 0, nullptr, espectre::test::replay::ReplayPacketMetadata{},
          empty.packet_size, DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

  const float fp_rate =
      metrics.static_presence_eval_count > 0
          ? 100.0f * metrics.fp / metrics.static_presence_eval_count
          : 0.0f;
  printf("Empty room Lightweight: fp=%.2f%%, effective_alarms=%d, evaluations=%d\n",
         fp_rate, metrics.effective_alarms, metrics.static_presence_eval_count);

  TEST_ASSERT_TRUE(metrics.static_presence_eval_count > 0);
  // Occupancy 70% can admit a single four-hit debounce burst. Two alarms on
  // one short empty file remain a defect. High Accuracy stays at zero alarms.
  TEST_ASSERT_TRUE(metrics.effective_alarms <= kLightweightEmptyMaxEffectiveAlarms);
  // Secondary regression guard on the raw per-evaluation rate. The corpus
  // maximum is 5.14%, so this bounds drift without tracking noise.
  TEST_ASSERT_TRUE(fp_rate < kLightweightEmptyMaxFpRate);
}

void test_ml_empty_room_has_no_effective_alarms(void) {
  const csi_test_data::CsiData& empty = csi_test_data::g_empty_data;
  const int8_t* const* packets = csi_test_data::g_empty_ptrs.data();
  const int8_t* rssi = empty.rssi_dbm.empty() ? nullptr : empty.rssi_dbm.data();
  const espectre::test::replay::ReplayPacketMetadata metadata = empty_metadata();
  const uint16_t window_size = espectre::test::replay::detector_window_packets(
      metadata, empty.num_packets);
  HighAccuracyDetector detector(window_size, HIGH_ACCURACY_DEFAULT_THRESHOLD);
  detector.configure_hampel(true);

  const espectre::test::replay::ReplayMetrics metrics =
      espectre::test::replay::evaluate_detector(
          detector, packets, empty.num_packets, rssi, metadata, nullptr, 0,
          nullptr, {}, empty.packet_size, DEFAULT_SUBCARRIERS,
          HT20_SELECTED_BAND_SIZE);

  TEST_ASSERT_TRUE(metrics.static_presence_eval_count > 0);
  TEST_ASSERT_EQUAL(0, metrics.effective_alarms);
  TEST_ASSERT_TRUE(metrics.fp_rate < 5.0f);
}

int process(const espectre::test::dataset_cli::Options& options) {
  int failures = 0;
  const int empty_count = csi_test_data::get_available_empty_room_count();
  if (empty_count <= 0) {
    printf("ERROR: No empty-room recordings available\n");
    return 1;
  }

  int selected_recordings = 0;
  for (int empty_index = 0; empty_index < empty_count; empty_index++) {
    const csi_test_data::ChipType chip = csi_test_data::empty_room_chip(empty_index);
    if (!espectre::test::dataset_cli::matches(options, chip)) {
      continue;
    }
    selected_recordings++;
    printf("\n========================================\n");
    printf("Running empty-room Lightweight with %s recording\n",
           csi_test_data::chip_name(chip));
    printf("Recording: %s\n", csi_test_data::empty_room_label(empty_index));
    printf("========================================\n");

    if (!csi_test_data::switch_empty_room_dataset(empty_index)) {
      printf("ERROR: Failed to load empty-room recording %d\n", empty_index);
      failures++;
      continue;
    }

    UNITY_BEGIN();
    RUN_TEST(test_classic_empty_room_stays_within_alarm_budget);
    RUN_TEST(test_ml_empty_room_has_no_effective_alarms);
    failures += UNITY_END();
  }

  if (!options.aggregate && selected_recordings == 0) {
    return espectre::test::dataset_cli::no_eligible_dataset(options);
  }

  return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) {
  espectre::test::dataset_cli::Options options;
  process(options);
}
#else
int main(int argc, char** argv) {
  espectre::test::dataset_cli::Options options;
  if (!espectre::test::dataset_cli::parse(argc, argv, "empty", options)) {
    return 2;
  }
  return process(options);
}
#endif
