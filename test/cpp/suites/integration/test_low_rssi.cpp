/*
 * ESPectre - Low-RSSI Integration Test
 *
 * Validates the production Lightweight startup and replay path on every real
 * low-RSSI pair exposed by dataset metadata.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "lightweight_detector.h"
#include "csi_replay_metrics.h"
#include "csi_test_data.h"
#include "runtime_sensing_schema.h"

#include <algorithm>
#include <vector>

using namespace espectre;

namespace {

std::vector<const int8_t*> packet_rows(const csi_test_data::CsiData& data) {
  std::vector<const int8_t*> rows;
  rows.reserve(data.packets.size());
  for (const auto& packet : data.packets) {
    rows.push_back(packet.data());
  }
  return rows;
}

espectre::test::replay::ReplayPacketMetadata packet_metadata(
    const csi_test_data::CsiData& data) {
  return {
      data.stream_seq_num.empty() ? nullptr : data.stream_seq_num.data(),
      data.device_ticks_us.empty() ? nullptr : data.device_ticks_us.data(),
      data.wifi_rx_ts_us.empty() ? nullptr : data.wifi_rx_ts_us.data(),
  };
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_classic_handles_loaded_low_rssi_pair(void) {
  const csi_test_data::CsiData baseline = csi_test_data::g_static_presence_data;
  const csi_test_data::CsiData motion = csi_test_data::g_motion_data;
  const std::vector<const int8_t*> baseline_rows = packet_rows(baseline);
  const std::vector<const int8_t*> motion_rows = packet_rows(motion);
  const espectre::test::replay::ReplayPacketMetadata baseline_metadata =
      packet_metadata(baseline);
  const espectre::test::replay::ReplayPacketMetadata motion_metadata =
      packet_metadata(motion);
  const uint16_t window_size = espectre::test::replay::detector_window_packets(
      baseline_metadata, baseline.num_packets);
  LightweightDetector detector(window_size, LIGHTWEIGHT_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);
  const int calibration_packets = std::min(
      baseline.num_packets,
      static_cast<int>(espectre::test::replay::calibration_packet_count(
          baseline_metadata, baseline.num_packets)));
  float adaptive_threshold = LIGHTWEIGHT_DEFAULT_THRESHOLD;
  TEST_ASSERT_TRUE(espectre::test::replay::calibrate_lightweight_detector(
      detector, calibration_packets, baseline_rows.data(), baseline.num_packets,
      baseline.rssi_dbm.empty() ? nullptr : baseline.rssi_dbm.data(),
      baseline_metadata, baseline.packet_size, DEFAULT_SUBCARRIERS,
      HT20_SELECTED_BAND_SIZE, adaptive_threshold));

  const espectre::test::replay::ReplayMetrics metrics =
      espectre::test::replay::evaluate_detector(
          detector, baseline_rows.data(), baseline.num_packets,
          baseline.rssi_dbm.empty() ? nullptr : baseline.rssi_dbm.data(),
          baseline_metadata, motion_rows.data(), motion.num_packets,
          motion.rssi_dbm.empty() ? nullptr : motion.rssi_dbm.data(),
          motion_metadata, baseline.packet_size, DEFAULT_SUBCARRIERS,
          HT20_SELECTED_BAND_SIZE);
  printf("Low-RSSI Lightweight: recall=%.2f%%, fp=%.2f%%, effective_alarms=%d\n",
         metrics.recall, metrics.fp_rate, metrics.effective_alarms);
  // Every weak-link pair clears this recall floor, but the margin on the
  // weakest C3 pair is under a point, so 85 is the level the corpus actually
  // supports today. Raise it to 90 only once the Lightweight feature work lands,
  // and never lower it to accommodate a regression.
  TEST_ASSERT_TRUE(metrics.recall >= 85.0f);
  // Sanity bound, not a false-positive gate. These baselines hold a stationary
  // person, whose breathing and small shifts are real channel motion, so a
  // share of these evaluations is the detector working rather than failing.
  // Zero alarms is asserted in test_empty_rooms, on the only recordings in the
  // corpus with nobody in the room. Corpus maximum is 10.6%.
  TEST_ASSERT_TRUE(metrics.fp_rate <= 12.0f);
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
    printf("Running low-RSSI Lightweight with %s dataset pair\n",
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
