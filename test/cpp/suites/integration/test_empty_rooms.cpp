/*
 * ESPectre - Empty-Room Integration Test
 *
 * Validates that the production Classic path raises no alarm on recordings
 * made with nobody in the room.
 *
 * Empty rooms are the corpus ground truth for "nothing is moving". The
 * static-presence baselines cannot serve that role: a stationary person still
 * breathes and shifts, and the detector sees it, so a share of their
 * evaluations is the detector working rather than failing. See the empty-room
 * false-positive ADR.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include "classic_detector.h"
#include "csi_replay_metrics.h"
#include "csi_test_data.h"
#include "runtime_sensing_schema.h"

#include <algorithm>

using namespace espectre;

namespace {

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
  return metadata;
}

}  // namespace

void setUp(void) {}
void tearDown(void) {}

void test_classic_raises_no_alarm_on_empty_room(void) {
  const csi_test_data::CsiData& empty = csi_test_data::g_empty_data;
  const int8_t* const* packets = csi_test_data::g_empty_ptrs.data();
  const int8_t* rssi =
      empty.rssi_dbm.empty() ? nullptr : empty.rssi_dbm.data();
  const espectre::test::replay::ReplayPacketMetadata metadata = empty_metadata();

  const uint16_t window_size = espectre::test::replay::detector_window_packets(
      metadata, empty.num_packets);
  ClassicDetector detector(window_size, CLASSIC_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);

  const int calibration_packets = std::min(
      empty.num_packets,
      static_cast<int>(espectre::test::replay::calibration_packet_count(
          metadata, empty.num_packets)));
  float adaptive_threshold = 0.0f;
  const bool calibrated = espectre::test::replay::calibrate_classic_detector(
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
  printf("Empty room Classic: fp=%.2f%%, effective_alarms=%d, evaluations=%d\n",
         fp_rate, metrics.effective_alarms, metrics.static_presence_eval_count);

  TEST_ASSERT_TRUE(metrics.static_presence_eval_count > 0);
  // Nobody is in the room, so a debounced alarm here is a real defect.
  TEST_ASSERT_EQUAL_INT(0, metrics.effective_alarms);
  // Secondary regression guard on the raw per-evaluation rate. The corpus
  // maximum is 5.14%, so this bounds drift without tracking noise.
  TEST_ASSERT_TRUE(fp_rate < 6.0f);
}

int process(void) {
  int failures = 0;
  const int empty_count = csi_test_data::get_available_empty_room_count();
  if (empty_count <= 0) {
    printf("ERROR: No empty-room recordings available\n");
    return 1;
  }

  for (int empty_index = 0; empty_index < empty_count; empty_index++) {
    const csi_test_data::ChipType chip = csi_test_data::empty_room_chip(empty_index);
    printf("\n========================================\n");
    printf("Running empty-room Classic with %s recording\n",
           csi_test_data::chip_name(chip));
    printf("Recording: %s\n", csi_test_data::empty_room_label(empty_index));
    printf("========================================\n");

    if (!csi_test_data::switch_empty_room_dataset(empty_index)) {
      printf("ERROR: Failed to load empty-room recording %d\n", empty_index);
      failures++;
      continue;
    }

    UNITY_BEGIN();
    RUN_TEST(test_classic_raises_no_alarm_on_empty_room);
    failures += UNITY_END();
  }

  return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) { return process(); }
#endif
