/*
 * ESPectre - Packet-rate adaptation regression test
 *
 * Replays 60-second prefixes of explicit high-rate pairs after synthetic
 * decimation and checks that the C++ Lightweight and High Accuracy implementations stay
 * robust across the supported 80-120 pps operating region.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "lightweight_detector.h"
#include "detector_limits.h"
#include "detector_timing.h"
#include "high_accuracy_detector.h"
#include "csi_test_data.h"
#include "csi_replay_metrics.h"

using namespace espectre;
namespace replay = espectre::test::replay;

namespace {

constexpr double kMinimumSourceAveragePacketRate = 500.0;
constexpr int kReplayDurationSeconds = 60;
constexpr int kTargetPps[] = {120, 100, 80};
constexpr size_t kTargetCount = sizeof(kTargetPps) / sizeof(kTargetPps[0]);

struct PacketRateSourceSelection {
  std::string pair_id;
  std::string static_presence_path;
  std::string motion_path;
  int nominal_pps{0};
  double average_packet_rate{0.0};
};

struct ReplayMetrics {
  DetectorTiming timing{};
  float threshold{0.0f};
  float recall{0.0f};
  float fp_rate{0.0f};
  float precision{0.0f};
  float f1{0.0f};
  int baseline_eval{0};
  int motion_eval{0};
};

struct RateResult {
  int target_pps{0};
  uint32_t measured_interval_us{0U};
  ReplayMetrics classic{};
  ReplayMetrics ml{};
};

double entry_average_packet_rate(JsonObjectConst entry) {
  const double average_packet_rate = entry["average_packet_rate"] | 0.0;
  if (average_packet_rate > 0.0) {
    return average_packet_rate;
  }
  const double duration_ms = entry["duration_ms"] | 0.0;
  const int num_packets = entry["num_packets"] | 0;
  if (duration_ms <= 0.0 || num_packets <= 0) {
    return 0.0;
  }
  return (static_cast<double>(num_packets) * 1000.0) / duration_ms;
}

int entry_nominal_packet_rate(JsonObjectConst entry, double average_packet_rate) {
  (void) average_packet_rate;
  const int nominal_packet_rate = entry["nominal_packet_rate"] | 0;
  if (nominal_packet_rate > 0) {
    return nominal_packet_rate;
  }
  return 0;
}

const std::vector<PacketRateSourceSelection>& source_pairs() {
  static const std::vector<PacketRateSourceSelection> selections = []() {
    const std::string dataset_info_path = "../../data/dataset_info.json";
    std::ifstream in(dataset_info_path);
    if (!in.is_open()) {
      throw std::runtime_error("Cannot open " + dataset_info_path);
    }

    DynamicJsonDocument doc(128 * 1024);
    const auto err = deserializeJson(doc, in);
    if (err) {
      throw std::runtime_error(
          std::string("Failed parsing dataset_info.json: ") + err.c_str());
    }

    struct MotionEntry {
      std::string path;
      double average_packet_rate{0.0};
    };
    std::unordered_map<std::string, MotionEntry> motion_by_filename;

    for (JsonObjectConst entry : doc["files"]["motion"].as<JsonArrayConst>()) {
      const char* filename = entry["filename"];
      if (filename == nullptr) {
        continue;
      }
      const double average_packet_rate = entry_average_packet_rate(entry);
      if (average_packet_rate < kMinimumSourceAveragePacketRate) {
        continue;
      }
      motion_by_filename.emplace(
          filename,
          MotionEntry{
              std::string("../../data/motion/") + filename,
              average_packet_rate,
          });
    }

    std::vector<PacketRateSourceSelection> loaded;
    for (JsonObjectConst entry : doc["files"]["static_presence"].as<JsonArrayConst>()) {
      const char* filename = entry["filename"];
      const char* motion_filename = entry["optimal_pair_motion_file"];
      if (filename == nullptr || motion_filename == nullptr) {
        continue;
      }
      const double average_packet_rate = entry_average_packet_rate(entry);
      if (average_packet_rate < kMinimumSourceAveragePacketRate) {
        continue;
      }
      const auto motion_it = motion_by_filename.find(motion_filename);
      if (motion_it == motion_by_filename.end()) {
        continue;
      }
      const int nominal_pps = entry_nominal_packet_rate(entry, average_packet_rate);
      if (nominal_pps <= 0) {
        continue;
      }
      loaded.push_back(PacketRateSourceSelection{
          std::string(filename),
          std::string("../../data/static_presence/") + filename,
          motion_it->second.path,
          nominal_pps,
          average_packet_rate,
      });
    }

    std::sort(loaded.begin(), loaded.end(), [](const PacketRateSourceSelection& a,
                                               const PacketRateSourceSelection& b) {
      if (a.nominal_pps != b.nominal_pps) {
        return a.nominal_pps < b.nominal_pps;
      }
      return a.pair_id < b.pair_id;
    });
    if (loaded.empty()) {
      throw std::runtime_error(
          "No explicit static_presence/motion pairs with average_packet_rate >= 500");
    }
    return loaded;
  }();
  return selections;
}

replay::ReplayPacketMetadata metadata_for_data(const csi_test_data::CsiData& data) {
  return {
      data.stream_seq_num.empty() ? nullptr : data.stream_seq_num.data(),
      data.device_ticks_us.empty() ? nullptr : data.device_ticks_us.data(),
      data.wifi_rx_ts_us.empty() ? nullptr : data.wifi_rx_ts_us.data(),
      data.csi_target_pps,
  };
}

const csi_test_data::CsiData& source_static_presence(
    const PacketRateSourceSelection& selection) {
  static std::unordered_map<std::string, csi_test_data::CsiData> cache;
  auto it = cache.find(selection.static_presence_path);
  if (it == cache.end()) {
    it = cache.emplace(
        selection.static_presence_path,
        csi_test_data::load_npz(selection.static_presence_path)).first;
  }
  return it->second;
}

const csi_test_data::CsiData& source_motion(const PacketRateSourceSelection& selection) {
  static std::unordered_map<std::string, csi_test_data::CsiData> cache;
  auto it = cache.find(selection.motion_path);
  if (it == cache.end()) {
    it = cache.emplace(selection.motion_path, csi_test_data::load_npz(selection.motion_path)).first;
  }
  return it->second;
}

csi_test_data::CsiData decimate_capture(const csi_test_data::CsiData& source,
                                        int source_pps,
                                        int target_pps) {
  csi_test_data::CsiData result;
  result.packet_size = source.packet_size;
  result.num_subcarriers = source.num_subcarriers;

  const int resolved_target_pps = std::min(source_pps, target_pps);
  result.csi_target_pps = static_cast<uint32_t>(resolved_target_pps);
  const int source_packet_limit = std::min(
      source.num_packets, source_pps * kReplayDurationSeconds);
  const double stride =
      static_cast<double>(source_pps) / static_cast<double>(resolved_target_pps);
  const uint32_t interval_us = static_cast<uint32_t>(
      std::lround(1000000.0 / static_cast<double>(resolved_target_pps)));
  double cursor = 0.0;
  bool seeded_seq = false;
  uint32_t next_seq_num = 0U;
  bool seeded_device_ticks = false;
  uint64_t next_device_ticks_us = 0U;
  bool seeded_wifi_rx_ts = false;
  uint32_t next_wifi_rx_ts_us = 0U;

  while (true) {
    const int source_index = static_cast<int>(std::llround(cursor));
    if (source_index >= source_packet_limit) {
      break;
    }

    result.packets.push_back(source.packets[static_cast<size_t>(source_index)]);
    if (!source.rssi_dbm.empty()) {
      result.rssi_dbm.push_back(source.rssi_dbm[static_cast<size_t>(source_index)]);
    }
    if (!source.device_ticks_us.empty()) {
      if (!seeded_device_ticks) {
        next_device_ticks_us = source.device_ticks_us[static_cast<size_t>(source_index)];
        seeded_device_ticks = true;
      } else {
        next_device_ticks_us += interval_us;
      }
      result.device_ticks_us.push_back(next_device_ticks_us);
    }
    if (!source.wifi_rx_ts_us.empty()) {
      if (!seeded_wifi_rx_ts) {
        next_wifi_rx_ts_us = source.wifi_rx_ts_us[static_cast<size_t>(source_index)];
        seeded_wifi_rx_ts = true;
      } else {
        next_wifi_rx_ts_us += interval_us;
      }
      result.wifi_rx_ts_us.push_back(next_wifi_rx_ts_us);
    }
    if (!source.stream_seq_num.empty()) {
      if (!seeded_seq) {
        next_seq_num = source.stream_seq_num[static_cast<size_t>(source_index)];
        seeded_seq = true;
      } else {
        next_seq_num++;
      }
      result.stream_seq_num.push_back(next_seq_num);
    }

    cursor += stride;
  }

  result.num_packets = static_cast<int>(result.packets.size());
  return result;
}

uint32_t measure_packet_interval_us(const csi_test_data::CsiData& data) {
  const uint32_t nominal = nominal_packet_interval_us(DETECTOR_DEFAULT_WINDOW_SIZE);
  if (data.num_packets < 2) {
    return nominal;
  }

  csi_replay_timing::PacketTimingTracker tracker(nominal);
  uint64_t total_us = 0U;
  int counted = 0;
  for (int i = 0; i < data.num_packets; i++) {
    const size_t index = static_cast<size_t>(i);
    const bool has_seq = index < data.stream_seq_num.size();
    const bool has_device_ticks = index < data.device_ticks_us.size();
    const bool has_wifi_rx_ts = index < data.wifi_rx_ts_us.size();
    const csi_replay_timing::TimingObservation timing = tracker.observe(
        has_seq ? data.stream_seq_num[index] : 0U,
        has_seq,
        has_device_ticks ? data.device_ticks_us[index] : 0U,
        has_device_ticks,
        has_wifi_rx_ts ? data.wifi_rx_ts_us[index] : 0U,
        has_wifi_rx_ts);
    if (i == 0 || timing.contaminated || timing.coverage_us == 0U) {
      continue;
    }
    total_us += timing.delta_us;
    counted++;
  }
  if (counted <= 0) {
    return nominal;
  }
  return static_cast<uint32_t>(std::llround(
      static_cast<double>(total_us) / static_cast<double>(counted)));
}

RateResult run_rate_case(const PacketRateSourceSelection& selection, int target_pps) {
  const csi_test_data::CsiData static_capture =
      decimate_capture(source_static_presence(selection), selection.nominal_pps, target_pps);
  const csi_test_data::CsiData motion_capture =
      decimate_capture(source_motion(selection), selection.nominal_pps, target_pps);
  const std::vector<const int8_t*> static_ptrs = csi_test_data::get_packet_pointers(static_capture);
  const std::vector<const int8_t*> motion_ptrs = csi_test_data::get_packet_pointers(motion_capture);
  const replay::ReplayPacketMetadata baseline_metadata = metadata_for_data(static_capture);
  const replay::ReplayPacketMetadata movement_metadata = metadata_for_data(motion_capture);

  const uint32_t measured_interval_us = measure_packet_interval_us(static_capture);
  const DetectorTiming timing = derive_detector_timing(measured_interval_us);

  LightweightDetector classic(
      timing.window_packets,
      LIGHTWEIGHT_DEFAULT_THRESHOLD,
      timing.autocorr_lag);
  classic.configure_hampel(true);
  float classic_threshold = LIGHTWEIGHT_DEFAULT_THRESHOLD;
  TEST_ASSERT_TRUE_MESSAGE(
      replay::calibrate_lightweight_detector(
          classic,
          replay::calibration_packet_count(
              baseline_metadata, static_capture.num_packets),
          static_ptrs.data(),
          static_capture.num_packets,
          static_capture.rssi_dbm.empty() ? nullptr : static_capture.rssi_dbm.data(),
          baseline_metadata,
          static_capture.packet_size,
          DEFAULT_SUBCARRIERS,
          12,
          classic_threshold),
      "Lightweight startup calibration failed");
  const replay::ReplayMetrics classic_replay = replay::evaluate_detector(
      classic,
      static_ptrs.data(),
      static_capture.num_packets,
      static_capture.rssi_dbm.empty() ? nullptr : static_capture.rssi_dbm.data(),
      baseline_metadata,
      motion_ptrs.data(),
      motion_capture.num_packets,
      motion_capture.rssi_dbm.empty() ? nullptr : motion_capture.rssi_dbm.data(),
      movement_metadata,
      static_capture.packet_size,
      DEFAULT_SUBCARRIERS,
      12);
  ReplayMetrics classic_metrics{};
  classic_metrics.timing = timing;
  classic_metrics.threshold = classic_threshold;
  classic_metrics.recall = classic_replay.recall;
  classic_metrics.fp_rate = classic_replay.fp_rate;
  classic_metrics.precision = classic_replay.precision;
  classic_metrics.f1 = classic_replay.f1;
  classic_metrics.baseline_eval = classic_replay.static_presence_eval_count;
  classic_metrics.motion_eval = classic_replay.motion_eval_count;

  HighAccuracyDetector ml(timing.window_packets, HIGH_ACCURACY_DEFAULT_THRESHOLD);
  ml.configure_hampel(true);
  const replay::ReplayMetrics ml_replay = replay::evaluate_detector(
      ml,
      static_ptrs.data(),
      static_capture.num_packets,
      nullptr,
      baseline_metadata,
      motion_ptrs.data(),
      motion_capture.num_packets,
      nullptr,
      movement_metadata,
      static_capture.packet_size,
      DEFAULT_SUBCARRIERS,
      12);
  ReplayMetrics ml_metrics{};
  ml_metrics.timing = timing;
  ml_metrics.threshold = HIGH_ACCURACY_DEFAULT_THRESHOLD;
  ml_metrics.recall = ml_replay.recall;
  ml_metrics.fp_rate = ml_replay.fp_rate;
  ml_metrics.precision = ml_replay.precision;
  ml_metrics.f1 = ml_replay.f1;
  ml_metrics.baseline_eval = ml_replay.static_presence_eval_count;
  ml_metrics.motion_eval = ml_replay.motion_eval_count;

  RateResult result{};
  result.target_pps = target_pps;
  result.measured_interval_us = measured_interval_us;
  result.classic = classic_metrics;
  result.ml = ml_metrics;
  return result;
}

void print_summary_table(const PacketRateSourceSelection& selection,
                         const std::vector<RateResult>& results) {
  printf("\n");
  printf("Pair: %s | nominal=%d pps | average=%.1f pps\n",
         selection.pair_id.c_str(),
         selection.nominal_pps,
         selection.average_packet_rate);
  printf("Replay prefix: %d seconds per phase\n", kReplayDurationSeconds);
  printf("                         PACKET-RATE ADAPTATION SUMMARY (C++)\n");
  printf("---------------------------------------------------------------------------------------------------------\n");
  printf("pps | timing      | Lightweight R/FP  | ML R/FP       | eval idle/motion\n");
  printf("----+-------------+---------------+---------------+-----------------\n");
  for (const RateResult& result : results) {
    printf("%-3d | w%-3u l%-2u a%-1u | %5.1f%% / %4.1f%% | %5.1f%% / %4.1f%% | %3d / %-3d\n",
           result.target_pps,
           static_cast<unsigned>(result.classic.timing.window_packets),
           static_cast<unsigned>(result.classic.timing.lag),
           static_cast<unsigned>(result.classic.timing.autocorr_lag),
           result.classic.recall,
           result.classic.fp_rate,
           result.ml.recall,
           result.ml.fp_rate,
           result.classic.baseline_eval,
           result.classic.motion_eval);
  }
  printf("----------------------------------------------------------------------------------------------------\n");
}

void test_packet_rate_adaptation_regression(void) {
  for (const PacketRateSourceSelection& selection : source_pairs()) {
    std::vector<RateResult> results;
    results.reserve(kTargetCount);
    for (size_t i = 0; i < kTargetCount; i++) {
      results.push_back(run_rate_case(selection, kTargetPps[i]));
    }

    print_summary_table(selection, results);

    int min_baseline_eval = results[0].classic.baseline_eval;
    int max_baseline_eval = results[0].classic.baseline_eval;
    int min_motion_eval = results[0].classic.motion_eval;
    int max_motion_eval = results[0].classic.motion_eval;

    for (size_t i = 0; i < results.size(); i++) {
      const RateResult& result = results[i];
      min_baseline_eval = std::min(min_baseline_eval, result.classic.baseline_eval);
      max_baseline_eval = std::max(max_baseline_eval, result.classic.baseline_eval);
      min_motion_eval = std::min(min_motion_eval, result.classic.motion_eval);
      max_motion_eval = std::max(max_motion_eval, result.classic.motion_eval);
    }

    for (size_t i = 0; i < results.size(); i++) {
      const uint32_t expected_interval_us = static_cast<uint32_t>(
          std::lround(1000000.0 / static_cast<double>(results[i].target_pps)));
      const DetectorTiming expected_timing = derive_detector_timing(expected_interval_us);
      const DetectorTiming actual_timing = results[i].classic.timing;
      TEST_ASSERT_TRUE(std::abs(
                           static_cast<int>(results[i].measured_interval_us) -
                           static_cast<int>(expected_interval_us)) <= 2);
      TEST_ASSERT_EQUAL(expected_timing.window_packets, actual_timing.window_packets);
      TEST_ASSERT_EQUAL(expected_timing.lag, actual_timing.lag);
      TEST_ASSERT_EQUAL(expected_timing.autocorr_lag, actual_timing.autocorr_lag);
    }

    TEST_ASSERT_TRUE(min_baseline_eval >= 220);
    TEST_ASSERT_TRUE(max_baseline_eval <= 245);
    TEST_ASSERT_TRUE((max_baseline_eval - min_baseline_eval) <= 20);
    TEST_ASSERT_TRUE(min_motion_eval >= 220);
    TEST_ASSERT_TRUE(max_motion_eval <= 245);
    TEST_ASSERT_TRUE((max_motion_eval - min_motion_eval) <= 20);

    for (const RateResult& result : results) {
      TEST_ASSERT_TRUE(result.classic.recall >= 95.0f);
      const float classic_fp_limit = result.target_pps <= 80 ? 1.2f : 1.0f;
      TEST_ASSERT_TRUE(result.classic.fp_rate <= classic_fp_limit);
      TEST_ASSERT_TRUE(result.ml.recall >= 95.0f);
      TEST_ASSERT_TRUE(result.ml.fp_rate <= 1.0f);
    }
  }
}

void test_detector_window_covers_the_configured_duration(void) {
  constexpr uint32_t kIntervalUs = 10723U;
  constexpr uint32_t kDurationUs = 1000000U;
  const DetectorTiming timing = derive_detector_timing(kIntervalUs, 1000U);

  TEST_ASSERT_EQUAL(94U, timing.window_packets);
  TEST_ASSERT_TRUE(static_cast<uint32_t>(timing.window_packets) * kIntervalUs >= kDurationUs);
  TEST_ASSERT_TRUE(static_cast<uint32_t>(timing.window_packets - 1U) * kIntervalUs < kDurationUs);
}

}  // namespace

int main(int argc, char** argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_detector_window_covers_the_configured_duration);
  RUN_TEST(test_packet_rate_adaptation_regression);
  return UNITY_END();
}
