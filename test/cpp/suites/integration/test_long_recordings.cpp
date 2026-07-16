/*
 * ESPectre - C++ Long Recording Tests
 *
 * Runs the same long recordings used by Python validation and prints
 * native Classic and ML metrics for manual comparison.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "csi_format.h"
#include "utils.h"
#include "classic_detector.h"
#include "ml_detector.h"
#include "runtime_sensing_schema.h"
#include "threshold.h"

using namespace espectre;

#include "csi_test_data.h"

struct LongRunMetrics {
  int static_presence_eval_count{0};
  int motion_eval_count{0};
  int tp{0};
  int fn{0};
  int fp{0};
  int tn{0};
  int effective_alarms{0};
  int false_motion_evaluations{0};
  float recall{0.0f};
  float precision{0.0f};
  float fp_rate{0.0f};
  float f1{0.0f};
  std::array<uint8_t, HT20_SELECTED_BAND_SIZE> selected_band{};
  uint8_t selected_band_size{0};
  float adaptive_threshold{0.0f};
};

struct DatasetLongRunResults {
  std::string dataset_name;
  const char *chip_name{nullptr};
  LongRunMetrics classic;
  LongRunMetrics ml;
  bool has_classic{false};
  bool has_ml{false};
};

static std::vector<DatasetLongRunResults> g_results;

struct LongRunAggregate {
  int count{0};
  float avg_fp_rate{0.0f};
  float max_fp_rate{0.0f};
  int effective_alarms{0};
  int false_motion_evaluations{0};
  bool valid{false};
};

static void compute_derived_metrics(LongRunMetrics &metrics) {
  metrics.recall = (metrics.tp + metrics.fn) > 0
                       ? static_cast<float>(metrics.tp) / static_cast<float>(metrics.tp + metrics.fn) * 100.0f
                       : 0.0f;
  metrics.precision = (metrics.tp + metrics.fp) > 0
                          ? static_cast<float>(metrics.tp) / static_cast<float>(metrics.tp + metrics.fp) * 100.0f
                          : 0.0f;
  metrics.fp_rate = metrics.static_presence_eval_count > 0
                        ? static_cast<float>(metrics.fp) / static_cast<float>(metrics.static_presence_eval_count) * 100.0f
                        : 0.0f;
  metrics.f1 = (metrics.precision + metrics.recall) > 0.0f
                   ? 2.0f * (metrics.precision / 100.0f) * (metrics.recall / 100.0f) /
                         ((metrics.precision + metrics.recall) / 100.0f) * 100.0f
                   : 0.0f;
}

static void record_result(const char *algorithm, const LongRunMetrics &metrics) {
  const char *dataset_name = csi_test_data::current_long_recording_name();
  const char *chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
  if (dataset_name == nullptr) {
    dataset_name = "unknown_long_recording";
  }

  if (g_results.empty() || g_results.back().dataset_name != dataset_name) {
    DatasetLongRunResults row{};
    row.dataset_name = dataset_name;
    row.chip_name = chip_name;
    g_results.push_back(row);
  }

  DatasetLongRunResults &current = g_results.back();
  if (std::strcmp(algorithm, "classic") == 0) {
    current.classic = metrics;
    current.has_classic = true;
  } else if (std::strcmp(algorithm, "ml") == 0) {
    current.ml = metrics;
    current.has_ml = true;
  }
}

static LongRunMetrics mean_result_for_chip(const char *chip_name, const char *algorithm, bool &has_value) {
  LongRunMetrics mean{};
  int count = 0;
  for (const auto &result : g_results) {
    if (std::strcmp(result.chip_name, chip_name) != 0) {
      continue;
    }
    const bool valid = std::strcmp(algorithm, "classic") == 0 ? result.has_classic : result.has_ml;
    if (!valid) {
      continue;
    }
    const LongRunMetrics &value = std::strcmp(algorithm, "classic") == 0 ? result.classic : result.ml;
    mean.recall += value.recall;
    mean.precision += value.precision;
    mean.fp_rate += value.fp_rate;
    mean.f1 += value.f1;
    count++;
  }
  has_value = count > 0;
  if (!has_value) {
    return mean;
  }
  mean.recall /= count;
  mean.precision /= count;
  mean.fp_rate /= count;
  mean.f1 /= count;
  return mean;
}

static int dataset_count_for_chip(const char *chip_name) {
  int count = 0;
  for (const auto &result : g_results) {
    if (std::strcmp(result.chip_name, chip_name) == 0) {
      count++;
    }
  }
  return count;
}

static LongRunAggregate summarize_long_metrics_for_chip(const char *chip_name, const char *algorithm) {
  LongRunAggregate aggregate;
  for (const auto &result : g_results) {
    if (std::strcmp(result.chip_name, chip_name) != 0) {
      continue;
    }
    const bool valid = std::strcmp(algorithm, "classic") == 0 ? result.has_classic : result.has_ml;
    if (!valid) {
      continue;
    }
    const LongRunMetrics &value = std::strcmp(algorithm, "classic") == 0 ? result.classic : result.ml;
    aggregate.avg_fp_rate += value.fp_rate;
    aggregate.max_fp_rate = aggregate.count == 0 ? value.fp_rate : std::max(aggregate.max_fp_rate, value.fp_rate);
    aggregate.effective_alarms += value.effective_alarms;
    aggregate.false_motion_evaluations += value.false_motion_evaluations;
    aggregate.count++;
  }
  if (aggregate.count > 0) {
    aggregate.avg_fp_rate /= static_cast<float>(aggregate.count);
    aggregate.valid = true;
  }
  return aggregate;
}

static void print_metrics(const char *label, const LongRunMetrics &metrics) {
  printf("%s: tp=%d fn=%d fp=%d tn=%d | recall=%.6f precision=%.6f fp_rate=%.6f f1=%.6f | alarms=%d false_motion_evals=%d\n",
         label, metrics.tp, metrics.fn, metrics.fp, metrics.tn, metrics.recall,
         metrics.precision, metrics.fp_rate, metrics.f1,
         metrics.effective_alarms, metrics.false_motion_evaluations);
  if (metrics.selected_band_size > 0) {
    printf("%s band: [", label);
    for (uint8_t i = 0; i < metrics.selected_band_size; i++) {
      printf("%u", metrics.selected_band[i]);
      if (i + 1 < metrics.selected_band_size) {
        printf(", ");
      }
    }
    printf("], threshold=%.6f\n", metrics.adaptive_threshold);
  }
}

static void assert_dataset_metadata_is_valid() {
  TEST_ASSERT_NOT_NULL_MESSAGE(csi_test_data::current_long_recording_name(), "Missing long-recording filename");
  TEST_ASSERT_TRUE_MESSAGE(csi_test_data::current_motion_start_packet() > 0, "Invalid motion_start_packet");
  TEST_ASSERT_EQUAL_INT(csi_test_data::current_motion_start_packet(), csi_test_data::num_static_presence());
  TEST_ASSERT_TRUE_MESSAGE(csi_test_data::num_motion() >= 0, "Movement split must be valid");
}

static void assert_metrics_are_valid(const LongRunMetrics &metrics) {
  TEST_ASSERT_TRUE(metrics.static_presence_eval_count >= 0);
  TEST_ASSERT_TRUE(metrics.motion_eval_count >= 0);
  TEST_ASSERT_EQUAL_INT(metrics.static_presence_eval_count, metrics.fp + metrics.tn);
  TEST_ASSERT_EQUAL_INT(metrics.motion_eval_count, metrics.tp + metrics.fn);
  TEST_ASSERT_TRUE(metrics.effective_alarms >= 0);
  TEST_ASSERT_TRUE(metrics.false_motion_evaluations >= 0);
  TEST_ASSERT_TRUE(metrics.recall >= 0.0f && metrics.recall <= 100.0f);
  TEST_ASSERT_TRUE(metrics.precision >= 0.0f && metrics.precision <= 100.0f);
  TEST_ASSERT_TRUE(metrics.fp_rate >= 0.0f && metrics.fp_rate <= 100.0f);
  TEST_ASSERT_TRUE(metrics.f1 >= 0.0f && metrics.f1 <= 100.0f);
}

static void apply_runtime_policy_metrics(const std::vector<bool> &raw_motion_states, LongRunMetrics &metrics) {
  MotionState effective_state = MotionState::IDLE;
  MotionState pending_state = MotionState::IDLE;
  uint8_t pending_hits = 0;

  for (bool raw_motion : raw_motion_states) {
    const MotionState detector_state = raw_motion ? MotionState::MOTION : MotionState::IDLE;
    const MotionState previous_state = effective_state;

    if (detector_state == effective_state) {
      pending_state = effective_state;
      pending_hits = 0;
    } else {
      if (detector_state != pending_state) {
        pending_state = detector_state;
        pending_hits = 1;
      } else if (pending_hits < UINT8_MAX) {
        pending_hits++;
      }

      const uint8_t required_hits =
          pending_state == MotionState::MOTION ? RUNTIME_MOTION_ON_HITS_DEFAULT : RUNTIME_MOTION_OFF_HITS_DEFAULT;
      if (pending_hits >= required_hits) {
        effective_state = pending_state;
        pending_hits = 0;
      }
    }

    const bool changed = effective_state != previous_state;
    if (changed && effective_state == MotionState::MOTION) {
      metrics.effective_alarms++;
    }
    if (effective_state == MotionState::MOTION) {
      metrics.false_motion_evaluations++;
    }
  }
}

static bool build_calibrated_classic_detector(ClassicDetector& detector, int calibration_packets,
                                              int pkt_size, float& out_threshold) {
  StartupThresholdCalibrator calibrator;
  calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());
  uint16_t packets_since_evaluation = 0;
  for (int i = 0; i < calibration_packets; i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    calibrator.observe(detector.is_ready(), detector.get_motion_metric(),
                       detector.get_startup_floor_metric(), packets_since_evaluation);
    packets_since_evaluation = 0;
    if (calibrator.is_complete()) {
      break;
    }
  }
  if (!calibrator.is_successful()) {
    out_threshold = CLASSIC_DEFAULT_THRESHOLD;
    return false;
  }
  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(calibrator.threshold_metric());
  out_threshold = detector.get_threshold();
  detector.clear_buffer();
  return true;
}

static void print_summary_table() {
  printf("\n");
  printf("========================================================================================================================\n");
  printf("                                   LONG RECORDING SUMMARY (C++, all datasets)\n");
  printf("========================================================================================================================\n");
  printf("| Chip   | Datasets | Classic                 | ML                      |\n");
  printf("|--------|----------|-------------------------|-------------------------|\n");

  for (auto chip : csi_test_data::get_supported_chips()) {
    const char *chip_name = csi_test_data::chip_name(chip);
    const int dataset_count = dataset_count_for_chip(chip_name);
    if (dataset_count == 0) {
      continue;
    }
    char classic_str[32] = "N/A";
    char ml_str[32] = "N/A";
    bool has_classic = false;
    bool has_ml = false;
    const LongRunMetrics classic = mean_result_for_chip(chip_name, "classic", has_classic);
    const LongRunMetrics ml = mean_result_for_chip(chip_name, "ml", has_ml);

    if (has_classic) {
      std::snprintf(classic_str, sizeof(classic_str), "%.1f%% R, %.1f%% FP",
                    classic.recall, classic.fp_rate);
    }
    if (has_ml) {
      std::snprintf(ml_str, sizeof(ml_str), "%.1f%% R, %.1f%% FP",
                    ml.recall, ml.fp_rate);
    }

    printf("| %-6s | %8d | %-23s | %-23s |\n", chip_name, dataset_count, classic_str, ml_str);
  }

  printf("------------------------------------------------------------------------------------------------------------------------\n");
  printf("Legend: R = Recall, FP = False Positive Rate\n");
}

static LongRunMetrics evaluate_ml_long_recording() {
  LongRunMetrics metrics;
  const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
  const int pkt_size = csi_test_data::packet_size();
  std::vector<bool> baseline_motion_states;

  MLDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, ML_DEFAULT_THRESHOLD);
  detector.configure_hampel(true);

  uint32_t packets_since_evaluation = 0;

  for (int i = 0; i < csi_test_data::num_static_presence(); i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    packets_since_evaluation = 0;
    if (i >= warmup) {
      metrics.static_presence_eval_count++;
      const bool raw_motion = detector.get_state() == MotionState::MOTION;
      baseline_motion_states.push_back(raw_motion);
      if (raw_motion) {
        metrics.fp++;
      }
    }
  }

  for (int i = 0; i < csi_test_data::num_motion(); i++) {
    detector.process_packet(csi_test_data::motion_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    packets_since_evaluation = 0;
    if (i >= warmup) {
      metrics.motion_eval_count++;
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.static_presence_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  apply_runtime_policy_metrics(baseline_motion_states, metrics);
  return metrics;
}

static LongRunMetrics evaluate_classic_long_recording() {
  LongRunMetrics metrics;
  const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
  const int pkt_size = csi_test_data::packet_size();
  std::vector<bool> baseline_motion_states;

  ClassicDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, CLASSIC_DEFAULT_THRESHOLD);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);

  const int calibration_packets = std::min(csi_test_data::num_static_presence(),
                                           static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
  float calibrated_threshold = CLASSIC_DEFAULT_THRESHOLD;
  build_calibrated_classic_detector(detector, calibration_packets, pkt_size, calibrated_threshold);

  metrics.selected_band_size = HT20_SELECTED_BAND_SIZE;
  std::copy(DEFAULT_SUBCARRIERS, DEFAULT_SUBCARRIERS + HT20_SELECTED_BAND_SIZE, metrics.selected_band.begin());
  metrics.adaptive_threshold = calibrated_threshold;
  uint32_t packets_since_evaluation = 0;

  for (int i = 0; i < csi_test_data::num_static_presence(); i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    packets_since_evaluation = 0;
    if (i >= warmup) {
      metrics.static_presence_eval_count++;
      const bool raw_motion = detector.get_state() == MotionState::MOTION;
      baseline_motion_states.push_back(raw_motion);
      if (raw_motion) {
        metrics.fp++;
      }
    }
  }

  for (int i = 0; i < csi_test_data::num_motion(); i++) {
    detector.process_packet(csi_test_data::motion_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    packets_since_evaluation++;
    if (packets_since_evaluation < RUNTIME_EVALUATION_INTERVAL_DEFAULT) {
      continue;
    }
    detector.update_state();
    packets_since_evaluation = 0;
    if (i >= warmup) {
      metrics.motion_eval_count++;
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.static_presence_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  apply_runtime_policy_metrics(baseline_motion_states, metrics);
  return metrics;
}

static void write_algorithm_json(FILE *handle, const char *algorithm) {
  bool first_chip = true;
  for (auto chip : csi_test_data::get_supported_chips()) {
    const char *chip_name = csi_test_data::chip_name(chip);
    const LongRunAggregate aggregate = summarize_long_metrics_for_chip(chip_name, algorithm);
    if (!aggregate.valid) {
      continue;
    }
    if (!first_chip) {
      fprintf(handle, ",");
    }
    first_chip = false;
    fprintf(handle,
            "\"%s\":{\"count\":%d,\"avg_fp_rate\":%.6f,\"max_fp_rate\":%.6f,"
            "\"effective_alarms\":%d,\"false_motion_evaluations\":%d}",
            chip_name,
            aggregate.count,
            aggregate.avg_fp_rate,
            aggregate.max_fp_rate,
            aggregate.effective_alarms,
            aggregate.false_motion_evaluations);
  }
}

static void write_parity_payload_if_requested() {
  const char *output_dir = getenv("ESPECTRE_PARITY_OUTPUT_DIR");
  if (output_dir == nullptr || output_dir[0] == '\0') {
    return;
  }

  std::string path = std::string(output_dir) + "/test_long_recordings.json";
  FILE *handle = fopen(path.c_str(), "w");
  if (handle == nullptr) {
    printf("WARNING: failed to open parity output path: %s\n", path.c_str());
    return;
  }

  fprintf(handle, "{");
  fprintf(handle, "\"suite\":\"test_long_recordings\",");
  fprintf(handle, "\"long_quiet\":{");
  fprintf(handle, "\"classic\":{");
  write_algorithm_json(handle, "classic");
  fprintf(handle, "},");
  fprintf(handle, "\"ml\":{");
  write_algorithm_json(handle, "ml");
  fprintf(handle, "}");
  fprintf(handle, "}");
  fprintf(handle, "}\n");
  fclose(handle);
  printf("Wrote parity metrics to %s\n", path.c_str());
}

void setUp(void) {}
void tearDown(void) {}

void test_long_recording_classic(void) {
  assert_dataset_metadata_is_valid();
  LongRunMetrics actual = evaluate_classic_long_recording();
  print_metrics("Classic actual", actual);
  assert_metrics_are_valid(actual);
  record_result("classic", actual);
}

void test_long_recording_ml(void) {
  assert_dataset_metadata_is_valid();
  LongRunMetrics actual = evaluate_ml_long_recording();
  print_metrics("ML actual", actual);
  assert_metrics_are_valid(actual);
  record_result("ml", actual);
}

int run_tests_for_long_recording(int recording_index) {
  const csi_test_data::ChipType chip = csi_test_data::long_recording_chip(recording_index);
  printf("\n========================================\n");
  printf("Running long-recording tests with %s\n", csi_test_data::chip_name(chip));
  printf("Dataset: %s\n", csi_test_data::long_recording_label(recording_index));
  printf("========================================\n");

  if (!csi_test_data::switch_long_recording_dataset_by_index(recording_index)) {
    printf("ERROR: Failed to load long recording %s\n", csi_test_data::long_recording_label(recording_index));
    return 1;
  }

  UNITY_BEGIN();
  RUN_TEST(test_long_recording_classic);
  RUN_TEST(test_long_recording_ml);
  return UNITY_END();
}

int process(void) {
  int failures = 0;
  g_results.clear();
  const int recording_count = csi_test_data::get_available_long_recording_count();
  if (recording_count <= 0) {
    printf("ERROR: No long recording datasets available\n");
    return 1;
  }
  for (int recording_index = 0; recording_index < recording_count; recording_index++) {
    failures += run_tests_for_long_recording(recording_index);
  }
  print_summary_table();
  write_parity_payload_if_requested();
  return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
