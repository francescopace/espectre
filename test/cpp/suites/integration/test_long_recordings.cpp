/*
 * ESPectre - C++ Long Recording Tests
 *
 * Runs the same long recordings used by Python validation and prints
 * native Classic and ML metrics for manual comparison.
 */

#include "test_harness.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "csi_format.h"
#include "utils.h"
#include "classic_detector.h"
#include "ml_detector.h"
#include "threshold.h"

using namespace esphome::espectre;

#include "csi_test_data.h"

struct LongRunMetrics {
  int static_presence_eval_count{0};
  int motion_eval_count{0};
  int tp{0};
  int fn{0};
  int fp{0};
  int tn{0};
  float recall{0.0f};
  float precision{0.0f};
  float fp_rate{0.0f};
  float f1{0.0f};
  std::array<uint8_t, HT20_SELECTED_BAND_SIZE> selected_band{};
  uint8_t selected_band_size{0};
  float adaptive_threshold{0.0f};
};

struct ChipLongRunResults {
  const char *chip_name{nullptr};
  LongRunMetrics classic;
  LongRunMetrics ml;
  bool has_classic{false};
  bool has_ml{false};
};

static ChipLongRunResults g_results[5];
static int g_results_count = 0;

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
  const char *chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
  if (g_results_count == 0 || std::strcmp(g_results[g_results_count - 1].chip_name, chip_name) != 0) {
    g_results[g_results_count] = ChipLongRunResults{};
    g_results[g_results_count].chip_name = chip_name;
    g_results_count++;
  }

  ChipLongRunResults &current = g_results[g_results_count - 1];
  if (std::strcmp(algorithm, "classic") == 0) {
    current.classic = metrics;
    current.has_classic = true;
  } else if (std::strcmp(algorithm, "ml") == 0) {
    current.ml = metrics;
    current.has_ml = true;
  }
}

static void print_metrics(const char *label, const LongRunMetrics &metrics) {
  printf("%s: tp=%d fn=%d fp=%d tn=%d | recall=%.6f precision=%.6f fp_rate=%.6f f1=%.6f\n",
         label, metrics.tp, metrics.fn, metrics.fp, metrics.tn, metrics.recall,
         metrics.precision, metrics.fp_rate, metrics.f1);
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
  TEST_ASSERT_TRUE_MESSAGE(csi_test_data::num_motion() > 0, "Movement split must not be empty");
}

static void assert_metrics_are_valid(const LongRunMetrics &metrics) {
  TEST_ASSERT_TRUE(metrics.static_presence_eval_count >= 0);
  TEST_ASSERT_TRUE(metrics.motion_eval_count >= 0);
  TEST_ASSERT_EQUAL_INT(metrics.static_presence_eval_count, metrics.fp + metrics.tn);
  TEST_ASSERT_EQUAL_INT(metrics.motion_eval_count, metrics.tp + metrics.fn);
  TEST_ASSERT_TRUE(metrics.recall >= 0.0f && metrics.recall <= 100.0f);
  TEST_ASSERT_TRUE(metrics.precision >= 0.0f && metrics.precision <= 100.0f);
  TEST_ASSERT_TRUE(metrics.fp_rate >= 0.0f && metrics.fp_rate <= 100.0f);
  TEST_ASSERT_TRUE(metrics.f1 >= 0.0f && metrics.f1 <= 100.0f);
}

static bool build_calibrated_classic_detector(ClassicDetector& detector, int calibration_packets,
                                              int pkt_size, float& out_threshold) {
  StartupThresholdCalibrator calibrator;
  calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());
  for (int i = 0; i < calibration_packets; i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    detector.update_state();
    calibrator.observe(detector.is_ready(), detector.get_motion_metric(),
                       detector.get_startup_floor_metric());
    if (calibrator.is_complete()) {
      break;
    }
  }
  if (!calibrator.is_successful()) {
    out_threshold = CLASSIC_DEFAULT_THRESHOLD;
    return false;
  }
  float variance_floor = 0.0f;
  bool vote_enabled = false;
  uint16_t floor_count = 0;
  calibrator.floor_snapshot(variance_floor, vote_enabled, floor_count);
  detector.apply_startup_floor(variance_floor, vote_enabled, floor_count);
  detector.on_startup_calibration_complete();
  out_threshold = calibrator.threshold_metric() *
                  get_threshold_factor(ThresholdMode::AUTO, detector.get_startup_threshold_factor());
  detector.set_threshold(out_threshold);
  detector.clear_buffer();
  return true;
}

static void print_summary_table() {
  printf("\n");
  printf("=====================================================================================================================\n");
  printf("                                     LONG RECORDING SUMMARY (C++)\n");
  printf("=====================================================================================================================\n");
  printf("| Chip   | Classic                 | ML                      |\n");
  printf("|--------|-------------------------|-------------------------|\n");

  for (int i = 0; i < g_results_count; i++) {
    const ChipLongRunResults &r = g_results[i];
    char classic_str[32] = "N/A";
    char ml_str[32] = "N/A";

    if (r.has_classic) {
      std::snprintf(classic_str, sizeof(classic_str), "%.1f%% R, %.1f%% FP",
                    r.classic.recall, r.classic.fp_rate);
    }
    if (r.has_ml) {
      std::snprintf(ml_str, sizeof(ml_str), "%.1f%% R, %.1f%% FP",
                    r.ml.recall, r.ml.fp_rate);
    }

    printf("| %-6s | %-23s | %-23s |\n", r.chip_name, classic_str, ml_str);
  }

  printf("---------------------------------------------------------------------------------------------------------------------\n");
  printf("Legend: R = Recall, FP = False Positive Rate\n");
}

static LongRunMetrics evaluate_ml_long_recording() {
  LongRunMetrics metrics;
  const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
  const int pkt_size = csi_test_data::packet_size();

  MLDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, ML_DEFAULT_THRESHOLD);
  detector.configure_hampel(true);

  metrics.static_presence_eval_count = std::max(csi_test_data::num_static_presence() - warmup, 0);
  metrics.motion_eval_count = std::max(csi_test_data::num_motion() - warmup, 0);

  for (int i = 0; i < csi_test_data::num_static_presence(); i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    detector.update_state();
    if (i >= warmup && detector.get_state() == MotionState::MOTION) {
      metrics.fp++;
    }
  }

  for (int i = 0; i < csi_test_data::num_motion(); i++) {
    detector.process_packet(csi_test_data::motion_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    detector.update_state();
    if (i >= warmup) {
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.static_presence_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  return metrics;
}

static LongRunMetrics evaluate_classic_long_recording() {
  LongRunMetrics metrics;
  const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
  const int pkt_size = csi_test_data::packet_size();

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
  metrics.static_presence_eval_count = std::max(csi_test_data::num_static_presence() - warmup, 0);
  metrics.motion_eval_count = std::max(csi_test_data::num_motion() - warmup, 0);

  for (int i = 0; i < csi_test_data::num_static_presence(); i++) {
    detector.process_packet(csi_test_data::static_presence_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    detector.update_state();
    if (i >= warmup && detector.get_state() == MotionState::MOTION) {
      metrics.fp++;
    }
  }

  for (int i = 0; i < csi_test_data::num_motion(); i++) {
    detector.process_packet(csi_test_data::motion_packets()[i], pkt_size, DEFAULT_SUBCARRIERS,
                            HT20_SELECTED_BAND_SIZE);
    detector.update_state();
    if (i >= warmup) {
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.static_presence_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  return metrics;
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

int run_tests_for_chip(csi_test_data::ChipType chip) {
  printf("\n========================================\n");
  printf("Running long-recording tests with %s\n", csi_test_data::chip_name(chip));
  printf("========================================\n");

  if (!csi_test_data::switch_long_recording_dataset(chip)) {
    printf("ERROR: Failed to load long recording for %s\n", csi_test_data::chip_name(chip));
    return 1;
  }

  UNITY_BEGIN();
  RUN_TEST(test_long_recording_classic);
  RUN_TEST(test_long_recording_ml);
  return UNITY_END();
}

int process(void) {
  int failures = 0;
  for (auto chip : csi_test_data::get_available_long_recording_chips()) {
    failures += run_tests_for_chip(chip);
  }
  print_summary_table();
  return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
