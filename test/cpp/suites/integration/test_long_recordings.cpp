/*
 * ESPectre - C++ Long Recording Tests
 *
 * Runs the same long recordings used by Python validation and prints
 * native MVS + NBVI and ML metrics for manual comparison.
 */

#include "test_harness.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "utils.h"
#include "mvs_detector.h"
#include "ml_detector.h"
#include "csi_manager.h"
#include "nbvi_calibrator.h"
#include "threshold.h"

using namespace esphome::espectre;

class WiFiCSIMock : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override { return ESP_OK; }
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override { return ESP_OK; }
  esp_err_t set_csi(bool enable) override { return ESP_OK; }
};
static WiFiCSIMock g_wifi_mock;

#include "csi_test_data.h"

struct LongRunMetrics {
  int baseline_eval_count{0};
  int movement_eval_count{0};
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
  bool use_cv_normalization{false};
};

struct ChipLongRunResults {
  const char *chip_name{nullptr};
  LongRunMetrics mvs_nbvi;
  LongRunMetrics ml;
  bool has_mvs_nbvi{false};
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
  metrics.fp_rate = metrics.baseline_eval_count > 0
                        ? static_cast<float>(metrics.fp) / static_cast<float>(metrics.baseline_eval_count) * 100.0f
                        : 0.0f;
  metrics.f1 = (metrics.precision + metrics.recall) > 0.0f
                   ? 2.0f * (metrics.precision / 100.0f) * (metrics.recall / 100.0f) /
                         ((metrics.precision + metrics.recall) / 100.0f) * 100.0f
                   : 0.0f;
}

static bool needs_cv_normalization() {
  if (csi_test_data::baseline_gain_locked_known()) {
    return !csi_test_data::baseline_gain_locked();
  }
  return csi_test_data::current_chip() == csi_test_data::ChipType::ESP32;
}

static void record_result(const char *algorithm, const LongRunMetrics &metrics) {
  const char *chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
  if (g_results_count == 0 || std::strcmp(g_results[g_results_count - 1].chip_name, chip_name) != 0) {
    g_results[g_results_count] = ChipLongRunResults{};
    g_results[g_results_count].chip_name = chip_name;
    g_results_count++;
  }

  ChipLongRunResults &current = g_results[g_results_count - 1];
  if (std::strcmp(algorithm, "mvs_nbvi") == 0) {
    current.mvs_nbvi = metrics;
    current.has_mvs_nbvi = true;
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
    printf("], threshold=%.6f, cv_norm=%s\n",
           metrics.adaptive_threshold,
           metrics.use_cv_normalization ? "ON" : "OFF");
  }
}

static void assert_dataset_metadata_is_valid() {
  TEST_ASSERT_NOT_NULL_MESSAGE(csi_test_data::current_long_recording_name(), "Missing long-recording filename");
  TEST_ASSERT_TRUE_MESSAGE(csi_test_data::current_motion_start_packet() > 0, "Invalid motion_start_packet");
  TEST_ASSERT_EQUAL_INT(csi_test_data::current_motion_start_packet(), csi_test_data::num_baseline());
  TEST_ASSERT_TRUE_MESSAGE(csi_test_data::num_movement() > 0, "Movement split must not be empty");
}

static void assert_metrics_are_valid(const LongRunMetrics &metrics) {
  TEST_ASSERT_TRUE(metrics.baseline_eval_count >= 0);
  TEST_ASSERT_TRUE(metrics.movement_eval_count >= 0);
  TEST_ASSERT_EQUAL_INT(metrics.baseline_eval_count, metrics.fp + metrics.tn);
  TEST_ASSERT_EQUAL_INT(metrics.movement_eval_count, metrics.tp + metrics.fn);
  TEST_ASSERT_TRUE(metrics.recall >= 0.0f && metrics.recall <= 100.0f);
  TEST_ASSERT_TRUE(metrics.precision >= 0.0f && metrics.precision <= 100.0f);
  TEST_ASSERT_TRUE(metrics.fp_rate >= 0.0f && metrics.fp_rate <= 100.0f);
  TEST_ASSERT_TRUE(metrics.f1 >= 0.0f && metrics.f1 <= 100.0f);
}

static void assert_mvs_metrics_are_valid(const LongRunMetrics &metrics) {
  assert_metrics_are_valid(metrics);
  TEST_ASSERT_EQUAL_UINT8(HT20_SELECTED_BAND_SIZE, metrics.selected_band_size);
  TEST_ASSERT_TRUE(metrics.adaptive_threshold >= 0.0f);
  TEST_ASSERT_TRUE(metrics.adaptive_threshold <= 10.0f);
}

static void print_summary_table() {
  printf("\n");
  printf("=====================================================================================================================\n");
  printf("                                     LONG RECORDING SUMMARY (C++)\n");
  printf("=====================================================================================================================\n");
  printf("| Chip   | MVS + NBVI              | ML                      |\n");
  printf("|--------|-------------------------|-------------------------|\n");

  for (int i = 0; i < g_results_count; i++) {
    const ChipLongRunResults &r = g_results[i];
    char mvs_str[32] = "N/A";
    char ml_str[32] = "N/A";

    if (r.has_mvs_nbvi) {
      std::snprintf(mvs_str, sizeof(mvs_str), "%.1f%% R, %.1f%% FP",
                    r.mvs_nbvi.recall, r.mvs_nbvi.fp_rate);
    }
    if (r.has_ml) {
      std::snprintf(ml_str, sizeof(ml_str), "%.1f%% R, %.1f%% FP",
                    r.ml.recall, r.ml.fp_rate);
    }

    printf("| %-6s | %-23s | %-23s |\n", r.chip_name, mvs_str, ml_str);
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

  metrics.baseline_eval_count = std::max(csi_test_data::num_baseline() - warmup, 0);
  metrics.movement_eval_count = std::max(csi_test_data::num_movement() - warmup, 0);

  for (int i = 0; i < csi_test_data::num_baseline(); i++) {
    detector.process_packet(csi_test_data::baseline_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    detector.update_state();
    if (i >= warmup && detector.get_state() == MotionState::MOTION) {
      metrics.fp++;
    }
  }

  for (int i = 0; i < csi_test_data::num_movement(); i++) {
    detector.process_packet(csi_test_data::movement_packets()[i], pkt_size, DEFAULT_SUBCARRIERS, 12);
    detector.update_state();
    if (i >= warmup) {
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.baseline_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  return metrics;
}

static LongRunMetrics evaluate_mvs_long_recording() {
  LongRunMetrics metrics;
  const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
  const int pkt_size = csi_test_data::packet_size();

  metrics.use_cv_normalization = needs_cv_normalization();

  CSIManager csi_manager;
  MVSDetector calibration_detector(DETECTOR_DEFAULT_WINDOW_SIZE, SEGMENTATION_DEFAULT_THRESHOLD);
  calibration_detector.configure_lowpass(false);
  calibration_detector.configure_hampel(true);
  calibration_detector.set_cv_normalization(metrics.use_cv_normalization);
  csi_manager.init(&calibration_detector, DEFAULT_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);

  const char *buffer_path = "/tmp/test_long_recordings_nbvi_buffer.bin";
  NBVICalibrator nbvi;
  nbvi.init(&csi_manager, buffer_path);
  nbvi.set_mvs_window_size(DETECTOR_DEFAULT_WINDOW_SIZE);
  nbvi.set_cv_normalization(metrics.use_cv_normalization);
  nbvi.configure_hampel(true);

  const uint16_t buffer_size = std::min(static_cast<int>(nbvi.get_buffer_size()), csi_test_data::num_baseline());
  nbvi.set_buffer_size(buffer_size);

  bool calibration_success = false;
  uint8_t percentile_tmp = 95;
  float calibrated_threshold = 1.0f;
  uint8_t calibrated_band[HT20_SELECTED_BAND_SIZE] = {0};
  uint8_t calibrated_size = 0;

  esp_err_t err = nbvi.start_calibration(
      DEFAULT_SUBCARRIERS, 12,
      [&](const uint8_t *band, uint8_t size, const std::vector<float> &mv_values, bool success) {
        if (success && size > 0) {
          std::memcpy(calibrated_band, band, size);
          calibrated_size = size;
          calculate_adaptive_threshold(mv_values, ThresholdMode::AUTO, calibrated_threshold, percentile_tmp);
        }
        calibration_success = success;
      });

  TEST_ASSERT_EQUAL_MESSAGE(ESP_OK, err, "NBVI calibration start failed");

  for (uint16_t i = 0; i < buffer_size; i++) {
    nbvi.add_packet(csi_test_data::baseline_packets()[i], pkt_size);
  }

  for (int wait = 0; wait < 500 && nbvi.is_calibrating(); wait++) {
    vTaskDelay(1);
  }

  TEST_ASSERT_TRUE_MESSAGE(calibration_success, "NBVI calibration failed");
  TEST_ASSERT_EQUAL_UINT8_MESSAGE(HT20_SELECTED_BAND_SIZE, calibrated_size, "NBVI band size mismatch");

  MVSDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, calibrated_threshold);
  detector.configure_lowpass(false);
  detector.configure_hampel(true);
  detector.set_cv_normalization(metrics.use_cv_normalization);

  metrics.selected_band_size = calibrated_size;
  std::copy(calibrated_band, calibrated_band + calibrated_size, metrics.selected_band.begin());
  metrics.adaptive_threshold = calibrated_threshold;
  metrics.baseline_eval_count = std::max(csi_test_data::num_baseline() - warmup, 0);
  metrics.movement_eval_count = std::max(csi_test_data::num_movement() - warmup, 0);

  for (int i = 0; i < csi_test_data::num_baseline(); i++) {
    detector.process_packet(csi_test_data::baseline_packets()[i], pkt_size, calibrated_band, calibrated_size);
    detector.update_state();
    if (i >= warmup && detector.get_state() == MotionState::MOTION) {
      metrics.fp++;
    }
  }

  for (int i = 0; i < csi_test_data::num_movement(); i++) {
    detector.process_packet(csi_test_data::movement_packets()[i], pkt_size, calibrated_band, calibrated_size);
    detector.update_state();
    if (i >= warmup) {
      if (detector.get_state() == MotionState::MOTION) {
        metrics.tp++;
      } else {
        metrics.fn++;
      }
    }
  }

  metrics.tn = std::max(metrics.baseline_eval_count - metrics.fp, 0);
  compute_derived_metrics(metrics);
  std::remove(buffer_path);
  return metrics;
}

void setUp(void) {}
void tearDown(void) {}

void test_long_recording_mvs_nbvi(void) {
  assert_dataset_metadata_is_valid();
  LongRunMetrics actual = evaluate_mvs_long_recording();
  print_metrics("MVS actual", actual);
  assert_mvs_metrics_are_valid(actual);
  record_result("mvs_nbvi", actual);
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
  RUN_TEST(test_long_recording_mvs_nbvi);
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
