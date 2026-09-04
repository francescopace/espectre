/*
 * ESPectre - Periodic Sensing Status Logger Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "espectre_log.h"
#include "periodic_sensing_status_logger.h"

#include <cstdarg>
#include <cstdio>
#include <string>

#include "esp_timer.h"

using namespace espectre;

namespace {

struct CapturedLog {
  int writes{0};
  std::string message;
};

bool log_enabled(void*, LogLevel level, const char*) {
  return level == LogLevel::INFO;
}

void log_write(void* context, LogLevel, const char*, int, const char* format,
               va_list args) {
  auto* capture = static_cast<CapturedLog*>(context);
  char message[512];
  std::vsnprintf(message, sizeof(message), format, args);
  capture->writes++;
  capture->message = message;
}

RuntimeSnapshot snapshot() {
  RuntimeSnapshot value;
  value.motion_state = MotionState::MOTION;
  value.movement_metric = 0.75f;
  value.threshold = 0.5f;
  value.link_channel = 6U;
  value.link_rssi_dbm = -48;
  return value;
}

}  // namespace

void setUp(void) {
  clear_log_sink();
  esp_timer_mock::reset(1000000, 0);
}

void tearDown(void) { clear_log_sink(); }

void test_null_tag_is_silent(void) {
  CapturedLog capture;
  TEST_ASSERT_TRUE(set_log_sink({&capture, &log_enabled, &log_write}));
  PeriodicSensingStatusLogger logger;

  logger.log_status(nullptr, snapshot(), 100U);

  TEST_ASSERT_EQUAL(0, capture.writes);
}

void test_diagnostics_snapshot_formats_all_rates_and_link_state(void) {
  CapturedLog capture;
  TEST_ASSERT_TRUE(set_log_sink({&capture, &log_enabled, &log_write}));
  PeriodicSensingStatusLogger logger;
  RuntimeDiagnosticsSample diagnostics;
  diagnostics.csi_admitted_pps = 99.8f;
  diagnostics.csi_accepted_pps = 101.2f;
  diagnostics.traffic_tx_pps = 100.9f;
  diagnostics.csi_missing_slots_pps = 1.1f;
  diagnostics.csi_excess_pps = 2.1f;
  diagnostics.csi_stale_pps = 3.1f;
  diagnostics.csi_out_of_order_pps = 4.1f;
  diagnostics.csi_occupancy_ratio = 0.805f;

  logger.log_status("runtime", snapshot(), 0U, &diagnostics);

  TEST_ASSERT_EQUAL(1, capture.writes);
  TEST_ASSERT_TRUE(capture.message.find("MOTION") != std::string::npos);
  TEST_ASSERT_TRUE(capture.message.find("csi:99/101 tx:100 occ:81%") != std::string::npos);
  TEST_ASSERT_TRUE(capture.message.find("miss:1 excess:2 stale:3 ooo:4") != std::string::npos);
  TEST_ASSERT_TRUE(capture.message.find("ch:6 rssi:-48") != std::string::npos);
}

void test_reset_restarts_fallback_packet_rate_window(void) {
  CapturedLog capture;
  TEST_ASSERT_TRUE(set_log_sink({&capture, &log_enabled, &log_write}));
  PeriodicSensingStatusLogger logger;
  logger.log_status("runtime", snapshot(), 50U);
  esp_timer_mock::advance(500000);
  logger.log_status("runtime", snapshot(), 50U);
  TEST_ASSERT_TRUE(capture.message.find("csi:100/0") != std::string::npos);

  logger.reset();
  esp_timer_mock::advance(500000);
  logger.log_status("runtime", snapshot(), 50U);
  TEST_ASSERT_TRUE(capture.message.find("csi:0/0") != std::string::npos);
}

void test_calibration_progress_is_clamped(void) {
  CapturedLog capture;
  TEST_ASSERT_TRUE(set_log_sink({&capture, &log_enabled, &log_write}));
  PeriodicSensingStatusLogger logger;
  RuntimeSnapshot value = snapshot();
  value.calibrating = true;
  value.calibration_packets = 200U;
  value.calibration_target_packets = 100U;

  logger.log_status("runtime", value, 0U);

  TEST_ASSERT_TRUE(capture.message.find("[####################]") != std::string::npos);
  TEST_ASSERT_TRUE(capture.message.find("CALIBRATING") != std::string::npos);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_null_tag_is_silent);
  RUN_TEST(test_diagnostics_snapshot_formats_all_rates_and_link_state);
  RUN_TEST(test_reset_restarts_fallback_packet_rate_window);
  RUN_TEST(test_calibration_progress_is_clamped);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return process();
}
#endif
