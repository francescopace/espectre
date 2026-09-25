/*
 * ESPectre - Runtime Detector Switch Unit Tests
 *
 * Unit tests for Runtime Detector Switch.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#define private public
#define protected public
#include "esp_idf_runtime.h"
#undef protected
#undef private

#include "esp_timer.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "csi_format.h"
#include "csi_traffic_fakes.h"
#include "nvs.h"
#include "runtime_detector_store.h"
#include "runtime_motion_hits_store.h"
#include "runtime_traffic_mode_store.h"
#include "runtime_config_utils.h"
#include "runtime_frontend_controller.h"
#include "frontend_ha_mqtt_helpers.h"

using namespace espectre;
using namespace espectre::test;

namespace {
bool reject_calibrator_allocation = false;
}

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  if (reject_calibrator_allocation && size == sizeof(StartupThresholdCalibrator)) return nullptr;
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

namespace {

class DetectorListener : public IRuntimeListener {
 public:
  void on_detector_changed(const RuntimeSnapshot &snapshot) override {
    detector_changes++;
    last_detector = snapshot.detector_name;
  }
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override {
    threshold_changes++;
    last_threshold = snapshot.threshold;
  }
  void on_calibration_started(const RuntimeSnapshot &) override { calibration_starts++; }
  void on_calibration_finished(const RuntimeSnapshot &, bool success) override {
    calibration_finishes++;
    last_calibration_success = success;
  }
  void on_runtime_fault(const char *) override { faults++; }

  void on_live_telemetry(float, float) override { live_updates++; }

  int live_updates{0};
  int detector_changes{0};
  int threshold_changes{0};
  int calibration_starts{0};
  int calibration_finishes{0};
  int faults{0};
  std::string last_detector;
  float last_threshold{0.0f};
  bool last_calibration_success{true};
};

bool accept_raw_packet(void *, const RawCsiPacketView &) { return true; }

// A real generator whose worker runs only when the test lets it exit.
unsigned traffic_sockets_opened = 0U;
int last_traffic_socket = -1;
unsigned traffic_delay_calls = 0U;

int open_test_socket(int, int, int) {
  ++traffic_sockets_opened;
  last_traffic_socket = open("/dev/null", O_RDWR);
  return last_traffic_socket;
}

bool socket_is_open(int sock) { return fcntl(sock, F_GETFD) >= 0; }

void count_traffic_delay() { ++traffic_delay_calls; }

void run_pending_traffic_task() {
  const auto function = g_freertos_task_mock.pending_function;
  g_freertos_task_mock.pending_function = nullptr;
  if (function != nullptr) function(g_freertos_task_mock.pending_argument);
}

void prepare_deferred_traffic_task() {
  traffic_sockets_opened = 0U;
  last_traffic_socket = -1;
  traffic_delay_calls = 0U;
  g_freertos_task_mock = {};
  g_freertos_task_mock.defer_execution = true;
  g_lwip_socket_mock_factory = open_test_socket;
}

// Shuts the runtime down and lets its last worker exit and be reaped, also
// when an assertion leaves the test early. The generator's destructor waits
// for its worker, and the mock runs that worker only when told to.
struct DeferredTrafficTaskScope {
  DeferredTrafficTaskScope(EspIdfRuntime &owner, TrafficGeneratorManager &traffic)
      : runtime(owner), generator(traffic) {}
  ~DeferredTrafficTaskScope() {
    runtime.shutdown();
    run_pending_traffic_task();
    generator.loop();
    g_freertos_task_mock = {};
    g_lwip_socket_mock_factory = nullptr;
  }
  EspIdfRuntime &runtime;
  TrafficGeneratorManager &generator;
};

// The same guarantee for a controller that owns its traffic generator.
struct ControllerTrafficTaskScope {
  explicit ControllerTrafficTaskScope(RuntimeFrontendController &owner) : controller(owner) {}
  ~ControllerTrafficTaskScope() {
    g_freertos_delay_hook = nullptr;
    controller.shutdown();
    run_pending_traffic_task();
    controller.loop();
    g_freertos_task_mock = {};
    g_lwip_socket_mock_factory = nullptr;
  }
  RuntimeFrontendController &controller;
};

void complete_csi_receive_path_refresh(EspIdfRuntime &runtime) {
  TEST_ASSERT_TRUE(runtime.csi_receive_path_refresh_in_progress_);
  wifi_event_sta_scan_done_t event{};
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_SCAN_DONE, &event);
  TEST_ASSERT_EQUAL(ESP_OK, runtime.wifi_lifecycle_.process_pending_events());
  TEST_ASSERT_FALSE(runtime.csi_receive_path_refresh_in_progress_);
}

#if defined(CONFIG_IDF_TARGET_ESP32C6) && CONFIG_IDF_TARGET_ESP32C6
constexpr bool kSupportsWifiRaw = false;
#else
constexpr bool kSupportsWifiRaw = true;
#endif

}  // namespace

void setUp(void) {
  reject_calibrator_allocation = false;
  nvs_mock_reset();
  esp_timer_mock::reset();
  esp_event_mock_reset();
  esp_netif_mock_reset();
  g_esp_netif_mock.ip_addr = 0U;
  esp_wifi_mock_reset();
}
void tearDown(void) {}

void test_runtime_healthy_startup_does_not_scan_even_when_csi_is_rejected(void) {
  for (const bool already_connected : {false, true}) {
    esp_event_mock_reset();
    esp_netif_mock_reset();
    esp_wifi_mock_reset();
    g_esp_netif_mock.ip_addr = already_connected ? 0x3701A8C0U : 0U;
    g_esp_wifi_mock.protocol_bitmap = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
    RuntimeConfig config;
    config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    DetectorListener listener;
    runtime.set_listener(&listener);
    TEST_ASSERT_TRUE(runtime.setup());
    if (already_connected) {
      runtime.loop();
    } else {
      esp_netif_ip_info_t ip{};
      ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
      runtime.on_wifi_connected_(ip);
    }
    TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_TRUE(generator.is_running());
    TEST_ASSERT_EQUAL(1, listener.calibration_starts);
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    // The callback counter includes packets that cannot enter the detector.
    g_esp_wifi_mock.csi_callback(g_esp_wifi_mock.csi_callback_context, nullptr);
    generator.send_successes = 1000U;
    esp_timer_mock::advance(6000000);
    runtime.loop();
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    TEST_ASSERT_FALSE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_EQUAL(0U, runtime.csi_pipeline_.accepted_packets_total());
    runtime.shutdown();
  }
}

void test_runtime_silent_startup_refreshes_once_then_resumes_on_failure_or_timeout(void) {
  for (const int completion : {0, 1, 2}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    RuntimeConfig config;
    config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    generator.send_successes = 1U;
    runtime.check_csi_receive_path_();
    generator.send_successes = 100U;
    esp_timer_mock::advance(999000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    TEST_ASSERT_TRUE(generator.is_running());
    esp_timer_mock::advance(1000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.scan_start_call_count);
    TEST_ASSERT_FALSE(generator.is_running());
    TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
    if (completion < 2) {
      wifi_event_sta_scan_done_t event{};
      event.status = completion;
      esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_SCAN_DONE, &event);
    } else {
      esp_timer_mock::advance(30001000);
    }
    runtime.loop();
    TEST_ASSERT_FALSE(runtime.csi_receive_path_refresh_in_progress_);
    TEST_ASSERT_TRUE(generator.is_running());
    TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_TRUE(runtime.is_calibrating());
    generator.send_successes += 1000U;
    esp_timer_mock::advance(60000000);
    runtime.loop();
    TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.scan_start_call_count);
    runtime.shutdown();
  }
}

void test_runtime_absent_or_stopped_traffic_does_not_trigger_a_refresh(void) {
  for (const auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::EXTERNAL}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    RuntimeConfig config;
    config.traffic_generator_mode = mode;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    esp_timer_mock::advance(60000000);
    runtime.loop();
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    generator.send_successes = 1U;
    ingress.packets_received = 1U;
    runtime.check_csi_receive_path_();
    esp_timer_mock::advance(500000);
    generator.send_successes = 20U;
    ingress.packets_received = 20U;
    runtime.check_csi_receive_path_();
    esp_timer_mock::advance(1000000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    generator.send_successes++;
    ingress.packets_received++;
    runtime.check_csi_receive_path_();
    esp_timer_mock::advance(1000000);
    generator.send_successes++;
    ingress.packets_received++;
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.scan_start_call_count);
    complete_csi_receive_path_refresh(runtime);
    runtime.shutdown();
  }
}

void test_runtime_external_wifi_stack_owns_recovery_scan_results(void) {
  for (const int completion : {0, 1, 2}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    RuntimeConfig config;
    config.wifi_scan_results_managed_externally = true;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    generator.send_successes = 1U;
    runtime.check_csi_receive_path_();
    generator.send_successes++;
    esp_timer_mock::advance(5000000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_TRUE(runtime.csi_receive_path_refresh_in_progress_);
    g_esp_wifi_mock.scan_ap_count = 6U;
    if (completion < 2) {
      wifi_event_sta_scan_done_t event{};
      event.status = completion;
      esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_SCAN_DONE, &event);
    } else {
      esp_timer_mock::advance(30001000);
    }
    runtime.loop();
    runtime.loop();
    TEST_ASSERT_FALSE(runtime.csi_receive_path_refresh_in_progress_);
    TEST_ASSERT_TRUE(generator.is_running());
    TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_FALSE(WiFiLifecycleManager::csi_receive_path_refresh_active());
    runtime.shutdown();
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.clear_ap_list_call_count);
    TEST_ASSERT_EQUAL(6U, g_esp_wifi_mock.scan_ap_count);
  }
}

void test_runtime_defers_busy_refresh_with_capture_running_and_a_bounded_request_window(void) {
  esp_timer_mock::reset(0, 0);
  RuntimeConfig config;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip);
  generator.send_successes = 1U;
  runtime.check_csi_receive_path_();
  g_esp_wifi_mock.set_scan_parameters_result = ESP_ERR_INVALID_STATE;
  generator.send_successes = 1000U;
  esp_timer_mock::advance(5000000);
  runtime.check_csi_receive_path_();
  TEST_ASSERT_TRUE(generator.is_running());
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
  esp_timer_mock::advance(10000000);
  runtime.check_csi_receive_path_();
  TEST_ASSERT_FALSE(runtime.csi_receive_path_check_pending_);
  g_esp_wifi_mock.set_scan_parameters_result = ESP_OK;
  esp_timer_mock::advance(1000000);
  runtime.check_csi_receive_path_();
  TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
  runtime.shutdown();
}

void test_runtime_profile_change_rechecks_the_receive_path(void) {
  if (!kSupportsWifiRaw) TEST_IGNORE_MESSAGE("wifi_raw is not supported on this target");
  for (const bool callbacks : {false, true}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    nvs_mock_reset();
    RuntimeConfig config;
    config.traffic_generator_mode = TrafficGeneratorMode::WIFI_RAW;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    TEST_ASSERT_EQUAL(CsiCaptureProfile::LLTF20, runtime.get_snapshot().csi_capture_profile);
    // LLTF20 delivers from the first arm, which closes the startup check.
    g_esp_wifi_mock.csi_callback(g_esp_wifi_mock.csi_callback_context, nullptr);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_FALSE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::PING));
    TEST_ASSERT_EQUAL(CsiCaptureProfile::HT20, runtime.get_snapshot().csi_capture_profile);
    TEST_ASSERT_TRUE(runtime.csi_receive_path_check_pending_);
    if (callbacks) g_esp_wifi_mock.csi_callback(g_esp_wifi_mock.csi_callback_context, nullptr);
    generator.send_successes++;
    runtime.check_csi_receive_path_();
    generator.send_successes += 100U;
    esp_timer_mock::advance(1000000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(callbacks ? 0 : 1, g_esp_wifi_mock.scan_start_call_count);
    if (!callbacks) complete_csi_receive_path_refresh(runtime);
    runtime.shutdown();
  }
}

void test_runtime_reports_a_receive_path_still_silent_after_its_refresh(void) {
  for (const bool callbacks : {false, true}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    RuntimeConfig config;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    DetectorListener listener;
    runtime.set_listener(&listener);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    generator.send_successes++;
    runtime.check_csi_receive_path_();
    generator.send_successes++;
    esp_timer_mock::advance(1000000);
    runtime.check_csi_receive_path_();
    complete_csi_receive_path_refresh(runtime);
    TEST_ASSERT_TRUE(generator.is_running());
    TEST_ASSERT_TRUE(runtime.csi_receive_path_check_pending_);
    if (callbacks) g_esp_wifi_mock.csi_callback(g_esp_wifi_mock.csi_callback_context, nullptr);
    generator.send_successes++;
    runtime.check_csi_receive_path_();
    generator.send_successes += 100U;
    esp_timer_mock::advance(999000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(0, listener.faults);
    esp_timer_mock::advance(1000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_EQUAL(callbacks ? 0 : 1, listener.faults);
    TEST_ASSERT_FALSE(runtime.csi_receive_path_check_pending_);
    // The fault is reported once, and the refresh is never repeated.
    generator.send_successes += 1000U;
    esp_timer_mock::advance(60000000);
    runtime.loop();
    TEST_ASSERT_EQUAL(callbacks ? 0 : 1, listener.faults);
    TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.scan_start_call_count);
    runtime.shutdown();
  }
}

void test_runtime_detector_switch_preserves_state_when_calibrator_allocation_fails(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_pipeline_.enabled_ = true;
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_TRUE(runtime.set_threshold(0.75f));
  const int threshold_changes = listener.threshold_changes;

  reject_calibrator_allocation = true;
  const bool changed = runtime.set_detection_algorithm(DetectionAlgorithm::LIGHTWEIGHT);
  reject_calibrator_allocation = false;
  TEST_ASSERT_FALSE(changed);
  TEST_ASSERT_EQUAL_STRING("high_accuracy", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(0.75f, runtime.get_snapshot().threshold);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(0, listener.detector_changes);
  TEST_ASSERT_EQUAL(0, listener.calibration_starts);
  TEST_ASSERT_EQUAL(threshold_changes, listener.threshold_changes);
  TEST_ASSERT_EQUAL(1, listener.faults);
  DetectionAlgorithm stored = DetectionAlgorithm::LIGHTWEIGHT;
  bool has_stored = false;
  TEST_ASSERT_EQUAL(ESP_OK, load_runtime_detection_algorithm(&stored, &has_stored));
  TEST_ASSERT_TRUE(has_stored);
  TEST_ASSERT_TRUE(stored == DetectionAlgorithm::HIGH_ACCURACY);
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::LIGHTWEIGHT));
  TEST_ASSERT_TRUE(runtime.is_calibrating());
}

void test_runtime_calibration_can_restart_from_completion_callback(void) {
  class RetryListener : public DetectorListener {
   public:
    EspIdfRuntime *runtime{nullptr};
    bool restart_on_threshold{false};
    bool restarted{false};
    void on_threshold_changed(const RuntimeSnapshot &snapshot) override {
      DetectorListener::on_threshold_changed(snapshot);
      if (restart_on_threshold && !restarted) restarted = runtime->trigger_recalibration();
    }
    void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override {
      TEST_ASSERT_FALSE(snapshot.calibrating);
      DetectorListener::on_calibration_finished(snapshot, success);
      if (!restart_on_threshold && !restarted) restarted = runtime->trigger_recalibration();
    }
  };
  for (bool success : {false, true}) {
    for (bool restart_on_threshold : {false, true}) {
      if (restart_on_threshold && !success) continue;
      EspIdfRuntime runtime(RuntimeConfig{});
      RetryListener listener;
      listener.runtime = &runtime;
      listener.restart_on_threshold = restart_on_threshold;
      runtime.set_listener(&listener);
      TEST_ASSERT_TRUE(runtime.configure_detector_());
      runtime.csi_pipeline_.init(runtime.detector_.get());
      TEST_ASSERT_TRUE(runtime.trigger_recalibration());
      runtime.finish_threshold_calibration_(success);
      TEST_ASSERT_TRUE(listener.restarted);
      TEST_ASSERT_TRUE(runtime.is_calibrating());
      int8_t csi[HT20_CSI_LEN]{};
      TEST_ASSERT_TRUE(runtime.handle_threshold_calibration_packet_(
          csi, sizeof(csi), -50, false, 1U, false));
      runtime.finish_threshold_calibration_(false);
      TEST_ASSERT_FALSE(runtime.is_calibrating());
      TEST_ASSERT_EQUAL(2, listener.calibration_starts);
      TEST_ASSERT_EQUAL(2, listener.calibration_finishes);
    }
  }
}

void test_runtime_calibration_allocation_failure_does_not_emit_started(void) {
  EspIdfRuntime runtime(RuntimeConfig{});
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  const auto before = runtime.get_snapshot();
  reject_calibrator_allocation = true;
  const bool started = runtime.trigger_recalibration();
  reject_calibrator_allocation = false;
  TEST_ASSERT_FALSE(started);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL_FLOAT(before.threshold, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL(0, listener.calibration_starts);
  TEST_ASSERT_EQUAL(1, listener.faults);
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
}

void test_runtime_detector_switch_updates_pipeline_threshold_and_calibration(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL_STRING("high_accuracy", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.detector_changes);
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);

  runtime.csi_pipeline_.enabled_ = true;
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::LIGHTWEIGHT));
  TEST_ASSERT_EQUAL_STRING("lightweight", runtime.get_snapshot().detector_name);
  TEST_ASSERT_EQUAL_FLOAT(LIGHTWEIGHT_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);

  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  TEST_ASSERT_FALSE(listener.last_calibration_success);

  TEST_ASSERT_TRUE(runtime.set_threshold(0.75f));
  TEST_ASSERT_EQUAL_FLOAT(0.75f, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, runtime.get_snapshot().threshold);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
}

void test_runtime_readiness_requires_valid_recent_csi_and_recovers_after_quality_gap(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  runtime.set_live_telemetry_enabled(true);
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  // This test isolates hardware admission and availability. Frame provenance
  // has its own suite, so the synthetic callback needs no fabricated IP header.
  runtime.csi_pipeline_.traffic_filter_configured_ = false;
  int8_t payload[128];
  std::fill_n(payload, 128, int8_t{20});
  for (uint8_t bin : HT20_CENTERED_ONLY_NULL_BINS) {
    payload[bin * 2] = payload[bin * 2 + 1] = 0;
  }
  wifi_csi_info_t info{};
  info.buf = payload;
  info.len = sizeof(payload);
  info.rx_ctrl.sig_mode = 1;
  info.rx_ctrl.channel = 1;
  info.rx_ctrl.rssi = -40;
  esp_timer_mock::reset(10000000, 0);
  const auto feed = [&](unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
      esp_timer_mock::advance(10000);
      info.rx_ctrl.timestamp = static_cast<uint32_t>(esp_timer_mock::time_us);
      runtime.csi_pipeline_.capture_service_.process_packet(&info);
      runtime.loop();
    }
  };
  feed(125);
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);
  const int live_updates = listener.live_updates;
  TEST_ASSERT_TRUE(live_updates > 0);
  const float last_score = runtime.get_snapshot().movement_metric;
  info.rx_ctrl.rx_state = 1;
  feed(110);
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(live_updates, listener.live_updates);
  TEST_ASSERT_EQUAL_FLOAT(last_score, runtime.get_snapshot().movement_metric);
  TEST_ASSERT_EQUAL(110, runtime.csi_pipeline_.capture_service_.rx_error_packets());
  info.rx_ctrl.rx_state = 0;
  feed(125);
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_TRUE(listener.live_updates > live_updates);
}

void test_runtime_readiness_holds_brief_coverage_dips_and_drops_on_real_loss(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  runtime.csi_pipeline_.traffic_filter_configured_ = false;
  int8_t payload[128];
  std::fill_n(payload, 128, int8_t{20});
  for (uint8_t bin : HT20_CENTERED_ONLY_NULL_BINS) {
    payload[bin * 2] = payload[bin * 2 + 1] = 0;
  }
  wifi_csi_info_t info{};
  info.buf = payload;
  info.len = sizeof(payload);
  info.rx_ctrl.sig_mode = 1;
  info.rx_ctrl.channel = 1;
  info.rx_ctrl.rssi = -40;
  esp_timer_mock::reset(10000000, 0);
  unsigned unready_loops = 0U;
  unsigned held_heartbeats = 0U;
  const auto feed = [&](unsigned count, uint32_t spacing_us) {
    for (unsigned i = 0; i < count; ++i) {
      esp_timer_mock::advance(spacing_us);
      info.rx_ctrl.timestamp = static_cast<uint32_t>(esp_timer_mock::time_us);
      const uint32_t heartbeat_ms = runtime.csi_pipeline_.last_heartbeat_ms_;
      const float metric = runtime.snapshot_.movement_metric;
      runtime.csi_pipeline_.capture_service_.process_packet(&info);
      runtime.loop();
      const RuntimeSnapshot snapshot = runtime.get_snapshot();
      if (!snapshot.ready_to_publish) {
        unready_loops++;
      } else if (!runtime.detector_->is_ready() &&
                 runtime.csi_pipeline_.last_heartbeat_ms_ != heartbeat_ms) {
        // A heartbeat inside a held dip keeps the last metric instead of
        // publishing the one the detector cleared.
        held_heartbeats++;
        TEST_ASSERT_EQUAL_FLOAT(metric, snapshot.movement_metric);
      }
    }
  };
  feed(125, 10000U);
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);

  // A 400 ms hole leaves the window at 60% valid slots, under the 70% floor,
  // for most of a second while input keeps arriving: readiness holds.
  unready_loops = 0U;
  esp_timer_mock::advance(400000);
  runtime.loop();
  // Stand in for the last published metric, which this flat input keeps at 0.
  runtime.snapshot_.movement_metric = 0.5f;
  feed(120, 10000U);
  TEST_ASSERT_EQUAL(0, unready_loops);
  TEST_ASSERT_TRUE(held_heartbeats > 0U);
  TEST_ASSERT_TRUE(runtime.detector_->is_ready());

  // Coverage that stays under the floor clears readiness one window after it
  // first dips: half rate reaches the floor after 600 ms, then 1 s of grace.
  unready_loops = 0U;
  feed(90, 20000U);
  TEST_ASSERT_TRUE(unready_loops > 0U && unready_loops <= 15U);
  TEST_ASSERT_FALSE(runtime.detector_->is_ready());
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_TRUE(runtime.readiness_gate_.reason() == SensingReadinessReason::LOW_COVERAGE);
  feed(125, 10000U);
  TEST_ASSERT_TRUE(runtime.get_snapshot().ready_to_publish);

  // Input that stops clears readiness once it is one window old.
  esp_timer_mock::advance(1000000);
  runtime.loop();
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_TRUE(runtime.readiness_gate_.reason() == SensingReadinessReason::INPUT_STALE);
  runtime.shutdown();
}

void test_runtime_detector_configuration_preserves_the_requested_threshold(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  config.threshold = 0.73f;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);

  TEST_ASSERT_TRUE(runtime.setup());
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.config_.threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.detector_->get_threshold());

  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;
  ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.detector_->get_threshold());

  TEST_ASSERT_TRUE(runtime.set_threshold(0.68f));
  runtime.on_wifi_disconnected_();
  runtime.on_wifi_connected_(ip_info);
  wifi_event_sta_scan_done_t scan_done{};
  esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_SCAN_DONE, &scan_done);
  TEST_ASSERT_EQUAL(ESP_OK, runtime.wifi_lifecycle_.process_pending_events());
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL_FLOAT(0.68f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.68f, runtime.detector_->get_threshold());

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_EQUAL_FLOAT(0.68f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.68f, runtime.detector_->get_threshold());
  runtime.shutdown();
}

void test_runtime_calibration_consumes_evaluations_resets_on_gaps_and_finishes(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;
  ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  const uint16_t target = runtime.get_snapshot().calibration_target_packets;
  int8_t csi[HT20_CSI_LEN];
  std::fill(std::begin(csi), std::end(csi), 8);
  TEST_ASSERT_FALSE(EspIdfRuntime::threshold_calibration_packet_callback_(
      nullptr, csi, sizeof(csi), -50, true, 1U, false));
  TEST_ASSERT_TRUE(runtime.handle_threshold_calibration_packet_(csi, sizeof(csi), -50, false, 1U, false));
  TEST_ASSERT_EQUAL(0, runtime.get_snapshot().calibration_packets);
  // An evaluation cannot count calibration packets before the detector is ready.
  TEST_ASSERT_TRUE(runtime.handle_threshold_calibration_packet_(csi, sizeof(csi), -50, true, 1U, false));
  TEST_ASSERT_EQUAL(0, runtime.get_snapshot().calibration_packets);
  for (uint32_t index = 0; index < target && runtime.get_snapshot().calibration_packets < 2U; ++index) {
    TEST_ASSERT_TRUE(EspIdfRuntime::threshold_calibration_packet_callback_(
        &runtime, csi, sizeof(csi), -50, true, 1U, false));
  }
  TEST_ASSERT_EQUAL(2, runtime.get_snapshot().calibration_packets);
  TEST_ASSERT_TRUE(runtime.handle_threshold_calibration_packet_(csi, sizeof(csi), -50, true, 1U, true));
  TEST_ASSERT_EQUAL(1, runtime.get_snapshot().calibration_packets);
  for (uint32_t index = 0; index < target && runtime.threshold_calibration_active_.load(); ++index) {
    TEST_ASSERT_TRUE(runtime.handle_threshold_calibration_packet_(csi, sizeof(csi), -50, true, 1U, false));
  }
  TEST_ASSERT_FALSE(runtime.threshold_calibration_active_.load());
  TEST_ASSERT_EQUAL(target, runtime.get_snapshot().calibration_packets);
  runtime.loop();
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
  TEST_ASSERT_EQUAL_FLOAT(runtime.detector_->get_threshold(), runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(runtime.get_snapshot().threshold, runtime.get_snapshot().startup_threshold);
  TEST_ASSERT_EQUAL(0, runtime.get_snapshot().calibration_target_packets);
  TEST_ASSERT_FALSE(runtime.handle_threshold_calibration_packet_(csi, sizeof(csi), -50, true, 1U, false));
  runtime.loop();
  TEST_ASSERT_EQUAL(1, listener.calibration_finishes);
  runtime.shutdown();
}

// One calibration evaluation of `kCalibrationStep` packets: flat CSI is a quiet
// room, and a slowly swinging spread across subcarriers is someone moving.
constexpr uint32_t kCalibrationStep = 25U;

void feed_calibration_evaluation(EspIdfRuntime &runtime, bool motion, uint32_t &packet_index) {
  int8_t csi[HT20_CSI_LEN];
  for (uint32_t step = 0U; step < kCalibrationStep; ++step, ++packet_index) {
    const float swing = motion ? 10.0f * (1.0f + std::sin(0.15f * static_cast<float>(packet_index))) : 0.0f;
    for (size_t index = 0U; index < sizeof(csi); ++index) {
      const float spread = ((index / 2U) % 2U == 0U) ? swing : -swing;
      csi[index] = static_cast<int8_t>(index % 2U == 0U ? 40.0f + spread : 0.0f);
    }
    (void) runtime.handle_threshold_calibration_packet_(
        csi, sizeof(csi), -50, step + 1U == kCalibrationStep, kCalibrationStep, false);
  }
}

float run_quiet_startup_calibration(EspIdfRuntime &runtime, uint32_t &packet_index) {
  for (uint32_t guard = 0U; guard < 400U && runtime.threshold_calibration_active_.load(); ++guard) {
    feed_calibration_evaluation(runtime, false, packet_index);
  }
  runtime.loop();
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  return runtime.detector_->get_threshold();
}

void test_runtime_recalibration_during_motion_keeps_the_live_threshold(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;
  ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  // Startup calibration has no trusted threshold; the detector ceiling guards it.
  TEST_ASSERT_TRUE(runtime.calibration_motion_guard_.active());
  TEST_ASSERT_EQUAL_FLOAT(runtime.detector_->calibration_motion_ceiling(),
                          runtime.calibration_motion_guard_.reference_threshold());
  uint32_t packet_index = 0U;
  const float live_threshold = run_quiet_startup_calibration(runtime, packet_index);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
  TEST_ASSERT_TRUE(live_threshold < 0.5f);

  // Someone keeps moving after pressing Recalibrate: every window restarts
  // until no full window fits in the budget, then the live threshold stays.
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
  const uint32_t target = runtime.get_snapshot().calibration_target_packets;
  uint32_t evaluations = 0U;
  for (; evaluations < 1000U && runtime.threshold_calibration_active_.load(); ++evaluations) {
    feed_calibration_evaluation(runtime, true, packet_index);
  }
  runtime.loop();
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL_FLOAT(live_threshold, runtime.detector_->get_threshold());
  TEST_ASSERT_FALSE(listener.last_calibration_success);
  TEST_ASSERT_TRUE(evaluations * kCalibrationStep <= CALIBRATION_MOTION_BUDGET_WINDOWS * target);
  TEST_ASSERT_EQUAL_FLOAT(live_threshold, runtime.get_snapshot().threshold);

  // Motion at the start restarts the window; the quiet remainder calibrates.
  TEST_ASSERT_TRUE(runtime.trigger_recalibration());
  for (uint32_t index = 0U; index < 12U; ++index) {
    feed_calibration_evaluation(runtime, true, packet_index);
  }
  TEST_ASSERT_TRUE(runtime.calibration_motion_guard_.restarts() > 0U);
  const float recalibrated = run_quiet_startup_calibration(runtime, packet_index);
  TEST_ASSERT_TRUE(listener.last_calibration_success);
  TEST_ASSERT_FLOAT_WITHIN(1e-4f, live_threshold, recalibrated);
  TEST_ASSERT_FALSE(runtime.calibration_motion_guard_.active());

  // A detector switch drops the live reference; the ceiling still guards.
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::LIGHTWEIGHT));
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_EQUAL_FLOAT(runtime.detector_->calibration_motion_ceiling(),
                          runtime.calibration_motion_guard_.reference_threshold());
  runtime.shutdown();
}

void test_runtime_startup_calibration_during_motion_keeps_the_default(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;
  ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  const float default_threshold = runtime.detector_->get_threshold();
  const uint32_t target = runtime.get_snapshot().calibration_target_packets;

  // Someone keeps moving through boot: no window stays under the ceiling, so
  // the calibration fails instead of learning a threshold near 1.0.
  uint32_t packet_index = 0U;
  uint32_t evaluations = 0U;
  for (; evaluations < 1000U && runtime.threshold_calibration_active_.load(); ++evaluations) {
    feed_calibration_evaluation(runtime, true, packet_index);
  }
  runtime.loop();
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_FALSE(listener.last_calibration_success);
  TEST_ASSERT_TRUE(evaluations * kCalibrationStep <=
                   CALIBRATION_MOTION_BUDGET_WINDOWS * target);
  TEST_ASSERT_EQUAL_FLOAT(default_threshold, runtime.detector_->get_threshold());
  TEST_ASSERT_EQUAL_FLOAT(default_threshold, runtime.get_snapshot().threshold);
  runtime.shutdown();
}

void test_runtime_rejects_invalid_detector_geometry_before_starting_services(void) {
  for (int field = 0; field < 3; ++field) {
    RuntimeConfig config;
    if (field == 0) config.csi_target_pps = RUNTIME_CSI_TARGET_PPS_MIN - 1U;
    if (field == 1) config.window_size_ms = RUNTIME_WINDOW_SIZE_MS_MAX + 1U;
    if (field == 2) config.threshold = -1.0f;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    DetectorListener listener;
    runtime.set_listener(&listener);
    TEST_ASSERT_FALSE(runtime.setup());
    TEST_ASSERT_EQUAL(1, listener.faults);
    TEST_ASSERT_EQUAL(0U, generator.start_calls);
    TEST_ASSERT_EQUAL(0U, ingress.start_calls);
    runtime.shutdown();
  }
}

void test_runtime_rejects_invalid_or_unpersisted_controls_without_changing_config(void) {
  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  TEST_ASSERT_TRUE(runtime.setup());
  TEST_ASSERT_FALSE(runtime.set_motion_hits(RUNTIME_MOTION_HITS_MIN - 1U, config.motion_off_hits));
  TEST_ASSERT_FALSE(runtime.set_motion_hits(config.motion_on_hits, RUNTIME_MOTION_HITS_MAX + 1U));
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(static_cast<TrafficGeneratorMode>(255)));
  TEST_ASSERT_FALSE(runtime.set_detection_algorithm(static_cast<DetectionAlgorithm>(255)));
  TEST_ASSERT_FALSE(runtime.set_threshold(-1.0f));
  nvs_mock_set_open_result(ESP_FAIL);
  TEST_ASSERT_FALSE(runtime.set_motion_hits(7U, 5U));
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::DNS));
  TEST_ASSERT_FALSE(runtime.set_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_TRUE(runtime.effective_config().traffic_generator_mode == config.traffic_generator_mode);
  TEST_ASSERT_TRUE(runtime.effective_config().detection_algorithm == config.detection_algorithm);
  TEST_ASSERT_EQUAL(config.motion_on_hits, runtime.effective_config().motion_on_hits);
  TEST_ASSERT_EQUAL(config.motion_off_hits, runtime.effective_config().motion_off_hits);
  runtime.shutdown();
}

void test_runtime_restores_internal_traffic_when_external_source_cannot_start(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = ip_info.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip_info);
  ingress.start_result = false;
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));
  TEST_ASSERT_TRUE(runtime.effective_config().traffic_generator_mode == TrafficGeneratorMode::PING);
  TEST_ASSERT_TRUE(generator.is_running());
  TEST_ASSERT_FALSE(ingress.is_running());
  TEST_ASSERT_EQUAL(1, listener.faults);
  generator.start_result = false;
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::DNS));
  TEST_ASSERT_TRUE(runtime.effective_config().traffic_generator_mode == config.traffic_generator_mode);
  TEST_ASSERT_FALSE(generator.is_running());
  TEST_ASSERT_TRUE(listener.faults > 1);
  runtime.shutdown();
}

void test_runtime_traffic_updates_roll_back_when_persistence_fails(void) {
  RuntimeConfig config;
  EspIdfRuntime runtime(config);
  nvs_mock_set_open_result(ESP_FAIL);

  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));
  TEST_ASSERT_TRUE(runtime.config_.traffic_generator_mode == TrafficGeneratorMode::PING);
  TEST_ASSERT_FALSE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::DNS_TCP));
  TEST_ASSERT_TRUE(runtime.config_.traffic_generator_mode == TrafficGeneratorMode::PING);
}

void test_runtime_detector_adaptation_emits_threshold_changed_without_live_telemetry(void) {
  RuntimeConfig config;
  config.threshold = 0.80f;
  EspIdfRuntime runtime(config);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.set_live_telemetry_enabled(false);

  TEST_ASSERT_TRUE(runtime.detector_->set_threshold(0.42f));
  runtime.loop();
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, listener.last_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, runtime.get_snapshot().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.42f, runtime.config_.threshold);

  runtime.loop();
  TEST_ASSERT_EQUAL(1, listener.threshold_changes);
}

void test_runtime_motion_hits_runtime_updates_pipeline_and_persists(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  EspIdfRuntime runtime(config);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());

  TEST_ASSERT_TRUE(runtime.set_motion_hits(8U, 6U));
  TEST_ASSERT_EQUAL_UINT8(8U, runtime.csi_pipeline_.motion_on_hits_);
  TEST_ASSERT_EQUAL_UINT8(6U, runtime.csi_pipeline_.motion_off_hits_);

  uint8_t saved_motion_on_hits = 0U;
  uint8_t saved_motion_off_hits = 0U;
  bool has_saved_value = false;
  TEST_ASSERT_EQUAL(ESP_OK,
                    load_runtime_motion_hits(&saved_motion_on_hits, &saved_motion_off_hits, &has_saved_value));
  TEST_ASSERT_TRUE(has_saved_value);
  TEST_ASSERT_EQUAL_UINT8(8U, saved_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(6U, saved_motion_off_hits);
}

void test_runtime_setup_loads_all_persisted_runtime_controls(void) {
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_motion_hits(8U, 6U));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));

  RuntimeConfig config;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.motion_on_hits = 4U;
  config.motion_off_hits = 3U;
  config.traffic_generator_mode = TrafficGeneratorMode::PING;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);

  TEST_ASSERT_TRUE(runtime.setup());
  const RuntimeConfig &effective = runtime.effective_config();
  TEST_ASSERT_TRUE(effective.detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY);
  TEST_ASSERT_EQUAL_FLOAT(HIGH_ACCURACY_DEFAULT_THRESHOLD, effective.threshold);
  TEST_ASSERT_EQUAL_UINT8(8U, effective.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(6U, effective.motion_off_hits);
  TEST_ASSERT_TRUE(effective.traffic_generator_mode == TrafficGeneratorMode::EXTERNAL);
  TEST_ASSERT_TRUE(runtime.csi_traffic_service_.mode() == TrafficGeneratorMode::EXTERNAL);
  runtime.shutdown();

  // A persisted detector matching the configured one keeps the configured threshold.
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  config.threshold = 0.73f;
  EspIdfRuntime same_detector_runtime(config, traffic_generator, traffic_ingress);
  TEST_ASSERT_TRUE(same_detector_runtime.setup());
  TEST_ASSERT_EQUAL_FLOAT(0.73f, same_detector_runtime.effective_config().threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.73f, same_detector_runtime.detector_->get_threshold());
  same_detector_runtime.shutdown();
}

void test_runtime_without_persistence_ignores_and_never_writes_saved_controls(void) {
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_motion_hits(8U, 6U));
  TEST_ASSERT_EQUAL(ESP_OK, save_runtime_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));

  RuntimeConfig config;
  config.persist_runtime_overrides = false;
  config.runtime_detector_selection_enabled = true;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.motion_on_hits = 4U;
  config.motion_off_hits = 3U;
  config.traffic_generator_mode = TrafficGeneratorMode::PING;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);

  TEST_ASSERT_TRUE(runtime.setup());
  const RuntimeConfig &effective = runtime.effective_config();
  TEST_ASSERT_TRUE(effective.detection_algorithm == DetectionAlgorithm::LIGHTWEIGHT);
  TEST_ASSERT_EQUAL_UINT8(4U, effective.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(3U, effective.motion_off_hits);
  TEST_ASSERT_TRUE(effective.traffic_generator_mode == TrafficGeneratorMode::PING);

  // Controls must not touch storage: they succeed even when NVS cannot open.
  nvs_mock_set_open_result(ESP_FAIL);
  TEST_ASSERT_TRUE(runtime.set_motion_hits(9U, 7U));
  TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::DNS));
  TEST_ASSERT_TRUE(runtime.set_detection_algorithm(DetectionAlgorithm::HIGH_ACCURACY));
  TEST_ASSERT_EQUAL_UINT8(9U, effective.motion_on_hits);
  TEST_ASSERT_TRUE(effective.traffic_generator_mode == TrafficGeneratorMode::DNS);
  TEST_ASSERT_TRUE(effective.detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY);
  runtime.shutdown();

  nvs_mock_set_open_result(ESP_OK);
  uint8_t saved_on = 0U;
  uint8_t saved_off = 0U;
  bool has_saved = false;
  TEST_ASSERT_EQUAL(ESP_OK, load_runtime_motion_hits(&saved_on, &saved_off, &has_saved));
  TEST_ASSERT_EQUAL_UINT8(8U, saved_on);
  TEST_ASSERT_EQUAL_UINT8(6U, saved_off);
}

void test_runtime_diagnostics_cache_current_wifi_association(void) {
  esp_wifi_mock_reset();
  RuntimeConfig config;
  EspIdfRuntime runtime(config);
  runtime.services_armed_ = false;
  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;

  runtime.on_wifi_connected_(ip_info);

  RuntimeDiagnosticsSnapshot diagnostics = runtime.get_diagnostics();
  TEST_ASSERT_EQUAL_UINT8(6U, diagnostics.link.channel);
  TEST_ASSERT_EQUAL_INT8(-55, diagnostics.link.rssi_dbm);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.get_ap_info_call_count);

  runtime.csi_pipeline_.last_channel_ = 11U;
  runtime.csi_pipeline_.last_rssi_dbm_ = -42;
  runtime.refresh_wifi_association_from_csi_();
  runtime.csi_pipeline_.last_channel_ = 0U;
  runtime.csi_pipeline_.last_rssi_dbm_ = INT8_MIN;

  diagnostics = runtime.get_diagnostics();
  TEST_ASSERT_EQUAL_UINT8(11U, diagnostics.link.channel);
  TEST_ASSERT_EQUAL_INT8(-42, diagnostics.link.rssi_dbm);
  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.get_ap_info_call_count);

  runtime.on_wifi_disconnected_();
  diagnostics = runtime.get_diagnostics();
  TEST_ASSERT_EQUAL_UINT8(0U, diagnostics.link.channel);
  TEST_ASSERT_EQUAL_INT8(INT8_MIN, diagnostics.link.rssi_dbm);
}

void test_runtime_channel_change_rearms_csi_and_restarts_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());

  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  runtime.on_csi_channel_changed_(8U, 10U);

  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(MotionState::IDLE, runtime.get_snapshot().motion_state);
  TEST_ASSERT_EQUAL(1, listener.calibration_starts);
  runtime.csi_traffic_service_.stop();
}

void test_runtime_services_armed_preserves_wifi_ip_and_restarts_capture(void) {
  RuntimeConfig config;
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    runtime.set_services_armed(false);
    TEST_ASSERT_TRUE(runtime.setup());
    runtime.on_wifi_connected_(ip);
    TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_EQUAL(cycle, generator.start_calls);
    runtime.set_services_armed(true);
    TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_TRUE(runtime.is_calibrating());
    TEST_ASSERT_EQUAL(cycle + 1U, generator.start_calls);
    TEST_ASSERT_TRUE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    runtime.set_services_armed(false);
    TEST_ASSERT_FALSE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_TRUE(runtime.wifi_ready_);
    TEST_ASSERT_EQUAL(ip.ip.addr, runtime.wifi_ip_info_.ip.addr);
    runtime.shutdown();
  }
}

void test_runtime_sensing_reasserts_promiscuous_disabled_before_capture(void) {
  esp_wifi_mock_reset();
  g_esp_wifi_mock.promiscuous = true;

  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));

  esp_netif_ip_info_t ip_info{};
  ip_info.ip.addr = 0x0101A8C0U;
  ip_info.gw.addr = 0x0101A8C0U;
  runtime.start_sensing_services_(ip_info);

  TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.set_promiscuous_call_count);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.last_promiscuous);
  TEST_ASSERT_FALSE(g_esp_wifi_mock.promiscuous);
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  runtime.csi_traffic_service_.stop();
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.disable());
}

void test_runtime_disconnect_or_disarm_cancels_refresh_and_discards_queued_completion(void) {
  for (const bool disconnect : {false, true}) {
    esp_timer_mock::reset(0, 0);
    esp_event_mock_reset();
    esp_wifi_mock_reset();
    RuntimeConfig config;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip{};
    ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip);
    generator.send_successes = 1U;
    runtime.check_csi_receive_path_();
    generator.send_successes++;
    esp_timer_mock::advance(5000000);
    runtime.check_csi_receive_path_();
    TEST_ASSERT_TRUE(runtime.csi_receive_path_refresh_in_progress_);
    TEST_ASSERT_FALSE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
    wifi_event_sta_scan_done_t stale{};
    esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_SCAN_DONE, &stale);
    if (disconnect) runtime.on_wifi_disconnected_();
    else runtime.set_services_armed(false);
    runtime.loop();
    TEST_ASSERT_FALSE(runtime.csi_receive_path_refresh_in_progress_);
    TEST_ASSERT_FALSE(generator.is_running());
    TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
    if (disconnect) runtime.on_wifi_connected_(ip);
    else runtime.set_services_armed(true);
    TEST_ASSERT_TRUE(generator.is_running());
    TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
    TEST_ASSERT_TRUE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_EQUAL(1, g_esp_wifi_mock.scan_start_call_count);
    runtime.shutdown();
  }
}

void test_runtime_raw_collection_restores_armed_and_disarmed_sensing(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::LIGHTWEIGHT;
  config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  DetectorListener listener;
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  runtime.capabilities_.supports_raw_csi = true;
  runtime.setup_complete_ = true;
  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());
  runtime.snapshot_.calibrating = true;
  runtime.snapshot_.ready_to_publish = true;

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, runtime.operation_state());
  TEST_ASSERT_EQUAL(TrafficGeneratorMode::EXTERNAL, runtime.csi_traffic_service_.mode());
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.snapshot_.calibrating);
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);
  TEST_ASSERT_FALSE(runtime.set_threshold(0.5f));

  runtime.set_services_armed(false);
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, runtime.operation_state());
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_EQUAL(TrafficGeneratorMode::EXTERNAL, runtime.csi_traffic_service_.mode());
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);

  const int calibration_starts = listener.calibration_starts;
  const int calibration_finishes = listener.calibration_finishes;
  TEST_ASSERT_FALSE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.raw_capture_active());
  TEST_ASSERT_FALSE(runtime.csi_traffic_service_.is_running());
  TEST_ASSERT_EQUAL(0U, traffic_generator.start_calls);
  TEST_ASSERT_EQUAL(0U, traffic_ingress.start_calls);
  TEST_ASSERT_FALSE(runtime.snapshot_.calibrating);
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);
  TEST_ASSERT_EQUAL(calibration_starts, listener.calibration_starts);
  TEST_ASSERT_EQUAL(calibration_finishes, listener.calibration_finishes);

  runtime.set_services_armed(true);
  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  runtime.set_services_armed(false);
  runtime.set_services_armed(true);
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, runtime.operation_state());
  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_TRUE(runtime.snapshot_.calibrating);
  TEST_ASSERT_TRUE(runtime.snapshot_.ready_to_publish);
  runtime.csi_traffic_service_.stop();
}

void test_runtime_disables_capture_only_after_the_traffic_task_exits(void) {
  prepare_deferred_traffic_task();
  RuntimeConfig config;
  TrafficGeneratorManager generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DeferredTrafficTaskScope scope(runtime, generator);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_TRUE(generator.has_live_worker());

  // Raw collection keeps traffic running through a disarm. Running traffic
  // must not park station radio work, or a reconfigure would never start.
  runtime.capabilities_.supports_raw_csi = true;
  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  runtime.set_services_armed(false);
  TEST_ASSERT_TRUE(generator.has_live_worker());
  TEST_ASSERT_TRUE(runtime.traffic_allows_radio_work());

  TEST_ASSERT_TRUE(runtime.stop_raw_collection(RawCsiStopReason::REQUESTED));
  TEST_ASSERT_FALSE(generator.has_live_worker());
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_FALSE(runtime.traffic_allows_radio_work());
  runtime.hold_pending_traffic_restart(true);
  runtime.loop();
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_FALSE(runtime.traffic_allows_radio_work());

  // A held restart does not hold the disable that the radio work waits for.
  run_pending_traffic_task();
  runtime.loop();
  TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_TRUE(runtime.traffic_allows_radio_work());
  runtime.hold_pending_traffic_restart(false);
  runtime.loop();
  TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
}

void test_runtime_channel_change_rearms_after_the_traffic_task_and_held_radio_work(void) {
  prepare_deferred_traffic_task();
  RuntimeConfig config;
  TrafficGeneratorManager generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DeferredTrafficTaskScope scope(runtime, generator);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip);
  TEST_ASSERT_TRUE(generator.has_live_worker());

  runtime.on_csi_channel_changed_(6U, 11U);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_FALSE(runtime.traffic_allows_radio_work());

  runtime.hold_pending_traffic_restart(true);
  run_pending_traffic_task();
  runtime.loop();
  TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_TRUE(runtime.traffic_allows_radio_work());
  runtime.loop();
  TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);

  runtime.hold_pending_traffic_restart(false);
  runtime.loop();
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(2U, g_freertos_task_mock.create_calls);
  TEST_ASSERT_TRUE(generator.has_live_worker());
  TEST_ASSERT_TRUE(runtime.is_calibrating());
}

void test_runtime_reconnect_while_traffic_stops_rearms_capture_after_its_disable(void) {
  prepare_deferred_traffic_task();
  RuntimeConfig config;
  TrafficGeneratorManager generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DeferredTrafficTaskScope scope(runtime, generator);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip);
  const int csi_calls = g_esp_wifi_mock.set_csi_call_count;

  // The reconnect arrives before the old sender exits. Capture must not stay
  // armed across the disconnect; it waits for its disable instead.
  runtime.on_wifi_disconnected_();
  runtime.on_wifi_connected_(ip);
  runtime.loop();
  TEST_ASSERT_EQUAL(csi_calls, g_esp_wifi_mock.set_csi_call_count);
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
  TEST_ASSERT_FALSE(runtime.snapshot_.ready_to_publish);

  run_pending_traffic_task();
  runtime.loop();
  TEST_ASSERT_TRUE(g_esp_wifi_mock.set_csi_call_count >= csi_calls + 2);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_EQUAL(2U, g_freertos_task_mock.create_calls);
  TEST_ASSERT_TRUE(generator.has_live_worker());
  TEST_ASSERT_TRUE(runtime.is_calibrating());
  TEST_ASSERT_TRUE(runtime.snapshot_.ready_to_publish);
}

void test_runtime_reports_a_traffic_task_that_does_not_stop_and_keeps_waiting(void) {
  prepare_deferred_traffic_task();
  RuntimeConfig config;
  DetectorListener listener;
  TrafficGeneratorManager generator;
  FakeCsiTrafficIngress ingress;
  EspIdfRuntime runtime(config, generator, ingress);
  DeferredTrafficTaskScope scope(runtime, generator);
  runtime.set_listener(&listener);
  TEST_ASSERT_TRUE(runtime.setup());
  esp_netif_ip_info_t ip{};
  ip.ip.addr = ip.gw.addr = 0x0101A8C0U;
  runtime.on_wifi_connected_(ip);

  runtime.on_wifi_disconnected_();
  const int faults = listener.faults;
  // A sender held for seconds by a Wi-Fi TX stall is not a fault.
  esp_timer_mock::advance(2000000);
  runtime.loop();
  TEST_ASSERT_EQUAL(faults, listener.faults);
  esp_timer_mock::advance(28000000);
  runtime.loop();
  TEST_ASSERT_EQUAL(faults + 1, listener.faults);
  runtime.loop();
  TEST_ASSERT_EQUAL(faults + 1, listener.faults);
  TEST_ASSERT_TRUE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_FALSE(runtime.traffic_allows_radio_work());
  TEST_ASSERT_EQUAL(0U, g_freertos_task_mock.delete_calls);

  run_pending_traffic_task();
  runtime.loop();
  TEST_ASSERT_FALSE(g_esp_wifi_mock.csi_enabled);
  TEST_ASSERT_TRUE(runtime.traffic_allows_radio_work());
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.delete_calls);
}

void test_controller_setup_after_shutdown_waits_for_the_previous_traffic_task(void) {
  prepare_deferred_traffic_task();
  g_esp_netif_mock.ip_addr = 0x3701A8C0U;
  g_esp_netif_mock.gw_addr = 0x0101A8C0U;
  g_esp_wifi_mock.protocol_bitmap = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
  RuntimeFrontendController controller;
  ControllerTrafficTaskScope scope(controller);
  TEST_ASSERT_TRUE(controller.traffic_allows_radio_work());
  TEST_ASSERT_TRUE(controller.setup(nullptr));
  controller.loop();
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
  const int first_sock = last_traffic_socket;
  TEST_ASSERT_TRUE(socket_is_open(first_sock));

  // Shutting down from the loop must not wait for a sender held in a socket
  // call. The backend goes away; the sender keeps its owner and its socket.
  g_freertos_delay_hook = count_traffic_delay;
  controller.shutdown();
  TEST_ASSERT_EQUAL(0U, traffic_delay_calls);
  g_freertos_delay_hook = nullptr;
  TEST_ASSERT_TRUE(socket_is_open(first_sock));
  // The backend is gone, but the sender is still inside its socket call.
  TEST_ASSERT_FALSE(controller.traffic_allows_radio_work());

  // A new backend waits for that sender: no second task and no second socket.
  TEST_ASSERT_TRUE(controller.setup(nullptr));
  controller.loop();
  controller.loop();
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
  TEST_ASSERT_EQUAL(1U, traffic_sockets_opened);
  TEST_ASSERT_FALSE(controller.traffic_allows_radio_work());

  run_pending_traffic_task();
  TEST_ASSERT_FALSE(socket_is_open(first_sock));
  controller.loop();
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.delete_calls);
  TEST_ASSERT_EQUAL(2U, g_freertos_task_mock.create_calls);
  TEST_ASSERT_EQUAL(2U, traffic_sockets_opened);
  TEST_ASSERT_TRUE(socket_is_open(last_traffic_socket));

  // Without a backend, the controller loop still reaps the last sender.
  controller.shutdown();
  TEST_ASSERT_FALSE(controller.traffic_allows_radio_work());
  run_pending_traffic_task();
  // The worker has left its send. The radio may move before loop() deletes it.
  TEST_ASSERT_TRUE(controller.traffic_allows_radio_work());
  controller.loop();
  TEST_ASSERT_EQUAL(2U, g_freertos_task_mock.delete_calls);
  TEST_ASSERT_TRUE(controller.traffic_allows_radio_work());
}

void test_recovered_traffic_stop_does_not_fault_the_next_runtime(void) {
  prepare_deferred_traffic_task();
  g_esp_netif_mock.ip_addr = 0x3701A8C0U;
  g_esp_netif_mock.gw_addr = 0x0101A8C0U;
  g_esp_wifi_mock.protocol_bitmap = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
  RuntimeFrontendController controller;
  ControllerTrafficTaskScope scope(controller);
  DetectorListener listener;
  TEST_ASSERT_TRUE(controller.setup(&listener));
  controller.loop();
  TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);

  // The stop times out with no backend to read it, then the worker exits.
  controller.shutdown();
  esp_timer_mock::advance(30000000);
  controller.loop();
  run_pending_traffic_task();
  controller.loop();

  TEST_ASSERT_TRUE(controller.setup(&listener));
  controller.loop();
  controller.loop();
  TEST_ASSERT_EQUAL(0, listener.faults);
}

void test_runtime_raw_collection_terminates_on_wifi_loss_and_channel_change(void) {
  RuntimeConfig config;
  config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  runtime.capabilities_.supports_raw_csi = true;
  runtime.setup_complete_ = true;
  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  runtime.on_csi_channel_changed_(6U, 11U);
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());

  TEST_ASSERT_TRUE(runtime.start_raw_collection(&accept_raw_packet, nullptr));
  runtime.on_wifi_disconnected_();
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, runtime.operation_state());
  TEST_ASSERT_FALSE(runtime.wifi_ready_);
  TEST_ASSERT_FALSE(runtime.csi_pipeline_.is_enabled());
  runtime.csi_traffic_service_.stop();
}

void test_runtime_channel_change_cold_resets_ml_without_calibration(void) {
  RuntimeConfig config;
  config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
  config.traffic_generator_mode = TrafficGeneratorMode::EXTERNAL;
  FakeCsiTrafficGenerator traffic_generator;
  FakeCsiTrafficIngress traffic_ingress;
  EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
  TEST_ASSERT_TRUE(runtime.configure_detector_());
  runtime.csi_pipeline_.init(runtime.detector_.get());
  runtime.csi_traffic_service_.init(to_csi_traffic_config(config));
  TEST_ASSERT_EQUAL(ESP_OK, runtime.csi_pipeline_.enable());

  int8_t csi_data[HT20_CSI_LEN] = {};
  runtime.detector_->process_packet(csi_data, sizeof(csi_data), DEFAULT_SUBCARRIERS,
                                    HT20_SELECTED_BAND_SIZE, -50);
  TEST_ASSERT_TRUE(runtime.detector_->get_buffer_count() > 0U);

  runtime.wifi_ready_ = true;
  runtime.wifi_ip_info_.ip.addr = 0x0101A8C0U;
  runtime.wifi_ip_info_.gw.addr = 0x0101A8C0U;
  runtime.on_csi_channel_changed_(8U, 10U);

  TEST_ASSERT_TRUE(runtime.csi_pipeline_.is_enabled());
  TEST_ASSERT_FALSE(runtime.is_calibrating());
  TEST_ASSERT_FALSE(runtime.get_snapshot().ready_to_publish);
  TEST_ASSERT_EQUAL(0U, runtime.detector_->get_buffer_count());
  runtime.csi_traffic_service_.stop();
}

void test_runtime_reassociation_restarts_traffic_without_ip_or_channel_change(void) {
  for (const bool reports_disconnect : {false, true}) {
    esp_event_mock_reset();
    esp_netif_mock_reset();
    esp_wifi_mock_reset();
    g_esp_netif_mock.ip_addr = 0U;
    RuntimeConfig config;
    config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
    config.threshold = 0.73f;
    config.traffic_generator_mode = kSupportsWifiRaw ? TrafficGeneratorMode::WIFI_RAW : TrafficGeneratorMode::PING;
    FakeCsiTrafficGenerator traffic_generator;
    FakeCsiTrafficIngress traffic_ingress;
    EspIdfRuntime runtime(config, traffic_generator, traffic_ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_START, nullptr);
    ip_event_got_ip_t ip{};
    ip.ip_info.ip.addr = 0x3701A8C0U;
    ip.ip_info.gw.addr = 0x0101A8C0U;
    ip.ip_info.netmask.addr = g_esp_netif_mock.netmask_addr;
    g_esp_netif_mock.ip_addr = ip.ip_info.ip.addr;
    g_esp_netif_mock.gw_addr = ip.ip_info.gw.addr;
    esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &ip);
    TEST_ASSERT_EQUAL(ESP_OK, runtime.wifi_lifecycle_.process_pending_events());
    TEST_ASSERT_TRUE(traffic_generator.is_running());
    const uint32_t starts = traffic_generator.start_calls;
    const uint32_t stops = traffic_generator.stop_calls;
    if (reports_disconnect) {
      wifi_event_sta_disconnected_t disconnect{};
      disconnect.reason = WIFI_REASON_ROAMING;
      esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &disconnect);
    }
    ++g_esp_wifi_mock.current_ap_info.bssid[5];
    esp_event_mock_emit(WIFI_EVENT, WIFI_EVENT_STA_CONNECTED, nullptr);
    TEST_ASSERT_EQUAL(ESP_OK, runtime.wifi_lifecycle_.process_pending_events());
    TEST_ASSERT_TRUE(traffic_generator.is_running());
    TEST_ASSERT_TRUE(traffic_generator.stop_calls > stops);
    TEST_ASSERT_TRUE(runtime.csi_receive_path_check_pending_);
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.scan_start_call_count);
    TEST_ASSERT_EQUAL(starts + 1U, traffic_generator.start_calls);
    TEST_ASSERT_EQUAL(config.traffic_generator_mode, traffic_generator.mode);
    TEST_ASSERT_EQUAL(ip.ip_info.gw.addr, traffic_generator.gateway_addr);
    TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.get_snapshot().threshold);
    TEST_ASSERT_FALSE(runtime.is_calibrating());
    esp_event_mock_emit(IP_EVENT, IP_EVENT_STA_GOT_IP, &ip);
    TEST_ASSERT_EQUAL(ESP_OK, runtime.wifi_lifecycle_.process_pending_events());
    TEST_ASSERT_EQUAL(starts + 1U, traffic_generator.start_calls);
    runtime.shutdown();
  }
}

void test_wifi_raw_switch_preserves_ml_threshold_and_recalibrates_lightweight_only(void) {
  for (const auto algorithm : {DetectionAlgorithm::HIGH_ACCURACY, DetectionAlgorithm::LIGHTWEIGHT}) {
    nvs_mock_reset();
    RuntimeConfig config;
    config.detection_algorithm = algorithm;
    config.threshold = 0.73f;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    DetectorListener listener;
    runtime.set_listener(&listener);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip_info{};
    ip_info.ip.addr = 0x1101A8C0U;
    ip_info.gw.addr = 0x0101A8C0U;
    runtime.on_wifi_connected_(ip_info);
    runtime.cancel_calibration_(false);
    const int calibration_starts = listener.calibration_starts;
    const int calibration_finishes = listener.calibration_finishes;
    const uint32_t configure_calls = runtime.csi_pipeline_.capture_service_.enable_attempts();
    TEST_ASSERT_EQUAL(CsiCaptureProfile::HT20, runtime.get_snapshot().csi_capture_profile);
    TEST_ASSERT_EQUAL(kSupportsWifiRaw, runtime.set_traffic_generator_mode(TrafficGeneratorMode::WIFI_RAW));
    if (!kSupportsWifiRaw) {
      TEST_ASSERT_EQUAL(TrafficGeneratorMode::PING, generator.mode);
      TEST_ASSERT_EQUAL(configure_calls, runtime.csi_pipeline_.capture_service_.enable_attempts());
      TEST_ASSERT_EQUAL(calibration_starts, listener.calibration_starts);
      TEST_ASSERT_EQUAL(calibration_finishes, listener.calibration_finishes);
      TrafficGeneratorMode saved{};
      bool has_saved = false;
      TEST_ASSERT_EQUAL(ESP_OK, load_runtime_traffic_generator_mode(&saved, &has_saved));
      TEST_ASSERT_FALSE(has_saved);
      runtime.shutdown();
      continue;
    }
    TEST_ASSERT_EQUAL(TrafficGeneratorMode::WIFI_RAW, generator.mode);
    TEST_ASSERT_EQUAL(CsiCaptureProfile::LLTF20, runtime.get_snapshot().csi_capture_profile);
    // AUTO starts on HT20; wifi_raw switches to LLTF20 and rearms CSI.
    TEST_ASSERT_EQUAL(configure_calls + 1U, runtime.csi_pipeline_.capture_service_.enable_attempts());
    TEST_ASSERT_EQUAL(algorithm == DetectionAlgorithm::LIGHTWEIGHT, runtime.is_calibrating());
    TEST_ASSERT_EQUAL(calibration_starts + (algorithm == DetectionAlgorithm::LIGHTWEIGHT ? 1 : 0),
                      listener.calibration_starts);
    TEST_ASSERT_EQUAL(calibration_finishes, listener.calibration_finishes);
    TEST_ASSERT_EQUAL_FLOAT(0.73f, runtime.get_snapshot().threshold);
    TrafficGeneratorMode saved{};
    bool has_saved = false;
    TEST_ASSERT_EQUAL(ESP_OK, load_runtime_traffic_generator_mode(&saved, &has_saved));
    TEST_ASSERT_TRUE(has_saved);
    TEST_ASSERT_EQUAL(TrafficGeneratorMode::WIFI_RAW, saved);
    TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));
    TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::PING));
    TEST_ASSERT_EQUAL(CsiCaptureProfile::HT20, runtime.get_snapshot().csi_capture_profile);
    runtime.shutdown();
  }
}

void test_runtime_traffic_destination_tracks_config_across_restarts_and_gateway_changes(void) {
  for (const char *target : {"", "192.168.1.53"}) {
    nvs_mock_reset();
    RuntimeConfig config;
    config.detection_algorithm = DetectionAlgorithm::HIGH_ACCURACY;
    config.traffic_generator_target_ip = target;
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    EspIdfRuntime runtime(config, generator, ingress);
    TEST_ASSERT_TRUE(runtime.setup());
    esp_netif_ip_info_t ip_info{};
    ip_info.ip.addr = 0x1101A8C0U;
    ip_info.gw.addr = 0x0101A8C0U;
    for (const auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS, TrafficGeneratorMode::DNS_TCP}) {
      runtime.stop_sensing_services_();
      ip_info.gw.addr += 0x01000000U;
      runtime.on_wifi_connected_(ip_info);
      TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(mode));
      const uint32_t expected = target[0] == '\0' ? ip_info.gw.addr : 0x3501A8C0U;
      TEST_ASSERT_TRUE(generator.is_running());
      TEST_ASSERT_EQUAL(expected, generator.gateway_addr);
      TEST_ASSERT_EQUAL(expected, runtime.csi_pipeline_.traffic_filter_.internal_target_ip_addr);
      TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(TrafficGeneratorMode::EXTERNAL));
      TEST_ASSERT_FALSE(generator.is_running());
      TEST_ASSERT_TRUE(runtime.set_traffic_generator_mode(mode));
      TEST_ASSERT_EQUAL(expected, generator.gateway_addr);
      TEST_ASSERT_EQUAL(expected, runtime.csi_pipeline_.traffic_filter_.internal_target_ip_addr);
    }
    runtime.shutdown();
  }
}

void test_traffic_source_target_support_applies_to_config_controls_persistence_and_discovery(void) {
  TEST_ASSERT_FALSE(runtime_traffic_generator_mode_supported(static_cast<TrafficGeneratorMode>(0xff)));
  for (const auto mode : {TrafficGeneratorMode::PING, TrafficGeneratorMode::DNS,
                          TrafficGeneratorMode::DNS_TCP, TrafficGeneratorMode::WIFI_RAW}) {
    nvs_mock_reset();
    esp_event_mock_reset();
    const bool supported = mode != TrafficGeneratorMode::WIFI_RAW || kSupportsWifiRaw;
    TEST_ASSERT_EQUAL(supported, runtime_traffic_generator_mode_supported(mode));
    RuntimeConfig config;
    config.traffic_generator_mode = mode;
    TEST_ASSERT_EQUAL(supported ? RuntimeConfigError::NONE : RuntimeConfigError::TRAFFIC_GENERATOR_MODE,
                      validate_runtime_config(config));
    RuntimeFrontendController controller;
    TEST_ASSERT_EQUAL(supported, controller.set_traffic_generator_mode(mode));
    TEST_ASSERT_EQUAL(supported ? mode : RuntimeConfig{}.traffic_generator_mode,
                      controller.config().traffic_generator_mode);
    FakeCsiTrafficGenerator generator;
    FakeCsiTrafficIngress ingress;
    {
      EspIdfRuntime runtime(config, generator, ingress);
      TEST_ASSERT_EQUAL(supported, runtime.setup());
      runtime.shutdown();
    }
    // Seed a value saved by older firmware, including a now-unsupported mode.
    TEST_ASSERT_EQUAL(ESP_OK, save_runtime_traffic_generator_mode(mode));
    esp_event_mock_reset();
    config.traffic_generator_mode = TrafficGeneratorMode::DNS;
    EspIdfRuntime restored(config, generator, ingress);
    TEST_ASSERT_TRUE(restored.setup());
    TEST_ASSERT_EQUAL(supported ? mode : config.traffic_generator_mode,
                      restored.config_.traffic_generator_mode);
    restored.shutdown();
  }
  FrontendHaMqttSettings settings;
  settings.traffic_generator_mode_object_id = "traffic_generator_mode";
  const auto messages = build_frontend_ha_discovery_messages(settings, {}, false, false, true);
  const auto message = std::find_if(messages.begin(), messages.end(), [](const auto &entry) {
    return entry.topic.find("/traffic_generator_mode/config") != std::string::npos;
  });
  TEST_ASSERT_TRUE(message != messages.end());
  const std::string &discovery = message->payload;
  TEST_ASSERT_EQUAL(kSupportsWifiRaw, discovery.find("\"wifi_raw\"") != std::string::npos);
  for (const char *mode : {"\"ping\"", "\"dns\"", "\"dns_tcp\""}) {
    TEST_ASSERT_TRUE(discovery.find(mode) != std::string::npos);
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  UNITY_BEGIN();
  RUN_TEST(test_runtime_healthy_startup_does_not_scan_even_when_csi_is_rejected);
  RUN_TEST(test_runtime_silent_startup_refreshes_once_then_resumes_on_failure_or_timeout);
  RUN_TEST(test_runtime_absent_or_stopped_traffic_does_not_trigger_a_refresh);
  RUN_TEST(test_runtime_external_wifi_stack_owns_recovery_scan_results);
  RUN_TEST(test_runtime_defers_busy_refresh_with_capture_running_and_a_bounded_request_window);
  RUN_TEST(test_runtime_profile_change_rechecks_the_receive_path);
  RUN_TEST(test_runtime_reports_a_receive_path_still_silent_after_its_refresh);
  RUN_TEST(test_runtime_detector_switch_preserves_state_when_calibrator_allocation_fails);
  RUN_TEST(test_runtime_calibration_can_restart_from_completion_callback);
  RUN_TEST(test_runtime_calibration_allocation_failure_does_not_emit_started);
  RUN_TEST(test_traffic_source_target_support_applies_to_config_controls_persistence_and_discovery);
  RUN_TEST(test_runtime_traffic_destination_tracks_config_across_restarts_and_gateway_changes);
  RUN_TEST(test_runtime_readiness_requires_valid_recent_csi_and_recovers_after_quality_gap);
  RUN_TEST(test_runtime_readiness_holds_brief_coverage_dips_and_drops_on_real_loss);
  RUN_TEST(test_runtime_reassociation_restarts_traffic_without_ip_or_channel_change);
  RUN_TEST(test_wifi_raw_switch_preserves_ml_threshold_and_recalibrates_lightweight_only);
  RUN_TEST(test_runtime_calibration_consumes_evaluations_resets_on_gaps_and_finishes);
  RUN_TEST(test_runtime_recalibration_during_motion_keeps_the_live_threshold);
  RUN_TEST(test_runtime_startup_calibration_during_motion_keeps_the_default);
  RUN_TEST(test_runtime_rejects_invalid_detector_geometry_before_starting_services);
  RUN_TEST(test_runtime_rejects_invalid_or_unpersisted_controls_without_changing_config);
  RUN_TEST(test_runtime_restores_internal_traffic_when_external_source_cannot_start);
  RUN_TEST(test_runtime_detector_switch_updates_pipeline_threshold_and_calibration);
  RUN_TEST(test_runtime_detector_configuration_preserves_the_requested_threshold);
  RUN_TEST(test_runtime_traffic_updates_roll_back_when_persistence_fails);
  RUN_TEST(test_runtime_detector_adaptation_emits_threshold_changed_without_live_telemetry);
  RUN_TEST(test_runtime_motion_hits_runtime_updates_pipeline_and_persists);
  RUN_TEST(test_runtime_setup_loads_all_persisted_runtime_controls);
  RUN_TEST(test_runtime_without_persistence_ignores_and_never_writes_saved_controls);
  RUN_TEST(test_runtime_diagnostics_cache_current_wifi_association);
  RUN_TEST(test_runtime_channel_change_rearms_csi_and_restarts_calibration);
  RUN_TEST(test_runtime_services_armed_preserves_wifi_ip_and_restarts_capture);
  RUN_TEST(test_runtime_sensing_reasserts_promiscuous_disabled_before_capture);
  RUN_TEST(test_runtime_disconnect_or_disarm_cancels_refresh_and_discards_queued_completion);
  RUN_TEST(test_runtime_raw_collection_restores_armed_and_disarmed_sensing);
  RUN_TEST(test_runtime_raw_collection_terminates_on_wifi_loss_and_channel_change);
  RUN_TEST(test_runtime_disables_capture_only_after_the_traffic_task_exits);
  RUN_TEST(test_runtime_channel_change_rearms_after_the_traffic_task_and_held_radio_work);
  RUN_TEST(test_runtime_reconnect_while_traffic_stops_rearms_capture_after_its_disable);
  RUN_TEST(test_runtime_reports_a_traffic_task_that_does_not_stop_and_keeps_waiting);
  RUN_TEST(test_controller_setup_after_shutdown_waits_for_the_previous_traffic_task);
  RUN_TEST(test_recovered_traffic_stop_does_not_fault_the_next_runtime);
  RUN_TEST(test_runtime_channel_change_cold_resets_ml_without_calibration);
  return UNITY_END();
}
