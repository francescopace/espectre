/*
 * ESPectre - SDK Surface Tests
 *
 * Guards the published embedding surface: that `espectre_sdk.h` alone reaches
 * every documented type, and that the defaults and ranges the SDK
 * documentation promises still hold.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

// Deliberately the only ESPectre include. An integrator following
// docs/EMBEDDING.md includes this and nothing else, so anything this suite
// needs and cannot reach is a hole in the facade.
#include "espectre_sdk.h"

#include "test_harness.h"

#include <cstring>

using namespace espectre;

namespace {

/** A listener that overrides one callback, as the documentation tells adopters to. */
class MinimalListener : public IRuntimeListener {
 public:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override {
    motion_changes++;
    last_state = snapshot.motion_state;
  }

  int motion_changes{0};
  MotionState last_state{MotionState::IDLE};
};

}  // namespace

void setUp(void) {}

void tearDown(void) {}

void test_sdk_version_macros_agree_with_each_other(void) {
  char expected[32];
  std::snprintf(expected, sizeof(expected), "%d.%d.%d", ESPECTRE_SDK_VERSION_MAJOR,
                ESPECTRE_SDK_VERSION_MINOR, ESPECTRE_SDK_VERSION_PATCH);

  TEST_ASSERT_EQUAL_STRING(expected, ESPECTRE_SDK_VERSION_STRING);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_SDK_VERSION_STRING, espectre_sdk_version());

  const int expected_number = (ESPECTRE_SDK_VERSION_MAJOR * 10000) +
                              (ESPECTRE_SDK_VERSION_MINOR * 100) + ESPECTRE_SDK_VERSION_PATCH;
  TEST_ASSERT_EQUAL_INT(expected_number, ESPECTRE_SDK_VERSION_NUMBER);

  // The guard integrators write must accept the running version and reject a
  // release that has not happened yet.
  TEST_ASSERT_TRUE(ESPECTRE_SDK_VERSION_AT_LEAST(ESPECTRE_SDK_VERSION_MAJOR, ESPECTRE_SDK_VERSION_MINOR,
                                                 ESPECTRE_SDK_VERSION_PATCH));
  TEST_ASSERT_FALSE(ESPECTRE_SDK_VERSION_AT_LEAST(ESPECTRE_SDK_VERSION_MAJOR + 1, 0, 0));
}

void test_default_runtime_config_is_a_working_sensing_config(void) {
  // docs/EMBEDDING.md tells integrators that `RuntimeConfig{}` is usable as is.
  // If a default drifts away from the schema, that promise silently breaks.
  const RuntimeConfig config;

  TEST_ASSERT_EQUAL(static_cast<int>(RuntimeProfile::SENSING), static_cast<int>(config.runtime_profile));
  TEST_ASSERT_EQUAL(static_cast<int>(WifiBandPolicy::BAND_2G),
                    static_cast<int>(config.wifi_band_policy));
  TEST_ASSERT_EQUAL(static_cast<int>(DetectionAlgorithm::CLASSIC),
                    static_cast<int>(config.detection_algorithm));
  TEST_ASSERT_EQUAL(static_cast<int>(CsiTrafficMode::INTERNAL), static_cast<int>(config.csi_traffic_mode));
  TEST_ASSERT_TRUE(runtime_detection_algorithm_valid(config.detection_algorithm));

  TEST_ASSERT_EQUAL_UINT8(RUNTIME_MOTION_ON_HITS_DEFAULT, config.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(RUNTIME_MOTION_OFF_HITS_DEFAULT, config.motion_off_hits);
  TEST_ASSERT_EQUAL(RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT,
                    config.segmentation_window_size_ms);
  TEST_ASSERT_EQUAL_INT(static_cast<int>(RUNTIME_TRAFFIC_GENERATOR_RATE_DEFAULT),
                        static_cast<int>(config.traffic_generator_rate));

  // Zero means "derive from the Wi-Fi MAC", which is what makes the default
  // config usable without the integrator supplying an identity.
  TEST_ASSERT_TRUE(config.device_id == ESPECTRE_DEFAULT_DEVICE_ID);
}

void test_documented_defaults_sit_inside_documented_ranges(void) {
  // Every tunable is documented as DEFAULT within [MIN, MAX]. A default outside
  // its own range would be rejected by the validators the same headers point
  // integrators at.
  const RuntimeConfig config;

  TEST_ASSERT_TRUE(validate_runtime_uint32(config.segmentation_window_size_ms,
                                           RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MIN,
                                           RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint32(config.traffic_generator_rate,
                                           RUNTIME_TRAFFIC_GENERATOR_RATE_MIN,
                                           RUNTIME_TRAFFIC_GENERATOR_RATE_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint32(config.publish_interval_ms,
                                           RUNTIME_PUBLISH_INTERVAL_MS_MIN,
                                           RUNTIME_PUBLISH_INTERVAL_MS_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint32(config.evaluation_interval_ms,
                                           RUNTIME_EVALUATION_INTERVAL_MS_MIN,
                                           RUNTIME_EVALUATION_INTERVAL_MS_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint8(config.motion_on_hits, RUNTIME_MOTION_HITS_MIN,
                                          RUNTIME_MOTION_HITS_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint8(config.motion_off_hits, RUNTIME_MOTION_HITS_MIN,
                                          RUNTIME_MOTION_HITS_MAX));
  TEST_ASSERT_TRUE(validate_runtime_uint8(config.hampel_window, RUNTIME_HAMPEL_WINDOW_MIN,
                                          RUNTIME_HAMPEL_WINDOW_MAX));
  TEST_ASSERT_TRUE(validate_runtime_float(config.hampel_threshold, RUNTIME_HAMPEL_THRESHOLD_MIN,
                                          RUNTIME_HAMPEL_THRESHOLD_MAX));
  TEST_ASSERT_TRUE(validate_runtime_float(config.lowpass_cutoff, RUNTIME_LOWPASS_CUTOFF_MIN,
                                          RUNTIME_LOWPASS_CUTOFF_MAX));

  // Per-detector defaults have to be acceptable to the per-detector validator,
  // or selecting a detector would immediately produce an invalid threshold.
  TEST_ASSERT_TRUE(validate_runtime_threshold_for_algorithm(
      runtime_default_threshold(DetectionAlgorithm::CLASSIC), DetectionAlgorithm::CLASSIC));
  TEST_ASSERT_TRUE(validate_runtime_threshold_for_algorithm(
      runtime_default_threshold(DetectionAlgorithm::ML), DetectionAlgorithm::ML));
}

void test_default_snapshot_is_not_publishable(void) {
  // The readiness gate is the first rule the SDK documentation gives adopters:
  // a snapshot must never arrive claiming it is safe to publish by default.
  const RuntimeSnapshot snapshot;

  TEST_ASSERT_FALSE(snapshot.ready_to_publish);
  TEST_ASSERT_FALSE(snapshot.calibrating);
  TEST_ASSERT_EQUAL(static_cast<int>(MotionState::IDLE), static_cast<int>(snapshot.motion_state));
  TEST_ASSERT_NOT_NULL(snapshot.detector_name);
}

void test_default_capabilities_advertise_nothing(void) {
  // Capabilities are opt-in by design, so a runtime that forgets to declare
  // one must fail closed rather than inherit a permissive default.
  const RuntimeCapabilities capabilities;

  TEST_ASSERT_FALSE(capabilities.supports_runtime_threshold_updates);
  TEST_ASSERT_FALSE(capabilities.supports_runtime_motion_hits_updates);
  TEST_ASSERT_FALSE(capabilities.supports_runtime_detector_selection);
  TEST_ASSERT_FALSE(capabilities.supports_manual_recalibration);
  TEST_ASSERT_FALSE(capabilities.supports_ble_telemetry);
  TEST_ASSERT_FALSE(capabilities.supports_extended_diagnostics);
  TEST_ASSERT_FALSE(capabilities.supports_traffic_control);
}

void test_listener_callbacks_default_to_no_ops(void) {
  // Adopters are told to override only what they use. That only holds if every
  // other callback is a safe no-op on the base class.
  MinimalListener listener;
  IRuntimeListener &base = listener;
  RuntimeSnapshot snapshot;
  snapshot.motion_state = MotionState::MOTION;

  base.on_periodic_update(snapshot, 100U);
  base.on_threshold_changed(snapshot);
  base.on_detector_changed(snapshot);
  base.on_calibration_started(snapshot);
  base.on_calibration_finished(snapshot, true);
  base.on_live_telemetry(0.5f, 0.7f);
  base.on_runtime_fault("surface test");
  TEST_ASSERT_EQUAL_INT(0, listener.motion_changes);

  base.on_motion_state_changed(snapshot);
  TEST_ASSERT_EQUAL_INT(1, listener.motion_changes);
  TEST_ASSERT_EQUAL(static_cast<int>(MotionState::MOTION), static_cast<int>(listener.last_state));
}

void test_detector_names_round_trip_through_the_protocol_form(void) {
  // RuntimeSnapshot::detector_name is documented as the protocol name that
  // parse_detection_algorithm() understands, not BaseDetector::get_name().
  for (const DetectionAlgorithm algorithm : {DetectionAlgorithm::CLASSIC, DetectionAlgorithm::ML}) {
    const char *name = detection_algorithm_name(algorithm);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL(static_cast<int>(algorithm), static_cast<int>(parse_detection_algorithm(name)));
  }

  TEST_ASSERT_EQUAL_STRING(RUNTIME_DETECTION_ALGORITHM_CLASSIC_NAME,
                           detection_algorithm_name(DetectionAlgorithm::CLASSIC));
  TEST_ASSERT_EQUAL_STRING(RUNTIME_DETECTION_ALGORITHM_ML_NAME,
                           detection_algorithm_name(DetectionAlgorithm::ML));
}

void test_core_only_detector_path_is_reachable_from_the_facade(void) {
  // The core-only integration path documented on ClassicDetector must be
  // usable with nothing but the facade include.
  ClassicDetector detector;

  TEST_ASSERT_FALSE(detector.is_ready());
  TEST_ASSERT_EQUAL(static_cast<int>(MotionState::IDLE), static_cast<int>(detector.get_state()));
  TEST_ASSERT_EQUAL_STRING("Classic", detector.get_name());
  TEST_ASSERT_TRUE(detector.set_threshold(CLASSIC_DEFAULT_THRESHOLD));
  TEST_ASSERT_EQUAL_FLOAT(CLASSIC_DEFAULT_THRESHOLD, detector.get_threshold());

  // The documented call shape for a caller that already owns CSI capture.
  const uint8_t subcarrier_count = static_cast<uint8_t>(HT20_SELECTED_BAND_SIZE);
  TEST_ASSERT_TRUE(subcarrier_count > 0U);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_sdk_version_macros_agree_with_each_other);
  RUN_TEST(test_default_runtime_config_is_a_working_sensing_config);
  RUN_TEST(test_documented_defaults_sit_inside_documented_ranges);
  RUN_TEST(test_default_snapshot_is_not_publishable);
  RUN_TEST(test_default_capabilities_advertise_nothing);
  RUN_TEST(test_listener_callbacks_default_to_no_ops);
  RUN_TEST(test_detector_names_round_trip_through_the_protocol_form);
  RUN_TEST(test_core_only_detector_path_is_reachable_from_the_facade);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
