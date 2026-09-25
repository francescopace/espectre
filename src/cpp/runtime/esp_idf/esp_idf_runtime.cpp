/*
 * ESPectre - ESP-IDF Runtime
 *
 * ESP-IDF runtime that wires Wi-Fi, CSI capture, calibration, and
 * detection together.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_idf_runtime.h"
#include "network_traffic.h"

#include "core/csi_format.h"
#include "csi_platform_config.h"
#include "esp_err.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "core/espectre_log.h"
#include "core/high_accuracy_detector.h"
#include "core/lightweight_detector.h"
#include "lwip/inet.h"
#include "runtime/runtime_config_utils.h"
#include "runtime/runtime_config_validation.h"
#include "runtime_detector_store.h"
#include "runtime_motion_hits_store.h"
#include "runtime/runtime_time.h"
#include "runtime_traffic_mode_store.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace espectre {

namespace {

static const char *const RUNTIME_TAG = "espectre.runtime";
static constexpr uint32_t CSI_ENABLE_SETTLE_MS = 100U;
// A working receive path reports the first managed packets within tens of
// milliseconds of arming. On S3 and C5 under ESPHome the first arm after
// association stays silent on every boot, also with fast_connect. Disabling and
// re-arming CSI does not recover it; the single-channel scan does. Waiting
// longer than one detector window before that scan only delays sensing, so the
// observation lasts one window and never less than this floor.
static constexpr uint32_t CSI_STARTUP_OBSERVATION_MS = 1000U;
static constexpr uint32_t CSI_STARTUP_TRAFFIC_IDLE_MS = 1000U;
static constexpr uint32_t CSI_REFRESH_RETRY_INTERVAL_MS = 500U;
static constexpr uint32_t CSI_REFRESH_REQUEST_WINDOW_MS = 15000U;
static constexpr SelectedSubcarriers SELECTED_SUBCARRIERS = make_default_subcarriers();

}  // namespace

void EspIdfRuntime::notify_threshold_if_changed_(float threshold) {
  if (threshold == snapshot_.threshold) {
    return;
  }
  snapshot_.threshold = threshold;
  config_.threshold = threshold;
  if (listener_ != nullptr) {
    listener_->on_threshold_changed(get_snapshot());
  }
}

void EspIdfRuntime::update_live_telemetry_callback_() {
  if (live_telemetry_enabled_) {
    csi_pipeline_.set_live_telemetry_callback([this](float movement, float threshold) {
      snapshot_.movement_metric = movement;
      notify_threshold_if_changed_(threshold);
      if (listener_ != nullptr) {
        listener_->on_live_telemetry(movement, threshold);
      }
    });
  } else {
    csi_pipeline_.set_live_telemetry_callback({});
  }
}

EspIdfRuntime::EspIdfRuntime(const RuntimeConfig &config)
    : EspIdfRuntimeBase(config, RUNTIME_TAG, "Unknown runtime fault"),
      csi_traffic_service_(traffic_generator_, traffic_ingress_) {
  initialize_runtime_state_();
}

EspIdfRuntime::EspIdfRuntime(const RuntimeConfig &config,
                             ICsiTrafficGenerator &traffic_generator,
                             ICsiTrafficIngress &traffic_ingress)
    : EspIdfRuntimeBase(config, RUNTIME_TAG, "Unknown runtime fault"),
      csi_traffic_service_(traffic_generator, traffic_ingress) {
  initialize_runtime_state_();
}

void EspIdfRuntime::initialize_runtime_state_() {
  detection_timing_supported_ = true;
  snapshot_.threshold = config_.threshold;
  // The sensing runtime owns a detector, so it can retune and recalibrate it,
  // and it drives the live-telemetry callback used by transport adapters.
  capabilities_.supports_runtime_threshold_updates = true;
  capabilities_.supports_runtime_motion_hits_updates = true;
  capabilities_.supports_manual_recalibration = true;
  capabilities_.supports_live_telemetry = true;
  capabilities_.supports_extended_diagnostics = true;
  capabilities_.supports_traffic_control = true;
  capabilities_.supports_runtime_detector_selection = config_.runtime_detector_selection_enabled;
  // Raw CSI uses the same ESP-IDF capture boundary on every supported chip.
  capabilities_.supports_raw_csi = true;
}

void EspIdfRuntime::load_persisted_overrides_() {
  if (config_.runtime_detector_selection_enabled) {
    DetectionAlgorithm saved_algorithm = config_.detection_algorithm;
    bool has_saved_value = false;
    const esp_err_t err = load_runtime_detection_algorithm(&saved_algorithm, &has_saved_value);
    if (err != ESP_OK) {
      ESPECTRE_LOGW(RUNTIME_TAG, "Failed to load persisted detector: %s", esp_err_to_name(err));
    } else if (has_saved_value && saved_algorithm != config_.detection_algorithm) {
      ESPECTRE_LOGI(RUNTIME_TAG, "Persisted detector override: configured=%s saved=%s",
                    detection_algorithm_name(config_.detection_algorithm),
                    detection_algorithm_name(saved_algorithm));
      // Thresholds are per-detector; keep the configured one when the detector is unchanged.
      config_.detection_algorithm = saved_algorithm;
      config_.threshold = runtime_default_threshold(saved_algorithm);
      snapshot_.threshold = config_.threshold;
    }
  }

  bool has_saved_generator_mode = false;
  TrafficGeneratorMode saved_generator_mode = config_.traffic_generator_mode;
  const esp_err_t generator_err =
      load_runtime_traffic_generator_mode(&saved_generator_mode, &has_saved_generator_mode);
  if (generator_err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Failed to load persisted traffic generator mode: %s", esp_err_to_name(generator_err));
  } else if (has_saved_generator_mode &&
             runtime_traffic_generator_mode_supported(saved_generator_mode) &&
             runtime_capture_profile_supports_traffic(config_.csi_capture_policy, saved_generator_mode)) {
    if (saved_generator_mode != config_.traffic_generator_mode) {
      ESPECTRE_LOGI(RUNTIME_TAG, "Persisted traffic generator mode override: configured=%s saved=%s",
                    traffic_generator_mode_name(config_.traffic_generator_mode),
                    traffic_generator_mode_name(saved_generator_mode));
    }
    config_.traffic_generator_mode = saved_generator_mode;
  } else if (has_saved_generator_mode) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Ignoring persisted traffic source incompatible with this target or CSI profile");
  }

  uint8_t saved_motion_on_hits = config_.motion_on_hits;
  uint8_t saved_motion_off_hits = config_.motion_off_hits;
  bool has_saved_motion_hits = false;
  const esp_err_t motion_hits_err =
      load_runtime_motion_hits(&saved_motion_on_hits, &saved_motion_off_hits, &has_saved_motion_hits);
  if (motion_hits_err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Failed to load persisted motion hits: %s", esp_err_to_name(motion_hits_err));
  } else if (has_saved_motion_hits) {
    if (saved_motion_on_hits != config_.motion_on_hits || saved_motion_off_hits != config_.motion_off_hits) {
      ESPECTRE_LOGI(RUNTIME_TAG, "Persisted motion hits override: configured=%u/%u saved=%u/%u",
                    static_cast<unsigned>(config_.motion_on_hits),
                    static_cast<unsigned>(config_.motion_off_hits),
                    static_cast<unsigned>(saved_motion_on_hits),
                    static_cast<unsigned>(saved_motion_off_hits));
    }
    config_.motion_on_hits = saved_motion_on_hits;
    config_.motion_off_hits = saved_motion_off_hits;
  }
}

bool EspIdfRuntime::setup() {
  if (setup_complete_) {
    return true;
  }

  csi_receive_path_check_pending_ = false;
  csi_receive_path_refresh_in_progress_ = false;

  const RuntimeConfigError config_error = validate_runtime_config(config_);
  if (config_error != RuntimeConfigError::NONE) {
    notify_fault_(runtime_config_error_message(config_error));
    return false;
  }

  ESPECTRE_LOGI(RUNTIME_TAG, "Initializing ESPectre runtime...");

  if (config_.persist_runtime_overrides) {
    load_persisted_overrides_();
  }

  if (!configure_detector_()) {
    return false;
  }

  // Traffic ownership is explicit in traffic_generator_mode; the positive
  // target only defines cadence.
  csi_traffic_service_.init(to_csi_traffic_config(config_));

  csi_pipeline_.init(detector_.get());
  csi_pipeline_.set_evaluation_interval_ms(config_.evaluation_interval_ms);
  if (!csi_pipeline_.set_csi_target_pps(config_.csi_target_pps) ||
      !csi_pipeline_.set_segmentation_window_size_ms(config_.window_size_ms)) {
    notify_fault_("Failed to allocate temporal CSI sampler");
    return false;
  }
  csi_pipeline_.set_motion_hit_thresholds(config_.motion_on_hits, config_.motion_off_hits);
  csi_pipeline_.set_channel_change_callback([this](uint8_t previous_channel, uint8_t current_channel) {
    on_csi_channel_changed_(previous_channel, current_channel);
  });
  update_live_telemetry_callback_();

  if (wifi_lifecycle_.register_handlers([this](const esp_netif_ip_info_t &ip_info) {
                                          on_wifi_connected_(ip_info);
                                        },
                                        [this]() { on_wifi_disconnected_(); },
                                        config_.wifi_band_policy) != ESP_OK) {
    notify_fault_("Failed to register Wi-Fi handlers");
    return false;
  }

  wifi_ready_ = false;
  wifi_ip_info_ = {};
  wifi_rssi_dbm_ = INT8_MIN;
  wifi_channel_ = 0U;
  setup_complete_ = true;
  performance_diagnostics_.reset();
  loop_step_timer_.reset();
  return true;
}

void EspIdfRuntime::shutdown() {
  if (!setup_complete_) {
    return;
  }

  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    (void) stop_raw_collection(RawCsiStopReason::SHUTDOWN);
  }
  set_services_armed(false);
  on_wifi_disconnected_();
  deferred_capture_action_ = DeferredCaptureAction::None;
  sensing_start_pending_ = false;
  arm_receive_path_check_when_traffic_starts_ = false;
  // Shutdown does not wait for a sender still inside a socket call: its owner
  // keeps the generator, and the next start waits for that sender to exit.
  capture_updates_suppressed_ = true;
  (void) csi_pipeline_.disable();
  csi_receive_path_check_pending_ = false;
  csi_receive_path_refresh_in_progress_ = false;
  wifi_lifecycle_.unregister_handlers();
  setup_complete_ = false;
}

void EspIdfRuntime::loop() {
  RuntimePerformanceLoopScope performance_scope(performance_diagnostics_);
  // Name the step that holds the loop when a stall recurs, and separate the
  // frontend's log sink and listener time from the runtime's own work.
  loop_step_timer_.begin();
  if (wifi_lifecycle_.process_pending_events() != ESP_OK) {
    notify_fault_("Wi-Fi lifecycle init failed");
  }
  loop_step_timer_.mark("wifi_events");
  bool calibration_success = false;
  if (calibration_finished_event_.take(calibration_success)) {
    finish_threshold_calibration_(calibration_success);
  }
  loop_step_timer_.mark("calibration");
  // Settle readiness before dispatching pipeline events, so their snapshots
  // agree with the one the frontend controller caches after this loop.
  update_sensing_readiness_();
  loop_step_timer_.mark("readiness");
  csi_pipeline_.loop();
  loop_step_timer_.mark("pipeline");
  refresh_wifi_association_from_csi_();
  // Detector-owned adaptation is a control-plane change, so keep the runtime
  // snapshot and listener event current even when high-rate live telemetry is
  // disabled because no frontend consumer is watching it.
  if (detector_ != nullptr) {
    notify_threshold_if_changed_(detector_->get_threshold());
  }
  loop_step_timer_.mark("threshold");
  // Keep the active traffic source healthy while raw capture bypasses the
  // sensing sampler. In external mode this drains the non-blocking UDP socket;
  // otherwise its receive queue fills during long raw sessions and the marker
  // traffic path cannot recover cleanly.
  service_deferred_capture_action_();
  loop_step_timer_.mark("traffic");
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    loop_step_timer_.finish(RUNTIME_TAG);
    return;
  }
  check_csi_receive_path_();
  loop_step_timer_.mark("receive_path");
  csi_pipeline_.heartbeat_if_due(monotonic_now_ms());
  loop_step_timer_.mark("heartbeat");
  DetectionTimingStats detection_timing;
  if (csi_pipeline_.take_detection_timing(&detection_timing)) {
    performance_diagnostics_.record_detection_timing(detection_timing.duration_sum_us,
                                                      detection_timing.samples,
                                                      detection_timing.minimum_us,
                                                      detection_timing.maximum_us);
  }
  loop_step_timer_.finish(RUNTIME_TAG);
}

RuntimeSnapshot EspIdfRuntime::get_snapshot() const {
  RuntimeSnapshot result = snapshot_;
  // The internal publishing gate also serves calibration and lifecycle work.
  // Public readiness additionally requires a current, valid detector window;
  // loop() folds brief coverage dips into readiness_gate_. The window check
  // stays live, so a detector cleared since the last loop() is never ready.
  result.ready_to_publish = result.ready_to_publish && !result.calibrating &&
      detector_ != nullptr && readiness_gate_.ready() &&
      detector_->get_buffer_count() >= detector_->get_window_size() &&
      csi_pipeline_.has_current_detector_input(static_cast<int64_t>(monotonic_now_us()));
  return result;
}

void EspIdfRuntime::update_sensing_readiness_() {
  SensingReadinessInputs inputs;
  inputs.sensing_active = snapshot_.ready_to_publish;
  inputs.calibrating = snapshot_.calibrating;
  inputs.has_detector = detector_ != nullptr;
  inputs.input_current =
      csi_pipeline_.has_current_detector_input(static_cast<int64_t>(monotonic_now_us()));
  inputs.window_full =
      detector_ != nullptr && detector_->get_buffer_count() >= detector_->get_window_size();
  inputs.detector_ready = detector_ != nullptr && detector_->is_ready();
  const SensingReadinessGate::Edge edge =
      readiness_gate_.update(inputs, monotonic_now_ms(), config_.window_size_ms);
  const unsigned valid_slots = detector_ != nullptr ? detector_->get_valid_buffer_count() : 0U;
  const unsigned window_slots = detector_ != nullptr ? detector_->get_window_size() : 0U;
  switch (edge) {
    case SensingReadinessGate::Edge::READY:
      ESPECTRE_LOGI(RUNTIME_TAG, "Sensing ready (%u/%u valid slots)", valid_slots, window_slots);
      break;
    case SensingReadinessGate::Edge::UNREADY:
      ESPECTRE_LOGI(RUNTIME_TAG, "Sensing not ready: %s (%u/%u valid slots)",
                    sensing_readiness_reason_name(readiness_gate_.reason()), valid_slots,
                    window_slots);
      break;
    case SensingReadinessGate::Edge::DIP_ABSORBED:
      ESPECTRE_LOGD(RUNTIME_TAG, "Held readiness through a %u ms coverage dip",
                    static_cast<unsigned>(readiness_gate_.last_dip_ms()));
      break;
    case SensingReadinessGate::Edge::NONE:
      break;
  }
}

RuntimeDiagnosticsSnapshot EspIdfRuntime::get_diagnostics() const {
  RuntimeDiagnosticsSnapshot diagnostics = EspIdfRuntimeBase::get_diagnostics();
  diagnostics.link.rssi_dbm = wifi_rssi_dbm_;
  diagnostics.link.channel = wifi_channel_;
  diagnostics.traffic.generator_packets_total = csi_traffic_service_.get_generator_packets_total();
  const NetworkTrafficSnapshot traffic = read_network_traffic();
  diagnostics.traffic.tx_packets_total = traffic.tx_packets;
  diagnostics.traffic.rx_packets_total = traffic.rx_packets;
  diagnostics.csi.callbacks_total = csi_pipeline_.capture_callback_invocations_total();
  diagnostics.csi.provenance_rejected_total = csi_pipeline_.traffic_rejected_packets_total();
  diagnostics.csi.accepted_total = csi_pipeline_.accepted_packets_total();
  diagnostics.csi.admitted_total = csi_pipeline_.detector_admitted_packets_total();
  diagnostics.csi.filtered_total = csi_pipeline_.capture_filtered_packets_total();
  diagnostics.csi.rx_error_total = csi_pipeline_.capture_rx_error_total();
  diagnostics.csi.rx_end_error_total = csi_pipeline_.capture_rx_end_error_total();
  diagnostics.csi.invalid_estimate_total = csi_pipeline_.capture_invalid_estimate_total();
  diagnostics.csi.invalid_first_word_total = csi_pipeline_.capture_invalid_first_word_total();
  diagnostics.csi.sanitized_first_word_total = csi_pipeline_.capture_sanitized_first_word_total();

  diagnostics.csi.pending_frame_drops_total = csi_pipeline_.pending_frame_drops_total();
  diagnostics.csi.missing_slots_total = csi_pipeline_.detector_missing_slots_total();
  diagnostics.csi.excess_total = csi_pipeline_.detector_excess_packets_total();
  diagnostics.csi.stale_total = csi_pipeline_.detector_stale_packets_total();
  diagnostics.csi.out_of_order_total = csi_pipeline_.detector_out_of_order_packets_total();
  diagnostics.csi.occupancy_slots = csi_pipeline_.detector_window_occupancy_slots();
  diagnostics.csi.window_slots = csi_pipeline_.detector_window_slots();
  diagnostics.csi.pending_frames = static_cast<uint32_t>(csi_pipeline_.pending_frame_count());
  diagnostics.csi.pending_frame_capacity =
      static_cast<uint32_t>(csi_pipeline_.pending_frame_capacity());
  return diagnostics;
}

const RuntimeDiagnosticsSample *EspIdfRuntime::get_diagnostics_sample() const {
  return &latest_diagnostics_;
}

void EspIdfRuntime::set_services_armed(bool armed) {
  if (services_armed_ == armed) {
    return;
  }

  services_armed_ = armed;
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    ESPECTRE_LOGI(RUNTIME_TAG, "Deferred sensing mutation until raw collection stops");
    return;
  }
  if (!setup_complete_) {
    return;
  }

  if (!services_armed_) {
    ESPECTRE_LOGI(RUNTIME_TAG, "CSI services disarmed");
    invalidate_csi_receive_path_refresh_();
    csi_receive_path_check_pending_ = false;
    stop_sensing_services_();
    return;
  }

  if (wifi_ready_ && wifi_ip_info_.ip.addr != 0U) {
    ESPECTRE_LOGI(RUNTIME_TAG, "CSI services armed, starting capture");
    maybe_resume_sensing_after_wifi_reconfigure_();
  } else {
    ESPECTRE_LOGI(RUNTIME_TAG, "CSI services armed, waiting for Wi-Fi IP");
  }
}

void EspIdfRuntime::set_live_telemetry_enabled(bool enabled) {
  live_telemetry_enabled_ = enabled;
  update_live_telemetry_callback_();
}

bool EspIdfRuntime::set_threshold(float threshold) {
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  if (!validate_runtime_threshold_for_algorithm(threshold, config_.detection_algorithm)) {
    ESPECTRE_LOGW(RUNTIME_TAG,
             "Rejected invalid runtime threshold: %.6f (detector=%s max=%.3f)",
             threshold,
             detection_algorithm_name(config_.detection_algorithm),
             static_cast<double>(runtime_threshold_max(config_.detection_algorithm)));
    return false;
  }
  if (!csi_pipeline_.set_threshold(threshold)) {
    return false;
  }
  config_.threshold = threshold;
  snapshot_.threshold = threshold;
  if (listener_ != nullptr) {
    listener_->on_threshold_changed(get_snapshot());
  }
  ESPECTRE_LOGD(RUNTIME_TAG, "Threshold updated to %.6f (session-only, recalculated at boot)", threshold);
  return true;
}

bool EspIdfRuntime::set_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  if (motion_on_hits < RUNTIME_MOTION_HITS_MIN || motion_on_hits > RUNTIME_MOTION_HITS_MAX ||
      motion_off_hits < RUNTIME_MOTION_HITS_MIN || motion_off_hits > RUNTIME_MOTION_HITS_MAX) {
    ESPECTRE_LOGW(RUNTIME_TAG,
             "Rejected invalid motion hits on=%u off=%u (range=%u-%u)",
             static_cast<unsigned>(motion_on_hits),
             static_cast<unsigned>(motion_off_hits),
             static_cast<unsigned>(RUNTIME_MOTION_HITS_MIN),
             static_cast<unsigned>(RUNTIME_MOTION_HITS_MAX));
    return false;
  }

  const esp_err_t persist_err = config_.persist_runtime_overrides
                                    ? save_runtime_motion_hits(motion_on_hits, motion_off_hits)
                                    : ESP_OK;
  if (persist_err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Failed to persist motion hits: %s", esp_err_to_name(persist_err));
    return false;
  }

  csi_pipeline_.set_motion_hit_thresholds(motion_on_hits, motion_off_hits, true);
  config_.motion_on_hits = motion_on_hits;
  config_.motion_off_hits = motion_off_hits;
  ESPECTRE_LOGI(RUNTIME_TAG,
           "Motion hit thresholds updated: on=%u off=%u",
           static_cast<unsigned>(motion_on_hits),
           static_cast<unsigned>(motion_off_hits));
  return true;
}

bool EspIdfRuntime::set_traffic_generator_mode(TrafficGeneratorMode mode) {
  if (!runtime_capture_profile_supports_traffic(config_.csi_capture_policy, mode)) {
    ESPECTRE_LOGW(RUNTIME_TAG, "wifi_raw requires auto or lltf CSI capture profile");
    return false;
  }
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  if (!runtime_traffic_generator_mode_supported(mode)) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Invalid or unsupported traffic generator mode");
    return false;
  }
  if (mode == config_.traffic_generator_mode) {
    return true;
  }
  const RuntimeConfig previous_config = config_;
  config_.traffic_generator_mode = mode;
  if (!apply_traffic_runtime_config_(true, false)) {
    restore_traffic_runtime_config_(previous_config);
    return false;
  }
  const esp_err_t persist_err =
      config_.persist_runtime_overrides ? save_runtime_traffic_generator_mode(mode) : ESP_OK;
  if (persist_err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Failed to persist traffic generator mode: %s", esp_err_to_name(persist_err));
    restore_traffic_runtime_config_(previous_config);
    return false;
  }
  if (deferred_capture_action_ == DeferredCaptureAction::FinishTrafficApply) {
    if (config_.detection_algorithm == DetectionAlgorithm::LIGHTWEIGHT) {
      deferred_traffic_recalibrate_ = true;
    }
  } else if (config_.detection_algorithm == DetectionAlgorithm::LIGHTWEIGHT) {
    (void) trigger_recalibration();
  }
  ESPECTRE_LOGI(RUNTIME_TAG, "Traffic generator mode updated to %s", traffic_generator_mode_name(mode));
  return true;
}

bool EspIdfRuntime::set_detection_algorithm(DetectionAlgorithm algorithm) {
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  if (!capabilities_.supports_runtime_detector_selection ||
      !runtime_detection_algorithm_valid(algorithm)) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Runtime detector selection is unavailable or invalid");
    return false;
  }
  if (algorithm == config_.detection_algorithm) {
    return true;
  }

  const float threshold = runtime_default_threshold(algorithm);
  std::unique_ptr<BaseDetector> next_detector =
      make_detector_(algorithm, threshold, resolved_window_packets_);
  if (next_detector == nullptr) {
    notify_fault_("Failed to configure runtime detector");
    return false;
  }
  std::unique_ptr<StartupThresholdCalibrator> next_calibrator;
  if (algorithm == DetectionAlgorithm::LIGHTWEIGHT && csi_pipeline_.is_enabled()) {
    next_calibrator.reset(new (std::nothrow) StartupThresholdCalibrator());
    if (!next_calibrator) {
      notify_fault_("Failed to allocate startup calibrator");
      return false;
    }
  }
  const esp_err_t persist_err =
      config_.persist_runtime_overrides ? save_runtime_detection_algorithm(algorithm) : ESP_OK;
  if (persist_err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Failed to persist detector: %s", esp_err_to_name(persist_err));
    return false;
  }

  cancel_calibration_(true);
  detector_ = std::move(next_detector);
  calibrated_setup_.valid = false;
  csi_pipeline_.set_detector(detector_.get());
  config_.detection_algorithm = algorithm;
  config_.threshold = threshold;
  snapshot_.detector_name = detection_algorithm_name(algorithm);
  snapshot_.threshold = threshold;
  snapshot_.startup_threshold = 0.0f;
  snapshot_.movement_metric = 0.0f;
  snapshot_.motion_state = MotionState::IDLE;

  if (listener_ != nullptr) {
    listener_->on_detector_changed(get_snapshot());
    listener_->on_threshold_changed(get_snapshot());
  }
  ESPECTRE_LOGI(RUNTIME_TAG, "Detector changed to %s", detection_algorithm_name(algorithm));

  if (algorithm == DetectionAlgorithm::LIGHTWEIGHT && csi_pipeline_.is_enabled()) {
    return start_calibration_(true, std::move(next_calibrator));
  }
  return true;
}

bool EspIdfRuntime::trigger_recalibration() {
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  if (snapshot_.calibrating) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Calibration already in progress");
    return false;
  }

  ESPECTRE_LOGI(RUNTIME_TAG, "Manual recalibration triggered");
  return start_calibration_();
}

bool EspIdfRuntime::is_calibrating() const { return snapshot_.calibrating; }

RuntimeOperationState EspIdfRuntime::operation_state() const {
  return operation_state_.load(std::memory_order_acquire);
}

bool EspIdfRuntime::start_raw_collection(raw_csi_packet_callback_t callback, void *context) {
  if (!capabilities_.supports_raw_csi || callback == nullptr || !setup_complete_ || !services_armed_ ||
      !wifi_ready_ || wifi_ip_info_.ip.addr == 0U ||
      operation_state() != RuntimeOperationState::SENSING || csi_receive_path_refresh_in_progress_) {
    return false;
  }

  operation_state_.store(RuntimeOperationState::RAW_COLLECTION, std::memory_order_release);
  cancel_calibration_(true);
  snapshot_.ready_to_publish = false;
  snapshot_.motion_state = MotionState::IDLE;
  csi_pipeline_.set_motion_state_callback({});
  csi_pipeline_.set_live_telemetry_callback({});
  performance_diagnostics_.reset();
  if (!csi_pipeline_.start_raw_capture(callback, context)) {
    operation_state_.store(RuntimeOperationState::SENSING, std::memory_order_release);
    begin_capture_shutdown_(false);
    update_live_telemetry_callback_();
    (void) schedule_capture_action_(DeferredCaptureAction::DisableThenResumeSensing);
    return false;
  }
  refresh_csi_local_identity_(wifi_ip_info_.ip.addr);

  if (!csi_pipeline_.is_enabled()) {
    const CsiCaptureProfile profile = sensing_capture_profile_();
    snapshot_.csi_capture_profile = profile;
    const esp_err_t err = csi_pipeline_.enable({}, profile);
    if (err != ESP_OK) {
      csi_pipeline_.stop_raw_capture();
      update_live_telemetry_callback_();
      operation_state_.store(RuntimeOperationState::SENSING, std::memory_order_release);
      begin_capture_shutdown_(false);
      (void) schedule_capture_action_(DeferredCaptureAction::DisableThenResumeSensing);
      char message[96];
      std::snprintf(message, sizeof(message), "Failed to enable raw CSI: %s", esp_err_to_name(err));
      notify_fault_(message);
      return false;
    }
  }

  csi_receive_path_check_pending_ = false;
  ESPECTRE_LOGI(RUNTIME_TAG, "Entered raw CSI collection mode");
  return true;
}

bool EspIdfRuntime::stop_raw_collection(RawCsiStopReason reason) {
  if (operation_state() != RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }

  operation_state_.store(RuntimeOperationState::SENSING, std::memory_order_release);
  csi_pipeline_.stop_raw_capture();
  csi_pipeline_.set_traffic_filter({});
  // Classic ESP32 can stop delivering CSI when traffic spans a CSI disable/
  // enable transition. Disable capture only after the generator has left its
  // send, immediately when it is already quiet and otherwise from loop().
  begin_capture_shutdown_(false);
  update_live_telemetry_callback_();

  ESPECTRE_LOGI(RUNTIME_TAG, "Exited raw CSI collection mode: %u", static_cast<unsigned>(reason));
  (void) schedule_capture_action_(DeferredCaptureAction::DisableThenResumeSensing);
  return true;
}

CsiCaptureProfile EspIdfRuntime::sensing_capture_profile_() const {
  const bool requires_lltf = config_.traffic_generator_mode == TrafficGeneratorMode::WIFI_RAW;
  return select_csi_capture_profile(wifi_channel_, requires_lltf, config_.csi_capture_policy);
}

bool EspIdfRuntime::traffic_config_can_start_() const {
  return setup_complete_ && wifi_ready_ && services_armed_ && wifi_ip_info_.ip.addr != 0U &&
         csi_pipeline_.is_enabled();
}

bool EspIdfRuntime::finish_traffic_apply_(bool recalibrate_if_active) {
  deferred_traffic_recalibrate_ = false;
  if (!traffic_config_can_start_()) {
    return true;
  }
  const CsiCaptureProfile profile = sensing_capture_profile_();
  const bool profile_changed = profile != csi_pipeline_.capture_profile();
  if (profile_changed) {
    cancel_calibration_(true);
    const esp_err_t err = csi_pipeline_.reconfigure_capture(profile);
    if (err != ESP_OK) {
      notify_fault_("Failed to change CSI capture profile");
      return false;
    }
    csi_receive_path_callbacks_at_start_ = csi_pipeline_.capture_callback_invocations_total();
    snapshot_.csi_capture_profile = profile;
    snapshot_.motion_state = MotionState::IDLE;
    snapshot_.movement_metric = 0.0f;
    vTaskDelay(pdMS_TO_TICKS(CSI_ENABLE_SETTLE_MS));
  }
  refresh_csi_local_identity_(wifi_ip_info_.ip.addr);
  if (!csi_traffic_service_.is_running() &&
      !csi_traffic_service_.start(runtime_traffic_target_addr(config_, wifi_ip_info_.gw.addr))) {
    notify_fault_("Failed to start CSI traffic service");
    return false;
  }
  // A deferred launch reports the fault from loop(). Calibration waits until
  // that launch has a live worker, via finish_pending_sensing_start_().
  if (!csi_traffic_service_.source_is_active()) {
    sensing_start_pending_ = true;
  }
  // A new profile is a new arm. LLTF20 callbacks at startup say nothing about
  // HT20, which can stay silent on S3 and C5 until the receive path refresh.
  if (profile_changed) {
    request_csi_receive_path_check_(false);
  }
  if (!csi_traffic_service_.source_is_active()) {
    return true;
  }
  if (recalibrate_if_active && !sensing_start_pending_ &&
      config_.detection_algorithm == DetectionAlgorithm::LIGHTWEIGHT) {
    (void) trigger_recalibration();
  }
  return true;
}

bool EspIdfRuntime::apply_traffic_runtime_config_(bool restart_service, bool recalibrate_if_active) {
  if (restart_service) {
    csi_traffic_service_.stop();
  }
  csi_traffic_service_.init(to_csi_traffic_config(config_));
  if (!traffic_config_can_start_()) {
    return true;
  }
  if (!csi_traffic_service_.is_quiescent()) {
    deferred_traffic_recalibrate_ = recalibrate_if_active;
    deferred_capture_action_ = DeferredCaptureAction::FinishTrafficApply;
    return true;
  }
  return finish_traffic_apply_(recalibrate_if_active);
}

void EspIdfRuntime::restore_traffic_runtime_config_(const RuntimeConfig &previous_config) {
  config_ = previous_config;
  if (!apply_traffic_runtime_config_(true, true)) {
    notify_fault_("Failed to restore CSI traffic configuration");
  }
}

bool EspIdfRuntime::configure_detector_() {
  if (!validate_runtime_uint32(config_.csi_target_pps,
                               RUNTIME_CSI_TARGET_PPS_MIN,
                               RUNTIME_CSI_TARGET_PPS_MAX)) {
    notify_fault_("Invalid CSI target PPS");
    return false;
  }
  if (!validate_runtime_uint32(config_.window_size_ms,
                               RUNTIME_WINDOW_SIZE_MS_MIN,
                               RUNTIME_WINDOW_SIZE_MS_MAX)) {
    notify_fault_("Invalid detector window duration");
    return false;
  }
  if (!validate_runtime_threshold_for_algorithm(config_.threshold,
                                                config_.detection_algorithm)) {
    notify_fault_("Invalid segmentation threshold");
    return false;
  }
  const float threshold = config_.threshold;
  snapshot_.threshold = threshold;
  resolved_window_packets_ = static_cast<uint16_t>(temporal_window_slots(
      config_.csi_target_pps, config_.window_size_ms));
  detector_ = make_detector_(config_.detection_algorithm, threshold, resolved_window_packets_);

  if (detector_ == nullptr) {
    notify_fault_("Failed to configure detector");
    return false;
  }

  snapshot_.detector_name = detection_algorithm_name(config_.detection_algorithm);
  return true;
}

std::unique_ptr<BaseDetector> EspIdfRuntime::make_detector_(DetectionAlgorithm algorithm,
                                                            float threshold,
                                                            uint16_t window_packets) {
  std::unique_ptr<BaseDetector> detector;
  if (algorithm == DetectionAlgorithm::HIGH_ACCURACY) {
    detector.reset(new (std::nothrow) HighAccuracyDetector(window_packets, threshold));
  } else if (algorithm == DetectionAlgorithm::LIGHTWEIGHT) {
    detector.reset(new (std::nothrow) LightweightDetector(window_packets, threshold));
  }
  if (detector != nullptr && !detector->is_valid()) {
    detector.reset();
  }
  if (detector != nullptr) {
    detector->configure_lowpass(config_.lowpass_enabled, config_.lowpass_cutoff);
    detector->configure_hampel(config_.hampel_enabled, config_.hampel_window, config_.hampel_threshold);
  }
  return detector;
}

void EspIdfRuntime::cancel_calibration_(bool notify_listener) {
  const bool was_calibrating = snapshot_.calibrating;
  threshold_calibration_active_.store(false, std::memory_order_relaxed);
  calibration_finished_event_.clear();
  csi_pipeline_.set_packet_interceptor(nullptr, nullptr);
  threshold_calibrator_.reset();
  calibration_motion_guard_.disable();
  if (was_calibrating && detector_ != nullptr) {
    detector_->on_startup_calibration_abandoned();
  }
  snapshot_.calibrating = false;
  snapshot_.calibration_packets = 0U;
  snapshot_.calibration_target_packets = 0U;
  if (was_calibrating && notify_listener && listener_ != nullptr) {
    listener_->on_calibration_finished(get_snapshot(), false);
  }
}

void EspIdfRuntime::on_wifi_connected_(const esp_netif_ip_info_t &ip_info) {
  if (ip_info.ip.addr == 0U) {
    return;
  }

  wifi_ready_ = true;
  wifi_ip_info_ = ip_info;
  wifi_ap_record_t ap_info{};
  if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
    wifi_rssi_dbm_ = ap_info.rssi;
    wifi_channel_ = ap_info.primary;
  } else {
    wifi_rssi_dbm_ = INT8_MIN;
    wifi_channel_ = 0U;
  }
  if (!services_armed_) {
    ESPECTRE_LOGI(RUNTIME_TAG, "Wi-Fi connected, CSI services not ready to resume");
    return;
  }

  maybe_resume_sensing_after_wifi_reconfigure_();
}

void EspIdfRuntime::on_wifi_disconnected_() {
  wifi_ready_ = false;
  wifi_ip_info_ = {};
  wifi_rssi_dbm_ = INT8_MIN;
  wifi_channel_ = 0U;
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    (void) stop_raw_collection(RawCsiStopReason::WIFI_LOST);
    return;
  }
  invalidate_csi_receive_path_refresh_();
  csi_receive_path_check_pending_ = false;
  stop_sensing_services_();
}

void EspIdfRuntime::invalidate_csi_receive_path_refresh_() {
  if (!csi_receive_path_refresh_in_progress_) {
    return;
  }

  wifi_lifecycle_.cancel_csi_receive_path_refresh();
  csi_receive_path_refresh_in_progress_ = false;
  csi_receive_path_check_pending_ = false;
}

void EspIdfRuntime::maybe_resume_sensing_after_wifi_reconfigure_() {
  if (!services_armed_ || !wifi_ready_ ||
      wifi_ip_info_.ip.addr == 0U || csi_receive_path_refresh_in_progress_) {
    return;
  }

  csi_receive_path_callbacks_at_start_ = csi_pipeline_.capture_callback_invocations_total();
  start_sensing_services_(wifi_ip_info_);
  request_csi_receive_path_check_(false);
}

void EspIdfRuntime::request_csi_receive_path_check_(bool after_refresh) {
  csi_receive_path_after_refresh_ = after_refresh;
  if (sensing_start_pending_) {
    arm_receive_path_check_when_traffic_starts_ = true;
    return;
  }
  arm_csi_receive_path_check_();
}

void EspIdfRuntime::check_csi_receive_path_() {
  if (!csi_receive_path_check_pending_ || csi_receive_path_refresh_in_progress_ ||
      !services_armed_ || !wifi_ready_ || !csi_pipeline_.is_enabled()) {
    return;
  }
  // Any hardware callback proves that the receive path works. Rejected packets
  // and a low movement score are not reasons to disturb the radio.
  if (csi_pipeline_.capture_callback_invocations_total() != csi_receive_path_callbacks_at_start_) {
    csi_receive_path_check_pending_ = false;
    return;
  }

  const uint64_t traffic = (csi_traffic_service_.mode() != TrafficGeneratorMode::EXTERNAL
                               ? csi_traffic_service_.get_generator_packets_total()
                               : csi_traffic_service_.get_packets_received());
  const uint32_t now = monotonic_now_ms();
  if (traffic < csi_receive_path_traffic_total_) {
    csi_receive_path_traffic_total_ = traffic;
    csi_receive_path_traffic_seen_ = false;
  }
  if (!csi_receive_path_traffic_seen_) {
    if (traffic == csi_receive_path_traffic_total_) return;
    csi_receive_path_traffic_seen_ = true;
    csi_receive_path_traffic_total_ = traffic;
    csi_receive_path_check_started_ms_ = now;
    csi_receive_path_last_traffic_ms_ = now;
    csi_receive_path_last_attempt_ms_ = now;
    return;
  }
  if (traffic != csi_receive_path_traffic_total_) {
    csi_receive_path_traffic_total_ = traffic;
    csi_receive_path_last_traffic_ms_ = now;
  }
  const uint32_t elapsed = now - csi_receive_path_check_started_ms_;
  if (elapsed >= CSI_REFRESH_REQUEST_WINDOW_MS) {
    csi_receive_path_check_pending_ = false;
    ESPECTRE_LOGW(RUNTIME_TAG, "CSI receive-path refresh deferred too long; continuing capture");
    return;
  }
  if (now - csi_receive_path_last_traffic_ms_ >= CSI_STARTUP_TRAFFIC_IDLE_MS) {
    // A short burst followed by silence is not a CSI receive-path fault.
    csi_receive_path_traffic_seen_ = false;
    return;
  }
  if (elapsed < std::max(CSI_STARTUP_OBSERVATION_MS, config_.window_size_ms)) return;
  if (csi_receive_path_after_refresh_) {
    // The refresh is the only recovery, so a path still silent under traffic
    // would otherwise leave sensing calibrating with no report at all.
    csi_receive_path_check_pending_ = false;
    notify_fault_("No CSI callbacks with active traffic after a receive-path refresh");
    return;
  }
  if (now - csi_receive_path_last_attempt_ms_ < CSI_REFRESH_RETRY_INTERVAL_MS) return;
  csi_receive_path_last_attempt_ms_ = now;

  const esp_err_t err = wifi_lifecycle_.refresh_csi_receive_path(
      [this](esp_err_t result) { finish_csi_receive_path_refresh_(result); },
      !config_.wifi_scan_results_managed_externally);
  if (err == ESP_ERR_INVALID_STATE || err == ESP_ERR_WIFI_STATE) {
    return;  // Leave the other scan/connection and the current capture intact.
  }
  // One accepted attempt per sensing session, including a failed completion.
  csi_receive_path_check_pending_ = false;
  if (err != ESP_OK) {
    ESPECTRE_LOGW(RUNTIME_TAG, "Could not refresh the CSI receive path: %s; continuing capture",
                  esp_err_to_name(err));
    return;
  }
  csi_receive_path_refresh_in_progress_ = true;
  stop_sensing_services_();
  ESPECTRE_LOGW(RUNTIME_TAG,
                "No CSI callbacks after %u ms of traffic; refreshing the receive path",
                static_cast<unsigned>(elapsed));
}

void EspIdfRuntime::finish_csi_receive_path_refresh_(esp_err_t result) {
  csi_receive_path_refresh_in_progress_ = false;
  if (result == ESP_OK) {
    ESPECTRE_LOGI(RUNTIME_TAG, "CSI receive-path refresh completed; resuming sensing");
  } else {
    ESPECTRE_LOGW(RUNTIME_TAG, "CSI receive-path refresh failed: %s; resuming sensing",
                  esp_err_to_name(result));
  }
  if (services_armed_ && wifi_ready_ && wifi_ip_info_.ip.addr != 0U) {
    csi_receive_path_callbacks_at_start_ = csi_pipeline_.capture_callback_invocations_total();
    start_sensing_services_(wifi_ip_info_);
    request_csi_receive_path_check_(true);
  }
}

void EspIdfRuntime::refresh_wifi_association_from_csi_() {
  if (!wifi_ready_) return;
  const int8_t rssi_dbm = csi_pipeline_.last_rssi_dbm();
  const uint8_t channel = csi_pipeline_.last_channel();
  if (rssi_dbm != INT8_MIN) wifi_rssi_dbm_ = rssi_dbm;
  if (channel != 0U) wifi_channel_ = channel;
}

void EspIdfRuntime::start_sensing_services_(const esp_netif_ip_info_t &ip_info) {
  // Capture must go through its pending disable before it is armed again, so
  // a start that arrives first runs after that disable, from loop(). A pending
  // rearm starts sensing itself once its disable has run.
  if (capture_action_disables_(deferred_capture_action_)) {
    if (deferred_capture_action_ == DeferredCaptureAction::Disable) {
      deferred_capture_action_ = DeferredCaptureAction::DisableThenResumeSensing;
    }
    sensing_start_pending_ = true;
    return;
  }
  if (deferred_capture_action_ == DeferredCaptureAction::ResumeSensing) {
    deferred_capture_action_ = DeferredCaptureAction::None;
  }
  capture_updates_suppressed_ = false;
  snapshot_.motion_state = MotionState::IDLE;
  snapshot_.ready_to_publish = false;

  csi_pipeline_.set_motion_state_callback([this](MotionState state) {
    snapshot_.motion_state = state;
    if (snapshot_.ready_to_publish && listener_ != nullptr) {
      listener_->on_motion_state_changed(get_snapshot());
    }
  });
  refresh_csi_local_identity_(ip_info.ip.addr);

  if (!csi_pipeline_.is_enabled()) {
    // Keep associated CSI capture strictly non-promiscuous, including after a
    // station stop/start cycle has rebuilt the Wi-Fi control block.
    const esp_err_t promiscuous_err = esp_wifi_set_promiscuous(false);
    if (promiscuous_err != ESP_OK) {
      char message[112];
      std::snprintf(message, sizeof(message), "Failed to disable promiscuous mode before CSI: %s",
                    esp_err_to_name(promiscuous_err));
      notify_fault_(message);
      return;
    }

    const CsiCaptureProfile profile = sensing_capture_profile_();
    snapshot_.csi_capture_profile = profile;
    const esp_err_t err = csi_pipeline_.enable([this](MotionState state, uint32_t packets_received) {
      if (capture_updates_suppressed_) {
        return;
      }
      snapshot_.motion_state = state;
      // A detector that is not ready has cleared its metric. Keep the last
      // one alongside the held state, as live telemetry does.
      if (detector_ == nullptr) {
        snapshot_.movement_metric = 0.0f;
      } else if (detector_->is_ready()) {
        snapshot_.movement_metric = detector_->get_motion_metric();
      }
      if (detector_ != nullptr) {
        notify_threshold_if_changed_(detector_->get_threshold());
      }
      snapshot_.link_rssi_dbm = csi_pipeline_.last_rssi_dbm();
      snapshot_.link_channel = csi_pipeline_.last_channel();

      if (snapshot_.ready_to_publish) {
        log_periodic_status_(packets_received);
        if (listener_ != nullptr) {
          listener_->on_periodic_update(get_snapshot(), packets_received);
        }
      }
    }, profile);
    if (err != ESP_OK) {
      char message[96];
      std::snprintf(message, sizeof(message), "Failed to enable CSI: %s", esp_err_to_name(err));
      notify_fault_(message);
      return;
    }
  }

  // esp_wifi_set_csi(true) returns before the classic ESP32 receive path is
  // always ready to associate new traffic with the CSI callback. Starting the
  // source in the same scheduler slice reproduces esp-csi#247 intermittently.
  // Yield once after arming so managed traffic cannot predate driver readiness.
  vTaskDelay(pdMS_TO_TICKS(CSI_ENABLE_SETTLE_MS));

  if (deferred_capture_action_ == DeferredCaptureAction::FinishTrafficApply) {
    sensing_start_pending_ = true;
    return;
  }
  if (!csi_traffic_service_.is_running() &&
      !csi_traffic_service_.start(runtime_traffic_target_addr(config_, ip_info.gw.addr))) {
    notify_fault_("Failed to start CSI traffic service");
    return;
  }
  if (!csi_traffic_service_.source_is_active()) {
    sensing_start_pending_ = true;
    return;
  }

  sensing_start_pending_ = false;
  start_calibration_(false);
  snapshot_.ready_to_publish = true;
  reset_periodic_status_logger_();
}

void EspIdfRuntime::begin_capture_shutdown_(bool notify_listener) {
  cancel_calibration_(false);
  csi_pipeline_.set_traffic_filter({});
  csi_pipeline_.set_motion_state_callback({});
  capture_updates_suppressed_ = true;
  csi_traffic_service_.stop();
  snapshot_.ready_to_publish = false;
  snapshot_.motion_state = MotionState::IDLE;
  if (notify_listener && listener_ != nullptr) {
    listener_->on_motion_state_changed(get_snapshot());
  }
}

void EspIdfRuntime::stop_sensing_services_() {
  begin_capture_shutdown_(true);
  (void) schedule_capture_action_(DeferredCaptureAction::Disable);
}

bool EspIdfRuntime::traffic_allows_radio_work() const {
  // Running traffic does not block the radio, as before stop() became
  // asynchronous; raw collection keeps it running across a reconfigure.
  return !csi_traffic_service_.generator_is_stopping() &&
         !capture_action_disables_(deferred_capture_action_);
}

void EspIdfRuntime::hold_pending_traffic_restart(bool hold) {
  hold_traffic_restart_ = hold;
}

void EspIdfRuntime::arm_csi_receive_path_check_() {
  csi_receive_path_check_pending_ = csi_pipeline_.is_enabled() && csi_traffic_service_.source_is_active();
  csi_receive_path_traffic_total_ = (csi_traffic_service_.mode() != TrafficGeneratorMode::EXTERNAL
                                         ? csi_traffic_service_.get_generator_packets_total()
                                         : csi_traffic_service_.get_packets_received());
  csi_receive_path_traffic_seen_ = false;
}

void EspIdfRuntime::finish_pending_sensing_start_() {
  if (!sensing_start_pending_ || !csi_traffic_service_.source_is_active()) {
    return;
  }
  sensing_start_pending_ = false;
  deferred_traffic_recalibrate_ = false;
  start_calibration_(false);
  snapshot_.ready_to_publish = true;
  reset_periodic_status_logger_();
  if (!arm_receive_path_check_when_traffic_starts_) {
    return;
  }
  arm_receive_path_check_when_traffic_starts_ = false;
  csi_receive_path_callbacks_at_start_ = csi_pipeline_.capture_callback_invocations_total();
  arm_csi_receive_path_check_();
}

EspIdfRuntime::CaptureActionResult EspIdfRuntime::schedule_capture_action_(DeferredCaptureAction action) {
  if (!csi_traffic_service_.is_quiescent()) {
    deferred_capture_action_ = action;
    return CaptureActionResult::Deferred;
  }
  deferred_capture_action_ = DeferredCaptureAction::None;
  return run_capture_action_(action);
}

bool EspIdfRuntime::capture_action_disables_(DeferredCaptureAction action) {
  return action == DeferredCaptureAction::Disable ||
         action == DeferredCaptureAction::DisableThenResumeSensing ||
         action == DeferredCaptureAction::DisableThenRearm;
}

EspIdfRuntime::DeferredCaptureAction EspIdfRuntime::disable_capture_for_(DeferredCaptureAction action) {
  const esp_err_t err = csi_pipeline_.disable();
  switch (action) {
    case DeferredCaptureAction::DisableThenResumeSensing:
      return DeferredCaptureAction::ResumeSensing;
    case DeferredCaptureAction::DisableThenRearm:
      if (err != ESP_OK && csi_pipeline_.is_enabled()) {
        char message[96];
        std::snprintf(message, sizeof(message), "Failed to rearm CSI after channel change: %s",
                      esp_err_to_name(err));
        notify_fault_(message);
        return DeferredCaptureAction::None;
      }
      return DeferredCaptureAction::Rearm;
    default:
      return DeferredCaptureAction::None;
  }
}

EspIdfRuntime::CaptureActionResult EspIdfRuntime::run_capture_action_(DeferredCaptureAction action) {
  switch (action) {
    case DeferredCaptureAction::None:
      return CaptureActionResult::Done;
    case DeferredCaptureAction::Disable:
    case DeferredCaptureAction::DisableThenResumeSensing:
    case DeferredCaptureAction::DisableThenRearm: {
      const DeferredCaptureAction next = disable_capture_for_(action);
      if (action == DeferredCaptureAction::DisableThenRearm && next == DeferredCaptureAction::None) {
        return CaptureActionResult::Failed;
      }
      return run_capture_action_(next);
    }
    case DeferredCaptureAction::ResumeSensing:
      if (services_armed_ && wifi_ready_ && wifi_ip_info_.ip.addr != 0U) {
        start_sensing_services_(wifi_ip_info_);
      } else {
        // A start deferred behind the disable no longer applies.
        sensing_start_pending_ = false;
        arm_receive_path_check_when_traffic_starts_ = false;
        if (listener_ != nullptr) {
          listener_->on_motion_state_changed(get_snapshot());
        }
      }
      return CaptureActionResult::Done;
    case DeferredCaptureAction::Rearm:
      // The rearm starts sensing again when it still applies.
      sensing_start_pending_ = false;
      arm_receive_path_check_when_traffic_starts_ = false;
      on_wifi_connected_(deferred_rearm_ip_);
      return CaptureActionResult::Done;
    case DeferredCaptureAction::FinishTrafficApply:
      return finish_traffic_apply_(deferred_traffic_recalibrate_) ? CaptureActionResult::Done
                                                                   : CaptureActionResult::Failed;
  }
  return CaptureActionResult::Done;
}

void EspIdfRuntime::service_deferred_capture_action_() {
  // A station reconfigure or scan that has not touched the driver yet must see
  // this iteration stay quiet. Launching here would put a sender back on the
  // radio before that driver call runs.
  const bool hold_launch = hold_traffic_restart_;
  const bool action_pending = deferred_capture_action_ != DeferredCaptureAction::None;
  csi_traffic_service_.hold_pending_restart(hold_launch || action_pending);
  csi_traffic_service_.loop();
  if (csi_traffic_service_.consume_generator_start_failure()) {
    sensing_start_pending_ = false;
    arm_receive_path_check_when_traffic_starts_ = false;
    notify_fault_("Failed to start CSI traffic service");
  }
  if (csi_traffic_service_.consume_generator_stop_timeout()) {
    notify_fault_("Traffic generator did not stop within 30 s");
  }
  if (action_pending && csi_traffic_service_.is_quiescent()) {
    // The held radio work waits for this disable, so it runs even while held.
    // Only the part that restarts traffic waits for the hold to clear.
    if (capture_action_disables_(deferred_capture_action_)) {
      deferred_capture_action_ = disable_capture_for_(deferred_capture_action_);
    }
    if (!hold_launch && deferred_capture_action_ != DeferredCaptureAction::None) {
      const DeferredCaptureAction action = deferred_capture_action_;
      deferred_capture_action_ = DeferredCaptureAction::None;
      (void) run_capture_action_(action);
    }
  }
  csi_traffic_service_.hold_pending_restart(
      hold_launch || deferred_capture_action_ != DeferredCaptureAction::None);
  if (!hold_launch && deferred_capture_action_ == DeferredCaptureAction::None) {
    csi_traffic_service_.loop();
  }
  finish_pending_sensing_start_();
}

void EspIdfRuntime::on_csi_channel_changed_(uint8_t previous_channel, uint8_t current_channel) {
  if (operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    ESPECTRE_LOGW(RUNTIME_TAG,
             "Ending raw collection after Wi-Fi channel change: %u -> %u",
             static_cast<unsigned>(previous_channel),
             static_cast<unsigned>(current_channel));
    (void) stop_raw_collection(RawCsiStopReason::CHANNEL_CHANGED);
    return;
  }
  if (!wifi_ready_ || !services_armed_ || wifi_ip_info_.ip.addr == 0U || !csi_pipeline_.is_enabled()) {
    return;
  }

  ESPECTRE_LOGW(RUNTIME_TAG,
           "Rearming CSI session after Wi-Fi channel change: %u -> %u",
           static_cast<unsigned>(previous_channel),
           static_cast<unsigned>(current_channel));

  const esp_netif_ip_info_t ip_info = wifi_ip_info_;
  begin_capture_shutdown_(true);
  const CaptureActionResult result = schedule_capture_action_(DeferredCaptureAction::Disable);
  if (result == CaptureActionResult::Deferred) {
    deferred_rearm_ip_ = ip_info;
    deferred_capture_action_ = DeferredCaptureAction::DisableThenRearm;
    return;
  }
  if (result == CaptureActionResult::Failed || csi_pipeline_.is_enabled()) {
    if (csi_pipeline_.is_enabled()) {
      notify_fault_("Failed to rearm CSI after channel change");
    }
    return;
  }
  on_wifi_connected_(ip_info);
}

bool EspIdfRuntime::start_calibration_(bool reset_high_accuracy_threshold,
                                      std::unique_ptr<StartupThresholdCalibrator> prepared) {
  if (config_.detection_algorithm == DetectionAlgorithm::LIGHTWEIGHT && !prepared) {
    prepared.reset(new (std::nothrow) StartupThresholdCalibrator());
    if (!prepared) {
      notify_fault_("Failed to allocate startup calibrator");
      return false;
    }
  }

  if (config_.detection_algorithm == DetectionAlgorithm::HIGH_ACCURACY) {
    const float threshold = reset_high_accuracy_threshold
                                ? runtime_default_threshold(DetectionAlgorithm::HIGH_ACCURACY)
                                : config_.threshold;
    if (detector_ != nullptr) {
      detector_->set_threshold(threshold);
    }
    config_.threshold = threshold;
    snapshot_.threshold = threshold;
    snapshot_.startup_threshold = threshold;
    snapshot_.calibrating = false;
    snapshot_.calibration_packets = 0U;
    snapshot_.calibration_target_packets = 0U;
    if (listener_ != nullptr) {
      listener_->on_threshold_changed(get_snapshot());
      listener_->on_calibration_finished(get_snapshot(), true);
    }
    return true;
  }

  const uint32_t calibration_duration_ms =
      config_.window_size_ms * CALIBRATION_NUM_WINDOWS;
  uint32_t calibration_target_packets = temporal_window_slots(
      config_.csi_target_pps, calibration_duration_ms);
  if (calibration_target_packets > UINT16_MAX) {
    calibration_target_packets = UINT16_MAX;
  }

  const bool notify_started = !snapshot_.calibrating;

  // Calibrate on the runtime detector itself (cold-cleared first), so the
  // observed metric matches the configured algorithm. Mirrors the Python
  // runtime calibration flow.
  threshold_calibrator_ = std::move(prepared);
  threshold_calibrator_->begin(static_cast<uint16_t>(calibration_target_packets),
                               detector_ != nullptr && detector_->startup_gate_enabled());
  // No calibration may learn its quiet baseline from motion. The detector's
  // absolute ceiling guards every calibration; a recalibration under the setup
  // that produced the live threshold also restarts on what that threshold
  // would report.
  float motion_reference =
      detector_ != nullptr ? detector_->calibration_motion_ceiling()
                           : std::numeric_limits<float>::infinity();
  const bool trusted_threshold =
      detector_ != nullptr && calibrated_setup_.valid &&
      calibrated_setup_.channel == wifi_channel_ &&
      calibrated_setup_.capture_profile == csi_pipeline_.capture_profile() &&
      calibrated_setup_.traffic_generator_mode == config_.traffic_generator_mode;
  if (trusted_threshold) {
    motion_reference = std::min(motion_reference, detector_->get_threshold());
  }
  if (std::isfinite(motion_reference)) {
    calibration_motion_guard_.begin(
        motion_reference, calibration_target_packets,
        calibration_target_packets * CALIBRATION_MOTION_BUDGET_WINDOWS);
  } else {
    calibration_motion_guard_.disable();
  }
  calibration_finished_event_.clear();
  snapshot_.calibrating = true;
  snapshot_.calibration_packets = 0U;
  snapshot_.calibration_target_packets = static_cast<uint16_t>(calibration_target_packets);
  threshold_calibration_active_.store(true, std::memory_order_relaxed);
  csi_pipeline_.clear_detector_buffer();
  if (detector_ != nullptr) {
    detector_->on_startup_calibration_begin();
  }
  csi_pipeline_.set_packet_interceptor(&EspIdfRuntime::threshold_calibration_packet_callback_, this);
  ESPECTRE_LOGI(RUNTIME_TAG, "Starting %s threshold calibration with fixed subcarriers",
           detector_ != nullptr ? detector_->get_name() : "detector");
  if (notify_started && listener_ != nullptr) {
    listener_->on_calibration_started(get_snapshot());
  }
  return true;
}

bool EspIdfRuntime::handle_threshold_calibration_packet_(const int8_t *csi_data, size_t csi_len,
                                                         int8_t rssi_dbm, bool evaluation_due,
                                                         uint32_t packets_in_window,
                                                         bool temporal_reset) {
  if (!threshold_calibration_active_.load(std::memory_order_relaxed) || detector_ == nullptr ||
      !threshold_calibrator_) {
    return false;
  }

  if (temporal_reset) {
    const uint16_t target_packets = threshold_calibrator_->base_target_packets();
    threshold_calibrator_->begin(target_packets, detector_->startup_gate_enabled());
    snapshot_.calibration_packets = 0U;
    snapshot_.calibration_target_packets = target_packets;
    detector_->on_startup_calibration_begin();
  }

  detector_->process_packet(csi_data, csi_len, SELECTED_SUBCARRIERS.data(),
                            HT20_SELECTED_BAND_SIZE, rssi_dbm);
  // The pipeline owns the cadence, so calibration evaluates exactly when
  // steady-state detection would and the threshold is fitted at the resolution
  // the detector will run at.
  if (!evaluation_due) {
    return true;
  }

  const uint16_t packet_weight =
      static_cast<uint16_t>(std::min<uint32_t>(std::max<uint32_t>(packets_in_window, 1U), UINT16_MAX));
  detector_->update_state();
  if (!detector_->is_ready()) {
    calibration_motion_guard_.skip(packet_weight);
    return true;
  }
  switch (calibration_motion_guard_.observe(detector_->get_motion_metric(), packet_weight)) {
    case CalibrationMotionGuard::Verdict::RESTART:
      // The window still holds the motion; refill it before collecting
      // quiet evidence again.
      threshold_calibrator_->begin(threshold_calibrator_->base_target_packets(),
                                   detector_->startup_gate_enabled());
      detector_->clear_buffer();
      detector_->on_startup_calibration_begin();
      snapshot_.calibration_packets = 0U;
      snapshot_.calibration_target_packets = threshold_calibrator_->target_packets();
      return true;
    case CalibrationMotionGuard::Verdict::REJECT:
      threshold_calibration_active_.store(false, std::memory_order_relaxed);
      calibration_finished_event_.post(false);
      return true;
    case CalibrationMotionGuard::Verdict::CONTINUE:
      break;
  }
  threshold_calibrator_->observe(true, detector_->get_motion_metric(), packet_weight);

  snapshot_.calibration_packets = threshold_calibrator_->packet_count();
  snapshot_.calibration_target_packets = threshold_calibrator_->target_packets();

  // Evidence the detector cannot read yet, such as a burst it has not seen
  // the end of, extends the calibration in steps instead of concluding.
  if (threshold_calibrator_->is_complete() &&
      threshold_calibrator_->extend_if_inconclusive(detector_->startup_calibration_conclusive())) {
    snapshot_.calibration_target_packets = threshold_calibrator_->target_packets();
    return true;
  }
  if (threshold_calibrator_->is_complete()) {
    snapshot_.calibration_packets = snapshot_.calibration_target_packets;
    threshold_calibration_active_.store(false, std::memory_order_relaxed);
    calibration_finished_event_.post(threshold_calibrator_->is_successful());
  }
  return true;
}

bool EspIdfRuntime::threshold_calibration_packet_callback_(void *context,
                                                           const int8_t *csi_data,
                                                           size_t csi_len,
                                                           int8_t rssi_dbm,
                                                           bool evaluation_due,
                                                           uint32_t packets_in_window,
                                                           bool temporal_reset) {
  auto *runtime = static_cast<EspIdfRuntime *>(context);
  return runtime != nullptr &&
         runtime->handle_threshold_calibration_packet_(csi_data, csi_len, rssi_dbm, evaluation_due,
                                                       packets_in_window, temporal_reset);
}

void EspIdfRuntime::finish_threshold_calibration_(bool success) {
  threshold_calibration_active_.store(false, std::memory_order_relaxed);
  csi_pipeline_.set_packet_interceptor(nullptr, nullptr);
  snapshot_.calibrating = false;
  snapshot_.calibration_packets = 0U;
  snapshot_.calibration_target_packets = 0U;

  bool threshold_changed = false;
  if (success && threshold_calibrator_) {
    const float auto_factor = detector_ != nullptr
                                  ? detector_->get_startup_threshold_factor()
                                  : DEFAULT_ADAPTIVE_FACTOR;
    const float adaptive_threshold = threshold_calibrator_->threshold_metric() * auto_factor;
    snapshot_.startup_threshold = adaptive_threshold;
    if (detector_ != nullptr) {
      detector_->on_startup_calibration_complete();
    }

    if (detector_ != nullptr) {
      detector_->set_adaptive_threshold(adaptive_threshold);
      const float applied_threshold = detector_->get_threshold();
      config_.threshold = applied_threshold;
      snapshot_.startup_threshold = applied_threshold;
      snapshot_.threshold = applied_threshold;
      threshold_changed = true;
      // A detector that owns its formula ignores the generic calibrator
      // metric, so only the applied threshold is meaningful.
      ESPECTRE_LOGD(RUNTIME_TAG, "Adaptive threshold: %.6f", applied_threshold);
    }
    csi_pipeline_.clear_detector_buffer();
    calibrated_setup_.valid = true;
    calibrated_setup_.channel = wifi_channel_;
    calibrated_setup_.capture_profile = csi_pipeline_.capture_profile();
    calibrated_setup_.traffic_generator_mode = config_.traffic_generator_mode;
  } else if (detector_ != nullptr) {
    detector_->on_startup_calibration_abandoned();
  }

  if (calibration_motion_guard_.rejected()) {
    ESPECTRE_LOGW(RUNTIME_TAG,
                  "Calibration rejected: motion kept crossing %.6f; keeping threshold %.6f",
                  calibration_motion_guard_.reference_threshold(),
                  detector_ != nullptr ? detector_->get_threshold() : 0.0f);
  } else if (calibration_motion_guard_.restarts() > 0U) {
    ESPECTRE_LOGI(RUNTIME_TAG, "Calibration restarted %u time(s) after motion",
                  static_cast<unsigned>(calibration_motion_guard_.restarts()));
  }
  calibration_motion_guard_.disable();
  ESPECTRE_LOGD(RUNTIME_TAG, "Calibration %s", success ? "completed successfully" : "failed");
  // Finish this operation before callbacks can start another calibration.
  threshold_calibrator_.reset();
  reset_periodic_status_logger_();
  const RuntimeSnapshot completed_snapshot = get_snapshot();
  if (threshold_changed && listener_ != nullptr) {
    listener_->on_threshold_changed(completed_snapshot);
  }
  if (listener_ != nullptr) {
    listener_->on_calibration_finished(completed_snapshot, success);
  }
}

void EspIdfRuntime::log_periodic_status_(uint32_t packets_received) {
  latest_diagnostics_ = diagnostics_sampler_.sample(get_diagnostics(), monotonic_now_ms());
  status_logger_.log_status(RUNTIME_TAG, snapshot_, packets_received, &latest_diagnostics_);
}

void EspIdfRuntime::reset_periodic_status_logger_() {
  const RuntimeDiagnosticsSnapshot diagnostics = get_diagnostics();
  const uint32_t now_ms = monotonic_now_ms();
  diagnostics_sampler_.reset(diagnostics, now_ms);
  latest_diagnostics_ = diagnostics_sampler_.sample(diagnostics, now_ms);
}

void EspIdfRuntime::refresh_csi_local_identity_(uint32_t local_ip_addr) {
  CsiFrameFilterConfig filter;
  filter.traffic_mode = config_.traffic_generator_mode;
  filter.local_ip_addr = local_ip_addr;
  filter.internal_target_ip_addr = runtime_traffic_target_addr(config_, wifi_ip_info_.gw.addr);
  filter.multicast_ip_addr = config_.csi_traffic_multicast_group.empty()
                                 ? 0U
                                 : inet_addr(config_.csi_traffic_multicast_group.c_str());
  filter.external_udp_port = config_.csi_traffic_udp_port;
  filter.internal_icmp_identifier = csi_traffic_service_.internal_icmp_identifier();
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK) {
    std::memcpy(filter.local_mac_addr, mac, sizeof(filter.local_mac_addr));
  }
  csi_pipeline_.set_traffic_filter(filter);
}

}  // namespace espectre
