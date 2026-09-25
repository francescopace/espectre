/*
 * ESPectre - Runtime Frontend Controller
 *
 * Owns runtime lifecycle and exposes a frontend-friendly control surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_frontend_controller.h"

#include "core/espectre_log.h"
#include "esp_idf_runtime.h"
#include "runtime/runtime_config_utils.h"
#include "runtime/runtime_time.h"
#include "runtime_performance_diagnostics.h"

#include <algorithm>
#include <cstdint>
#include <new>

namespace espectre {

namespace {

static const char *const TAG = "espectre.runtime";
static constexpr SelectedSubcarriers SELECTED_SUBCARRIERS = make_default_subcarriers();

}  // namespace

// Traffic sources that outlive each backend, so a generator worker still
// inside a socket call after shutdown() keeps a valid owner.
struct RuntimeTrafficSources {
  TrafficGeneratorManager generator;
  UDPListener ingress;
};

RuntimeFrontendController::RuntimeFrontendController() = default;

// The listener may already be partly destroyed, so scope exit sends no callback.
RuntimeFrontendController::~RuntimeFrontendController() { shutdown_(false); }

const SelectedSubcarriers &RuntimeFrontendController::subcarriers() const { return SELECTED_SUBCARRIERS; }

void RuntimeFrontendController::set_config(const RuntimeConfig &config) {
  if (runtime_) {
    return;
  }
  config_ = config;
  snapshot_.threshold = config_.threshold;
}

bool RuntimeFrontendController::setup(IRuntimeListener *listener) {
  if (setup_complete_) {
    return true;
  }

  const RuntimeConfigError config_error = validate_runtime_config(config_);
  if (config_error != RuntimeConfigError::NONE) {
    const char *message = runtime_config_error_message(config_error);
    ESPECTRE_LOGE(TAG, "Rejected runtime configuration: %s", message);
    if (listener != nullptr) {
      listener->on_runtime_fault(message);
    }
    return false;
  }

  active_config_ = config_;
  if (traffic_sources_ == nullptr) {
    traffic_sources_.reset(new (std::nothrow) RuntimeTrafficSources());
  }
  auto *backend = traffic_sources_ != nullptr
                      ? new (std::nothrow) EspIdfRuntime(active_config_, traffic_sources_->generator,
                                                         traffic_sources_->ingress)
                      : nullptr;
  if (backend == nullptr) {
    constexpr const char *message = "Failed to allocate runtime backend";
    ESPECTRE_LOGE(TAG, "%s", message);
    if (listener != nullptr) {
      listener->on_runtime_fault(message);
    }
    return false;
  }
  listener_ = listener;
  runtime_.reset(backend);
  runtime_->set_listener(this);
  runtime_->set_services_armed(services_armed_);
  runtime_->set_live_telemetry_enabled(live_telemetry_enabled_);
  if (!runtime_->setup()) {
    runtime_.reset();
    listener_ = nullptr;
    apply_deferred_shutdown_();
    return false;
  }

  active_config_ = backend->effective_config();
  config_ = active_config_;
  snapshot_ = runtime_->get_snapshot();
  last_sensing_ready_ = snapshot_.ready_to_publish;
  capabilities_ = runtime_->get_capabilities();
  setup_complete_ = true;
  apply_deferred_shutdown_();
  return setup_complete_;
}

bool RuntimeFrontendController::traffic_allows_radio_work() const {
  if (runtime_ != nullptr) {
    return runtime_->traffic_allows_radio_work();
  }
  // The generator outlives the backend. A sender still inside a socket call
  // keeps station reconfigure and scan parked after shutdown(); as in the
  // runtime, a worker that has not been stopped does not.
  if (traffic_sources_ == nullptr) {
    return true;
  }
  const TrafficGeneratorManager &generator = traffic_sources_->generator;
  return generator.is_quiescent() || generator.has_live_worker();
}

void RuntimeFrontendController::hold_pending_traffic_restart(bool hold) {
  if (runtime_ != nullptr) {
    runtime_->hold_pending_traffic_restart(hold);
  }
}

void RuntimeFrontendController::loop() {
  if (!runtime_ && traffic_sources_ != nullptr) {
    // Reap a worker that outlived the last backend, between shutdown and setup.
    traffic_sources_->generator.loop();
  }
  if (runtime_) {
    runtime_->loop();
    cache_snapshot_(runtime_->get_snapshot());
    if (snapshot_.ready_to_publish != last_sensing_ready_) {
      last_sensing_ready_ = snapshot_.ready_to_publish;
      if (listener_ != nullptr) {
        begin_callback_();
        listener_->on_sensing_readiness_changed(snapshot_);
        end_callback_();
      }
    }
  }
  apply_deferred_shutdown_();
}

void RuntimeFrontendController::shutdown() { shutdown_(true); }

void RuntimeFrontendController::shutdown_(bool notify_listener) {
  if (callback_depth_ > 0U) {
    shutdown_requested_ = true;
    return;
  }
  if (runtime_) {
    runtime_->shutdown();
    runtime_.reset();
  }
  const bool was_ready = last_sensing_ready_;
  setup_complete_ = false;
  capabilities_ = {};
  snapshot_.motion_state = MotionState::IDLE;
  snapshot_.calibrating = false;
  snapshot_.ready_to_publish = false;
  last_sensing_ready_ = false;
  // Close the availability edge the listener saw; shutdown is already underway.
  if (notify_listener && was_ready && listener_ != nullptr) {
    begin_callback_();
    listener_->on_sensing_readiness_changed(snapshot_);
    end_callback_();
  }
  listener_ = nullptr;
  shutdown_requested_ = false;
}

void RuntimeFrontendController::set_services_armed(bool armed) {
  services_armed_ = armed;
  if (runtime_ && runtime_->operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    runtime_->set_services_armed(armed);
    ESPECTRE_LOGI(TAG, "Deferred sensing mutation until raw collection stops");
    return;
  }
  if (runtime_) {
    runtime_->set_services_armed(armed);
    snapshot_ = runtime_->get_snapshot();
  }
  apply_deferred_shutdown_();
}

void RuntimeFrontendController::set_live_telemetry_enabled(bool enabled) {
  live_telemetry_enabled_ = enabled;
  if (runtime_) {
    runtime_->set_live_telemetry_enabled(enabled);
  }
  apply_deferred_shutdown_();
}

void RuntimeFrontendController::quiesce() {
  set_live_telemetry_enabled(false);
  if (runtime_ && runtime_->operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    set_services_armed(false);
    if (!stop_raw_collection(RawCsiStopReason::SHUTDOWN)) {
      ESPECTRE_LOGE(TAG, "Failed to stop raw collection while quiescing the runtime");
    }
    return;
  }
  set_services_armed(false);
}

bool RuntimeFrontendController::set_threshold(float threshold) {
  const RuntimeConfig &effective_config = runtime_ ? active_config_ : config_;
  if (!validate_runtime_threshold_for_algorithm(threshold, effective_config.detection_algorithm)) {
    return false;
  }
  if (runtime_) {
    if (!capabilities_.supports_runtime_threshold_updates ||
        !runtime_->set_threshold(threshold)) {
      apply_deferred_shutdown_();
      return false;
    }
  } else {
    snapshot_.threshold = threshold;
  }
  if (runtime_) {
    adopt_effective_threshold_(threshold);
  } else {
    config_.threshold = threshold;
  }
  snapshot_.threshold = threshold;
  apply_deferred_shutdown_();
  return true;
}

bool RuntimeFrontendController::set_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (motion_on_hits < RUNTIME_MOTION_HITS_MIN || motion_on_hits > RUNTIME_MOTION_HITS_MAX ||
      motion_off_hits < RUNTIME_MOTION_HITS_MIN || motion_off_hits > RUNTIME_MOTION_HITS_MAX) {
    return false;
  }
  const bool staged_motion_on_hits =
      runtime_ && config_.motion_on_hits != active_config_.motion_on_hits;
  const bool staged_motion_off_hits =
      runtime_ && config_.motion_off_hits != active_config_.motion_off_hits;
  if (runtime_) {
    if (!capabilities_.supports_runtime_motion_hits_updates ||
        !runtime_->set_motion_hits(motion_on_hits, motion_off_hits)) {
      apply_deferred_shutdown_();
      return false;
    }
  }
  if (!staged_motion_on_hits) config_.motion_on_hits = motion_on_hits;
  if (!staged_motion_off_hits) config_.motion_off_hits = motion_off_hits;
  if (runtime_) {
    active_config_.motion_on_hits = motion_on_hits;
    active_config_.motion_off_hits = motion_off_hits;
  }
  apply_deferred_shutdown_();
  return true;
}

bool RuntimeFrontendController::set_traffic_generator_mode(TrafficGeneratorMode mode) {
  const RuntimeConfig &effective_config = runtime_ ? active_config_ : config_;
  if (!runtime_capture_profile_supports_traffic(effective_config.csi_capture_policy, mode)) {
    return false;
  }
  if (!runtime_traffic_generator_mode_supported(mode)) {
    return false;
  }
  const bool staged_for_next_setup =
      runtime_ && config_.traffic_generator_mode != active_config_.traffic_generator_mode;
  if (runtime_) {
    if (!capabilities_.supports_traffic_control || !runtime_->set_traffic_generator_mode(mode)) {
      apply_deferred_shutdown_();
      return false;
    }
  }
  if (!staged_for_next_setup) config_.traffic_generator_mode = mode;
  if (runtime_) active_config_.traffic_generator_mode = mode;
  apply_deferred_shutdown_();
  return true;
}

bool RuntimeFrontendController::set_detection_algorithm(DetectionAlgorithm algorithm) {
  if (!runtime_detection_algorithm_valid(algorithm)) {
    return false;
  }
  if (runtime_) {
    if (!capabilities_.supports_runtime_detector_selection ||
        !runtime_->set_detection_algorithm(algorithm)) {
      apply_deferred_shutdown_();
      return false;
    }
    snapshot_ = runtime_->get_snapshot();
  } else {
    config_.detection_algorithm = algorithm;
    config_.threshold = runtime_default_threshold(algorithm);
    snapshot_.threshold = config_.threshold;
    snapshot_.detector_name = detection_algorithm_name(algorithm);
  }
  if (runtime_) {
    adopt_effective_detector_(algorithm);
    adopt_effective_threshold_(snapshot_.threshold);
  } else {
    config_.detection_algorithm = algorithm;
    config_.threshold = snapshot_.threshold;
  }
  apply_deferred_shutdown_();
  return true;
}

bool RuntimeFrontendController::validate_control_update(const RuntimeControlUpdate &update,
                                                        std::string *message) const {
  const auto reject = [message](const char *reason) {
    if (message != nullptr) *message = reason;
    return false;
  };
  if (runtime_) {
    if (runtime_->operation_state() == RuntimeOperationState::RAW_COLLECTION) {
      return reject("mutation is unavailable during raw CSI collection");
    }
    if ((update.has_detection_algorithm && !capabilities_.supports_runtime_detector_selection) ||
        (update.has_threshold && !capabilities_.supports_runtime_threshold_updates) ||
        (update.has_motion_hits && !capabilities_.supports_runtime_motion_hits_updates) ||
        (update.has_traffic_generator_mode && !capabilities_.supports_traffic_control)) {
      return reject("sensing control is unsupported by the active runtime");
    }
  }
  const RuntimeConfig &effective_config = runtime_ ? active_config_ : config_;
  const RuntimeConfigError error =
      validate_runtime_config(apply_runtime_control_update(effective_config, update));
  // A staged configuration can already be invalid before setup; report only
  // errors this update introduces, and let the setters check the rest.
  if (error == RuntimeConfigError::NONE || error == validate_runtime_config(effective_config)) {
    return true;
  }
  return reject(runtime_config_error_message(error));
}

bool RuntimeFrontendController::trigger_recalibration() {
  if (!capabilities_.supports_manual_recalibration || !runtime_) {
    return false;
  }
  const bool started = runtime_->trigger_recalibration();
  apply_deferred_shutdown_();
  return started;
}

bool RuntimeFrontendController::is_calibrating() const {
  return runtime_ != nullptr && runtime_->is_calibrating();
}

bool RuntimeFrontendController::start_raw_collection(raw_csi_packet_callback_t callback,
                                                     void *context) {
  if (!runtime_ || !capabilities_.supports_raw_csi || callback == nullptr) {
    return false;
  }
  const bool started = runtime_->start_raw_collection(callback, context);
  if (started) {
    cache_snapshot_(runtime_->get_snapshot());
  }
  apply_deferred_shutdown_();
  return started;
}

bool RuntimeFrontendController::stop_raw_collection(RawCsiStopReason reason) {
  if (!runtime_ || runtime_->operation_state() != RuntimeOperationState::RAW_COLLECTION) {
    return false;
  }
  const bool stopped = runtime_->stop_raw_collection(reason);
  if (stopped) {
    runtime_->set_services_armed(services_armed_);
    cache_snapshot_(runtime_->get_snapshot());
  }
  apply_deferred_shutdown_();
  return stopped;
}

RuntimeOperationState RuntimeFrontendController::operation_state() const {
  return runtime_ != nullptr ? runtime_->operation_state() : RuntimeOperationState::SENSING;
}

RuntimeDiagnosticsSnapshot RuntimeFrontendController::diagnostics() const {
  return runtime_ != nullptr ? runtime_->get_diagnostics() : RuntimeDiagnosticsSnapshot{};
}

const RuntimeDiagnosticsSample *RuntimeFrontendController::diagnostics_sample() const {
  return runtime_ != nullptr ? runtime_->get_diagnostics_sample() : nullptr;
}

void RuntimeFrontendController::cache_snapshot_(const RuntimeSnapshot &snapshot) {
  snapshot_ = snapshot;
}

void RuntimeFrontendController::adopt_effective_threshold_(float threshold) {
  const bool staged_for_next_setup =
      config_.threshold != active_config_.threshold;
  active_config_.threshold = threshold;
  if (!staged_for_next_setup) {
    config_.threshold = threshold;
  }
}

void RuntimeFrontendController::adopt_effective_detector_(DetectionAlgorithm algorithm) {
  const bool staged_for_next_setup =
      config_.detection_algorithm != active_config_.detection_algorithm;
  active_config_.detection_algorithm = algorithm;
  if (!staged_for_next_setup) {
    config_.detection_algorithm = algorithm;
  }
}

void RuntimeFrontendController::begin_callback_() {
  if (callback_depth_++ == 0U) {
    callback_started_us_ = monotonic_now_us();
    callback_sink_us_ = detail::log_sink_total_us();
  }
}

void RuntimeFrontendController::end_callback_() {
  if (callback_depth_ > 0U && --callback_depth_ == 0U) {
    // Listener time excludes the log sink, so a callback that logs is not
    // counted both as listener work and as sink work.
    const uint64_t elapsed_us = monotonic_now_us() - callback_started_us_;
    const uint64_t sink_now_us = detail::log_sink_total_us();
    const uint64_t sink_us = sink_now_us >= callback_sink_us_ ? sink_now_us - callback_sink_us_ : 0U;
    const uint64_t listener_us = elapsed_us > sink_us ? elapsed_us - sink_us : 0U;
    RuntimeLoopStepTimer::record_listener_time(
        static_cast<uint32_t>(std::min<uint64_t>(listener_us, UINT32_MAX)));
  }
}

void RuntimeFrontendController::apply_deferred_shutdown_() {
  if (shutdown_requested_ && callback_depth_ == 0U) {
    shutdown();
  }
}

void RuntimeFrontendController::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  cache_snapshot_(snapshot);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_motion_state_changed(snapshot);
    end_callback_();
  }
}

void RuntimeFrontendController::on_periodic_update(const RuntimeSnapshot &snapshot,
                                                   uint32_t csi_accepted) {
  cache_snapshot_(snapshot);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_periodic_update(snapshot, csi_accepted);
    end_callback_();
  }
}

void RuntimeFrontendController::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  cache_snapshot_(snapshot);
  adopt_effective_threshold_(snapshot.threshold);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_threshold_changed(snapshot);
    end_callback_();
  }
}

void RuntimeFrontendController::on_detector_changed(const RuntimeSnapshot &snapshot) {
  cache_snapshot_(snapshot);
  adopt_effective_detector_(parse_detection_algorithm(snapshot.detector_name));
  adopt_effective_threshold_(snapshot.threshold);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_detector_changed(snapshot);
    end_callback_();
  }
}

void RuntimeFrontendController::on_calibration_started(const RuntimeSnapshot &snapshot) {
  cache_snapshot_(snapshot);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_calibration_started(snapshot);
    end_callback_();
  }
}

void RuntimeFrontendController::on_calibration_finished(const RuntimeSnapshot &snapshot,
                                                        bool success) {
  cache_snapshot_(snapshot);
  adopt_effective_threshold_(snapshot.threshold);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_calibration_finished(snapshot, success);
    end_callback_();
  }
}

void RuntimeFrontendController::on_live_telemetry(float movement, float threshold) {
  snapshot_.movement_metric = movement;
  snapshot_.threshold = threshold;
  adopt_effective_threshold_(threshold);
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_live_telemetry(movement, threshold);
    end_callback_();
  }
}

void RuntimeFrontendController::on_runtime_fault(const char *message) {
  if (listener_ != nullptr) {
    begin_callback_();
    listener_->on_runtime_fault(message);
    end_callback_();
  }
}

}  // namespace espectre
