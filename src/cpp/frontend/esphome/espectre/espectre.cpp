/*
 * ESPectre - Main Component Implementation
 *
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "espectre.h"
#include "threshold_number.h"
#include "calibrate_switch.h"
#include "detector_select.h"

#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"

#include "debug_telemetry_log_helpers.h"
#include "espectre_banner.h"
#include "runtime_listener_utils.h"
#include "sdkconfig.h"

#include <cmath>

namespace esphome {
namespace espectre_component {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  espectre::configure_debug_telemetry_log_levels();

  if (!this->runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    this->mark_failed();
    return;
  }
  if (this->threshold_number_ != nullptr) {
    static_cast<ESpectreThresholdNumber *>(this->threshold_number_)
        ->update_detector_range(this->runtime_.config().detection_algorithm);
  }

  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  this->runtime_.shutdown();
}

void ESpectreComponent::loop() {
  this->runtime_.loop();
#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
  this->publish_diagnostics_if_due_();
#endif
}

#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
void ESpectreComponent::publish_diagnostics_if_due_() {
  const bool has_diagnostic_sensor = this->traffic_rate_sensor_ != nullptr ||
                                     this->csi_callback_rate_sensor_ != nullptr ||
                                     this->csi_accepted_rate_sensor_ != nullptr ||
                                     this->csi_filtered_rate_sensor_ != nullptr ||
                                     this->wifi_channel_sensor_ != nullptr ||
                                     this->wifi_rssi_sensor_ != nullptr ||
                                     this->csi_channel_changes_sensor_ != nullptr;
  if (!has_diagnostic_sensor) {
    return;
  }

  const uint32_t now_ms = millis();
  const ::espectre::RuntimeDiagnosticsSnapshot diagnostics = this->runtime_.diagnostics();
  if (!this->diagnostic_baseline_ready_) {
    this->diagnostic_baseline_ready_ = true;
    this->last_diagnostic_publish_ms_ = now_ms;
    this->previous_traffic_packets_total_ = diagnostics.traffic_packets_total;
    this->previous_csi_callbacks_total_ = diagnostics.csi_callbacks_total;
    this->previous_csi_accepted_total_ = diagnostics.csi_accepted_total;
    this->previous_csi_filtered_total_ = diagnostics.csi_filtered_total;
    if (this->traffic_rate_sensor_ != nullptr) this->traffic_rate_sensor_->publish_state(0.0f);
    if (this->csi_callback_rate_sensor_ != nullptr) this->csi_callback_rate_sensor_->publish_state(0.0f);
    if (this->csi_accepted_rate_sensor_ != nullptr) this->csi_accepted_rate_sensor_->publish_state(0.0f);
    if (this->csi_filtered_rate_sensor_ != nullptr) this->csi_filtered_rate_sensor_->publish_state(0.0f);
    if (this->wifi_channel_sensor_ != nullptr) this->wifi_channel_sensor_->publish_state(diagnostics.wifi_channel);
    if (this->wifi_rssi_sensor_ != nullptr) {
      this->wifi_rssi_sensor_->publish_state(diagnostics.wifi_rssi_dbm == INT8_MIN
                                                 ? NAN
                                                 : static_cast<float>(diagnostics.wifi_rssi_dbm));
    }
    if (this->csi_channel_changes_sensor_ != nullptr) {
      this->csi_channel_changes_sensor_->publish_state(diagnostics.channel_changes_total);
    }
    return;
  }

  const uint32_t elapsed_ms = now_ms - this->last_diagnostic_publish_ms_;
  if (elapsed_ms < 5000U) {
    return;
  }
  const auto counter_delta = [](uint64_t current, uint64_t previous) {
    return current >= previous ? current - previous : current;
  };
  const auto to_rate = [elapsed_ms](uint64_t delta) {
    return static_cast<float>(delta) * 1000.0f / static_cast<float>(elapsed_ms);
  };

  if (this->traffic_rate_sensor_ != nullptr) {
    this->traffic_rate_sensor_->publish_state(to_rate(counter_delta(
        diagnostics.traffic_packets_total, this->previous_traffic_packets_total_)));
  }
  if (this->csi_callback_rate_sensor_ != nullptr) {
    this->csi_callback_rate_sensor_->publish_state(to_rate(counter_delta(
        diagnostics.csi_callbacks_total, this->previous_csi_callbacks_total_)));
  }
  if (this->csi_accepted_rate_sensor_ != nullptr) {
    this->csi_accepted_rate_sensor_->publish_state(to_rate(counter_delta(
        diagnostics.csi_accepted_total, this->previous_csi_accepted_total_)));
  }
  if (this->csi_filtered_rate_sensor_ != nullptr) {
    this->csi_filtered_rate_sensor_->publish_state(to_rate(counter_delta(
        diagnostics.csi_filtered_total, this->previous_csi_filtered_total_)));
  }
  if (this->wifi_channel_sensor_ != nullptr) this->wifi_channel_sensor_->publish_state(diagnostics.wifi_channel);
  if (this->wifi_rssi_sensor_ != nullptr) {
    this->wifi_rssi_sensor_->publish_state(diagnostics.wifi_rssi_dbm == INT8_MIN
                                               ? NAN
                                               : static_cast<float>(diagnostics.wifi_rssi_dbm));
  }
  if (this->csi_channel_changes_sensor_ != nullptr) {
    this->csi_channel_changes_sensor_->publish_state(diagnostics.channel_changes_total);
  }

  this->last_diagnostic_publish_ms_ = now_ms;
  this->previous_traffic_packets_total_ = diagnostics.traffic_packets_total;
  this->previous_csi_callbacks_total_ = diagnostics.csi_callbacks_total;
  this->previous_csi_accepted_total_ = diagnostics.csi_accepted_total;
  this->previous_csi_filtered_total_ = diagnostics.csi_filtered_total;
}
#endif

void ESpectreComponent::set_threshold_runtime(float threshold) {
  this->runtime_.set_threshold_runtime(threshold);
}

void ESpectreComponent::set_detection_algorithm_runtime(const std::string &algorithm) {
  this->runtime_.set_detection_algorithm_runtime(parse_detection_algorithm(algorithm.c_str()));
}

void ESpectreComponent::trigger_recalibration() {
  this->runtime_.trigger_recalibration();
}

void ESpectreComponent::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
    this->detector_republished_ = false;
  }
  this->runtime_.record_snapshot(snapshot);
  if (snapshot.ready_to_publish) {
    this->sensor_publisher_.publish_motion_binary(snapshot.motion_state);
  }
}

void ESpectreComponent::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  if (!this->runtime_.snapshot().ready_to_publish && snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
  }
  this->runtime_.record_snapshot(snapshot);
  if (!snapshot.ready_to_publish) {
    return;
  }

  if (!this->threshold_republished_ && this->threshold_number_ != nullptr) {
    auto *threshold_num = static_cast<ESpectreThresholdNumber *>(this->threshold_number_);
    threshold_num->republish_state();
    this->threshold_republished_ = true;
  }
  if (!this->detector_republished_ && this->detector_select_ != nullptr) {
    static_cast<ESpectreDetectorSelect *>(this->detector_select_)->republish_state();
    this->detector_republished_ = true;
  }

  this->sensor_publisher_.log_status(TAG, snapshot, packets_received);
  this->sensor_publisher_.publish_movement_metric(snapshot.movement_metric);
  this->sensor_publisher_.publish_intensity(snapshot.movement_metric, snapshot.threshold);
}

void ESpectreComponent::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  apply_threshold_snapshot(this->runtime_, snapshot);
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(snapshot.threshold);
  }
  this->sensor_publisher_.publish_intensity(snapshot.movement_metric, snapshot.threshold);
}

void ESpectreComponent::on_detector_changed(const RuntimeSnapshot &snapshot) {
  apply_detector_snapshot(this->runtime_, snapshot);
  if (this->detector_select_ != nullptr) {
    this->detector_select_->publish_state(detection_algorithm_name(this->runtime_.config().detection_algorithm));
  }
  if (this->threshold_number_ != nullptr) {
    static_cast<ESpectreThresholdNumber *>(this->threshold_number_)
        ->update_detector_range(this->runtime_.config().detection_algorithm);
  }
}

void ESpectreComponent::on_calibration_started(const RuntimeSnapshot &snapshot) {
  this->runtime_.record_snapshot(snapshot);
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
}

void ESpectreComponent::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
  }
  finalize_frontend_calibration(this->runtime_, snapshot,
                                [this]() { this->sensor_publisher_.reset_rate_counter(); },
                                success, TAG);
}

void ESpectreComponent::on_runtime_fault(const char *message) {
  (void)message;
}

void ESpectreComponent::dump_config() {
  log_espectre_banner([](const char *line) { ESP_LOGCONFIG(TAG, "%s", line); });
  const RuntimeConfig &config = this->runtime_.config();
  const RuntimeSnapshot &snapshot = this->runtime_.snapshot();
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Detector ........... %s", snapshot.detector_name);
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.6f", snapshot.threshold);
  ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", config.segmentation_window_size);
  ESP_LOGCONFIG(TAG, " └─ Startup threshold .. %.6f", snapshot.startup_threshold);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                snapshot.fixed_subcarriers[0], snapshot.fixed_subcarriers[1],
                snapshot.fixed_subcarriers[2], snapshot.fixed_subcarriers[3],
                snapshot.fixed_subcarriers[4], snapshot.fixed_subcarriers[5],
                snapshot.fixed_subcarriers[6], snapshot.fixed_subcarriers[7],
                snapshot.fixed_subcarriers[8], snapshot.fixed_subcarriers[9],
                snapshot.fixed_subcarriers[10], snapshot.fixed_subcarriers[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", subcarrier_source_name(snapshot.subcarrier_source));
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  if (config.traffic_generator_rate > 0) {
    ESP_LOGCONFIG(TAG, " ├─ Mode ............... %s", traffic_mode_name(config.traffic_generator_mode));
    ESP_LOGCONFIG(TAG, " ├─ Target ............. %u valid CSI pps", config.traffic_generator_rate);
    ESP_LOGCONFIG(TAG, " ├─ Adaptive ........... %s", config.traffic_generator_adaptive ? "YES" : "NO");
    ESP_LOGCONFIG(TAG, " └─ Status ............. %s", snapshot.ready_to_publish ? "[ACTIVE]" : "[IDLE]");
  } else {
    ESP_LOGCONFIG(TAG, " └─ Mode ............... External Traffic");
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " PUBLISH INTERVAL");
  ESP_LOGCONFIG(TAG, " └─ Packets ............ %u", config.publish_interval);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " EVALUATION");
  ESP_LOGCONFIG(TAG, " ├─ Interval ........... %u pkts", config.evaluation_interval);
  ESP_LOGCONFIG(TAG, " └─ Hits on/off ........ %u / %u", config.motion_on_hits, config.motion_off_hits);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " LOW-PASS FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.lowpass_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.lowpass_enabled) {
    ESP_LOGCONFIG(TAG, " └─ Cutoff ............. %.1f Hz", config.lowpass_cutoff);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.hampel_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.hampel_enabled) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", config.hampel_window);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", config.hampel_threshold);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SENSORS");
  ESP_LOGCONFIG(TAG, " ├─ Movement ........... %s",
                this->sensor_publisher_.has_movement_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " ├─ Intensity .......... %s",
                this->sensor_publisher_.has_intensity_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " └─ Motion Binary ...... %s",
                this->sensor_publisher_.has_motion_binary_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, "");
}

}  // namespace espectre_component
}  // namespace esphome
