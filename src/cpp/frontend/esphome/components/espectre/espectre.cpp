/*
 * ESPectre - Main Component Implementation
 *
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "espectre.h"
#include "threshold_number.h"
#include "motion_hits_number.h"
#include "calibrate_switch.h"
#include "detector_select.h"
#include "traffic_mode_select.h"

#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"

#include "debug_telemetry_log_helpers.h"
#include "espectre_banner.h"
#include "runtime_motion_hits_store.h"
#include "runtime_traffic_mode_store.h"
#include "sdkconfig.h"

#include <cmath>

namespace esphome {
namespace espectre_component {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  espectre::configure_debug_telemetry_log_levels();

  this->runtime_.set_live_telemetry_enabled(this->sensor_publisher_.has_movement_sensor());
  uint8_t saved_motion_on_hits = 0U;
  uint8_t saved_motion_off_hits = 0U;
  bool has_saved_motion_hits = false;
  const esp_err_t motion_hits_err =
      espectre::load_runtime_motion_hits(&saved_motion_on_hits, &saved_motion_off_hits, &has_saved_motion_hits);
  if (motion_hits_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to load persisted motion hits: %s", esp_err_to_name(motion_hits_err));
  } else if (has_saved_motion_hits) {
    this->runtime_.config().motion_on_hits = saved_motion_on_hits;
    this->runtime_.config().motion_off_hits = saved_motion_off_hits;
  }
  bool has_saved_csi_traffic_mode = false;
  CsiTrafficMode saved_csi_traffic_mode = this->runtime_.config().csi_traffic_mode;
  const esp_err_t csi_traffic_err =
      espectre::load_runtime_csi_traffic_mode(&saved_csi_traffic_mode, &has_saved_csi_traffic_mode);
  if (csi_traffic_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to load persisted CSI traffic mode: %s", esp_err_to_name(csi_traffic_err));
  } else if (has_saved_csi_traffic_mode) {
    this->runtime_.config().csi_traffic_mode = saved_csi_traffic_mode;
  }
  bool has_saved_generator_mode = false;
  RuntimeTrafficMode saved_generator_mode = this->runtime_.config().traffic_generator_mode;
  const esp_err_t generator_err =
      espectre::load_runtime_traffic_generator_mode(&saved_generator_mode, &has_saved_generator_mode);
  if (generator_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to load persisted traffic generator mode: %s", esp_err_to_name(generator_err));
  } else if (has_saved_generator_mode) {
    this->runtime_.config().traffic_generator_mode = saved_generator_mode;
  }
  if (!this->runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    this->mark_failed();
    return;
  }
  const uint32_t diagnostics_now_ms = millis();
  const RuntimeDiagnosticsSnapshot diagnostics = this->runtime_.diagnostics();
  this->diagnostics_sampler_.reset(diagnostics, diagnostics_now_ms);
  this->latest_diagnostics_ = this->diagnostics_sampler_.sample(diagnostics, diagnostics_now_ms);
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
}

void ESpectreComponent::sample_diagnostics_() {
  const uint32_t now_ms = millis();
  this->latest_diagnostics_ = this->diagnostics_sampler_.sample(this->runtime_.diagnostics(), now_ms);
}

void ESpectreComponent::publish_cached_diagnostics_() {
  const RuntimeDiagnosticsSample &sample = this->latest_diagnostics_;

  if (this->traffic_rate_sensor_ != nullptr) {
    this->traffic_rate_sensor_->publish_state(sample.traffic_tx_pps);
  }
  if (this->csi_callback_rate_sensor_ != nullptr) {
    this->csi_callback_rate_sensor_->publish_state(sample.csi_callback_pps);
  }
  if (this->csi_accepted_rate_sensor_ != nullptr) {
    this->csi_accepted_rate_sensor_->publish_state(sample.csi_accepted_pps);
  }
  if (this->csi_admitted_rate_sensor_ != nullptr) {
    this->csi_admitted_rate_sensor_->publish_state(sample.csi_admitted_pps);
  }
  if (this->csi_filtered_rate_sensor_ != nullptr) {
    this->csi_filtered_rate_sensor_->publish_state(sample.csi_filtered_pps);
  }
  if (this->csi_missing_rate_sensor_ != nullptr) {
    this->csi_missing_rate_sensor_->publish_state(sample.csi_missing_slots_pps);
  }
  if (this->csi_excess_rate_sensor_ != nullptr) {
    this->csi_excess_rate_sensor_->publish_state(sample.csi_excess_pps);
  }
  if (this->csi_stale_rate_sensor_ != nullptr) {
    this->csi_stale_rate_sensor_->publish_state(sample.csi_stale_pps);
  }
  if (this->csi_out_of_order_rate_sensor_ != nullptr) {
    this->csi_out_of_order_rate_sensor_->publish_state(sample.csi_out_of_order_pps);
  }
  if (this->csi_occupancy_sensor_ != nullptr) {
    this->csi_occupancy_sensor_->publish_state(sample.csi_occupancy_ratio * 100.0f);
  }
  if (this->wifi_channel_sensor_ != nullptr) this->wifi_channel_sensor_->publish_state(sample.wifi_channel);
  if (this->wifi_rssi_sensor_ != nullptr) {
    this->wifi_rssi_sensor_->publish_state(sample.wifi_rssi_dbm == INT8_MIN
                                               ? NAN
                                               : static_cast<float>(sample.wifi_rssi_dbm));
  }
}

void ESpectreComponent::publish_diagnostics_on_demand() {
  this->publish_cached_diagnostics_();
}

void ESpectreComponent::set_threshold_runtime(float threshold) {
  this->runtime_.set_threshold_runtime(threshold);
}

void ESpectreComponent::set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (!this->runtime_.set_motion_hits_runtime(motion_on_hits, motion_off_hits)) {
    return;
  }
  if (this->motion_on_hits_number_ != nullptr) {
    this->motion_on_hits_number_->publish_state(this->runtime_.config().motion_on_hits);
  }
  if (this->motion_off_hits_number_ != nullptr) {
    this->motion_off_hits_number_->publish_state(this->runtime_.config().motion_off_hits);
  }
}

void ESpectreComponent::set_detection_algorithm_runtime(const std::string &algorithm) {
  this->runtime_.set_detection_algorithm_runtime(parse_detection_algorithm(algorithm.c_str()));
}

void ESpectreComponent::set_csi_traffic_mode_runtime(const std::string &mode) {
  if (!this->runtime_.set_csi_traffic_mode_runtime(parse_csi_traffic_mode(mode.c_str()))) {
    return;
  }
  if (this->csi_traffic_mode_select_ != nullptr) {
    this->csi_traffic_mode_select_->publish_state(csi_traffic_mode_name(this->runtime_.config().csi_traffic_mode));
  }
}

void ESpectreComponent::set_traffic_generator_mode_runtime(const std::string &mode) {
  if (!this->runtime_.set_traffic_generator_mode_runtime(parse_traffic_mode(mode.c_str()))) {
    return;
  }
  if (this->traffic_generator_mode_select_ != nullptr) {
    this->traffic_generator_mode_select_->publish_state(
        traffic_mode_name(this->runtime_.config().traffic_generator_mode));
  }
}

void ESpectreComponent::trigger_recalibration() {
  this->runtime_.trigger_recalibration();
}

void ESpectreComponent::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
    this->detector_republished_ = false;
    this->motion_hits_republished_ = false;
    this->traffic_mode_republished_ = false;
  }
  if (snapshot.ready_to_publish) {
    this->sensor_publisher_.publish_motion_binary(snapshot.motion_state);
  }
}

void ESpectreComponent::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  if (!snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
    this->detector_republished_ = false;
    this->motion_hits_republished_ = false;
    this->traffic_mode_republished_ = false;
  }
  this->sample_diagnostics_();
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
  if (!this->motion_hits_republished_ && (this->motion_on_hits_number_ != nullptr || this->motion_off_hits_number_ != nullptr)) {
    if (this->motion_on_hits_number_ != nullptr) {
      static_cast<ESpectreMotionHitsNumber *>(this->motion_on_hits_number_)->republish_state();
    }
    if (this->motion_off_hits_number_ != nullptr) {
      static_cast<ESpectreMotionHitsNumber *>(this->motion_off_hits_number_)->republish_state();
    }
    this->motion_hits_republished_ = true;
  }
  if (!this->traffic_mode_republished_ &&
      (this->csi_traffic_mode_select_ != nullptr || this->traffic_generator_mode_select_ != nullptr)) {
    if (this->csi_traffic_mode_select_ != nullptr) {
      static_cast<ESpectreTrafficModeSelect *>(this->csi_traffic_mode_select_)->republish_state();
    }
    if (this->traffic_generator_mode_select_ != nullptr) {
      static_cast<ESpectreTrafficModeSelect *>(this->traffic_generator_mode_select_)->republish_state();
    }
    this->traffic_mode_republished_ = true;
  }

}

void ESpectreComponent::on_live_telemetry(float movement, float threshold) {
  (void) threshold;
  if (!this->runtime_.snapshot().ready_to_publish) {
    return;
  }
  this->sensor_publisher_.publish_movement_metric(movement);
}

void ESpectreComponent::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(snapshot.threshold);
  }
}

void ESpectreComponent::on_detector_changed(const RuntimeSnapshot &snapshot) {
  if (this->detector_select_ != nullptr) {
    this->detector_select_->publish_state(detection_algorithm_name(this->runtime_.config().detection_algorithm));
  }
  if (this->threshold_number_ != nullptr) {
    static_cast<ESpectreThresholdNumber *>(this->threshold_number_)
        ->update_detector_range(this->runtime_.config().detection_algorithm);
  }
}

void ESpectreComponent::on_calibration_started(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
}

void ESpectreComponent::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  (void) snapshot;
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
  }
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void ESpectreComponent::on_runtime_fault(const char *message) {
  (void)message;
}

void ESpectreComponent::dump_config() {
  log_espectre_banner([](const char *line) { ESP_LOGCONFIG(TAG, "%s", line); });
  const RuntimeConfig &config = this->runtime_.config();
  const RuntimeSnapshot &snapshot = this->runtime_.snapshot();
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Wi-Fi band ......... %s", wifi_band_policy_name(config.wifi_band_policy));
  ESP_LOGCONFIG(TAG, " ├─ Detector ........... %s", snapshot.detector_name);
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.6f", snapshot.threshold);
  ESP_LOGCONFIG(TAG, " ├─ Window ............. %u ms",
                static_cast<unsigned>(config.segmentation_window_size_ms));
  ESP_LOGCONFIG(TAG, " └─ Startup threshold .. %.6f", snapshot.startup_threshold);
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                snapshot.fixed_subcarriers[0], snapshot.fixed_subcarriers[1],
                snapshot.fixed_subcarriers[2], snapshot.fixed_subcarriers[3],
                snapshot.fixed_subcarriers[4], snapshot.fixed_subcarriers[5],
                snapshot.fixed_subcarriers[6], snapshot.fixed_subcarriers[7],
                snapshot.fixed_subcarriers[8], snapshot.fixed_subcarriers[9],
                snapshot.fixed_subcarriers[10], snapshot.fixed_subcarriers[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", subcarrier_source_name(snapshot.subcarrier_source));
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  ESP_LOGCONFIG(TAG, " ├─ Mode ............... %s", traffic_mode_name(config.traffic_generator_mode));
  ESP_LOGCONFIG(TAG, " ├─ CSI target ......... %u pps",
                static_cast<unsigned>(config.csi_target_pps));
  ESP_LOGCONFIG(TAG, " ├─ CSI traffic ........ %s", csi_traffic_mode_name(config.csi_traffic_mode));
  ESP_LOGCONFIG(TAG, " ├─ Multicast join ..... %s",
                config.csi_traffic_multicast_group.empty() ? "[disabled]"
                                                          : config.csi_traffic_multicast_group.c_str());
  ESP_LOGCONFIG(TAG, " └─ Status ............. %s", snapshot.ready_to_publish ? "[ACTIVE]" : "[IDLE]");
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " PUBLISH INTERVAL");
  ESP_LOGCONFIG(TAG, " └─ Status log ......... %u ms",
                static_cast<unsigned>(config.publish_interval_ms));
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " EVALUATION");
  ESP_LOGCONFIG(TAG, " ├─ Interval ........... %u ms",
                static_cast<unsigned>(config.evaluation_interval_ms));
  ESP_LOGCONFIG(TAG, " └─ Hits on/off ........ %u / %u",
                static_cast<unsigned>(config.motion_on_hits),
                static_cast<unsigned>(config.motion_off_hits));
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " LOW-PASS FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.lowpass_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.lowpass_enabled) {
    ESP_LOGCONFIG(TAG, " └─ Cutoff ............. %.1f Hz", config.lowpass_cutoff);
  }
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.hampel_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.hampel_enabled) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", config.hampel_window);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", config.hampel_threshold);
  }
  ESP_LOGCONFIG(TAG, " ");
  ESP_LOGCONFIG(TAG, " SENSORS");
  ESP_LOGCONFIG(TAG, " ├─ Movement ........... %s",
                this->sensor_publisher_.has_movement_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " └─ Motion Binary ...... %s",
                this->sensor_publisher_.has_motion_binary_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " ");
}

}  // namespace espectre_component
}  // namespace esphome
