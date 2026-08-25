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
#include "sensing_switch.h"
#include "detector_select.h"
#include "traffic_mode_select.h"

#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"

#include "runtime_log_helpers.h"
#include "device_identity.h"
#include "direct_http_protocol.h"
#include "espectre_banner.h"
#include "espectre_protocol.h"
#include "firmware_version.h"
#include "protocol_json.h"
#include "runtime_motion_hits_store.h"
#include "runtime_traffic_mode_store.h"
#include "sdkconfig.h"

#include <esp_netif.h>

#include <cmath>

namespace esphome {
namespace espectre_component {

namespace {

constexpr uint32_t kMdnsRetryIntervalMs = 5000U;

}  // namespace

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");
  espectre::configure_runtime_log_levels();
  this->runtime_.config().device_id = espectre::derive_runtime_device_id();
  if (global_preferences != nullptr) {
    this->device_label_preference_ =
        global_preferences->make_preference<StoredDeviceLabel>(fnv1_hash("espectre_device_label"));
    StoredDeviceLabel stored;
    if (this->device_label_preference_.load(&stored) && stored.version == 1U) {
      stored.value.back() = '\0';
      this->device_label_override_ = stored.value.data();
    }
  }

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

  if (!this->direct_bridge_.setup(
          &this->direct_service_,
          &this->runtime_,
          RuntimeDirectHttpBridgeConfig{
              "esphome",
              this->device_name_(),
              std::string(App.get_name()),
              espectre_firmware_version(),
              CONFIG_IDF_TARGET,
              this->runtime_.config().device_id,
              espectre::ESPECTRE_DIRECT_HTTP_PORT,
              true,
              false,
              [this]() { return this->device_name_(); },
              [this](const std::string &label, std::string *message) {
                return this->set_device_name_(label, message);
              },
              {},
              &this->peer_discovery_,
          },
          [this]() { this->sync_direct_config_(); })) {
    ESP_LOGE(TAG, "ESPHome Direct HTTP setup failed");
    this->runtime_.shutdown();
    this->mark_failed();
    return;
  }
  if (!this->mdns_bootstrap_responder_.setup()) {
    ESP_LOGE(TAG, "ESPHome mDNS bootstrap responder setup failed");
    this->direct_bridge_.shutdown();
    this->runtime_.shutdown();
    this->mark_failed();
    return;
  }

  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  this->mdns_bootstrap_responder_.shutdown();
  this->mdns_discovery_.shutdown();
  this->direct_bridge_.shutdown();
  this->runtime_.shutdown();
}

void ESpectreComponent::loop() {
  this->runtime_.loop();
  this->direct_bridge_.loop();
  esp_netif_ip_info_t ip_info{};
  esp_netif_t *station = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  (void) this->mdns_bootstrap_responder_.update(
      station != nullptr && esp_netif_get_ip_info(station, &ip_info) == ESP_OK
          ? ip_info.ip.addr
          : 0U);
  this->mdns_bootstrap_responder_.loop();
  if (!this->mdns_discovery_.service_enabled() && millis() >= this->next_mdns_setup_ms_) {
    this->setup_mdns_discovery_();
  }
}

std::string ESpectreComponent::device_name_() const {
  if (!this->device_label_override_.empty()) return this->device_label_override_;
  return "ESPectre ESPHome " + format_espectre_device_id(this->runtime_.config().device_id);
}

MdnsTxtRecords ESpectreComponent::mdns_txt_records_() const {
  return {
      {"device_id", format_espectre_device_id(this->runtime_.config().device_id)},
      {"name", this->device_name_()},
      {"frontend", "esphome"},
      {"txtvers", ESPECTRE_DIRECT_DISCOVERY_TXT_VERSION},
      {"protovers", "1"},
      {"transport", ESPECTRE_DIRECT_HTTP_TRANSPORT},
      {"path", ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT},
      {"events", ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT},
      {"firmware", espectre_firmware_version()},
      {"chip", CONFIG_IDF_TARGET},
      {"capabilities", "config,monitor,raw_csi"},
  };
}

bool ESpectreComponent::set_device_name_(const std::string &device_name, std::string *message) {
  if (device_name.size() > ESPECTRE_DEVICE_LABEL_MAX_LENGTH ||
      device_name.find_first_of("\r\n") != std::string::npos) {
    if (message != nullptr) *message = "device label must be at most 32 bytes and one line";
    return false;
  }
  StoredDeviceLabel stored;
  std::copy(device_name.begin(), device_name.end(), stored.value.begin());
  if (!this->device_label_preference_.save(&stored)) {
    if (message != nullptr) *message = "device label could not be persisted";
    return false;
  }
  this->device_label_override_ = device_name;
  if (this->mdns_discovery_.initialized()) {
    (void) this->mdns_discovery_.update_txt(this->mdns_txt_records_());
  }
  if (message != nullptr) *message = "ESPectre label updated";
  return true;
}

void ESpectreComponent::setup_mdns_discovery_() {
  const std::string device_id = format_espectre_device_id(this->runtime_.config().device_id);
  const MdnsTxtRecords txt_records = this->mdns_txt_records_();
  if (!this->mdns_discovery_.setup(MdnsDiscoveryServiceConfig{
          "",
          this->device_name_() + " " + device_id,
          "_espectre",
          "_tcp",
          espectre::ESPECTRE_DIRECT_HTTP_PORT,
          txt_records,
          MdnsResponderMode::USE_EXISTING_RESPONDER,
      })) {
    this->next_mdns_setup_ms_ = millis() + kMdnsRetryIntervalMs;
    return;
  }
  ESP_LOGI(TAG, "Direct HTTP discovery published on port %u", espectre::ESPECTRE_DIRECT_HTTP_PORT);
}

void ESpectreComponent::sync_direct_config_() {
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(this->runtime_.snapshot().threshold);
  }
  if (this->detector_select_ != nullptr) {
    this->detector_select_->publish_state(detection_algorithm_name(this->runtime_.config().detection_algorithm));
  }
  if (this->motion_on_hits_number_ != nullptr) {
    this->motion_on_hits_number_->publish_state(this->runtime_.config().motion_on_hits);
  }
  if (this->motion_off_hits_number_ != nullptr) {
    this->motion_off_hits_number_->publish_state(this->runtime_.config().motion_off_hits);
  }
  if (this->csi_traffic_mode_select_ != nullptr) {
    this->csi_traffic_mode_select_->publish_state(csi_traffic_mode_name(this->runtime_.config().csi_traffic_mode));
  }
  if (this->traffic_generator_mode_select_ != nullptr) {
    this->traffic_generator_mode_select_->publish_state(
        traffic_mode_name(this->runtime_.config().traffic_generator_mode));
  }
  if (this->sensing_switch_ != nullptr) {
    static_cast<ESpectreSensingSwitch *>(this->sensing_switch_)->republish_state();
  }
  if (this->calibration_active_sensor_ != nullptr) {
    this->calibration_active_sensor_->publish_state(this->runtime_.is_calibrating());
  }
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
  EspectreCommand command;
  command.command = "diagnostics";
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (result.accepted) this->publish_cached_diagnostics_();
}

FrontendCommandResult ESpectreComponent::execute_entity_command_(const EspectreCommand &command) {
  const RuntimeCapabilities &runtime_capabilities = this->runtime_.capabilities();
  FrontendCommandCapabilities capabilities;
  using Method = EspectreDirectMethod;
  capabilities.set(Method::CAPABILITIES);
  capabilities.set(Method::INFO);
  capabilities.set(Method::STATUS);
  capabilities.set(Method::CONFIG);
  capabilities.set(Method::DIAGNOSTICS, runtime_capabilities.supports_extended_diagnostics);
  capabilities.set(Method::SET_SENSING);
  capabilities.set(Method::SET_THRESHOLD, runtime_capabilities.supports_runtime_threshold_updates);
  capabilities.set(Method::SET_MOTION_HITS, runtime_capabilities.supports_runtime_motion_hits_updates);
  capabilities.set(Method::SET_DETECTOR, runtime_capabilities.supports_runtime_detector_selection);
  capabilities.set(Method::RECALIBRATE, runtime_capabilities.supports_manual_recalibration);
  capabilities.set(Method::SET_CSI_TRAFFIC_MODE, runtime_capabilities.supports_traffic_control);
  capabilities.set(Method::SET_TRAFFIC_GENERATOR_MODE, runtime_capabilities.supports_traffic_control);
  capabilities.set(EspectreConfigSection::RUNTIME);
  FrontendCommandResult result = this->command_engine_.execute(
      command,
      FrontendCommandContext{FrontendCommandOrigin::ESPHOME},
      nullptr,
      espectre_firmware_version(),
      capabilities,
      [this, capabilities](const EspectreCommand &read) {
        EspectreDeviceConfig device;
        device.device_id = this->runtime_.config().device_id;
        EspectreDeviceInfo info;
        info.frontend = "esphome";
        info.firmware_version = espectre_firmware_version();
        info.chip = CONFIG_IDF_TARGET;
        info.detector = detection_algorithm_name(this->runtime_.config().detection_algorithm);
        info.supports_diagnostics = capabilities.supports(Method::DIAGNOSTICS);
        info.supports_runtime_threshold = capabilities.supports(Method::SET_THRESHOLD);
        info.supports_runtime_motion_hits = capabilities.supports(Method::SET_MOTION_HITS);
        info.supports_runtime_detector = capabilities.supports(Method::SET_DETECTOR);
        info.supports_manual_recalibration = capabilities.supports(Method::RECALIBRATE);
        info.supports_traffic_control = capabilities.supports(Method::SET_CSI_TRAFFIC_MODE) &&
                                        capabilities.supports(Method::SET_TRAFFIC_GENERATOR_MODE);
        if (read.command == "capabilities") {
          return espectre_capabilities_payload(device, info, capabilities);
        }
        if (read.command == "info") return espectre_info_payload(device, info);
        if (read.command == "diagnostics") {
          return espectre_diagnostics_payload(device,
                                               this->runtime_.snapshot(),
                                               millis(),
                                               millis() / 1000U,
                                               0.0f,
                                               0.0f,
                                               &this->latest_diagnostics_);
        }
        if (read.command == "status") {
          const RuntimeSnapshot &snapshot = this->runtime_.snapshot();
          std::string out = espectre_status_payload(device, true, millis());
          out.pop_back();
          out += std::string(",\"sensing_enabled\":") +
                 (this->runtime_.services_armed() ? "true" : "false");
          out += std::string(",\"ready_to_publish\":") +
                 (snapshot.ready_to_publish ? "true" : "false");
          out += std::string(",\"calibrating\":") +
                 (this->runtime_.is_calibrating() ? "true" : "false");
          out += "}";
          return out;
        }
        if (read.command == "config") {
          const RuntimeConfig &config = this->runtime_.config();
          std::string out{"{\"runtime\":{"};
          out += "\"threshold\":" + std::to_string(this->runtime_.snapshot().threshold);
          append_json_pair(&out, "detector", detection_algorithm_name(config.detection_algorithm));
          out += ",\"motion_on_hits\":" + std::to_string(config.motion_on_hits);
          out += ",\"motion_off_hits\":" + std::to_string(config.motion_off_hits);
          append_json_pair(&out, "csi_traffic_mode", csi_traffic_mode_name(config.csi_traffic_mode));
          append_json_pair(&out, "traffic_generator_mode", traffic_mode_name(config.traffic_generator_mode));
          out += "}}";
          return out;
        }
        return std::string{};
      },
      {},
      [this](float value, std::string *) { return this->runtime_.set_threshold_runtime(value); },
      [this](uint8_t on, uint8_t off, std::string *) { return this->runtime_.set_motion_hits_runtime(on, off); },
      [this](CsiTrafficMode mode, std::string *) { return this->runtime_.set_csi_traffic_mode_runtime(mode); },
      [this](RuntimeTrafficMode mode, std::string *) {
        return this->runtime_.set_traffic_generator_mode_runtime(mode);
      },
      [this](DetectionAlgorithm algorithm, std::string *) {
        return this->runtime_.set_detection_algorithm_runtime(algorithm);
      },
      [this](std::string *) { return this->runtime_.trigger_recalibration(); },
      {},
      {},
      [this](bool enabled, std::string *) {
        this->runtime_.set_services_armed(enabled);
        return true;
      });
  if (result.accepted && result.changes != FrontendCommandChange::NONE) {
    (void) this->direct_bridge_.publish_changes(result.changes);
    this->sync_direct_config_();
    this->threshold_republished_ = true;
    this->detector_republished_ = true;
    this->motion_hits_republished_ = true;
    this->traffic_mode_republished_ = true;
  }
  return result;
}

bool ESpectreComponent::set_threshold_runtime(float threshold) {
  EspectreCommand command;
  command.command = "set_threshold";
  command.threshold = threshold;
  command.has_threshold = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

bool ESpectreComponent::set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  EspectreCommand command;
  command.command = "set_motion_hits";
  command.motion_on_hits = motion_on_hits;
  command.motion_off_hits = motion_off_hits;
  command.has_motion_hits = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

bool ESpectreComponent::set_detection_algorithm_runtime(const std::string &algorithm) {
  EspectreCommand command;
  command.command = "set_detector";
  command.detector = algorithm;
  command.has_detector = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

bool ESpectreComponent::set_sensing_runtime(bool enabled) {
  EspectreCommand command;
  command.command = "set_sensing";
  command.sensing_enabled = enabled;
  command.has_sensing_enabled = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

bool ESpectreComponent::set_csi_traffic_mode_runtime(const std::string &mode) {
  EspectreCommand command;
  command.command = "set_csi_traffic_mode";
  command.csi_traffic_mode = mode;
  command.has_csi_traffic_mode = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

bool ESpectreComponent::set_traffic_generator_mode_runtime(const std::string &mode) {
  EspectreCommand command;
  command.command = "set_traffic_generator_mode";
  command.traffic_generator_mode = mode;
  command.has_traffic_generator_mode = true;
  const FrontendCommandResult result = this->execute_entity_command_(command);
  if (!result.accepted) this->sync_direct_config_();
  return result.accepted;
}

void ESpectreComponent::trigger_recalibration() {
  EspectreCommand command;
  command.command = "recalibrate";
  (void) this->execute_entity_command_(command);
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
    (void) this->direct_bridge_.publish_telemetry(snapshot);
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
  if (!this->runtime_.snapshot().ready_to_publish) {
    return;
  }
  this->sensor_publisher_.publish_movement_metric(movement);
  (void) movement;
  (void) threshold;
  (void) this->direct_bridge_.publish_telemetry(this->runtime_.snapshot());
}

void ESpectreComponent::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(snapshot.threshold);
  }
  (void) this->direct_bridge_.publish_changes(FrontendCommandChange::CONFIG);
}

void ESpectreComponent::on_detector_changed(const RuntimeSnapshot &snapshot) {
  if (this->detector_select_ != nullptr) {
    this->detector_select_->publish_state(detection_algorithm_name(this->runtime_.config().detection_algorithm));
  }
  if (this->threshold_number_ != nullptr) {
    static_cast<ESpectreThresholdNumber *>(this->threshold_number_)
        ->update_detector_range(this->runtime_.config().detection_algorithm);
  }
  (void) snapshot;
  (void) this->direct_bridge_.publish_changes(FrontendCommandChange::CONFIG);
}

void ESpectreComponent::on_calibration_started(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
  if (this->calibration_active_sensor_ != nullptr) this->calibration_active_sensor_->publish_state(true);
  (void) this->direct_bridge_.publish_changes(FrontendCommandChange::STATUS);
}

void ESpectreComponent::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  (void) snapshot;
  if (this->calibration_active_sensor_ != nullptr) this->calibration_active_sensor_->publish_state(false);
  FrontendCommandChange changes = FrontendCommandChange::STATUS;
  if (success) changes = changes | FrontendCommandChange::CONFIG;
  (void) this->direct_bridge_.publish_changes(changes);
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void ESpectreComponent::on_runtime_fault(const char *message) {
  std::string data{"{"};
  append_json_pair(&data, "message", message != nullptr ? message : "runtime fault", true);
  data += "}";
  (void) this->direct_bridge_.publish_event("fault", data);
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
