/*
 * ESPectre - Direct API add-on for the espectre component built into ESPHome
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "espectre_direct.h"

#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"
#include "esphome/core/log.h"
#include "esphome/core/version.h"

#include "esp_netif.h"
#include "sdkconfig.h"

namespace esphome::espectre_direct {

static const char *const TAG = "espectre_direct";

namespace {

const char *firmware_version() {
#ifdef ESPHOME_PROJECT_VERSION
  return ESPHOME_PROJECT_VERSION;
#else
  return ESPHOME_VERSION;
#endif
}

uint32_t station_ipv4() {
  esp_netif_t *station = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  esp_netif_ip_info_t info{};
  if (station == nullptr || esp_netif_get_ip_info(station, &info) != ESP_OK) {
    return 0;
  }
  return info.ip.addr;
}

}  // namespace

void ESPectreDirect::loop() {
  if (!this->started_) {
    const uint32_t now = millis();
    // The built-in component starts its runtime on its own schedule, and restarts it after a fault.
    if (static_cast<int32_t>(now - this->next_start_ms_) < 0)
      return;
    if (!this->start_()) {
      this->next_start_ms_ = now + RETRY_INTERVAL_MS;
      return;
    }
  }
  // The built-in component has looped once since the last call, so a change queued then is applied by now.
  this->bridge_.loop();
  this->update_discovery_();
}

bool ESPectreDirect::start_() {
  const ::espectre::RuntimeFrontendController *runtime = ::espectre::RuntimeGuestAccess::runtime();
  if (this->parent_ == nullptr || runtime == nullptr || !runtime->is_setup_complete())
    return false;

  this->device_id_ = runtime->config().device_id;
  if (this->device_id_ == 0)
    this->device_id_ = ::espectre::derive_runtime_device_id();

  ::espectre::RuntimeDirectHttpBridgeConfig config;
  config.frontend = "esphome";
  config.device_name = ::espectre::espectre_device_name(this->device_id_, CONFIG_IDF_TARGET);
  config.hostname = App.get_name().c_str();
  config.firmware_version = firmware_version();
  config.chip = CONFIG_IDF_TARGET;
  config.device_id = this->device_id_;
  config.port = this->port_;
  config.raw_csi = this->raw_csi_;
  config.device_label_getter = []() { return std::string(App.get_friendly_name().c_str()); };
  config.peer_discovery = &this->peer_discovery_;
  config.diagnostics_sample_getter = []() -> const ::espectre::RuntimeDiagnosticsSample * {
    const ::espectre::RuntimeFrontendController *current = ::espectre::RuntimeGuestAccess::runtime();
    return current != nullptr ? current->diagnostics_sample() : nullptr;
  };
  if (!this->bridge_.setup_guest(&this->service_, config, this->control_sink_())) {
    ESP_LOGW(TAG, "Direct API could not start; retrying");
    return false;
  }
  this->observing_ = ::espectre::RuntimeGuestAccess::set_observer(this);
  if (!this->observing_) {
    ESP_LOGW(TAG, "Another observer follows the runtime; live events are not published");
  }
  if (!this->mdns_bootstrap_responder_.setup()) {
    ESP_LOGW(TAG, "Browser discovery is not available");
  }
  this->started_ = true;
  ESP_LOGI(TAG, "Direct API started on port %u", this->port_);
  return true;
}

// Each callback only queues the value: the built-in component applies it in its own loop, republishes its
// entity, and saves it. A control whose entity type is not compiled in, or that this ESPHome version does
// not take, is left out and so not offered.
::espectre::RuntimeControlSink ESPectreDirect::control_sink_() {
  ::espectre::RuntimeControlSink sink;
  espectre::ESPectreComponent *parent = this->parent_;
  sink.queued = true;
  sink.recalibrate = [parent](std::string *) {
    parent->recalibrate();
    return true;
  };
#ifdef USE_SELECT
  sink.traffic_generator_mode = [parent](::espectre::TrafficGeneratorMode mode, std::string *) {
    parent->request_traffic_generator_mode(mode);
    return true;
  };
#endif
// The built-in component takes the other controls from ESPHome 2026.11.
#if ESPHOME_VERSION_CODE >= VERSION_CODE(2026, 11, 0)
#ifdef USE_NUMBER
  sink.threshold = [parent](float threshold, std::string *) {
    parent->request_threshold(threshold);
    return true;
  };
  sink.motion_hits = [parent](uint8_t on_hits, uint8_t off_hits, std::string *) {
    parent->request_motion_on_hits(on_hits);
    parent->request_motion_off_hits(off_hits);
    return true;
  };
#endif
#ifdef USE_SELECT
  sink.detection_algorithm = [parent](::espectre::DetectionAlgorithm algorithm, std::string *) {
    parent->request_detection_algorithm(::espectre::detection_algorithm_name(algorithm));
    return true;
  };
  sink.csi_sample_spacing = [parent](::espectre::TemporalCsiSpacing spacing, std::string *) {
    parent->request_csi_sample_spacing(::espectre::csi_sample_spacing_name(spacing));
    return true;
  };
#endif
#ifdef USE_SWITCH
  sink.sensing_enabled = [parent](bool enabled, std::string *) {
    parent->request_sensing(enabled);
    return true;
  };
#endif
#endif
  return sink;
}

void ESPectreDirect::update_discovery_() {
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - this->next_address_poll_ms_) >= 0) {
    this->next_address_poll_ms_ = now + ADDRESS_POLL_INTERVAL_MS;
    const uint32_t ipv4 = station_ipv4();
    if (ipv4 != this->ipv4_) {
      this->ipv4_ = ipv4;
      (void) this->mdns_bootstrap_responder_.update(ipv4);
    }
  }
  this->mdns_bootstrap_responder_.loop();
  if (this->mdns_discovery_.service_enabled() || static_cast<int32_t>(now - this->next_mdns_setup_ms_) < 0)
    return;
  ::espectre::MdnsDiscoveryServiceConfig discovery;
  discovery.instance_name = "ESPectre " + ::espectre::format_espectre_device_id(this->device_id_);
  discovery.service_type = "_espectre";
  discovery.service_protocol = "_tcp";
  discovery.port = this->port_;
  discovery.txt_records = this->mdns_txt_records_();
  // ESPHome's mdns component owns the responder.
  discovery.responder_mode = ::espectre::MdnsResponderMode::USE_EXISTING_RESPONDER;
  if (!this->mdns_discovery_.setup(discovery)) {
    this->next_mdns_setup_ms_ = now + RETRY_INTERVAL_MS;
    return;
  }
  ESP_LOGI(TAG, "Discovery published on port %u", this->port_);
}

::espectre::MdnsTxtRecords ESPectreDirect::mdns_txt_records_() const {
  return {
      {"device_id", ::espectre::format_espectre_device_id(this->device_id_)},
      {"mac_address", ::espectre::read_direct_station_mac_address()},
      {"name", App.get_friendly_name().c_str()},
      {"frontend", "esphome"},
      {"txtvers", ::espectre::ESPECTRE_DNS_SD_TXT_SCHEMA_VERSION},
      {"protovers", ::espectre::ESPECTRE_PROTOCOL_VERSION},
      {"transport", ::espectre::ESPECTRE_DIRECT_HTTP_TRANSPORT},
      {"path", ::espectre::ESPECTRE_DIRECT_HTTP_BASE_ENDPOINT},
      {"firmware", firmware_version()},
      {"chip", CONFIG_IDF_TARGET},
      {"capabilities", this->raw_csi_ ? "config,monitor,csi" : "config,monitor"},
  };
}

void ESPectreDirect::publish_sensing_() {
  (void) this->bridge_.publish_changes(::espectre::FrontendCommandChange::SENSING);
}

void ESPectreDirect::on_sensing_readiness_changed(const ::espectre::RuntimeSnapshot &snapshot) {
  this->publish_sensing_();
}

void ESPectreDirect::on_motion_state_changed(const ::espectre::RuntimeSnapshot &snapshot) {
  if (snapshot.ready_to_publish)
    (void) this->bridge_.publish_motion(snapshot);
}

void ESPectreDirect::on_threshold_changed(const ::espectre::RuntimeSnapshot &snapshot) { this->publish_sensing_(); }

void ESPectreDirect::on_detector_changed(const ::espectre::RuntimeSnapshot &snapshot) { this->publish_sensing_(); }

void ESPectreDirect::on_calibration_started(const ::espectre::RuntimeSnapshot &snapshot) { this->publish_sensing_(); }

void ESPectreDirect::on_calibration_finished(const ::espectre::RuntimeSnapshot &snapshot, bool success) {
  this->publish_sensing_();
}

void ESPectreDirect::on_live_telemetry(const ::espectre::RuntimeSnapshot &snapshot) {
  if (snapshot.ready_to_publish)
    (void) this->bridge_.publish_motion(snapshot);
}

void ESPectreDirect::on_runtime_fault(const char *message) {
  ::espectre::EspectreDeviceConfig device;
  device.device_id = this->device_id_;
  (void) this->bridge_.publish_event("fault", ::espectre::espectre_fault_payload(device, message, millis()));
}

void ESPectreDirect::on_shutdown() {
  // Another guest's observer is not this component's to remove.
  if (this->observing_) {
    (void) ::espectre::RuntimeGuestAccess::set_observer(nullptr);
    this->observing_ = false;
  }
  this->mdns_bootstrap_responder_.shutdown();
  this->mdns_discovery_.shutdown();
  this->bridge_.shutdown();
  this->started_ = false;
}

void ESPectreDirect::dump_config() {
  ESP_LOGCONFIG(TAG,
                "ESPectre Direct API:\n"
                "  Port: %u\n"
                "  Raw CSI collection: %s\n"
                "  Running: %s",
                this->port_, YESNO(this->raw_csi_), YESNO(this->started_));
}

}  // namespace esphome::espectre_direct
