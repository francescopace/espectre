#include "frontend_mqtt_helpers.h"

#include <utility>

#include "espectre_log.h"

namespace esphome {
namespace espectre {

bool setup_frontend_mqtt_transport(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   IMqttTransport::CommandCallback command_callback,
                                   FrontendMqttConnectedCallback connected_callback,
                                   const char *log_tag) {
  if (transport == nullptr) {
    return false;
  }
  if (config.mqtt_host.empty()) {
    transport->shutdown();
    return false;
  }

  transport->set_command_callback(std::move(command_callback));
  transport->set_connection_callback([connected_callback = std::move(connected_callback)](bool connected) {
    if (connected && connected_callback) {
      connected_callback();
    }
  });
  if (!transport->setup(config)) {
    ESP_LOGW(log_tag != nullptr ? log_tag : "espectre.mqtt", "MQTT transport setup failed");
    return false;
  }
  return true;
}

bool publish_frontend_mqtt_message(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   const char *suffix,
                                   const std::string &payload,
                                   bool retain) {
  if (transport == nullptr || !transport->connected()) {
    return false;
  }
  (void) config;
  return transport->publish_suffix(suffix, payload, retain);
}

bool publish_frontend_mqtt_status(IMqttTransport *transport,
                                  const EspectreDeviceConfig &config,
                                  bool online,
                                  uint32_t timestamp_ms) {
  return publish_frontend_mqtt_message(
      transport, config, "status", espectre_status_payload(config, online, timestamp_ms), false);
}

bool publish_frontend_mqtt_command_result(IMqttTransport *transport,
                                          const EspectreDeviceConfig &config,
                                          const EspectreCommand &command,
                                          bool accepted,
                                          const char *message) {
  return publish_frontend_mqtt_message(transport,
                                       config,
                                       accepted ? "commands/accepted" : "commands/rejected",
                                       espectre_command_result_payload(config, command, accepted, message),
                                       false);
}

bool publish_frontend_mqtt_ota_status(IMqttTransport *transport,
                                      const EspectreDeviceConfig &config,
                                      const EspectreOtaStatus &status,
                                      uint32_t timestamp_ms) {
  return publish_frontend_mqtt_message(
      transport, config, "ota/state", espectre_ota_status_payload(config, status, timestamp_ms), false);
}

FrontendMqttCommandResult handle_frontend_mqtt_command(const std::string &payload,
                                                       IOtaService *ota_service,
                                                       const char *current_version,
                                                       const FrontendMqttCommandCapabilities &capabilities,
                                                       FrontendMqttInfoCallback info_callback,
                                                       FrontendMqttStatsCallback stats_callback,
                                                       FrontendMqttThresholdCallback threshold_callback,
                                                       FrontendMqttOtaStatusCallback ota_status_callback) {
  FrontendMqttCommandResult result;
  result.handled = true;
  if (!parse_espectre_command(payload, &result.command, &result.message)) {
    result.command.command = "unknown";
    result.accepted = false;
    return result;
  }

  if (result.command.command == "info") {
    if (!capabilities.supports_info || !info_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    info_callback();
    result.accepted = true;
    result.message = "info published";
    return result;
  }

  if (result.command.command == "stats") {
    if (!capabilities.supports_stats || !stats_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    stats_callback();
    result.accepted = true;
    result.message = "stats published";
    return result;
  }

  if (result.command.command == "set_threshold") {
    if (!capabilities.supports_threshold || !threshold_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_threshold || !validate_runtime_threshold(result.command.threshold)) {
      result.accepted = false;
      result.message = "invalid threshold";
      return result;
    }
    result.accepted = threshold_callback(result.command.threshold, &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "threshold updated" : "threshold rejected";
    }
    return result;
  }

  if (result.command.command == "ota_status" || result.command.command == "ota_check" || result.command.command == "ota_start") {
    if (!capabilities.supports_ota || ota_service == nullptr) {
      result.accepted = false;
      result.message = "ota unavailable";
      return result;
    }

    const std::string normalized_current_version =
        (current_version == nullptr || current_version[0] == '\0') ? "unknown" : current_version;

    if (result.command.command == "ota_status") {
      if (ota_status_callback) {
        ota_status_callback(ota_service->status());
      }
      result.accepted = true;
      result.message = "ota status published";
      return result;
    }

    if (result.command.command == "ota_check") {
      result.accepted = result.command.has_manifest_url &&
                        ota_service->start_check(result.command.manifest_url, normalized_current_version);
      result.message = result.accepted ? "ota check started" : "ota check rejected";
      return result;
    }

    result.accepted = ota_service->start_update(result.command.has_manifest_url ? result.command.manifest_url : "",
                                                result.command.has_image_url ? result.command.image_url : "",
                                                result.command.has_version ? result.command.version : "",
                                                normalized_current_version);
    result.message = result.accepted ? "ota update started" : "ota update rejected";
    return result;
  }

  result.accepted = false;
  result.message = "unsupported command";
  return result;
}

}  // namespace espectre
}  // namespace esphome
