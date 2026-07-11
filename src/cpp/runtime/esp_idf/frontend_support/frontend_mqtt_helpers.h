#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "espectre_protocol.h"
#include "mqtt_transport.h"
#include "ota_service.h"
#include "runtime_config_utils.h"

namespace espectre {

using FrontendMqttConnectedCallback = std::function<void()>;
using FrontendMqttInfoCallback = std::function<void()>;
using FrontendMqttStatsCallback = std::function<void()>;
using FrontendMqttThresholdCallback = std::function<bool(float threshold, std::string *message)>;
using FrontendMqttOtaStatusCallback = std::function<void(const EspectreOtaStatus &status)>;

struct FrontendMqttCommandCapabilities {
  bool supports_info{true};
  bool supports_stats{false};
  bool supports_threshold{false};
  bool supports_ota{false};
};

struct FrontendMqttCommandResult {
  bool handled{false};
  bool accepted{false};
  EspectreCommand command{};
  std::string message;
};

bool setup_frontend_mqtt_transport(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   IMqttTransport::CommandCallback command_callback,
                                   FrontendMqttConnectedCallback connected_callback,
                                   const char *log_tag);

bool publish_frontend_mqtt_message(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   const char *suffix,
                                   const std::string &payload,
                                   bool retain);

bool publish_frontend_mqtt_status(IMqttTransport *transport,
                                  const EspectreDeviceConfig &config,
                                  bool online,
                                  uint32_t timestamp_ms);

bool publish_frontend_mqtt_command_result(IMqttTransport *transport,
                                          const EspectreDeviceConfig &config,
                                          const EspectreCommand &command,
                                          bool accepted,
                                          const char *message);

bool publish_frontend_mqtt_ota_status(IMqttTransport *transport,
                                      const EspectreDeviceConfig &config,
                                      const EspectreOtaStatus &status,
                                      uint32_t timestamp_ms);

FrontendMqttCommandResult handle_frontend_mqtt_command(const std::string &payload,
                                                       IOtaService *ota_service,
                                                       const char *current_version,
                                                       const FrontendMqttCommandCapabilities &capabilities,
                                                       FrontendMqttInfoCallback info_callback,
                                                       FrontendMqttStatsCallback stats_callback,
                                                       FrontendMqttThresholdCallback threshold_callback,
                                                       FrontendMqttOtaStatusCallback ota_status_callback);

}  // namespace espectre
