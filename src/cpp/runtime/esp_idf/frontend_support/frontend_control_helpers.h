#pragma once

#include <functional>
#include <string>

#include "espectre_protocol.h"

namespace esphome {
namespace espectre {

struct DeviceConfigBleCommandResult {
  bool handled{false};
  bool accepted{false};
  bool config_changed{false};
  EspectreDeviceConfig config{};
  std::string message;
};

using DeviceConfigClearHandler = std::function<bool(EspectreDeviceConfig *cleared_config, std::string *message)>;
using DeviceConfigUpdateHandler = std::function<bool(EspectreDeviceConfig *updated_config, std::string *message)>;

DeviceConfigBleCommandResult handle_ble_device_config_command(const std::string &command,
                                                              const EspectreDeviceConfig &current_config,
                                                              DeviceConfigClearHandler clear_handler,
                                                              DeviceConfigUpdateHandler update_handler);

}  // namespace espectre
}  // namespace esphome
