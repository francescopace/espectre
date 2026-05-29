#pragma once

namespace esphome {
namespace espectre {

struct RuntimeCapabilities {
  bool supports_runtime_threshold_updates{true};
  bool supports_manual_recalibration{true};
  bool supports_ble_telemetry{true};
  bool supports_extended_diagnostics{true};
  bool supports_traffic_control{true};
};

}  // namespace espectre
}  // namespace esphome
