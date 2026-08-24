/*
 * ESPectre - Command Capabilities Probe
 *
 * Emits the normalized C++ capabilities catalog used by the cross-language
 * parity gate.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include <iostream>

#include "runtime/espectre_protocol.h"

int main(int argc, char **) {
  espectre::EspectreDeviceConfig config;
  espectre::EspectreDeviceInfo info;
  info.supports_info = true;
  info.supports_diagnostics = true;
  info.supports_runtime_threshold = true;
  info.supports_runtime_motion_hits = true;
  info.supports_runtime_detector = true;
  info.supports_manual_recalibration = true;
  info.supports_traffic_control = true;

  const bool native_profile = argc > 1;
  info.supports_device_config = native_profile;
  info.supports_ota = native_profile;
  std::cout << espectre::espectre_capabilities_payload(config,
                                                       info,
                                                       true,
                                                       true,
                                                       native_profile,
                                                       native_profile,
                                                       native_profile,
                                                       native_profile)
            << '\n';
  return 0;
}
