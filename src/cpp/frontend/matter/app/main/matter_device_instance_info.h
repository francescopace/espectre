/*
 * ESPectre - Matter Device Instance Info Provider
 *
 * Reports a MAC-derived serial number on builds without factory data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#pragma once

#include <platform/CHIPDeviceLayer.h>
#include <platform/ESP32/ESP32Config.h>
#include <platform/internal/GenericDeviceInstanceInfoProvider.h>

namespace espectre {

/**
 * esp-matter's default instance info, except for the serial number, which
 * otherwise falls back to the shared `TEST_SN`. Factory-data builds use
 * esp-matter's factory provider instead.
 */
class MatterDeviceInstanceInfoProvider final
    : public chip::DeviceLayer::Internal::GenericDeviceInstanceInfoProvider<
          chip::DeviceLayer::Internal::ESP32Config> {
 public:
  MatterDeviceInstanceInfoProvider();

  CHIP_ERROR GetSerialNumber(char *buf, size_t buf_size) override;
};

}  // namespace espectre
