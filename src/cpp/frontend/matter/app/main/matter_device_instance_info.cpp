/*
 * ESPectre - Matter Device Instance Info Provider
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "matter_device_instance_info.h"

#include <cstring>
#include <string>

#include <esp_mac.h>
#include <platform/ESP32/ConfigurationManagerImpl.h>

#include "matter_surface.h"

namespace espectre {

MatterDeviceInstanceInfoProvider::MatterDeviceInstanceInfoProvider()
    : GenericDeviceInstanceInfoProvider(chip::DeviceLayer::ConfigurationManagerImpl::GetDefaultInstance()) {}

CHIP_ERROR MatterDeviceInstanceInfoProvider::GetSerialNumber(char *buf, size_t buf_size) {
  uint8_t mac[6] = {};
  if (esp_read_mac(mac, ESP_MAC_BASE) != ESP_OK) {
    return GenericDeviceInstanceInfoProvider::GetSerialNumber(buf, buf_size);
  }
  const std::string serial = matter_serial_number_from_mac(mac);
  VerifyOrReturnError(serial.size() < buf_size, CHIP_ERROR_BUFFER_TOO_SMALL);
  std::memcpy(buf, serial.c_str(), serial.size() + 1U);
  return CHIP_NO_ERROR;
}

}  // namespace espectre
