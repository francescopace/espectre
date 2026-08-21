/*
 * ESPectre - BLE Protocol Constants
 *
 * UUIDs and device-name constants for the ESPectre BLE setup surface.
 *
 * Native uses sysinfo and control for provisioning, device labels, and OTA.
 * `ESPECTRE_BLE_TELEMETRY_UUID` remains in the GATT table for discovery
 * compatibility; Native does not notify on it.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

inline constexpr const char *ESPECTRE_BLE_SERVICE_UUID = "d33ff46b-2203-4775-bc6f-b3a2c36af8f0";
inline constexpr const char *ESPECTRE_BLE_TELEMETRY_UUID = "119d5cac-48da-4bd9-bfc3-169805868258";
inline constexpr const char *ESPECTRE_BLE_SYSINFO_UUID = "c8c89ffa-c401-461f-9ffc-942fa04adfe3";
inline constexpr const char *ESPECTRE_BLE_CONTROL_UUID = "33ed9214-a8d7-40e8-82d1-c82747dcdc71";
inline constexpr const char *ESPECTRE_BLE_DEVICE_NAME = "ESPectre BLE";

}  // namespace espectre
