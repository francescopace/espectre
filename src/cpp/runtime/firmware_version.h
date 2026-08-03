/*
 * ESPectre - Firmware Version
 *
 * Firmware version string helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

namespace espectre {

/**
 * The running application's version, as your build system defined it.
 *
 * Resolved from `APP_PROJECT_VER` when set, otherwise from the ESP-IDF
 * application descriptor, and `"unknown"` on a host build. In an SDK
 * integration this is *your* product version, not ESPectre's: it is what the
 * ESPectre Protocol reports as `firmware_version` and what OTA compares
 * against. For the version of the ESPectre sources you compiled against, use
 * `ESPECTRE_SDK_VERSION_STRING` from `espectre_sdk_version.h`.
 *
 * @return A static string, valid for the process lifetime. Never `nullptr`.
 */
const char *espectre_firmware_version();

}  // namespace espectre
