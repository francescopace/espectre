/*
 * ESPectre - ESP-IDF Primary Console
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "primary_console.h"

#include <sdkconfig.h>

#include <sys/types.h>

#if CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE
#include <tinyusb.h>
#include <tinyusb_cdc_acm.h>
#include <tinyusb_console.h>
#include <tinyusb_default_config.h>

namespace {

constexpr uint8_t kPrimaryCdcInterface = static_cast<uint8_t>(TINYUSB_CDC_ACM_0);

}  // namespace
#endif

namespace espectre {

esp_err_t initialize_primary_console() {
#if CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE
  const tinyusb_config_t tinyusb_config = TINYUSB_DEFAULT_CONFIG();
  esp_err_t result = tinyusb_driver_install(&tinyusb_config);
  if (result != ESP_OK) {
    return result;
  }

  const tinyusb_config_cdcacm_t cdc_config = {};
  result = tinyusb_cdcacm_init(&cdc_config);
  if (result != ESP_OK) {
    return result;
  }
  return tinyusb_console_init(TINYUSB_CDC_ACM_0);
#else
  return ESP_OK;
#endif
}

}  // namespace espectre

#if CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE
// Compatibility adapters may target ESP-IDF's ROM CDC primitives. Link those
// calls to the configured TinyUSB endpoint so they retain their wire protocol
// without owning the USB transport.
extern "C" ssize_t __wrap_esp_usb_console_available_for_read() {
  return static_cast<ssize_t>(tud_cdc_n_available(kPrimaryCdcInterface));
}

extern "C" ssize_t __wrap_esp_usb_console_read_buf(char *buffer, size_t size) {
  return static_cast<ssize_t>(
      tud_cdc_n_read(kPrimaryCdcInterface, buffer, static_cast<uint32_t>(size)));
}

extern "C" ssize_t __wrap_esp_usb_console_write_buf(const char *buffer, size_t size) {
  const uint32_t written =
      tud_cdc_n_write(kPrimaryCdcInterface, buffer, static_cast<uint32_t>(size));
  tud_cdc_n_write_flush(kPrimaryCdcInterface);
  return static_cast<ssize_t>(written);
}
#endif
