/*
 * ESPectre - Firmware Primary Console
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "primary_console.h"

#include <sdkconfig.h>

#include <sys/types.h>

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <driver/usb_serial_jtag.h>
#include <esp_log.h>

namespace {

// ESP-IDF log v2 writes the prefix, message, and line ending in separate
// calls. Assemble each line so a full USB buffer drops it whole instead of
// splicing fragments of consecutive messages.
constexpr size_t kLogLineCapacity = 512U;
// Log v2 sends the line ending, at most a color reset and a newline, as its
// own call. Flush a longer line early so that ending is never truncated.
constexpr size_t kLineEndingReserve = sizeof(LOG_ANSI_COLOR_RESET "\n") - 1U;
char log_line[kLogLineCapacity];
size_t log_line_length = 0U;
bool discarding_log_line = false;

bool ends_with_newline(const char *text, size_t length) {
  return length > 0U && text[length - 1U] == '\n';
}

int nonblocking_usb_log_printf(const char *format, va_list args) {
  // The driver reports misuse through the log, so never call it uninstalled.
  if (!usb_serial_jtag_is_driver_installed()) return vprintf(format, args);
  // Log v2 holds this recursive lock across one message's calls, but
  // esp_log_write() callers such as Wi-Fi do not. Taking it here keeps their
  // lines out of another task's partially assembled line.
  flockfile(stdout);
  const size_t offset = discarding_log_line ? 0U : log_line_length;
  const size_t available = kLogLineCapacity - offset;
  const int length = vsnprintf(log_line + offset, available, format, args);
  if (length >= 0) {
    const bool truncated = static_cast<size_t>(length) >= available;
    const size_t end = truncated ? kLogLineCapacity - 1U : offset + static_cast<size_t>(length);
    // The reserve keeps log v2 line endings whole, so only a single-call line
    // from log v1 or esp_log_write() can be truncated; its format keeps the
    // newline.
    const bool line_end = truncated ? ends_with_newline(format, std::strlen(format))
                                    : ends_with_newline(log_line, end);
    if (discarding_log_line) {
      // Drop the remainder of an overlong line through its line ending.
      discarding_log_line = !line_end;
      log_line_length = 0U;
    } else if (line_end || end + kLineEndingReserve >= kLogLineCapacity) {
      log_line[end - 1U] = '\n';
      // ESP-IDF's console VFS can wait for USB TX even with O_NONBLOCK. A zero
      // driver timeout drops the line when the host is not reading.
      (void) usb_serial_jtag_write_bytes(log_line, end, 0);
      log_line_length = 0U;
      discarding_log_line = !line_end;
    } else {
      log_line_length = end;
    }
  }
  funlockfile(stdout);
  return length;
}

}  // namespace
#endif

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

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
void use_nonblocking_usb_log_output() {
  esp_log_set_vprintf(&nonblocking_usb_log_printf);
}
#endif

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
