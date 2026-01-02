/*
 * ESPectre - Mock ESPHome Defines
 *
 * Minimal mock for esphome/core/defines.h used in testing.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

// ESPHome build defines - add as needed for tests

// Logger configuration - ESPHome defines these based on logger config:
// - USE_LOGGER_UART_SELECTION_USB_SERIAL_JTAG: logger is using USB Serial JTAG (default on C3/C6/S3/H2)
// - USE_LOGGER_USB_CDC: USB CDC is available (always defined on chips with USB)
// - USE_LOGGER_USB_SERIAL_JTAG: USB Serial JTAG driver is available (always defined on supported chips)
//
// Note: In tests, we don't enable USB Serial JTAG functionality
// #define USE_LOGGER_UART_SELECTION_USB_SERIAL_JTAG


