// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#define MICROPY_HW_BOARD_NAME "ESPectre ESP32-S3"
#define MICROPY_HW_MCU_NAME "ESP32-S3"

// Keep the REPL on USB Serial/JTAG, the CLI's canonical S3 console. TinyUSB
// CDC would take the shared USB PHY and leave mpremote on the JTAG port.
#define MICROPY_HW_ENABLE_USBDEV (0)
#define MICROPY_HW_ENABLE_UART_REPL (1)

#include "../mpconfigboard_common.h"
