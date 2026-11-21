# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Route Improv Serial through the ESP32-S2 native USB connector."""

from pathlib import Path

import esphome.codegen as cg
from esphome.components import esp32, logger
import esphome.config_validation as cv
from esphome.const import CONF_HARDWARE_UART, CONF_LOGGER
from esphome.core import CORE
from esphome.coroutine import CoroPriority, coroutine_with_priority

DEPENDENCIES = ["logger", "improv_serial"]
CONFIG_SCHEMA = cv.All(cv.Schema({}), esp32.only_on_variant(supported=esp32.VARIANT_ESP32S2))


@coroutine_with_priority(CoroPriority.WORKAROUNDS)
async def to_code(config):
    if CORE.config[CONF_LOGGER][CONF_HARDWARE_UART] != logger.USB_CDC:
        return

    esp32.add_idf_component(name="espectre_usb_console", path=str(Path(__file__).resolve().parent))
    esp32.add_idf_sdkconfig_option("CONFIG_ESP_CONSOLE_USB_CDC", False)
    esp32.add_idf_sdkconfig_option("CONFIG_ESP_CONSOLE_NONE", True)
    esp32.add_idf_sdkconfig_option("CONFIG_TINYUSB_CDC_ENABLED", True)
    esp32.add_idf_sdkconfig_option("CONFIG_TINYUSB_DESC_SERIAL_STRING", "0")
    # Compile ESPHome's CDC calls while the IDF console itself stays disabled.
    cg.add_define("CONFIG_ESP_CONSOLE_USB_CDC", 1)
    cg.add_global(cg.RawStatement('#include "primary_console.h"'))
    cg.add(cg.RawStatement("ESP_ERROR_CHECK_WITHOUT_ABORT(::espectre::initialize_primary_console());"))
