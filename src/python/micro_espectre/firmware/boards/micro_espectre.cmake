# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.

# Shared Micro-ESPectre firmware profile. Target wrappers append their upstream
# SoC defaults first; this file owns every cross-chip project setting.
list(APPEND SDKCONFIG_DEFAULTS
    ${CMAKE_CURRENT_LIST_DIR}/sdkconfig.micro_espectre
)

if(IDF_TARGET STREQUAL "esp32")
    list(APPEND SDKCONFIG_DEFAULTS
        ${CMAKE_CURRENT_LIST_DIR}/ESP32_MICRO_ESPECTRE/sdkconfig.override
    )
endif()
