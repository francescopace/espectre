set(IDF_TARGET esp32)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.csi
)

include(${CMAKE_CURRENT_LIST_DIR}/../micro_espectre.cmake)
