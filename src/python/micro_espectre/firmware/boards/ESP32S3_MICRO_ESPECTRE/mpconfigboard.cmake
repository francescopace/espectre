set(IDF_TARGET esp32s3)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.flash_qio_80m
    boards/sdkconfig.csi
)

include(${CMAKE_CURRENT_LIST_DIR}/../micro_espectre.cmake)
