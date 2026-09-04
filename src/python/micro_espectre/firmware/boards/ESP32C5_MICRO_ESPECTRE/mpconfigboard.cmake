set(IDF_TARGET esp32c5)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.riscv
    boards/sdkconfig.240mhz
    boards/sdkconfig.spiram_quad
    boards/sdkconfig.flash_qio_80m
    boards/sdkconfig.csi
)

include(${CMAKE_CURRENT_LIST_DIR}/../micro_espectre.cmake)
