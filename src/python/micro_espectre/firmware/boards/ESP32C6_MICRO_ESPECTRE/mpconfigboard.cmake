set(IDF_TARGET esp32c6)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.riscv
    boards/sdkconfig.c6
    boards/sdkconfig.csi
)

include(${CMAKE_CURRENT_LIST_DIR}/../micro_espectre.cmake)
