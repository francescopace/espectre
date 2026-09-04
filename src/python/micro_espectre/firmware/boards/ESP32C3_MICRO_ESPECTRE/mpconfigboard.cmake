set(IDF_TARGET esp32c3)

set(SDKCONFIG_DEFAULTS
    boards/sdkconfig.base
    boards/sdkconfig.riscv
    boards/sdkconfig.c3
    boards/sdkconfig.csi
)

include(${CMAKE_CURRENT_LIST_DIR}/../micro_espectre.cmake)
