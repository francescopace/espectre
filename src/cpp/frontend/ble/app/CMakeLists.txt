# ESPectre BLE firmware build
#
# Prerequisites:
#   - ESP-IDF v5.5+
#
# Build:
#   cd src/cpp/frontend/ble/app
#   idf.py set-target esp32c3
#   idf.py build

cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../espectre"
)

project(espectre-ble)
