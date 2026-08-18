# ESPectre native firmware build
#
# Prerequisites:
#   - ESP-IDF v5.5+
#
# Build:
#   cd src/cpp/frontend/native/app
#   idf.py set-target esp32c3
#   idf.py build

cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_git_version.cmake")
set(PROJECT_VER "${ESPECTRE_GIT_VERSION}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../espectre"
    "${CMAKE_CURRENT_LIST_DIR}/../../../runtime/esp_idf/espectre_config"
)

project(espectre-native)
