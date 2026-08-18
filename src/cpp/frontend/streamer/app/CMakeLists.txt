# ESPectre streamer firmware build
#
# Build:
#   cd src/cpp/frontend/streamer/app
#   idf.py set-target esp32c3
#   idf.py build

cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_git_version.cmake")
set(PROJECT_VER "${ESPECTRE_GIT_VERSION}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../espectre"
)

project(espectre-streamer)
