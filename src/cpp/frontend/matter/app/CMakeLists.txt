# ESPectre Matter firmware build
#
# Prerequisites:
#   - ESP-IDF (see espressif/esp_matter component requirements)
#
# Build:
#   cd src/cpp/frontend/matter/app
#   idf.py set-target esp32
#   idf.py build
#
# esp_matter is fetched automatically by the IDF Component Manager into managed_components/.

cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../espectre"
    "${CMAKE_CURRENT_LIST_DIR}/../../../runtime/esp_idf/espectre_config"
)

# Compile only main and its transitive component dependencies. Matter declares
# a large component graph, so processing every available ESP-IDF component
# adds substantial clean-build work without contributing to the firmware.
idf_build_set_property(MINIMAL_BUILD ON)

project(espectre-matter)

idf_build_set_property(CXX_COMPILE_OPTIONS "-std=gnu++17;-DCHIP_HAVE_CONFIG_H" APPEND)
idf_build_set_property(COMPILE_OPTIONS "-Wno-format-nonliteral;-Wno-format-security" APPEND)
