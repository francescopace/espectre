# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.

add_library(usermod_espectre_native_components INTERFACE)

target_sources(usermod_espectre_native_components INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/native_mqtt.c
    ${CMAKE_CURRENT_LIST_DIR}/native_traffic.c
)

target_include_directories(usermod_espectre_native_components INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE usermod_espectre_native_components)

# MicroPython gathers user-module sources and includes recursively immediately
# after this file is loaded. Defer the ESP-IDF component edge so its generator
# expression include paths are not mistaken for literal source directories.
cmake_language(DEFER CALL target_link_libraries usermod INTERFACE idf::mqtt)
