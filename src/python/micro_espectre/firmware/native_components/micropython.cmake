# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.

add_library(usermod_espectre_native_components INTERFACE)

if(NOT ESPECTRE_CORE_SDK_ROOT)
    message(FATAL_ERROR "ESPECTRE_CORE_SDK_ROOT must point to the ESPectre C++ SDK")
endif()

target_sources(usermod_espectre_native_components INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/native_direct.c
    ${CMAKE_CURRENT_LIST_DIR}/native_features.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native_features_module.c
    ${CMAKE_CURRENT_LIST_DIR}/native_log_sink.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native_traffic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/native_traffic.c
)

target_include_directories(usermod_espectre_native_components INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${ESPECTRE_CORE_SDK_ROOT}
    ${ESPECTRE_CORE_SDK_ROOT}/core
)

target_link_libraries(usermod INTERFACE usermod_espectre_native_components)

# MicroPython gathers user-module sources and includes recursively immediately
# after this file is loaded. Defer the ESP-IDF component edge so its generator
# expression include paths are not mistaken for literal source directories.
cmake_language(
    DEFER CALL target_link_libraries usermod INTERFACE
    idf::espectre_core
    idf::espectre_runtime_traffic
    idf::esp_http_server
    idf::json
    idf::log
    idf::espressif__mdns
)
