include(FetchContent)

find_package(ZLIB REQUIRED)

find_path(ARDUINOJSON_INCLUDE_DIR ArduinoJson.h)
if(NOT ARDUINOJSON_INCLUDE_DIR)
    FetchContent_Declare(
        ArduinoJson
        URL https://github.com/bblanchon/ArduinoJson/releases/download/v6.21.4/ArduinoJson-v6.21.4.hpp
        DOWNLOAD_NO_EXTRACT TRUE
        DOWNLOAD_NAME ArduinoJson.h
    )
    FetchContent_MakeAvailable(ArduinoJson)
    set(ARDUINOJSON_INCLUDE_DIR "${arduinojson_SOURCE_DIR}")
endif()

add_library(espectre_test_framework STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/support/test_harness.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mocks/esp_idf/esp_event_mock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mocks/esp_idf/esp_wifi_mock.cpp"
)
target_include_directories(espectre_test_framework
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/support"
)
target_link_libraries(espectre_test_framework
    PUBLIC
        espectre_test_mocks
)

add_library(espectre_test_support STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/support/cnpy.cpp"
)
target_include_directories(espectre_test_support
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/support"
        "${ARDUINOJSON_INCLUDE_DIR}"
)
target_link_libraries(espectre_test_support
    PUBLIC
        espectre_test_mocks
        ZLIB::ZLIB
)

add_library(espectre_core_testlib STATIC
    "${ESPECTRE_REPO_ROOT}/src/core/base_detector.cpp"
    "${ESPECTRE_REPO_ROOT}/src/core/csi_filters.cpp"
    "${ESPECTRE_REPO_ROOT}/src/core/ml_detector.cpp"
    "${ESPECTRE_REPO_ROOT}/src/core/mvs_detector.cpp"
)
target_link_libraries(espectre_core_testlib
    PUBLIC
        espectre_test_mocks
)

add_library(espectre_runtime_testlib STATIC
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/calibration_file_buffer.cpp"
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/csi_manager.cpp"
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/gain_controller.cpp"
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/nbvi_calibrator.cpp"
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/traffic_generator_manager.cpp"
    "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf/wifi_lifecycle.cpp"
)
target_link_libraries(espectre_runtime_testlib
    PUBLIC
        espectre_core_testlib
        espectre_test_mocks
)

add_library(espectre_frontend_esphome_testlib STATIC
    "${ESPECTRE_REPO_ROOT}/src/frontend/esphome/espectre/calibrate_switch.cpp"
    "${ESPECTRE_REPO_ROOT}/src/frontend/esphome/espectre/espectre.cpp"
    "${ESPECTRE_REPO_ROOT}/src/frontend/esphome/espectre/sensor_publisher.cpp"
    "${ESPECTRE_REPO_ROOT}/src/frontend/esphome/espectre/threshold_number.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/support/frontend_runtime_shim.cpp"
)
target_link_libraries(espectre_frontend_esphome_testlib
    PUBLIC
        espectre_runtime_testlib
        espectre_test_mocks
)

foreach(target_name
        espectre_test_framework
        espectre_test_support
        espectre_core_testlib
        espectre_runtime_testlib
        espectre_frontend_esphome_testlib)
    espectre_apply_coverage("${target_name}")
endforeach()
