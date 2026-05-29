add_library(espectre_test_mocks INTERFACE)

target_include_directories(espectre_test_mocks
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/mocks"
        "${CMAKE_CURRENT_SOURCE_DIR}/mocks/esp_idf"
        "${CMAKE_CURRENT_SOURCE_DIR}/mocks/esphome"
        "${CMAKE_CURRENT_SOURCE_DIR}/support"
        "${ESPECTRE_REPO_ROOT}/src/core"
        "${ESPECTRE_REPO_ROOT}/src/runtime"
        "${ESPECTRE_REPO_ROOT}/src/runtime/esp_idf"
        "${ESPECTRE_REPO_ROOT}/src/frontend/esphome/espectre"
)

target_compile_definitions(espectre_test_mocks
    INTERFACE
        CONFIG_FREERTOS_HZ=100
)
