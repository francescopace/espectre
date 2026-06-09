if(NOT DEFINED ESPECTRE_CPP_ROOT)
    get_filename_component(ESPECTRE_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
endif()

set(ESPECTRE_CORE_SOURCES
    "${ESPECTRE_CPP_ROOT}/core/base_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/csi_filters.cpp"
    "${ESPECTRE_CPP_ROOT}/core/ml_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/mvs_detector.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_manager.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_payload_normalizer.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_platform_config.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/esp_idf_runtime.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/gain_controller.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/udp_listener.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_lifecycle.cpp"
)

set(ESPECTRE_FRONTEND_ESPHOME_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/calibrate_switch.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/espectre.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/sensor_publisher.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/threshold_number.cpp"
)

set(ESPECTRE_FRONTEND_MATTER_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/matter/espectre/matter_frontend.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/matter/espectre/matter_surface.cpp"
)

set(ESPECTRE_FRONTEND_STREAMER_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/streamer/espectre/csi_udp_sender.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/streamer/espectre/stream_frontend.cpp"
)

set(ESPECTRE_CORE_INCLUDE_DIRS
    "${ESPECTRE_CPP_ROOT}/core"
)

set(ESPECTRE_RUNTIME_INCLUDE_DIRS
    "${ESPECTRE_CPP_ROOT}/runtime"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf"
)

set(ESPECTRE_SHARED_INCLUDE_DIRS
    ${ESPECTRE_CORE_INCLUDE_DIRS}
    ${ESPECTRE_RUNTIME_INCLUDE_DIRS}
)
