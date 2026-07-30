if(NOT DEFINED ESPECTRE_CPP_ROOT)
    get_filename_component(ESPECTRE_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
endif()

set(ESPECTRE_CORE_SOURCES
    "${ESPECTRE_CPP_ROOT}/core/base_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/classic_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/filters.cpp"
    "${ESPECTRE_CPP_ROOT}/core/ml_detector.cpp"
)

set(ESPECTRE_RUNTIME_COMMON_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/periodic_sensing_status_logger.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/espectre_protocol.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/firmware_version.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/protocol_json.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_config_utils.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_diagnostics.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_time.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/traffic_rate_controller.cpp"
)

set(ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/frontend_bootstrap_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/frontend_control_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/frontend_mqtt_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/frontend_sysinfo_helpers.cpp"
)

set(ESPECTRE_RUNTIME_STREAMER_FRONTEND_SUPPORT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/streamer_discovery_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/stream_esp_idf_runtime.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_PLATFORM_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/debug_telemetry_log_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/device_identity.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_capture_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_stream_transport.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_pipeline.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_payload_normalizer.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_platform_config.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/esp_idf_runtime.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/nvs_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_frontend_controller.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_detector_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_motion_hits_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_debug_telemetry.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_sensing_kconfig.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/stream_runtime_factory.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/standalone_wifi_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_frame_identity.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_traffic_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/udp_listener.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_lifecycle.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_SOURCES
    ${ESPECTRE_RUNTIME_COMMON_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_PLATFORM_SOURCES}
)

set(ESPECTRE_RUNTIME_STREAMER_SOURCES
    ${ESPECTRE_RUNTIME_ESP_IDF_SOURCES}
    ${ESPECTRE_RUNTIME_STREAMER_FRONTEND_SUPPORT_SOURCES}
)

set(ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/ota_service_https.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_BLE_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/ble_bindings_nimble.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/device_config_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support/mqtt_transport_esp_idf.cpp"
)

set(ESPECTRE_FRONTEND_ESPHOME_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/calibrate_switch.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/detector_select.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/espectre.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/sensor_publisher.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/esphome/espectre/threshold_number.cpp"
)

set(ESPECTRE_FRONTEND_MATTER_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/matter/espectre/matter_frontend.cpp"
    "${ESPECTRE_CPP_ROOT}/frontend/matter/espectre/matter_surface.cpp"
)

set(ESPECTRE_FRONTEND_NATIVE_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/native/espectre/native_frontend.cpp"
)

set(ESPECTRE_FRONTEND_STREAMER_SOURCES
    "${ESPECTRE_CPP_ROOT}/frontend/streamer/espectre/streamer_frontend.cpp"
)

set(ESPECTRE_CORE_INCLUDE_DIRS
    "${ESPECTRE_CPP_ROOT}/core"
)

set(ESPECTRE_RUNTIME_INCLUDE_DIRS
    "${ESPECTRE_CPP_ROOT}/runtime"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_support"
)

set(ESPECTRE_SHARED_INCLUDE_DIRS
    ${ESPECTRE_CORE_INCLUDE_DIRS}
    ${ESPECTRE_RUNTIME_INCLUDE_DIRS}
)
