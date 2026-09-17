# First-party firmware sources, kept outside the distributed SDK.
get_filename_component(ESPECTRE_FRONTEND_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

if(NOT ESPECTRE_SDK_ROOT)
    if(DEFINED ENV{ESPECTRE_SDK_ROOT} AND NOT "$ENV{ESPECTRE_SDK_ROOT}" STREQUAL "")
        set(ESPECTRE_SDK_ROOT "$ENV{ESPECTRE_SDK_ROOT}")
    else()
        get_filename_component(ESPECTRE_SDK_ROOT "${ESPECTRE_FRONTEND_ROOT}/.." ABSOLUTE)
    endif()
endif()
get_filename_component(ESPECTRE_SDK_ROOT "${ESPECTRE_SDK_ROOT}" ABSOLUTE)
set(ENV{ESPECTRE_SDK_ROOT} "${ESPECTRE_SDK_ROOT}")
set(ESPECTRE_CPP_ROOT "${ESPECTRE_SDK_ROOT}")
include("${ESPECTRE_SDK_ROOT}/espectre_sources.cmake")

set(ESPECTRE_FRONTEND_IMPROV_SOURCES
    "${ESPECTRE_FRONTEND_ROOT}/improv_serial_service.cpp"
)

set(ESPECTRE_FRONTEND_OTA_PROTOCOL_SOURCES
    "${ESPECTRE_FRONTEND_ROOT}/ota_protocol.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/ota_version.cpp"
)

set(ESPECTRE_FRONTEND_OTA_SOURCES
    ${ESPECTRE_FRONTEND_OTA_PROTOCOL_SOURCES}
    "${ESPECTRE_FRONTEND_ROOT}/ota_service_https.cpp"
)

set(ESPECTRE_FRONTEND_COMMON_SOURCES
    "${ESPECTRE_FRONTEND_ROOT}/frontend_firmware_version.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/primary_console.cpp"
)

set(ESPECTRE_FRONTEND_ESPHOME_SOURCES
    ${ESPECTRE_FRONTEND_COMMON_SOURCES}
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/esphome_log_sink.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/recalibrate_button.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/sensing_switch.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/detector_select.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/diagnostics_button.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/espectre.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/motion_hits_number.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/sensor_publisher.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/threshold_number.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/esphome/components/espectre/traffic_mode_select.cpp"
)

set(ESPECTRE_FRONTEND_MATTER_SOURCES
    ${ESPECTRE_FRONTEND_COMMON_SOURCES}
    "${ESPECTRE_FRONTEND_ROOT}/matter/espectre/matter_frontend.cpp"
)

set(ESPECTRE_FRONTEND_NATIVE_SOURCES
    ${ESPECTRE_FRONTEND_COMMON_SOURCES}
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/home_assistant_mqtt_frontend.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/native_command_bindings.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/native_direct_frontend.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/recovery_button_service.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/native_mqtt_frontend.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/native/espectre/native_frontend.cpp"
)

set(ESPECTRE_FRONTEND_NATIVE_ESP_IDF_SOURCES)

