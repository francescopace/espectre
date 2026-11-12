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

# Enabled by CONFIG_ESPECTRE_EMULATOR_CSI_REPLAY; inert outside esp-emu.
set(ESPECTRE_FRONTEND_EMULATOR_REPLAY_SOURCES
    "${ESPECTRE_FRONTEND_ROOT}/csi_replay_frame.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/emulator_csi_replay.cpp"
)

set(ESPECTRE_FRONTEND_EMULATOR_REPLAY_LINK_OPTIONS
    "LINKER:--wrap=esp_wifi_set_csi_config"
    "LINKER:--wrap=esp_wifi_set_csi_rx_cb"
    "LINKER:--wrap=esp_wifi_set_csi"
)

# Link the emulator CSI replay into a frontend component when Kconfig enables it.
function(espectre_link_emulator_csi_replay component_lib)
    if(CONFIG_ESPECTRE_EMULATOR_CSI_REPLAY)
        target_sources(${component_lib} PRIVATE ${ESPECTRE_FRONTEND_EMULATOR_REPLAY_SOURCES})
        target_link_options(${component_lib} INTERFACE ${ESPECTRE_FRONTEND_EMULATOR_REPLAY_LINK_OPTIONS})
    endif()
endfunction()

set(ESPECTRE_FRONTEND_COMMON_SOURCES
    "${ESPECTRE_FRONTEND_ROOT}/frontend_firmware_version.cpp"
    "${ESPECTRE_FRONTEND_ROOT}/primary_console.cpp"
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

