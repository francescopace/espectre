# ESPHome compiles this directory's adapter sources in its generated `src`
# component. Register the canonical shared ESPectre sources as a separate,
# native ESP-IDF component through the SDK build definition.
set(ESPECTRE_SDK_ADDITIONAL_REQUIRES esp_http_server espressif__mdns esp_app_format)
include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_frontend_sources.cmake")
# Reject a generated Python schema paired with a different SDK contract.
set(_espectre_schema_digests "")
foreach(_espectre_schema_header IN ITEMS
        core/detector_types.h core/filter_config.h runtime/runtime_sensing_schema.h)
    file(SHA256 "${ESPECTRE_SDK_ROOT}/${_espectre_schema_header}" _espectre_schema_digest)
    string(APPEND _espectre_schema_digests "${_espectre_schema_digest}")
endforeach()
string(SHA256 _espectre_schema_hash "${_espectre_schema_digests}")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/sensing_schema.py" _espectre_schema_version
     REGEX "^SDK_SCHEMA_SHA256 = ")
if(NOT _espectre_schema_version STREQUAL "SDK_SCHEMA_SHA256 = \"${_espectre_schema_hash}\"")
    message(FATAL_ERROR "ESPHome sensing schema does not match this SDK. Regenerate sensing_schema.py for the selected SDK.")
endif()

include("${ESPECTRE_SDK_ROOT}/CMakeLists.txt")

# ESPHome copies adapter sources from this directory only. Compile the shared
# frontend helpers here, without adding them to the standalone SDK source groups.
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    target_sources(${COMPONENT_LIB} PRIVATE ${ESPECTRE_FRONTEND_COMMON_SOURCES})
    target_include_directories(${COMPONENT_LIB} PUBLIC "${ESPECTRE_FRONTEND_ROOT}")
    idf_component_optional_requires(PRIVATE esp_tinyusb espressif__esp_tinyusb)

    if(CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE)
        target_link_options(
            ${COMPONENT_LIB}
            INTERFACE
                "LINKER:--wrap=esp_usb_console_available_for_read"
                "LINKER:--wrap=esp_usb_console_read_buf"
                "LINKER:--wrap=esp_usb_console_write_buf"
        )
    endif()
endif()
