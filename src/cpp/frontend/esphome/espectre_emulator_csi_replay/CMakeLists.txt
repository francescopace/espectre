# The emulator CSI replay as an ESP-IDF component of its own.
#
# ESPHome builds the `espectre` component from its own tree and the SDK from
# the registry, and neither carries replay code. CI adds this directory to the
# published ESPHome images through `esp32.framework.components`, so the same
# replay that Native links also runs there.
include("${CMAKE_CURRENT_LIST_DIR}/../../espectre_frontend_sources.cmake")

# Nothing references the wrappers until the SDK's wrapped calls are resolved,
# and by then the linker is past this archive, so link all of it.
set(_espectre_replay_sources "")
set(_espectre_replay_whole_archive "")
if(CONFIG_ESPECTRE_EMULATOR_CSI_REPLAY)
    set(_espectre_replay_sources ${ESPECTRE_FRONTEND_EMULATOR_REPLAY_SOURCES})
    set(_espectre_replay_whole_archive WHOLE_ARCHIVE)
endif()

idf_component_register(
    SRCS
        ${_espectre_replay_sources}
    INCLUDE_DIRS
        "${ESPECTRE_FRONTEND_ROOT}"
    PRIV_REQUIRES
        esp_wifi
        esp_netif
        esp_timer
        esp_driver_uart
        freertos
        log
    ${_espectre_replay_whole_archive}
)

if(CONFIG_ESPECTRE_EMULATOR_CSI_REPLAY)
    # The SDK is `espectre` from a local path and `francescopace__espectre` from the registry.
    idf_component_optional_requires(PRIVATE espectre francescopace__espectre)
    target_link_options(${COMPONENT_LIB} INTERFACE ${ESPECTRE_FRONTEND_EMULATOR_REPLAY_LINK_OPTIONS})
endif()
