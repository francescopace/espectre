include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_sources.cmake")

idf_component_register(
    SRCS
        ${ESPECTRE_FRONTEND_STREAMER_SOURCES}
        ${ESPECTRE_CORE_SOURCES}
        ${ESPECTRE_RUNTIME_STREAMER_SOURCES}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
    REQUIRES
        esp_wifi
        esp_event
        esp_netif
        esp_timer
        mdns
        nvs_flash
        lwip
        driver
        freertos
        heap
        log
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ESPECTRE_ENABLE_STREAM_RUNTIME=1)
