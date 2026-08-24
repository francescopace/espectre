include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_sources.cmake")

set(espectre_streamer_sources
    ${ESPECTRE_FRONTEND_STREAMER_SOURCES}
    ${ESPECTRE_CORE_SOURCES}
    ${ESPECTRE_RUNTIME_STREAMER_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES}
)
list(REMOVE_DUPLICATES espectre_streamer_sources)

idf_component_register(
    SRCS
        ${espectre_streamer_sources}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
    REQUIRES
        esp_wifi
        esp_event
        esp_netif
        esp_timer
        mdns
        esp_http_server
        nvs_flash
        lwip
        driver
        freertos
        heap
        log
        mbedtls
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ESPECTRE_ENABLE_STREAM_RUNTIME=1)
