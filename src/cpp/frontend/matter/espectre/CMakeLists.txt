include("${CMAKE_CURRENT_LIST_DIR}/../../espectre_frontend_sources.cmake")

idf_component_register(
    SRCS
        ${ESPECTRE_FRONTEND_MATTER_SOURCES}
        ${ESPECTRE_CORE_SOURCES}
        ${ESPECTRE_RUNTIME_ESP_IDF_SOURCES}
        ${ESPECTRE_FRONTEND_IMPROV_SOURCES}
        ${ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
        "${ESPECTRE_FRONTEND_ROOT}"
    REQUIRES
        esp_wifi
        esp_event
        esp_netif
        esp_timer
        nvs_flash
        lwip
        driver
        freertos
        heap
        log
        improv__improv
        mbedtls
        mdns
        esp_http_server
    PRIV_REQUIRES
        espectre_config
)

idf_component_get_property(espectre_mdns_dir espressif__mdns COMPONENT_DIR)
target_include_directories(${COMPONENT_LIB} PRIVATE "${espectre_mdns_dir}/private_include")
target_link_options(${COMPONENT_LIB} INTERFACE "LINKER:--wrap=mdns_priv_receive_action")
