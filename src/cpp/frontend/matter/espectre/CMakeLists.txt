include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_sources.cmake")

idf_component_register(
    SRCS
        ${ESPECTRE_FRONTEND_MATTER_SOURCES}
        ${ESPECTRE_CORE_SOURCES}
        ${ESPECTRE_RUNTIME_ESP_IDF_SOURCES}
        ${ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
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
        mbedtls
        mdns
        esp_http_server
)

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

idf_component_get_property(espectre_mdns_dir espressif__mdns COMPONENT_DIR)
target_include_directories(${COMPONENT_LIB} PRIVATE "${espectre_mdns_dir}/private_include")
target_link_options(${COMPONENT_LIB} INTERFACE "LINKER:--wrap=mdns_priv_receive_action")
