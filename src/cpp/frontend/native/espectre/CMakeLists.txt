include("${CMAKE_CURRENT_LIST_DIR}/../../../espectre_sources.cmake")

set(espectre_native_sources
    ${ESPECTRE_FRONTEND_NATIVE_SOURCES}
    ${ESPECTRE_CORE_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES}
)

set(espectre_native_requires
    esp_wifi
    esp_event
    esp_netif
    esp_timer
    mqtt
    app_update
    esp_http_client
    esp_https_ota
    esp-tls
    mbedtls
    nvs_flash
    lwip
    driver
    freertos
    heap
    log
)

if(CONFIG_BT_ENABLED)
    list(APPEND espectre_native_sources ${ESPECTRE_RUNTIME_ESP_IDF_BLE_SOURCES})
    list(APPEND espectre_native_requires bt)
endif()

idf_component_register(
    SRCS
        ${espectre_native_sources}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
    REQUIRES
        ${espectre_native_requires}
)
