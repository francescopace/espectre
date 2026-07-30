include("${CMAKE_CURRENT_LIST_DIR}/espectre_sources.cmake")

set(espectre_sdk_sources
    ${ESPECTRE_CORE_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_SOURCES}
)

set(espectre_sdk_requires
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
)

idf_component_register(
    SRCS
        ${espectre_sdk_sources}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
    REQUIRES
        ${espectre_sdk_requires}
)
