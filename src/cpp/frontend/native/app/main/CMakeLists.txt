idf_component_register(
    SRCS
        "app_main.cpp"
    INCLUDE_DIRS
        "."
        "../../espectre"
        "../../../../core"
        "../../../../runtime"
        "../../../../runtime/esp_idf"
    REQUIRES
        espectre
        bt
        esp_event
        esp_netif
        esp_wifi
        freertos
        lwip
        mqtt
        app_update
        esp_http_client
        esp_https_ota
        esp-tls
        mbedtls
        nvs_flash
)

if("$ENV{NATIVE_OTA_CHANNEL}" STREQUAL "preview")
    target_compile_definitions(
        ${COMPONENT_LIB}
        PRIVATE ESPECTRE_OTA_PREVIEW_BUILD=1
    )
elseif("$ENV{NATIVE_OTA_CHANNEL}" STREQUAL "develop")
    target_compile_definitions(
        ${COMPONENT_LIB}
        PRIVATE ESPECTRE_OTA_DEVELOP_BUILD=1
    )
endif()
