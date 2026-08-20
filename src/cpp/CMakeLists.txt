# ESPectre SDK - ESP-IDF component
#
# Registers the shared core and ESP-IDF runtime as a vendored component. Drop
# this tree into your project's `components/` directory and add `espectre` to
# your own component's REQUIRES.
#
# The sensing runtime is always built. The optional capability groups are
# opt-in through menuconfig ("ESPectre SDK") so a minimal integration does not
# pay for transports it never calls. Consumers that include
# `espectre_sources.cmake` directly select the same groups by listing the
# matching ESPECTRE_RUNTIME_ESP_IDF_*_SOURCES variables instead.

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
    mbedtls
)

if(CONFIG_ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES})
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_MQTT)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES})
    list(APPEND espectre_sdk_requires mqtt)
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_BLE)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_BLE_SOURCES})
    list(APPEND espectre_sdk_requires bt)
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_PROVISIONING)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES})
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_OTA)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES})
    list(APPEND espectre_sdk_requires app_update esp_http_client esp_https_ota esp-tls mbedtls)
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_STREAM_RUNTIME)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_STREAMER_FRONTEND_SUPPORT_SOURCES})
    list(APPEND espectre_sdk_requires mdns)
endif()

list(REMOVE_DUPLICATES espectre_sdk_sources)
list(REMOVE_DUPLICATES espectre_sdk_requires)

idf_component_register(
    SRCS
        ${espectre_sdk_sources}
    INCLUDE_DIRS
        "."
        ${ESPECTRE_SHARED_INCLUDE_DIRS}
    REQUIRES
        ${espectre_sdk_requires}
)

# ESPHome keeps CONFIG_LOG_DEFAULT_LEVEL at ERROR so Wi-Fi and lwIP debug stay
# compiled out. Compile INFO/DEBUG only in this component (4 == ESP_LOG_DEBUG)
# so detection status and telemetry can be enabled per-tag at runtime.
target_compile_definitions(${COMPONENT_LIB} PRIVATE LOG_LOCAL_LEVEL=4)

include("${CMAKE_CURRENT_LIST_DIR}/espectre_git_version.cmake")
if(ESPECTRE_GIT_VERSION AND NOT ESPECTRE_GIT_VERSION_STAMPED)
    target_compile_definitions(
        ${COMPONENT_LIB}
        PUBLIC
            ESPECTRE_SDK_VERSION_STRING="${ESPECTRE_GIT_VERSION}"
            ESPECTRE_SDK_VERSION_MAJOR=${ESPECTRE_SDK_VERSION_MAJOR}
            ESPECTRE_SDK_VERSION_MINOR=${ESPECTRE_SDK_VERSION_MINOR}
            ESPECTRE_SDK_VERSION_PATCH=${ESPECTRE_SDK_VERSION_PATCH}
    )
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_STREAM_RUNTIME)
    # make_stream_runtime() returns nullptr unless the profile is compiled in,
    # so the option has to reach the sources as well as the source list.
    target_compile_definitions(${COMPONENT_LIB} PRIVATE ESPECTRE_ENABLE_STREAM_RUNTIME=1)
endif()
