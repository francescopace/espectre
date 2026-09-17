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
    mqtt
    esp_http_server
    nvs_flash
    lwip
    freertos
    heap
    mbedtls
    ${ESPECTRE_SDK_ADDITIONAL_REQUIRES}
)

if(ESPECTRE_SDK_EXCLUDED_REQUIRES)
    list(REMOVE_ITEM espectre_sdk_requires ${ESPECTRE_SDK_EXCLUDED_REQUIRES})
endif()

# ESP-IDF expands component requirements before menuconfig values are
# available. Declare the built-in dependencies needed by the public headers
# here. The manifest resolves external stacks through Kconfig conditions;
# the source switches below keep unused service code out of the firmware.

if(CONFIG_ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES})
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_MQTT)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES})
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_PROVISIONING)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES})
endif()

if(CONFIG_ESPECTRE_SDK_ENABLE_DIRECT)
    list(APPEND espectre_sdk_sources ${ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES})
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

if(CONFIG_ESPECTRE_SDK_ENABLE_DIRECT)
    idf_component_get_property(espectre_mdns_dir espressif__mdns COMPONENT_DIR)
    target_include_directories(${COMPONENT_LIB} PRIVATE "${espectre_mdns_dir}/private_include")
    target_link_options(${COMPONENT_LIB} INTERFACE "LINKER:--wrap=mdns_priv_receive_action")
endif()
