if(NOT DEFINED ESPECTRE_CPP_ROOT)
    get_filename_component(ESPECTRE_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
endif()

set(ESPECTRE_CORE_SOURCES
    "${ESPECTRE_CPP_ROOT}/core/espectre_log.cpp"
    "${ESPECTRE_CPP_ROOT}/core/base_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/temporal_csi_sampler.cpp"
    "${ESPECTRE_CPP_ROOT}/core/lightweight_detector.cpp"
    "${ESPECTRE_CPP_ROOT}/core/filters.cpp"
    "${ESPECTRE_CPP_ROOT}/core/filtered_turbulence_ring.cpp"
    "${ESPECTRE_CPP_ROOT}/core/high_accuracy_detector.cpp"
)

set(ESPECTRE_RUNTIME_COMMON_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/csi_traffic_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/periodic_sensing_status_logger.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/espectre_protocol.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/direct_http_protocol.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/peer_discovery.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/protocol_json.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_config_utils.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_diagnostics.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_diagnostics_protocol.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_time.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/frontend_command_engine.cpp"
)

set(ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_bootstrap_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_ha_mqtt_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_mqtt_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/frontend_sysinfo_helpers.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS
    "LINKER:--wrap=esp_netif_receive"
    "LINKER:--wrap=esp_netif_transmit_wrap"
)

set(ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/network_traffic.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/sta_socket_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_tx_rate.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_PLATFORM_SOURCES
    ${ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES}
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/device_identity.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/direct_wifi_snapshot_esp_idf.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_capture_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_pipeline.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_payload_normalizer.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_platform_config.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/esp_idf_runtime.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/esp_idf_runtime_base.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/nvs_helpers.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_frontend_controller.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_detector_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_motion_hits_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_traffic_mode_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_performance_diagnostics.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_sensing_kconfig.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/standalone_wifi_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_frame_identity.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/udp_datagram_socket_esp_idf.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/udp_listener.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_lifecycle.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_SOURCES
    ${ESPECTRE_RUNTIME_COMMON_SOURCES}
    ${ESPECTRE_RUNTIME_ESP_IDF_PLATFORM_SOURCES}
)

set(ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/device_config_store.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_provisioning_service.cpp"
)

set(ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/mqtt_transport_esp_idf.cpp"
)

# The bootstrap responder wraps the mDNS receive path and uses private mDNS
# headers. Source-list targets link with these options and add the
# `private_include` directory of the `espressif__mdns` component.
set(ESPECTRE_RUNTIME_ESP_IDF_DIRECT_LINK_OPTIONS
    "LINKER:--wrap=mdns_priv_receive_action"
)

set(ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/direct_http_service_esp_idf.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/mdns_bootstrap_responder.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/mdns_discovery_service.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/peer_discovery_service_esp_idf.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/raw_csi_session_controller.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/runtime_direct_http_bridge.cpp"
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_bssid_pin_service.cpp"
)

# SDK root, the only include directory. It exposes the facades and
# layer-prefixed includes such as "runtime/runtime_config.h". SDK sources
# reach other layers only through this root, so the layer directories stay off
# the consumer's search path and generic basenames such as `utils.h` cannot
# collide with the consuming project in either direction.
set(ESPECTRE_SDK_ROOT_INCLUDE_DIRS
    "${ESPECTRE_CPP_ROOT}"
)

set(ESPECTRE_SHARED_INCLUDE_DIRS
    ${ESPECTRE_SDK_ROOT_INCLUDE_DIRS}
)
