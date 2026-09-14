function(add_espectre_test target_name source_path)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_executable("${target_name}" "${source_path}")
    target_link_libraries("${target_name}"
        PRIVATE
            espectre_test_support
            ${ARG_LIBRARIES}
    )
    add_test(NAME "${target_name}" COMMAND "${target_name}")
    set_tests_properties("${target_name}" PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        LABELS "unit"
    )
    espectre_apply_coverage("${target_name}")
endfunction()

function(add_espectre_dataset_cases target_name gate_name)
    set(shared_chips ESP32 C3 S2 S3 C5 C6)
    set_tests_properties("${target_name}" PROPERTIES
        LABELS "integration;performance;aggregate;dataset:${gate_name}"
    )
    foreach(chip IN LISTS shared_chips)
        string(TOLOWER "${chip}" chip_id)
        set(test_name "${target_name}.${gate_name}.${chip_id}")
        add_test(
            NAME "${test_name}"
            COMMAND "${target_name}" --gate "${gate_name}" --chip "${chip}"
        )
        set_tests_properties("${test_name}" PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            SKIP_RETURN_CODE 77
            LABELS "integration;performance;chip:${chip_id};dataset:${gate_name}"
        )
    endforeach()
endfunction()

add_espectre_test(test_utils
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test_utils.cpp"
    LIBRARIES espectre_core_testlib
)

add_espectre_test(test_core_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test_core_helpers.cpp"
    LIBRARIES espectre_core_testlib
)

add_espectre_test(test_hampel_filter
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test_hampel_filter.cpp"
    LIBRARIES espectre_core_testlib
)

add_espectre_test(test_lightweight_detector
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test_lightweight_detector.cpp"
    LIBRARIES espectre_core_testlib
)

add_espectre_test(test_high_accuracy_detector
    "${CMAKE_CURRENT_SOURCE_DIR}/core/test_high_accuracy_detector.cpp"
    LIBRARIES espectre_core_testlib
)

add_espectre_test(test_traffic_generator
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_traffic_generator.cpp"
    LIBRARIES espectre_runtime_testlib
)

# Verify the shared raw injection default on another legacy-API target.
add_espectre_test(test_traffic_generator_s2
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_traffic_generator.cpp"
    LIBRARIES espectre_runtime_testlib
)
target_sources(test_traffic_generator_s2 PRIVATE
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
)
target_compile_definitions(test_traffic_generator_s2 PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32S2=1
)

# Raw injection remains OFDM when station TX A-MPDU is enabled.
add_espectre_test(test_traffic_generator_ampdu
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_traffic_generator.cpp"
    LIBRARIES espectre_runtime_testlib
)
target_sources(test_traffic_generator_ampdu PRIVATE
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
)
target_compile_definitions(test_traffic_generator_ampdu PRIVATE
    CONFIG_ESP_WIFI_AMPDU_TX_ENABLED=1
)

foreach(rate IN ITEMS 12 24)
    add_espectre_test(test_traffic_generator_${rate}m
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_traffic_generator.cpp"
        LIBRARIES espectre_runtime_testlib
    )
    target_sources(test_traffic_generator_${rate}m PRIVATE
        "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
    )
    target_compile_definitions(test_traffic_generator_${rate}m PRIVATE
        CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=${rate}
        ESPECTRE_TEST_EXPECTED_TX_RATE=WIFI_PHY_RATE_${rate}M
    )
endforeach()

# Explicit choices override the shared default; C5 exercises the modern raw API.
foreach(variant IN ITEMS auto s2_auto c5_auto c5_fixed)
    add_espectre_test(test_traffic_generator_${variant}
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_traffic_generator.cpp"
        LIBRARIES espectre_runtime_testlib
    )
    target_sources(test_traffic_generator_${variant} PRIVATE
        "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/traffic_generator_manager.cpp"
    )
endforeach()
target_compile_definitions(test_traffic_generator_auto PRIVATE
    CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=0
)
target_compile_definitions(test_traffic_generator_s2_auto PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32S2=1
    CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=0
)
target_compile_definitions(test_traffic_generator_c5_auto PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32C5=1
    CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=0
)
target_compile_definitions(test_traffic_generator_c5_fixed PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32C5=1
    CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=24 ESPECTRE_TEST_EXPECTED_TX_RATE=WIFI_PHY_RATE_24M
)

add_espectre_test(test_runtime_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_helpers.cpp"
    LIBRARIES espectre_runtime_testlib
)

# Exercise the HE metadata branch in the same capture admission test owner.
add_espectre_test(test_runtime_helpers_he
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_helpers.cpp"
    LIBRARIES espectre_runtime_testlib
)
target_sources(test_runtime_helpers_he PRIVATE
    "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/csi_capture_service.cpp"
)
target_compile_definitions(test_runtime_helpers_he PRIVATE CONFIG_SOC_WIFI_HE_SUPPORT=1)

add_espectre_test(test_frontend_sysinfo_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_frontend_sysinfo_helpers.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_periodic_sensing_status_logger
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_periodic_sensing_status_logger.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_device_identity
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_device_identity.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_ota_service_https
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_ota_service_https.cpp"
    LIBRARIES espectre_ota_https_testlib
)

add_espectre_test(test_runtime_frontend_controller
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_frontend_controller.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_sdk_surface
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_sdk_surface.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_runtime_sensing_invalid_kconfig
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_sensing_invalid_kconfig.cpp"
    LIBRARIES espectre_runtime_invalid_kconfig_testlib
)

add_espectre_test(test_runtime_detector_switch
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_detector_switch.cpp"
    LIBRARIES espectre_runtime_testlib
)

# Exercise C6 traffic eligibility through the same runtime and frontend paths.
add_espectre_test(test_runtime_detector_switch_c6
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_detector_switch.cpp"
    LIBRARIES espectre_runtime_testlib
)
target_sources(test_runtime_detector_switch_c6 PRIVATE
    "${ESPECTRE_CPP_ROOT}/runtime/runtime_config_utils.cpp"
)
target_compile_definitions(test_runtime_detector_switch_c6 PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32C6=1
)

add_espectre_test(test_wifi_lifecycle
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_lifecycle.cpp"
    LIBRARIES espectre_runtime_testlib
)

# Station rates belong to Wi-Fi startup, independently of traffic generation.
foreach(variant IN ITEMS s2 auto ampdu 12m 24m)
    add_espectre_test(test_wifi_lifecycle_${variant}
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_lifecycle.cpp"
        LIBRARIES espectre_runtime_testlib
    )
    target_sources(test_wifi_lifecycle_${variant} PRIVATE
        "${ESPECTRE_CPP_ROOT}/runtime/esp_idf/wifi_lifecycle.cpp"
    )
endforeach()
target_compile_definitions(test_wifi_lifecycle_s2 PRIVATE
    CONFIG_IDF_TARGET_ESP32=0 CONFIG_IDF_TARGET_ESP32S2=1
)
target_compile_definitions(test_wifi_lifecycle_auto PRIVATE
    CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=0
)
target_compile_definitions(test_wifi_lifecycle_ampdu PRIVATE
    CONFIG_ESP_WIFI_AMPDU_TX_ENABLED=1
)
foreach(rate IN ITEMS 12 24)
    target_compile_definitions(test_wifi_lifecycle_${rate}m PRIVATE
        CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS=${rate}
    )
endforeach()

# Exercise the MicroPython Wi-Fi bridge against the same station-rate policy.
foreach(target IN ITEMS test_wifi_lifecycle test_wifi_lifecycle_s2 test_wifi_lifecycle_auto
                        test_wifi_lifecycle_ampdu test_wifi_lifecycle_12m test_wifi_lifecycle_24m)
    target_sources(${target} PRIVATE
        "${ESPECTRE_REPO_ROOT}/src/python/micro_espectre/firmware/native_components/native_wifi.cpp"
    )
    target_include_directories(${target} PRIVATE
        "${ESPECTRE_REPO_ROOT}/src/python/micro_espectre/firmware/native_components"
    )
endforeach()

add_espectre_test(test_wifi_lifecycle_dual_band
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_lifecycle_dual_band.cpp"
    LIBRARIES espectre_runtime_dual_band_testlib
)

add_espectre_test(test_mdns_discovery_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_mdns_discovery_service.cpp"
    LIBRARIES espectre_runtime_testlib espectre_native_mdns_bootstrap_testlib
)

add_espectre_test(test_direct_http_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_direct_http_service.cpp"
    LIBRARIES espectre_direct_service_testlib
)

add_espectre_test(test_mqtt_transport_esp_idf
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_mqtt_transport_esp_idf.cpp"
    LIBRARIES espectre_mqtt_transport_testlib
)

add_espectre_test(test_pending_event
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_pending_event.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_wifi_provisioning_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_provisioning_service.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_wifi_bssid_pin_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_bssid_pin_service.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_device_config_store
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_device_config_store.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_espectre_protocol
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_espectre_protocol.cpp"
    LIBRARIES espectre_frontend_ota_protocol_testlib
)

add_espectre_test(test_csi_pipeline
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_csi_pipeline.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_csi_frame_identity
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_csi_frame_identity.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_csi_traffic_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_csi_traffic_service.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_udp_listener
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_udp_listener.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_motion_detection
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_motion_detection.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_long_recordings
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_long_recordings.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_low_rssi
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_low_rssi.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_empty_rooms
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_empty_rooms.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_packet_rate_adaptation
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_packet_rate_adaptation.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_dataset_cases(test_motion_detection normal)
add_espectre_dataset_cases(test_motion_detection reserved)
add_espectre_dataset_cases(test_long_recordings long)
add_espectre_dataset_cases(test_low_rssi weak)
add_espectre_dataset_cases(test_empty_rooms empty)
add_espectre_dataset_cases(test_packet_rate_adaptation packet_rate)

add_espectre_test(test_sensor_publisher
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_sensor_publisher.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_frontend_controls
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_frontend_controls.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_native_frontend_lifecycle
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_native_frontend_lifecycle.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_native_direct_frontend
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_native_direct_frontend.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_native_mqtt_frontend
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_native_mqtt_frontend.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_home_assistant_mqtt_frontend
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_home_assistant_mqtt_frontend.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_native_frontend_ota
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_native_frontend_ota.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_recovery_button_service
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_recovery_button_service.cpp"
    LIBRARIES espectre_frontend_native_testlib
)

add_espectre_test(test_matter_frontend
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_matter_frontend.cpp"
    LIBRARIES espectre_frontend_matter_testlib
)

# This is intentionally a probe rather than a CTest case: the Python owner
# suite executes it and compares its JSON with the independent MicroPython
# registry.
add_executable(espectre_capabilities_probe
    "${ESPECTRE_TEST_ROOT}/support/espectre_capabilities_probe.cpp"
)
target_link_libraries(espectre_capabilities_probe
    PRIVATE
        espectre_frontend_ota_protocol_testlib
)
espectre_apply_coverage(espectre_capabilities_probe)
