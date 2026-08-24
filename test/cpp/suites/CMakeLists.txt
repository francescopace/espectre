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
    )
    espectre_apply_coverage("${target_name}")
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

add_espectre_test(test_runtime_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_helpers.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_runtime_frontend_controller
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_frontend_controller.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_sdk_surface
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_sdk_surface.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_runtime_detector_switch
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_runtime_detector_switch.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_wifi_lifecycle
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_lifecycle.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_wifi_lifecycle_dual_band
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_wifi_lifecycle_dual_band.cpp"
    LIBRARIES espectre_runtime_dual_band_testlib
)

add_espectre_test(test_mdns_discovery_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_mdns_discovery_service.cpp"
    LIBRARIES espectre_runtime_testlib espectre_native_mdns_alias_testlib
)

add_espectre_test(test_direct_websocket_service
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_direct_websocket_service.cpp"
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

add_espectre_test(test_device_config_store
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_device_config_store.cpp"
    LIBRARIES espectre_runtime_testlib
)

add_espectre_test(test_espectre_protocol
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/test_espectre_protocol.cpp"
    LIBRARIES espectre_runtime_testlib
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

add_espectre_test(test_sensor_publisher
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_sensor_publisher.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_frontend_controls
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_frontend_controls.cpp"
    LIBRARIES espectre_frontend_esphome_testlib
)

add_espectre_test(test_native_frontend
    "${CMAKE_CURRENT_SOURCE_DIR}/frontend/test_native_frontend.cpp"
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
        espectre_runtime_testlib
)
