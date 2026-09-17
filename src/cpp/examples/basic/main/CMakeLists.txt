idf_component_register(
    SRCS "app_main.cpp" "optional_services.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_event esp_netif esp_wifi esp_timer nvs_flash log
)
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
