idf_component_register(SRCS "espectre.c" "nvs_storage.c" "filters.c" "wavelet.c" "csi_processor.c" "csi_features.c" "mqtt_handler.c" "mqtt_commands.c" "config_manager.c" "traffic_generator.c" "validation.c" "nbvi_calibrator.c"
                      INCLUDE_DIRS "."
                      REQUIRES esp_wifi mqtt nvs_flash esp_timer esp_netif esp_event)
