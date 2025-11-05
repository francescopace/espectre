idf_component_register(SRCS "espectre.c" "calibration.c" "nvs_storage.c" "filters.c" "wavelet.c" "csi_processor.c" "detection_engine.c" "mqtt_handler.c" "mqtt_commands.c" "wifi_manager.c" "statistics.c" "config_manager.c" "traffic_generator.c"
                      INCLUDE_DIRS "."
                      REQUIRES esp_wifi mqtt nvs_flash esp_timer)
