idf_component_register(SRCS "espectre.c"
                      INCLUDE_DIRS "."
                      REQUIRES esp_wifi mqtt nvs_flash esp_timer)
