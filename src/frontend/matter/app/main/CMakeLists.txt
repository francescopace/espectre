idf_component_register(
    SRCS
        "app_main.cpp"
        "matter_bindings_esp_matter.cpp"
    INCLUDE_DIRS
        "."
        "../../espectre"
        "../../../../core"
        "../../../../runtime"
        "../../../../runtime/esp_idf"
    REQUIRES
        espectre
        esp_matter
        nvs_flash
)

target_compile_options(${COMPONENT_LIB} PRIVATE "-DCHIP_HAVE_CONFIG_H")
