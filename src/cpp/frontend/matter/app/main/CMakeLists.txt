idf_component_register(
    SRCS
        "app_main.cpp"
        "matter_bindings_esp_matter.cpp"
        "matter_commissioning_data.cpp"
    INCLUDE_DIRS
        "."
        "../../espectre"
        "../../../../core"
        "../../../../runtime"
        "../../../../runtime/esp_idf"
    REQUIRES
        espectre
        bootloader_support
        esp_matter
        esp_partition
        nvs_flash
)

target_compile_options(${COMPONENT_LIB} PRIVATE "-DCHIP_HAVE_CONFIG_H")
