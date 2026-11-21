set(frontend_dir "${CMAKE_CURRENT_LIST_DIR}/../../..")
idf_component_register(
    SRCS "${frontend_dir}/primary_console.cpp"
    INCLUDE_DIRS "${frontend_dir}"
    PRIV_REQUIRES espressif__esp_tinyusb
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE=1)
target_link_options(${COMPONENT_LIB} INTERFACE
    "LINKER:--wrap=esp_usb_console_available_for_read"
    "LINKER:--wrap=esp_usb_console_read_buf"
    "LINKER:--wrap=esp_usb_console_write_buf"
)
