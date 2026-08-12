# ESPHome compiles this directory's adapter sources in its generated `src`
# component. Register the canonical shared ESPectre sources as a separate,
# native ESP-IDF component through the SDK build definition.
include("${CMAKE_CURRENT_LIST_DIR}/../../../../CMakeLists.txt")
