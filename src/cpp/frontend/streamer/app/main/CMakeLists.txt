include("${CMAKE_CURRENT_LIST_DIR}/../../../../espectre_sources.cmake")

set(STREAMER_MAIN_SRCS
    "app_main.cpp"
)
set(STREAMER_MAIN_REQUIRES
    espectre
)
set(STREAMER_MAIN_INCLUDE_DIRS
    "."
    "../../espectre"
    "../../../../core"
    "../../../../runtime"
    "../../../../runtime/esp_idf"
    "../../../../runtime/esp_idf/protocol"
)

idf_component_register(
    SRCS
        ${STREAMER_MAIN_SRCS}
    INCLUDE_DIRS
        ${STREAMER_MAIN_INCLUDE_DIRS}
    REQUIRES
        ${STREAMER_MAIN_REQUIRES}
)
