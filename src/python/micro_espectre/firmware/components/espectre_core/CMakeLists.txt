# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.

# Core-only ESP-IDF component for the Micro-ESPectre firmware. Keep the source
# registry in the C++ SDK so the native binding cannot drift onto a copied subset.
if(NOT ESPECTRE_CORE_SDK_ROOT AND DEFINED ENV{ESPECTRE_CORE_SDK_ROOT})
    set(ESPECTRE_CORE_SDK_ROOT "$ENV{ESPECTRE_CORE_SDK_ROOT}")
endif()

if(NOT ESPECTRE_CORE_SDK_ROOT)
    message(FATAL_ERROR "ESPECTRE_CORE_SDK_ROOT must point to the ESPectre C++ SDK")
endif()

include("${ESPECTRE_CORE_SDK_ROOT}/espectre_sources.cmake")

idf_component_register(
    SRCS
        ${ESPECTRE_CORE_SOURCES}
    INCLUDE_DIRS
        ${ESPECTRE_SDK_ROOT_INCLUDE_DIRS}
        ${ESPECTRE_CORE_INCLUDE_DIRS}
    REQUIRES
        log
)

include("${ESPECTRE_CORE_SDK_ROOT}/espectre_git_version.cmake")
if(ESPECTRE_GIT_VERSION AND NOT ESPECTRE_GIT_VERSION_STAMPED)
    target_compile_definitions(
        ${COMPONENT_LIB}
        PUBLIC
            ESPECTRE_SDK_VERSION_STRING="${ESPECTRE_GIT_VERSION}"
            ESPECTRE_SDK_VERSION_MAJOR=${ESPECTRE_SDK_VERSION_MAJOR}
            ESPECTRE_SDK_VERSION_MINOR=${ESPECTRE_SDK_VERSION_MINOR}
            ESPECTRE_SDK_VERSION_PATCH=${ESPECTRE_SDK_VERSION_PATCH}
    )
endif()
