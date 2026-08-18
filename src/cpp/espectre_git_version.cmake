# ESPectre git-describe version
#
# Resolves ESPECTRE_GIT_VERSION from numeric git tags. Rolling GitHub tags such
# as `snapshot` and `snapshot-dev` are ignored. Keep the git arguments in sync with
# `.github/scripts/detect_git_version.py`.
#
# Git history is required for first-party builds. ESPHome GitHub clones are
# shallow and have no numeric tags, so they pass the identity through
# `ESPECTRE_GIT_VERSION` in the environment. A stamped SDK bundle already
# contains the version macros in `runtime/espectre_sdk_version.h` and does not
# need `.git`. There is no numeric fallback.

if(NOT DEFINED ESPECTRE_GIT_VERSION OR ESPECTRE_GIT_VERSION STREQUAL "")
    if(DEFINED ENV{ESPECTRE_GIT_VERSION} AND NOT "$ENV{ESPECTRE_GIT_VERSION}" STREQUAL "")
        set(ESPECTRE_GIT_VERSION "$ENV{ESPECTRE_GIT_VERSION}")
        set(ESPECTRE_GIT_VERSION_FROM_GIT FALSE)
        set(ESPECTRE_GIT_VERSION_STAMPED FALSE)
    endif()
endif()

if(NOT DEFINED ESPECTRE_GIT_VERSION OR ESPECTRE_GIT_VERSION STREQUAL "")
    get_filename_component(_espectre_git_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    set(_espectre_git_describe_dirs "${_espectre_git_root}")
    if(DEFINED ENV{GITHUB_WORKSPACE} AND NOT "$ENV{GITHUB_WORKSPACE}" STREQUAL "")
        get_filename_component(_espectre_github_workspace "$ENV{GITHUB_WORKSPACE}" ABSOLUTE)
        if(NOT _espectre_github_workspace STREQUAL _espectre_git_root)
            list(APPEND _espectre_git_describe_dirs "${_espectre_github_workspace}")
        endif()
        unset(_espectre_github_workspace)
    endif()

    set(ESPECTRE_GIT_DESCRIBE_ERROR "")
    set(_espectre_described "")
    foreach(_espectre_git_dir IN LISTS _espectre_git_describe_dirs)
        if(_espectre_described STREQUAL "")
            execute_process(
                COMMAND git describe --tags --match "[0-9]*" --abbrev=7
                WORKING_DIRECTORY "${_espectre_git_dir}"
                OUTPUT_VARIABLE _espectre_described
                ERROR_VARIABLE ESPECTRE_GIT_DESCRIBE_ERROR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE ESPECTRE_GIT_DESCRIBE_RESULT
            )
            if(NOT ESPECTRE_GIT_DESCRIBE_RESULT EQUAL 0 OR _espectre_described STREQUAL "")
                set(_espectre_described "")
            endif()
        endif()
    endforeach()

    if(NOT _espectre_described STREQUAL "")
        set(ESPECTRE_GIT_VERSION "${_espectre_described}")
        set(ESPECTRE_GIT_VERSION_FROM_GIT TRUE)
        set(ESPECTRE_GIT_VERSION_STAMPED FALSE)
    else()
        set(ESPECTRE_GIT_VERSION "")
        file(READ "${CMAKE_CURRENT_LIST_DIR}/runtime/espectre_sdk_version.h" _espectre_version_header)
        if(_espectre_version_header MATCHES "#define[ \t]+ESPECTRE_SDK_VERSION_STRING[ \t]+\"([^\"]+)\"")
            set(ESPECTRE_GIT_VERSION "${CMAKE_MATCH_1}")
            set(ESPECTRE_GIT_VERSION_FROM_GIT FALSE)
            set(ESPECTRE_GIT_VERSION_STAMPED TRUE)
        else()
            message(FATAL_ERROR
                "Unable to resolve ESPectre version from git describe, and the SDK "
                "header is not stamped. Fetch numeric tags (`git fetch --tags`), pass "
                "-DESPECTRE_GIT_VERSION=..., or set ESPECTRE_GIT_VERSION in the "
                "environment. ${ESPECTRE_GIT_DESCRIBE_ERROR}"
            )
        endif()
        unset(_espectre_version_header)
    endif()
    unset(_espectre_git_root)
    unset(_espectre_git_describe_dirs)
    unset(_espectre_git_dir)
    unset(_espectre_described)
    unset(ESPECTRE_GIT_DESCRIBE_ERROR)
    unset(ESPECTRE_GIT_DESCRIBE_RESULT)
endif()

if(NOT ESPECTRE_GIT_VERSION)
    message(FATAL_ERROR "ESPECTRE_GIT_VERSION is empty")
endif()
if(ESPECTRE_GIT_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(ESPECTRE_SDK_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(ESPECTRE_SDK_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(ESPECTRE_SDK_VERSION_PATCH "${CMAKE_MATCH_3}")
else()
    message(FATAL_ERROR "ESPECTRE_GIT_VERSION ${ESPECTRE_GIT_VERSION} is not MAJOR.MINOR.PATCH")
endif()
